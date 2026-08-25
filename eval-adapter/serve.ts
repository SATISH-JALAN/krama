/**
 * Eval adapter -- a sidecar for BeaconBandhu/rag-local-eval-loop, and NOT part
 * of Krama's product surface. It exists because that suite's only supported
 * target is a pair of importable *Python* modules (its TARGET_INTERFACE.md;
 * the `eval.http_target` / EVAL_HTTP_CONFIG "HTTP service" branch described in
 * the task runbook does not exist in the repo -- verified against the single
 * commit on its only branch). Krama is TypeScript, so something has to bridge
 * the two, and this is the smallest honest bridge: it imports Krama's REAL
 * embedder and REAL synthesis path rather than reimplementing either, so the
 * numbers the suite prints are about Krama and not about a lookalike.
 *
 * server/ (the production API) is deliberately untouched -- no eval-only route
 * ships to prod. Run this separately, alongside or instead of `bun run dev`.
 *
 * Two routes, matching exactly what the suite asks a target for:
 *
 *   POST /embed     { texts: string[], kind: "query" | "passage" }
 *                -> { vectors: number[][], dim: number }
 *   POST /generate  { query: string, passages: [{text, source}], lang? }
 *                -> { text, grounded, generation_ms, model, gate }
 *
 * WHY `kind` MATTERS: e5 is asymmetric. The suite calls embed() on PASSAGE
 * chunks to build its own throwaway index (eval/index_build.py) and
 * embed_one() on QUERIES (eval/pipeline.py). Krama's runtime embed() only ever
 * saw queries, so it hardcoded "query: "; routing the suite's passages through
 * that would tank Recall@k for a reason that has nothing to do with Krama.
 * ghana/embed.ts now takes the kind explicitly -- see its own note.
 *
 * WHICH GUARDRAILS RUN HERE, and why it is not all of them:
 *   L0 input guard   -- RUNS. Cheap, query-only, corpus-independent.
 *   L1 safety guard  -- SKIPPED. Needs the safety exemplar embeddings from
 *                       Krama's own artifacts, and MSMARCO-XI queries are
 *                       ordinary web questions; it would be a no-op that
 *                       costs a full bootFromDisk().
 *   L2 OOD guard     -- SKIPPED ON PURPOSE, and this one is a real judgement
 *                       call worth stating out loud: it is a *corpus
 *                       membership* test (cosine to Krama's own passage
 *                       centroid). The suite's passages come from MSMARCO-XI,
 *                       not Krama's corpus, so running it would refuse
 *                       essentially every query and report a fake 0% lying
 *                       factor. Refusing everything is not a guardrail win.
 *   L3 rerank gate   -- RUNS, and is the one that carries the reliability
 *                       signal. A cross-encoder scoring (query, passage)
 *                       jointly is corpus-agnostic: it transfers to someone
 *                       else's passages in a way a centroid never can. This is
 *                       what has to catch Krama fabricating on an unanswerable
 *                       query, so `grounded` below is driven by it.
 */
import { Hono } from "hono";
import * as embed from "../server/ghana/embed";
import * as rerank from "../server/ghana/rerank";
import { checkL0 } from "../server/maun/input";
import { checkRerankScore } from "../server/maun/rerank_guard";
import { synthesizeAnswer } from "../server/llm/synthesize";
import { CircuitBreaker } from "../server/harness/breaker";
import { createGeminiProvider, GEMINI_MODEL_CHAIN } from "../server/llm/gemini";
import { createCerebrasProvider } from "../server/llm/cerebras";
import { loadThresholds } from "../server/index";
import type { LlmProvider } from "../server/llm/chain";
import type { RetryOptions } from "../server/harness/retry";

const RETRY_OPTS: RetryOptions = { maxAttempts: 2, baseDelayMs: 300, deadlineMs: 30_000 };

const providers: LlmProvider[] = [];
let breakers = new Map<string, CircuitBreaker>();
let rerankMinScore = 0;
let rerankerEnabled = false;
let modelLabel = "none";

async function boot() {
  await embed.boot(process.env.KRAMA_ONNX_DIR ?? "artifacts/onnx");

  const rerankerDir = process.env.KRAMA_RERANKER_DIR ?? "artifacts/onnx_reranker";
  try {
    await rerank.boot(rerankerDir, "model_int8.onnx");
    rerankerEnabled = true;
  } catch (e) {
    // Loud, not silent: without the reranker there is no relevance gate, so
    // every answer would come back grounded=true and the suite's reliability
    // check would report a flattering number that means nothing.
    console.warn(`[eval-adapter] reranker did NOT boot from ${rerankerDir}: ${e}`);
    console.warn("[eval-adapter] L3 gate is OFF -- reliability numbers will be meaningless. Fix this before trusting a run.");
  }

  rerankMinScore = loadThresholds(process.env.KRAMA_THRESHOLDS ?? "artifacts/thresholds.json").rerankMinScore;

  if (process.env.GEMINI_API_KEY) {
    for (const model of GEMINI_MODEL_CHAIN) {
      providers.push(createGeminiProvider(process.env.GEMINI_API_KEY, { model }));
    }
  }
  if (process.env.CEREBRAS_API_KEY) providers.push(createCerebrasProvider(process.env.CEREBRAS_API_KEY));
  breakers = new Map(providers.map((p) => [p.name, new CircuitBreaker()]));
  modelLabel = providers[0]?.name ?? "none";

  if (providers.length === 0) {
    console.warn("[eval-adapter] no GEMINI_API_KEY / CEREBRAS_API_KEY -- /generate will always decline.");
    console.warn("[eval-adapter] Retrieval + latency stay real; faithfulness/correctness will score a permanent refusal.");
  }
  console.log(
    `[eval-adapter] booted. reranker=${rerankerEnabled} minScore=${rerankMinScore} providers=[${providers.map((p) => p.name).join(", ")}]`,
  );
}

const app = new Hono();

app.get("/health", (c) => c.json({ ok: true, rerankerEnabled, providers: providers.map((p) => p.name) }));

app.post("/embed", async (c) => {
  const { texts, kind } = (await c.req.json()) as { texts: string[]; kind?: "query" | "passage" };
  if (!Array.isArray(texts)) return c.json({ error: "texts must be an array" }, 400);
  const k = kind === "passage" ? "passage" : "query";
  // Sequential, not Promise.all: one shared onnxruntime session with
  // intraOpNumThreads=2 (ghana/embed.ts) gains nothing from concurrent calls
  // and contends on the same graph. The suite already batches its own calls.
  const vectors: number[][] = [];
  for (const t of texts) vectors.push(Array.from(await embed.embed(t, k)));
  return c.json({ vectors, dim: vectors[0]?.length ?? 0 });
});

app.post("/generate", async (c) => {
  const body = (await c.req.json()) as {
    query: string;
    passages: { text: string; source: string }[];
    lang?: string;
  };
  const lang = body.lang ?? "en";
  const t0 = performance.now();

  const decline = (why: string, gate: string) =>
    c.json({ text: "", grounded: false, generation_ms: performance.now() - t0, model: modelLabel, gate, detail: why });

  const l0 = checkL0(body.query, lang);
  if (l0.refused) return decline(l0.detail ?? "L0 refused", "l0_input_guard");

  const passages = (body.passages ?? []).filter((p) => p?.text?.trim());
  if (passages.length === 0) return decline("no candidate passages supplied", "no_candidates");

  // L3 -- Krama's real cross-encoder relevance gate, replayed over the
  // suite's passages instead of Krama's own retrieval output.
  let best = passages[0]!;
  if (rerankerEnabled) {
    const scores = await rerank.scoreBatch(
      body.query,
      passages.map((p) => p.text),
    );
    let bestIdx = 0;
    for (let i = 1; i < scores.length; i++) if (scores[i]! > scores[bestIdx]!) bestIdx = i;
    best = passages[bestIdx]!;
    const l3 = checkRerankScore(scores[bestIdx]!, rerankMinScore);
    if (l3.refused) return decline(l3.detail ?? `rerank ${scores[bestIdx]} < ${rerankMinScore}`, "l3_rerank_guard");
  }

  if (providers.length === 0) return decline("no LLM provider configured", "no_provider");

  const result = await synthesizeAnswer(
    [{ chunkId: best.source, text: best.text }],
    body.query,
    lang,
    providers,
    breakers,
    RETRY_OPTS,
  );

  // null = every provider failed outright (network/quota). Distinct from the
  // LLM successfully judging the passage ungrounded, which comes back as a
  // real result with an empty answer -- both are grounded=false here, but only
  // the second is Krama correctly declining.
  if (!result) return decline("all LLM providers failed", "provider_failure");
  const answer = result.answer.trim();
  if (answer.length === 0) return decline("LLM declined -- passage does not support an answer", "llm_declined");

  return c.json({
    text: answer,
    grounded: true,
    generation_ms: performance.now() - t0,
    model: result.provider,
    gate: "passed",
  });
});

await boot();
const port = Number(process.env.EVAL_ADAPTER_PORT ?? 3100);
console.log(`[eval-adapter] listening on http://127.0.0.1:${port}`);
export default { port, fetch: app.fetch, idleTimeout: 255 };
