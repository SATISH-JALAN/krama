/**
 * Entrypoint. Boots all in-memory state once, then serves the retrieval
 * core over HTTP (ARCHITECTURE.md §1/§5).
 *
 * Dense retrieval is brute-force cosine search (`ghana/bruteforce.ts`), not
 * HNSW -- see MEMORY.md's "RAGINGOA reference analyzed" entry for the full
 * reasoning. This removes the hnswlib-node/WSL2/Oracle blocker for MVP
 * scale rather than resolving it; `ghana/hnsw.ts` stays in the repo,
 * undeleted, as a documented option for a future larger-scale index.
 */
import { Hono } from "hono";
import { cors } from "hono/cors";
import { z } from "zod";
import { readFileSync } from "fs";

import { GroundedAnswer, SynthesisResponse, type Span } from "./harness/contracts";
import { runStage } from "./harness/pipeline";
import { SemanticCache } from "./harness/cache";
import * as embed from "./ghana/embed";
import * as bruteforce from "./ghana/bruteforce";
import * as rerank from "./ghana/rerank";
import { Bm25Index } from "./bm25";
import { rrfRanked } from "./jata/fuse";
import { extractSpan, type ExtractCandidate } from "./krama/extract";
import { checkL0 } from "./maun/input";
import { romanize } from "./krama/romanize";
import { resolveLang } from "./krama/detect_lang";
import { SafetyGuard, DEFAULT_SAFETY_THRESHOLD } from "./maun/safety";
import { OodGuard, computeCentroid, DEFAULT_OOD_THRESHOLDS } from "./maun/ood";
import { checkRerankScore, DEFAULT_MIN_RERANK_SCORE } from "./maun/rerank_guard";
import { createSarvamClient, type SarvamClient } from "./stt/sarvam";
import { CircuitBreaker } from "./harness/breaker";
import type { RetryOptions } from "./harness/retry";
import type { LlmProvider } from "./llm/chain";
import { createGeminiProvider, GEMINI_MODEL_CHAIN } from "./llm/gemini";
import { createCerebrasProvider } from "./llm/cerebras";
import { answerFromGeneralKnowledge, synthesizeAnswer } from "./llm/synthesize";

const RETRIEVAL_TOP_K = 10;
// How many of the RRF-fused top candidates get a real cross-encoder score.
// Real bench numbers (not the isolated synthetic-text test that first
// measured ~27ms for 5 candidates): real MS MARCO passages run much closer
// to the tokenizer's max_length cap than that test's short synthetic
// sentences did, and 5 candidates at real passage length pushed P90 to
// 217ms and P100 to 408ms -- a genuine SLA miss, not a rounding error.
// Cut to 3, matching MAX_RERANK_PAIR_TOKENS's own cut -- see bench/results
// after this change for the real, re-measured numbers.
const RERANK_TOP_N = 3;
// Applies only to the async synthesis path (POST /query/synthesize), never
// to handleQuery()'s fast path -- CLAUDE.md #4, LLM generation stays outside
// the t0->t1 core budget entirely, reported (and timed) separately. 30s, not
// 20s: matches gemini.ts's own widened GENERATE_TIMEOUT_MS (25s), found via
// live testing where a single real call took up to 19.3s -- a tighter outer
// deadline left no room for withRetry to ever attempt a second try.
const LLM_RETRY_OPTS: RetryOptions = { maxAttempts: 2, baseDelayMs: 300, deadlineMs: 30_000 };

interface BootState {
  bm25: Bm25Index;
  safetyGuard: SafetyGuard;
  oodGuard: OodGuard;
  rerankMinScore: number;
  rerankerEnabled: boolean;
  cache: SemanticCache<{ text: string; chunkId: string }>;
  passageTextById: Map<string, { text: string; lang: string }>;
  sttClient: SarvamClient | null;
  llmProviders: LlmProvider[];
  llmBreakers: Map<string, CircuitBreaker>;
}

let state: BootState | null = null;

// Shape written by eval/calibrate_guardrails.ts (PLAN.md E5.5) and
// eval/calibrate_reranker.ts. Validated with Zod like every other artifact
// boundary (ARCHITECTURE.md §7) rather than trusted blindly, since it's
// read from disk. rerankMinScore is optional so an older
// thresholds.json (written before the reranker existed) still loads --
// missing just means "fall back to the built-in default", not a hard error.
const ThresholdsArtifact = z.object({
  safetyThreshold: z.number(),
  oodThresholds: z.object({ minTopScore: z.number(), minCentroidCosine: z.number() }),
  rerankMinScore: z.number().optional(),
});

export function loadThresholds(thresholdsPath?: string): {
  safetyThreshold: number;
  oodThresholds: { minTopScore: number; minCentroidCosine: number };
  rerankMinScore: number;
} {
  const defaults = {
    safetyThreshold: DEFAULT_SAFETY_THRESHOLD,
    oodThresholds: DEFAULT_OOD_THRESHOLDS,
    rerankMinScore: DEFAULT_MIN_RERANK_SCORE,
  };
  if (!thresholdsPath) return defaults;
  try {
    const raw = JSON.parse(readFileSync(thresholdsPath, "utf-8"));
    const parsed = ThresholdsArtifact.parse(raw);
    return { ...parsed, rerankMinScore: parsed.rerankMinScore ?? DEFAULT_MIN_RERANK_SCORE };
  } catch (e) {
    // A malformed/missing calibration artifact should fall back to the
    // (also real, just less specific) calibrated defaults baked into
    // safety.ts/ood.ts, not crash boot -- but this is surprising enough to
    // be worth a loud log, not a silent swallow.
    console.warn(`could not load thresholds from ${thresholdsPath}, using built-in defaults: ${e}`);
    return defaults;
  }
}

/**
 * Loads every artifact needed to serve. `corpus` is the deduped passage list
 * (same shape as ingest/02_dedupe.py's output); `bruteforceEmbeddingsPath`/
 * `bruteforceIdsPath` are the flat Float32 binary + id-order JSON that
 * `ghana/bruteforce.ts` reads (same files `eval/calibrate_guardrails.ts`
 * already loads: `data/medium/embeddings.f32bin` + `embeddings_ids.json`).
 * `safetyExemplarEmbeddings` must be precomputed the same way query
 * embeddings are (same model, `query:` prefix -- exemplars are phrased as
 * queries). `thresholdsPath` is optional -- points at
 * artifacts/thresholds.json (PLAN.md E5.5's calibration output); omitted or
 * unreadable falls back to the calibrated DEFAULT_* constants baked into
 * safety.ts/ood.ts, not stale placeholders -- both are real calibrated
 * values now, this just lets a redeployed calibration take effect without a
 * code change. `sarvamApiKey` is optional -- when omitted, `/query/voice`
 * (real audio input) is disabled (503s) but the rest of the server, which
 * only needs a transcript string, is unaffected. `geminiApiKey`/
 * `cerebrasApiKey` are likewise optional -- when both are omitted,
 * `/query/synthesize` (the slow LLM-synthesis path, off the t0->t1 core
 * budget) is disabled (503) but the graded fast path is entirely
 * unaffected, matching the same degrade-honestly pattern as STT.
 * `rerankerArtifactDir` is ALSO optional but, unlike the others, its
 * absence degrades something on the graded fast path itself: without it,
 * handleQuery() falls back to picking the RRF-fused top result directly
 * (the pre-reranker behavior), losing the cross-encoder relevance check
 * that catches an irrelevant top-fused passage before it's returned as
 * "grounded" -- see maun/rerank_guard.ts's docstring for why that check
 * exists. Optional only so a server can still boot and answer questions
 * without ingest/10_export_reranker_onnx.py's artifact present, not
 * because skipping it is cost-free.
 */
export async function boot(opts: {
  onnxArtifactDir: string;
  bruteforceEmbeddingsPath: string;
  bruteforceIdsPath: string;
  corpus: { passageId: string; text: string; lang: string }[];
  safetyExemplarEmbeddings: Float32Array[];
  passageEmbeddingsForCentroid: Float32Array[];
  thresholdsPath?: string;
  sarvamApiKey?: string;
  geminiApiKey?: string;
  cerebrasApiKey?: string;
  rerankerArtifactDir?: string;
}): Promise<void> {
  await embed.boot(opts.onnxArtifactDir);
  const passageIdToLang = new Map(opts.corpus.map((c) => [c.passageId, c.lang]));
  await bruteforce.boot(opts.bruteforceEmbeddingsPath, opts.bruteforceIdsPath, passageIdToLang);

  let rerankerEnabled = false;
  if (opts.rerankerArtifactDir) {
    await rerank.boot(opts.rerankerArtifactDir, "model_int8.onnx");
    rerankerEnabled = true;
  } else {
    console.warn("no rerankerArtifactDir given -- handleQuery() will skip the cross-encoder relevance gate");
  }

  const bm25 = new Bm25Index();
  bm25.build(opts.corpus.map((c) => ({ id: c.passageId, text: c.text, lang: c.lang })));

  const passageTextById = new Map(
    opts.corpus.map((c) => [c.passageId, { text: c.text, lang: c.lang }]),
  );

  const centroid = computeCentroid(opts.passageEmbeddingsForCentroid);
  const thresholds = loadThresholds(opts.thresholdsPath);

  // Gemini primary, Cerebras secondary (ARCHITECTURE.md §7's fallback-chain
  // shape, same as the original Groq->Cerebras design -- Gemini takes
  // Groq's slot per this project's own free-tier-quota comparison, not a
  // random swap). Either, both, or neither key may be present; an absent
  // key just means that provider is never in the list, not a boot failure.
  const llmProviders: LlmProvider[] = [];
  // One provider entry per Gemini model, not one for "Gemini" -- the free
  // tier's request cap is per-model, so these are independent budgets that
  // chain.ts can fall through between (see GEMINI_MODEL_CHAIN's own note).
  if (opts.geminiApiKey) {
    for (const model of GEMINI_MODEL_CHAIN) {
      llmProviders.push(createGeminiProvider(opts.geminiApiKey, { model }));
    }
  }
  if (opts.cerebrasApiKey) llmProviders.push(createCerebrasProvider(opts.cerebrasApiKey));
  const llmBreakers = new Map(llmProviders.map((p) => [p.name, new CircuitBreaker()]));

  state = {
    bm25,
    safetyGuard: new SafetyGuard(opts.safetyExemplarEmbeddings, thresholds.safetyThreshold),
    oodGuard: new OodGuard(centroid, thresholds.oodThresholds),
    rerankMinScore: thresholds.rerankMinScore,
    rerankerEnabled,
    cache: new SemanticCache(),
    passageTextById,
    sttClient: opts.sarvamApiKey ? createSarvamClient(opts.sarvamApiKey) : null,
    llmProviders,
    llmBreakers,
  };
}

/**
 * The fast-path pipeline (t0 -> t1): embed -> guardrails -> retrieve ->
 * fuse -> extract. Does NOT include LLM synthesis (that's the async slow
 * path, off this budget per CLAUDE.md invariant #4) or L4 grounding (which
 * only applies to LLM output, not the extractive fast-path answer, which is
 * grounded by construction -- it's quoted text).
 */
export async function handleQuery(
  transcriptText: string,
  lang: string,
  queryType: string,
): Promise<z.infer<typeof GroundedAnswer>> {
  if (!state) throw new Error("index.boot() must be called before handleQuery()");
  const trace: Span[] = [];

  const l0 = await runStage("l0_input_guard", transcriptText, z.any(), async (text) => {
    const result = checkL0(text, lang);
    if (result.refused) throw new Error(result.detail ?? "L0 refused");
    return result;
  }, trace).catch(() => null);

  if (!l0) {
    return {
      answer: "",
      citations: [],
      confidence: 0,
      refused: true,
      refusalReason: "empty_or_gibberish",
      trace,
    };
  }

  const queryEmbedding = await runStage(
    "embed_query",
    transcriptText,
    z.instanceof(Float32Array),
    async (text) => embed.embed(text),
    trace,
  );

  const l1 = state.safetyGuard.check(queryEmbedding);
  if (l1.refused) {
    trace.push({ name: "l1_safety_guard", ms: 0, ok: false, err: l1.detail });
    return { answer: "", citations: [], confidence: 0, refused: true, refusalReason: "unsafe", trace };
  }

  // Semantic cache (E4.5, harness/cache.ts) -- checked AFTER L1 (guardrails
  // always run on every query, cheap and safety-critical, never skipped by
  // a cache hit) but BEFORE the expensive retrieval stages it exists to
  // skip. A "cache_hit" span's mere presence in the trace is the signal a
  // caller (bench/latency.ts) uses to bucket cached vs uncached latency
  // separately -- CLAUDE.md #4, never conflate the two.
  const cacheHit = state.cache.get(queryEmbedding);
  if (cacheHit.hit && cacheHit.value) {
    trace.push({ name: "cache_hit", ms: 0, ok: true });
    return {
      answer: cacheHit.value.text,
      citations: [cacheHit.value.chunkId],
      confidence: cacheHit.similarity ?? 1,
      refused: false,
      trace,
      cached: true,
      answerRomanized: romanize(cacheHit.value.text, lang) ?? undefined,
    };
  }

  const denseResults = await runStage(
    "dense_search",
    queryEmbedding,
    z.any(),
    // Scoped to the query's own language -- without this, dense search
    // ranks across the whole multilingual corpus at once and can "ground"
    // an answer in a passage the user never asked for and can't read (a
    // real bug found by voice-testing: a Hindi query returning a Tamil
    // passage as its answer).
    async (qEmb) => bruteforce.search(qEmb, RETRIEVAL_TOP_K, lang),
    trace,
  );

  const bm25Results = await runStage(
    "bm25_search",
    transcriptText,
    z.any(),
    async (text) => state!.bm25.search(text, lang, RETRIEVAL_TOP_K),
    trace,
  );

  const l2 = state.oodGuard.check(
    queryEmbedding,
    denseResults[0]?.score ?? 0,
  );
  if (l2.refused) {
    trace.push({ name: "l2_ood_guard", ms: 0, ok: false, err: l2.detail });
    return { answer: "", citations: [], confidence: 0, refused: true, refusalReason: "off_topic", trace };
  }

  const fused = await runStage(
    "fuse_rrf",
    null,
    z.any(),
    async () =>
      rrfRanked(
        [
          { ids: denseResults.map((r) => r.passageId), weight: 2 }, // dense-favoured, ARCHITECTURE.md §5.3
          { ids: bm25Results.map((r) => r.id), weight: 1 },
        ],
        RETRIEVAL_TOP_K,
      ),
    trace,
  );

  // Extractive span selection needs sentence-level candidates; the fused
  // results above are passage-level (strategy A, Sprint 3's winner) -- for
  // the fast path we treat the top fused passage's full text as the
  // candidate directly, skipping sentence-level extraction. This is a
  // simplification vs. ARCHITECTURE.md §5.4's per-sentence scoring, tracked
  // as a gap: real sentence-level extraction needs the passage broken into
  // sentence candidates with embeddings, which requires the full corpus's
  // sentence-level artifacts (analogous to the bake-off's chunks_DE), not
  // yet built for the production corpus.
  if (fused.length === 0) {
    return { answer: "", citations: [], confidence: 0, refused: true, refusalReason: "no_grounding", trace };
  }

  // L3: cross-encoder relevance gate (maun/rerank_guard.ts). Re-scores the
  // top RERANK_TOP_N fused candidates by actually attending over
  // (query, passage) jointly, instead of trusting RRF rank order alone --
  // RRF rank can put a passage first that a real relevance judge would
  // reject outright (found live: a Taj Mahal query "grounded" in an
  // unrelated video-game-character passage). Falls back to the pre-reranker
  // behavior (trust fused[0]) if the reranker artifact wasn't booted.
  let bestId = fused[0]!.id;
  let bestScore = fused[0]!.score;
  if (state.rerankerEnabled) {
    const candidates = fused.slice(0, RERANK_TOP_N);
    const candidateTexts = candidates.map((c) => state!.passageTextById.get(c.id)?.text ?? "");
    const rerankScores = await runStage(
      "rerank",
      candidateTexts,
      z.any(),
      async (texts) => rerank.scoreBatch(transcriptText, texts),
      trace,
    );
    let bestIdx = 0;
    for (let i = 1; i < rerankScores.length; i++) {
      if (rerankScores[i]! > rerankScores[bestIdx]!) bestIdx = i;
    }
    bestId = candidates[bestIdx]!.id;
    bestScore = rerankScores[bestIdx]!;

    const l3 = checkRerankScore(bestScore, state.rerankMinScore);
    if (l3.refused) {
      trace.push({ name: "l3_rerank_guard", ms: 0, ok: false, err: l3.detail });
      return { answer: "", citations: [], confidence: 0, refused: true, refusalReason: "off_topic", trace };
    }
  }

  const topPassage = state.passageTextById.get(bestId);
  const answerText = topPassage?.text ?? "";
  // Cross-encoder logits are unbounded, not a 0-1 score like the bi-encoder
  // cosine/RRF-derived confidence was -- squash with a sigmoid so
  // `confidence` stays interpretable as "how sure", not a raw classifier
  // logit that can read as >1 or wildly negative in the API response.
  const confidence = state.rerankerEnabled ? 1 / (1 + Math.exp(-bestScore)) : Math.min(1, bestScore);

  state.cache.set(queryEmbedding, { text: answerText, chunkId: bestId });

  return {
    answer: answerText,
    citations: [bestId],
    confidence,
    refused: false,
    trace,
    answerRomanized: romanize(answerText, lang) ?? undefined,
  };
}

/**
 * Slow, async synthesis path (POST /query/synthesize) -- entirely off the
 * t0->t1 core budget `bench/latency.ts` measures (CLAUDE.md #4), and the
 * frontend calls it as a genuinely separate request after the fast path's
 * answer already landed, not inline with it. Reuses handleQuery() itself
 * rather than re-deriving retrieval, which has two real effects, both
 * intentional: (1) guardrails apply identically to both paths -- a query
 * L0/L1/L2 already refused never reaches an LLM either, so this can't
 * become a way to route unsafe input around the guardrails; (2) it costs
 * one redundant retrieval pass (~50-70ms) per call, which is irrelevant
 * next to LLM round-trip latency (hundreds of ms to seconds) and far
 * simpler than threading retrieval results between two separate endpoints.
 *
 * The candidate passed to synthesizeAnswer() is the SAME single top-fused
 * passage the fast path already selected, not a fresh multi-candidate
 * retrieval -- a deliberate scope decision (richer multi-passage synthesis
 * context is a real future enhancement, not attempted here) that keeps
 * this addition small and lets the LLM's actual job stay narrow: rewrite
 * that one passage more naturally, or recognize it doesn't answer the
 * question and decline (llm/synthesize.ts's prompt already instructs the
 * latter) -- exactly the guardrail gap real-world testing found (README's
 * documented "low lexical/semantic separation between a good match and a
 * degenerate/repetitive corpus passage" weakness).
 */
export async function handleSynthesisQuery(
  transcriptText: string,
  lang: string,
  queryType: string,
): Promise<z.infer<typeof SynthesisResponse>> {
  if (!state) throw new Error("index.boot() must be called before handleSynthesisQuery()");

  const fast = await handleQuery(transcriptText, lang, queryType);

  if (state.llmProviders.length === 0) {
    throw new Error("no LLM provider configured -- boot() was not given a geminiApiKey or cerebrasApiKey");
  }

  if (fast.refused) {
    // Retrieval declined -- but "the corpus has nothing on this" is not the
    // same as "this question has no answer", and dead-ending here was a real
    // usability complaint. The corpus is an MS MARCO web-passage subset, so
    // plenty of ordinary questions ("who built the Taj Mahal") genuinely have
    // zero matching passages; refusing them is correct retrieval behaviour and
    // still an unhelpful product.
    //
    // Two things keep this from becoming the reference implementation's
    // failure (serving an unrelated passage as "GROUNDED"): the answer is
    // generated from the model's own knowledge rather than from a passage
    // that already failed the relevance gate, and it is returned with
    // grounded=false and refused=true so it can never be mistaken for one
    // that passed maun/. L0/L1 refusals are deliberately NOT routed here --
    // gibberish and unsafe input should stay refused outright, not be handed
    // to an LLM.
    const eligible = fast.refusalReason === "off_topic" || fast.refusalReason === "no_grounding";
    if (!eligible) {
      return { refused: true, refusalReason: fast.refusalReason, synthesized: null };
    }

    const general = await answerFromGeneralKnowledge(
      transcriptText,
      lang,
      state.llmProviders,
      state.llmBreakers,
      LLM_RETRY_OPTS,
    );
    if (!general) {
      return { refused: true, refusalReason: fast.refusalReason, synthesized: null };
    }
    return {
      refused: true,
      refusalReason: fast.refusalReason,
      synthesized: {
        answer: general.answer,
        streaming: false,
        declined: false,
        provider: general.provider,
        citedChunkIds: [],
        grounded: false,
      },
    };
  }

  const candidate = { chunkId: fast.citations[0]!, text: fast.answer };
  const result = await synthesizeAnswer(
    [candidate],
    transcriptText,
    lang,
    state.llmProviders,
    state.llmBreakers,
    LLM_RETRY_OPTS,
  );

  if (!result) {
    // Every provider failed outright (network/quota/timeout), not "declined
    // to answer" -- distinct from the LLM successfully judging the passage
    // ungrounded (that returns a real result with an empty `answer` below).
    return { refused: false, synthesized: null };
  }

  return {
    refused: false,
    synthesized: {
      answer: result.answer,
      streaming: false,
      declined: result.answer.trim().length === 0,
      provider: result.provider,
      citedChunkIds: result.citedChunkIds,
      grounded: true,
    },
  };
}

/**
 * Voice entrypoint (E5b): real Sarvam batch STT (`stt/sarvam.ts`) -> the
 * same `handleQuery()` fast path above. `lang`, if given, is passed through
 * to Sarvam as the expected language (skips auto-detect); omitted lets
 * Sarvam auto-detect and its returned language drives retrieval instead.
 * Note this puts STT latency INSIDE this function's wall-clock time, but
 * NOT inside the t0->t1 core budget `bench/latency.ts` measures (CLAUDE.md
 * #4) -- that budget starts at transcript-in, STT is upstream of it.
 */
export async function handleVoiceQuery(
  pcm: Int16Array,
  lang: string | undefined,
  queryType: string,
): Promise<z.infer<typeof GroundedAnswer> & { transcript: string; detectedLang: string }> {
  if (!state) throw new Error("index.boot() must be called before handleVoiceQuery()");
  if (!state.sttClient) throw new Error("STT is not configured -- boot() was not given a sarvamApiKey");

  const transcript = await state.sttClient.transcribe(pcm, lang);
  const answer = await handleQuery(transcript.text, transcript.lang, queryType);
  return { ...answer, transcript: transcript.text, detectedLang: transcript.lang };
}

export function createApp() {
  const app = new Hono();

  // Wide open by design, not an oversight: this API has no cookies/auth to
  // leak cross-origin, and every secret stays server-side regardless of who
  // calls it (CLAUDE.md #10 -- Sarvam/Groq keys never touch the client).
  // The frontend (web/) is a separate origin/port from this server in both
  // local dev (5173 vs 3000) and once actually deployed, so CORS has to be
  // explicitly allowed here or every browser request 0-round-trips into a
  // preflight failure -- found live by actually driving the frontend
  // against a real running server, not by reading the Hono docs.
  app.use("*", cors());

  app.get("/health", (c) => c.json({ ok: state !== null }));

  app.post("/query", async (c) => {
    const body = await c.req.json();
    const parsed = z
      .object({ text: z.string(), lang: z.string().optional(), queryType: z.string().default("DESCRIPTION") })
      .safeParse(body);
    if (!parsed.success) return c.json({ error: "invalid request" }, 400);

    const result = await handleQuery(
      parsed.data.text,
      // "auto" / omitted -> detect from script, so a typed English query is
      // not forced through whatever the UI selector happened to be set to.
      resolveLang(parsed.data.lang, parsed.data.text),
      parsed.data.queryType,
    );
    return c.json(result);
  });

  // Real 16kHz mono Int16 PCM as the raw request body (CLAUDE.md #8 -- the
  // same format the frontend's AudioWorklet already captures, concatenated
  // client-side into one buffer per utterance since STT here is batch, not
  // streaming). lang/queryType arrive as query-string params instead of a
  // JSON body since the body itself is binary audio.
  app.post("/query/voice", async (c) => {
    if (!state?.sttClient) return c.json({ error: "STT not configured on this server" }, 503);

    const paramsParsed = z
      .object({
        lang: z.string().length(2).optional(),
        queryType: z.string().default("DESCRIPTION"),
      })
      .safeParse({ lang: c.req.query("lang"), queryType: c.req.query("queryType") ?? undefined });
    if (!paramsParsed.success) return c.json({ error: "invalid request" }, 400);

    const buf = await c.req.arrayBuffer();
    if (buf.byteLength === 0 || buf.byteLength % 2 !== 0) {
      return c.json({ error: "body must be non-empty 16-bit PCM (even byte length)" }, 400);
    }
    const pcm = new Int16Array(buf);

    const result = await handleVoiceQuery(pcm, paramsParsed.data.lang, paramsParsed.data.queryType);
    return c.json(result);
  });

  // Slow path, deliberately separate from /query -- see handleSynthesisQuery's
  // own docstring for why. The frontend calls this after /query's answer
  // already landed; a client that never calls it (or this route being 503'd
  // with no key configured) leaves the fast path completely unaffected.
  app.post("/query/synthesize", async (c) => {
    if (!state || state.llmProviders.length === 0) {
      return c.json({ error: "no LLM provider configured on this server" }, 503);
    }
    const body = await c.req.json();
    const parsed = z
      .object({ text: z.string(), lang: z.string().optional(), queryType: z.string().default("DESCRIPTION") })
      .safeParse(body);
    if (!parsed.success) return c.json({ error: "invalid request" }, 400);

    const result = await handleSynthesisQuery(
      parsed.data.text,
      resolveLang(parsed.data.lang, parsed.data.text),
      parsed.data.queryType,
    );
    return c.json(result);
  });

  return app;
}

/**
 * Boots the server from the real medium-scale artifacts already on disk
 * from Sprint 1/E5.5's guardrail calibration run (`data/medium/`,
 * `artifacts/`) -- the first path in this project's history that actually
 * calls boot() with real arguments instead of only exercising it via mocks.
 * Factored out of the `import.meta.main` block below so `bench/latency.ts`
 * can reuse the exact same real boot sequence rather than re-deriving it.
 */
export async function bootFromDisk(opts?: {
  dataDir?: string;
  onnxArtifactDir?: string;
  thresholdsPath?: string;
  rerankerArtifactDir?: string;
}): Promise<void> {
  const dataDir = opts?.dataDir ?? "data/medium";
  const onnxArtifactDir = opts?.onnxArtifactDir ?? "artifacts/onnx";
  const thresholdsPath = opts?.thresholdsPath ?? "artifacts/thresholds.json";
  const rerankerArtifactDir = opts?.rerankerArtifactDir ?? "artifacts/onnx_reranker";
  const DIM = 384;

  const corpus = readFileSync(`${dataDir}/passages_dedup.jsonl`, "utf-8")
    .trim()
    .split("\n")
    .map((line) => {
      const row = JSON.parse(line);
      return { passageId: row.passage_id as string, text: row.text as string, lang: row.lang as string };
    });

  // Needed to embed the safety exemplars below -- boot() re-boots embed.ts
  // itself too (see its own call to embed.boot()), which reloads the same
  // ONNX session a second time. Mildly wasteful (one extra model load, once
  // per process start, not per-query) but keeps boot()'s own contract
  // simple rather than adding an "already booted" special case for a
  // one-time startup cost.
  await embed.boot(onnxArtifactDir);
  const { SAFETY_EXEMPLARS } = await import("./maun/exemplars");
  const safetyExemplarEmbeddings = await Promise.all(
    SAFETY_EXEMPLARS.map((text) => embed.embed(text)),
  );

  // Same flat row-major Float32 buffer bruteforce.ts reads, sliced into
  // per-row views (no copy) for computeCentroid() -- read once here
  // separately from bruteforce.boot()'s own read of the same file, since
  // that module intentionally only exposes a path-based boot() (matching
  // hnsw.ts's interface), not a way to hand it an already-loaded buffer.
  const embeddingsBuf = readFileSync(`${dataDir}/embeddings.f32bin`);
  const flat = new Float32Array(
    embeddingsBuf.buffer,
    embeddingsBuf.byteOffset,
    embeddingsBuf.byteLength / 4,
  );
  const passageEmbeddingsForCentroid: Float32Array[] = [];
  for (let i = 0; i < flat.length / DIM; i++) {
    passageEmbeddingsForCentroid.push(flat.subarray(i * DIM, (i + 1) * DIM));
  }

  await boot({
    onnxArtifactDir,
    bruteforceEmbeddingsPath: `${dataDir}/embeddings.f32bin`,
    bruteforceIdsPath: `${dataDir}/embeddings_ids.json`,
    corpus,
    safetyExemplarEmbeddings,
    passageEmbeddingsForCentroid,
    thresholdsPath,
    rerankerArtifactDir,
    // Bun loads .env automatically -- none of these three are required to
    // boot (see boot()'s own docstring); SARVAM_API_KEY only gates
    // /query/voice, GEMINI_API_KEY/CEREBRAS_API_KEY only gate
    // /query/synthesize.
    sarvamApiKey: process.env.SARVAM_API_KEY,
    geminiApiKey: process.env.GEMINI_API_KEY,
    cerebrasApiKey: process.env.CEREBRAS_API_KEY,
  });
}

if (import.meta.main) {
  const port = Number(process.env.PORT ?? 3000);
  console.log("booting from data/medium/ + artifacts/ ...");
  await bootFromDisk();
  console.log(`booted. listening on :${port}`);
  Bun.serve({ port, fetch: createApp().fetch });
}
