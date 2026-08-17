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

import { GroundedAnswer, type Span } from "./harness/contracts";
import { runStage } from "./harness/pipeline";
import { SemanticCache } from "./harness/cache";
import * as embed from "./ghana/embed";
import * as bruteforce from "./ghana/bruteforce";
import { Bm25Index } from "./bm25";
import { rrfRanked } from "./jata/fuse";
import { extractSpan, type ExtractCandidate } from "./krama/extract";
import { checkL0 } from "./maun/input";
import { SafetyGuard, DEFAULT_SAFETY_THRESHOLD } from "./maun/safety";
import { OodGuard, computeCentroid, DEFAULT_OOD_THRESHOLDS } from "./maun/ood";
import { createSarvamClient, type SarvamClient } from "./stt/sarvam";

const RETRIEVAL_TOP_K = 10;

interface BootState {
  bm25: Bm25Index;
  safetyGuard: SafetyGuard;
  oodGuard: OodGuard;
  cache: SemanticCache<{ text: string; chunkId: string }>;
  passageTextById: Map<string, { text: string; lang: string }>;
  sttClient: SarvamClient | null;
}

let state: BootState | null = null;

// Shape written by eval/calibrate_guardrails.ts (PLAN.md E5.5). Validated
// with Zod like every other artifact boundary (ARCHITECTURE.md §7) rather
// than trusted blindly, since it's read from disk.
const ThresholdsArtifact = z.object({
  safetyThreshold: z.number(),
  oodThresholds: z.object({ minTopScore: z.number(), minCentroidCosine: z.number() }),
});

export function loadThresholds(thresholdsPath?: string): {
  safetyThreshold: number;
  oodThresholds: { minTopScore: number; minCentroidCosine: number };
} {
  if (!thresholdsPath) {
    return { safetyThreshold: DEFAULT_SAFETY_THRESHOLD, oodThresholds: DEFAULT_OOD_THRESHOLDS };
  }
  try {
    const raw = JSON.parse(readFileSync(thresholdsPath, "utf-8"));
    return ThresholdsArtifact.parse(raw);
  } catch (e) {
    // A malformed/missing calibration artifact should fall back to the
    // (also real, just less specific) calibrated defaults baked into
    // safety.ts/ood.ts, not crash boot -- but this is surprising enough to
    // be worth a loud log, not a silent swallow.
    console.warn(`could not load thresholds from ${thresholdsPath}, using built-in defaults: ${e}`);
    return { safetyThreshold: DEFAULT_SAFETY_THRESHOLD, oodThresholds: DEFAULT_OOD_THRESHOLDS };
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
 * only needs a transcript string, is unaffected.
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
}): Promise<void> {
  await embed.boot(opts.onnxArtifactDir);
  await bruteforce.boot(opts.bruteforceEmbeddingsPath, opts.bruteforceIdsPath);

  const bm25 = new Bm25Index();
  bm25.build(opts.corpus.map((c) => ({ id: c.passageId, text: c.text, lang: c.lang })));

  const passageTextById = new Map(
    opts.corpus.map((c) => [c.passageId, { text: c.text, lang: c.lang }]),
  );

  const centroid = computeCentroid(opts.passageEmbeddingsForCentroid);
  const thresholds = loadThresholds(opts.thresholdsPath);

  state = {
    bm25,
    safetyGuard: new SafetyGuard(opts.safetyExemplarEmbeddings, thresholds.safetyThreshold),
    oodGuard: new OodGuard(centroid, thresholds.oodThresholds),
    cache: new SemanticCache(),
    passageTextById,
    sttClient: opts.sarvamApiKey ? createSarvamClient(opts.sarvamApiKey) : null,
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
    };
  }

  const denseResults = await runStage(
    "dense_search",
    queryEmbedding,
    z.any(),
    async (qEmb) => bruteforce.search(qEmb, RETRIEVAL_TOP_K),
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
  const topFused = fused[0];
  if (!topFused) {
    return { answer: "", citations: [], confidence: 0, refused: true, refusalReason: "no_grounding", trace };
  }
  const topPassage = state.passageTextById.get(topFused.id);
  const answerText = topPassage?.text ?? "";

  state.cache.set(queryEmbedding, { text: answerText, chunkId: topFused.id });

  return {
    answer: answerText,
    citations: [topFused.id],
    confidence: Math.min(1, topFused.score),
    refused: false,
    trace,
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
      .object({ text: z.string(), lang: z.string().length(2), queryType: z.string().default("DESCRIPTION") })
      .safeParse(body);
    if (!parsed.success) return c.json({ error: "invalid request" }, 400);

    const result = await handleQuery(parsed.data.text, parsed.data.lang, parsed.data.queryType);
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
}): Promise<void> {
  const dataDir = opts?.dataDir ?? "data/medium";
  const onnxArtifactDir = opts?.onnxArtifactDir ?? "artifacts/onnx";
  const thresholdsPath = opts?.thresholdsPath ?? "artifacts/thresholds.json";
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
    // Bun loads .env automatically -- SARVAM_API_KEY isn't required to boot
    // (see boot()'s own docstring), only to serve /query/voice.
    sarvamApiKey: process.env.SARVAM_API_KEY,
  });
}

if (import.meta.main) {
  const port = Number(process.env.PORT ?? 3000);
  console.log("booting from data/medium/ + artifacts/ ...");
  await bootFromDisk();
  console.log(`booted. listening on :${port}`);
  Bun.serve({ port, fetch: createApp().fetch });
}
