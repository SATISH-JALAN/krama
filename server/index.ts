/**
 * Entrypoint. Boots all in-memory state once, then serves the retrieval
 * core over HTTP (ARCHITECTURE.md §1/§5).
 *
 * HONEST STATUS, not glossed over: this wires together every piece that
 * exists (embed, bm25, fuse, extract, guardrails, cache), but it has NOT
 * been run end-to-end on this machine. Two real blockers:
 *   1. hnswlib-node cannot build here (no MSVC Build Tools) -- ghana/hnsw.ts
 *      is itself unverified, see its own docstring.
 *   2. There is no production artifacts/ directory yet -- the ONNX model
 *      exists (artifacts/onnx/), but there's no hnsw.bin or a full corpus's
 *      worth of passage data; only bake-off/medium-scale scratch data under
 *      data/, which was never meant to be served, just evaluated offline.
 * Both need WSL2/the Oracle VM and a real full-corpus ingest run before this
 * file can actually boot successfully. Written now because the wiring logic
 * itself doesn't depend on either blocker being resolved -- validate the
 * plumbing today, validate the artifacts once they exist.
 */
import { Hono } from "hono";
import { z } from "zod";
import { readFileSync } from "fs";

import { GroundedAnswer, type Span } from "./harness/contracts";
import { runStage } from "./harness/pipeline";
import { SemanticCache } from "./harness/cache";
import * as embed from "./ghana/embed";
import * as hnsw from "./ghana/hnsw";
import { Bm25Index } from "./bm25";
import { rrfRanked } from "./jata/fuse";
import { extractSpan, type ExtractCandidate } from "./krama/extract";
import { checkL0 } from "./maun/input";
import { SafetyGuard, DEFAULT_SAFETY_THRESHOLD } from "./maun/safety";
import { OodGuard, computeCentroid, DEFAULT_OOD_THRESHOLDS } from "./maun/ood";

const RETRIEVAL_TOP_K = 10;

interface BootState {
  bm25: Bm25Index;
  safetyGuard: SafetyGuard;
  oodGuard: OodGuard;
  cache: SemanticCache<{ text: string; chunkId: string }>;
  passageTextById: Map<string, { text: string; lang: string }>;
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
 * Loads every artifact needed to serve. UNVERIFIED end-to-end on this
 * machine -- see module docstring. `corpus` is the deduped passage list
 * (same shape as ingest/02_dedupe.py's output); `safetyExemplarEmbeddings`
 * must be precomputed the same way query embeddings are (same model,
 * `query:` prefix -- exemplars are phrased as queries). `thresholdsPath`
 * is optional -- points at artifacts/thresholds.json (PLAN.md E5.5's
 * calibration output); omitted or unreadable falls back to the calibrated
 * DEFAULT_* constants baked into safety.ts/ood.ts, not stale placeholders --
 * both are real calibrated values now, this just lets a redeployed
 * calibration take effect without a code change.
 */
export async function boot(opts: {
  onnxArtifactDir: string;
  hnswIndexPath: string;
  hnswIdMapPath: string;
  corpus: { passageId: string; text: string; lang: string }[];
  safetyExemplarEmbeddings: Float32Array[];
  passageEmbeddingsForCentroid: Float32Array[];
  thresholdsPath?: string;
}): Promise<void> {
  await embed.boot(opts.onnxArtifactDir);
  await hnsw.boot(opts.hnswIndexPath, opts.hnswIdMapPath);

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
    return { answer: "", citations: [], confidence: 0, refused: true, refusalReason: "unsafe" };
  }

  const denseResults = await runStage(
    "hnsw_search",
    queryEmbedding,
    z.any(),
    async (qEmb) => hnsw.search(qEmb, RETRIEVAL_TOP_K),
    trace,
  );

  const bm25Results = state.bm25.search(transcriptText, lang, RETRIEVAL_TOP_K);

  const l2 = state.oodGuard.check(
    queryEmbedding,
    denseResults[0]?.score ?? 0,
  );
  if (l2.refused) {
    trace.push({ name: "l2_ood_guard", ms: 0, ok: false, err: l2.detail });
    return { answer: "", citations: [], confidence: 0, refused: true, refusalReason: "off_topic" };
  }

  const fused = rrfRanked(
    [
      { ids: denseResults.map((r) => r.passageId), weight: 2 }, // dense-favoured, ARCHITECTURE.md §5.3
      { ids: bm25Results.map((r) => r.id), weight: 1 },
    ],
    RETRIEVAL_TOP_K,
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
    return { answer: "", citations: [], confidence: 0, refused: true, refusalReason: "no_grounding" };
  }
  const topPassage = state.passageTextById.get(topFused.id);

  return {
    answer: topPassage?.text ?? "",
    citations: [topFused.id],
    confidence: Math.min(1, topFused.score),
    refused: false,
  };
}

export function createApp() {
  const app = new Hono();

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

  return app;
}

if (import.meta.main) {
  throw new Error(
    "server/index.ts is not runnable yet -- boot() needs real artifacts " +
      "(HNSW index, full corpus, safety exemplar embeddings) that don't " +
      "exist on this machine. See this file's module docstring.",
  );
}
