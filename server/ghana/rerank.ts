/**
 * ghana -- cross-encoder relevance reranker. Local ONNX, no API call (same
 * CLAUDE.md #5 discipline as embed.ts).
 *
 * Why this exists: real voice-testing found the bi-encoder cosine score
 * (dense_search's topScore, two independently-computed embeddings compared
 * after the fact) doesn't reliably separate a genuinely relevant passage
 * from an irrelevant one -- an unrelated corpus passage about video-game
 * characters scored high enough to be returned as the "grounded" answer to
 * a Taj Mahal question. A cross-encoder jointly encodes (query, passage) as
 * ONE sequence through the transformer, so the model can actually attend
 * across both -- a much better-calibrated relevance signal, and the same
 * technique the reference implementation studied for this task uses for
 * its own reranking stage.
 *
 * Architecturally different from embed.ts, not just a config change:
 *   - embed.ts: ONE segment in, a pooled sentence-embedding vector out,
 *     compared to OTHER embeddings later via cosine similarity.
 *   - this module: TWO segments (query + passage) in as a single joined
 *     sequence, a single relevance LOGIT out directly from a
 *     classification head -- token_type_ids here is NOT all-zeros, it
 *     meaningfully marks which tokens belong to which segment.
 *
 * Model artifact: ingest/10_export_reranker_onnx.py exports
 * cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (mMARCO-trained, verified live
 * to include Hindi in its 14 training languages; Bengali/Tamil are NOT
 * among them, so their scores rely on the base multilingual model's own
 * cross-lingual transfer -- verify this holds before trusting it, see
 * eval/verify_reranker.py, don't assume it from lore).
 */
import * as ort from "onnxruntime-node";
import { resolve } from "path";
import { AutoTokenizer, type PreTrainedTokenizer } from "@huggingface/transformers";

let session: ort.InferenceSession | null = null;
let tokenizer: PreTrainedTokenizer | null = null;

// query + passage combined. Cut from 256 (the export script's validation
// default) to 128 after a real bench run: cross-encoder attention cost
// scales with sequence length, and a handful of long real MS MARCO
// passages running close to 256 tokens pushed P100 to 250ms -- over the
// 200ms budget. 128 tokens is still generous for a relevance judgment (a
// passage's first ~100 tokens almost always carry enough topical signal to
// classify relevance, even if truncated before its last sentence) and is a
// standard cap for production cross-encoder rerankers for this exact
// latency reason, not a KRAMA-specific compromise.
const MAX_PAIR_TOKENS = 128;

/**
 * @param artifactDir directory containing model.onnx (or model_int8.onnx)
 *   AND tokenizer.json/tokenizer_config.json --
 *   ingest/10_export_reranker_onnx.py writes both, e.g. "artifacts/onnx_reranker".
 */
export async function boot(
  artifactDir: string,
  modelFile: string = "model.onnx",
): Promise<void> {
  session = await ort.InferenceSession.create(`${artifactDir}/${modelFile}`, {
    executionProviders: ["cpu"],
    // 4, not embed.ts's 2, and deliberately different: embed.ts runs ONE
    // short query at batch size 1, where extra threads cost more in
    // coordination than they save (ARCHITECTURE.md 5.1). This session runs
    // RERANK_TOP_N=3 pairs at up to MAX_PAIR_TOKENS=128 through a 12-layer
    // cross-encoder -- matmuls large enough that intra-op parallelism
    // actually pays. Measured on the Oracle Ampere A1 deploy target, this
    // stage was ~160ms of a 212ms P50, i.e. 75% of the whole budget.
    intraOpNumThreads: 4,
    interOpNumThreads: 1,
    graphOptimizationLevel: "all",
  });
  // Same absolute-path requirement as embed.ts -- from_pretrained() treats
  // a relative "owner/repo"-shaped path as a Hub id, not a local directory.
  tokenizer = await AutoTokenizer.from_pretrained(resolve(artifactDir));
}

/** Scores one (query, passage) pair. Higher = more relevant; unbounded logit, not a 0-1 probability. */
export async function score(query: string, passageText: string): Promise<number> {
  const scores = await scoreBatch(query, [passageText]);
  return scores[0]!;
}

/**
 * Scores several passages against the same query in ONE graph run, not N --
 * this runs on the fast path (handleQuery(), <200ms budget), and the whole
 * reason it exists is to re-score the top-K RRF candidates (K~5), so paying
 * per-call session overhead K times instead of once is a real, avoidable
 * cost, not a micro-optimization. `padding: true` pads every pair in this
 * call to the same length so they can share one batched tensor.
 */
export async function scoreBatch(query: string, passages: string[]): Promise<number[]> {
  if (!session || !tokenizer) throw new Error("rerank.boot() must be called before score()/scoreBatch()");
  if (passages.length === 0) return [];

  const encoded = tokenizer(Array(passages.length).fill(query), {
    text_pair: passages,
    padding: true,
    truncation: true,
    max_length: MAX_PAIR_TOKENS,
    return_token_type_ids: true,
  });
  const inputIds = encoded.input_ids.data as BigInt64Array;
  const attentionMask = encoded.attention_mask.data as BigInt64Array;
  const tokenTypeIds = encoded.token_type_ids.data as BigInt64Array;
  const dims = encoded.input_ids.dims as number[];
  const [batchSize, seqLen] = dims;

  const feeds: Record<string, ort.Tensor> = {
    input_ids: new ort.Tensor("int64", inputIds, [batchSize!, seqLen!]),
    attention_mask: new ort.Tensor("int64", attentionMask, [batchSize!, seqLen!]),
    token_type_ids: new ort.Tensor("int64", tokenTypeIds, [batchSize!, seqLen!]),
  };

  const results = await session.run(feeds);
  const logits = results.logits ?? Object.values(results)[0];
  return Array.from(logits.data as Float32Array);
}

export function isBooted(): boolean {
  return session !== null;
}
