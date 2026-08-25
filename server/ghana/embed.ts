/**
 * ghana -- dense index module. Local ONNX query embedding (CLAUDE.md #5: no
 * embedding API calls, ever).
 *
 * Model artifact and pooling method must exactly match ingest/06_export_onnx.py
 * and ingest/04_embed.py (CLAUDE.md #2, same model both sides). In particular:
 *   - token_type_ids IS a required graph input (all zeros for single-segment
 *     text) -- discovered the hard way when 06_export_onnx.py's own validation
 *     step omitted it and onnxruntime rejected the run with "Required inputs
 *     (['token_type_ids']) are missing". Don't drop it here too.
 *   - pooling is mean-pooling over token embeddings weighted by attention
 *     mask, NOT the [CLS] token -- confirmed correct for this model via
 *     research; sentence-transformers' own e5 config uses mean pooling.
 *   - every query text MUST be prefixed with "query: " before tokenizing
 *     (CLAUDE.md #1) -- see addQueryPrefix.
 *
 * Tokenizer: CLAUDE.md originally specified the native `tokenizers` npm
 * package. CORRECTED here -- that package does not publish a
 * `tokenizers-linux-arm64-gnu` binary (verified directly against the npm
 * registry: 404, while linux-x64-gnu and win32-x64-msvc both 200), so it
 * cannot be installed at all on the actual deployment target (Oracle Ampere
 * A1, aarch64). This was a real gap in the earlier ARM research, which
 * checked hnswlib-node and onnxruntime-node but not this package. Using
 * `@huggingface/transformers` instead, purely for its tokenizer -- it's
 * WASM-based (no native platform binary needed, confirmed no 404s on
 * install), loads our locally-exported tokenizer.json directly, and its
 * output tensors are already int64/BigInt64Array-typed, matching what
 * onnxruntime-node's Tensor constructor needs with no manual conversion.
 * We are NOT using its model-loading/inference machinery -- onnxruntime-node
 * still does the actual embedding inference directly, per ARCHITECTURE.md's
 * original "not transformers.js" decision, which was scoped to inference,
 * not tokenization.
 */
import * as ort from "onnxruntime-node";
import { resolve } from "path";
import { AutoTokenizer, type PreTrainedTokenizer } from "@huggingface/transformers";

const QUERY_PREFIX = "query: ";
const PASSAGE_PREFIX = "passage: ";
export function addQueryPrefix(text: string): string {
  return `${QUERY_PREFIX}${text}`;
}

/**
 * e5 is an ASYMMETRIC model: queries are prefixed "query: " and passages
 * "passage: " (ingest/prefixing.py, CLAUDE.md #1). At runtime this module
 * only ever embedded queries -- the corpus was embedded offline in Python
 * by ingest/04_embed.py -- so `query: ` was hardcoded below. Anything that
 * needs to embed a PASSAGE with this same session (the eval adapter, which
 * has to build a throwaway index out of someone else's corpus) must use
 * this prefix instead, or every passage vector lands in the wrong half of
 * the asymmetric space and recall collapses for a reason that has nothing
 * to do with the model's actual quality.
 */
export function addPassagePrefix(text: string): string {
  return `${PASSAGE_PREFIX}${text}`;
}

let session: ort.InferenceSession | null = null;
let tokenizer: PreTrainedTokenizer | null = null;

/**
 * @param artifactDir directory containing model.onnx (or model_int8.onnx)
 *   AND tokenizer.json/tokenizer_config.json -- ingest/06_export_onnx.py
 *   writes both into the same directory, e.g. "artifacts/onnx".
 * @param modelFile which ONNX file to load. Defaults to fp32 -- int8 was
 *   measured to fail the 0.995 cosine-agreement validation (0.99199), see
 *   MEMORY.md, so fp32 is the correct default until/unless that's revisited.
 */
export async function boot(
  artifactDir: string,
  modelFile: string = "model.onnx",
): Promise<void> {
  session = await ort.InferenceSession.create(`${artifactDir}/${modelFile}`, {
    executionProviders: ["cpu"],
    intraOpNumThreads: 2, // >2 hurts for batch-size-1 short sequences (ARCHITECTURE.md §5.1)
    interOpNumThreads: 1,
    graphOptimizationLevel: "all",
  });
  // @huggingface/transformers' from_pretrained() treats any relative path
  // that happens to match its Hub repo-id regex (owner/repo -- a single
  // "/", word chars only) as a Hub id, not a local directory. "artifacts/
  // onnx" matches that shape, so it silently looked for tokenizer files
  // under its own package-relative model cache instead of our directory,
  // failing with "tokenizerConfig.tokenizer_class" on undefined -- found by
  // actually running boot() against real artifacts, not from reading the
  // library's docs. Resolving to an absolute path first (which contains a
  // drive letter/colon on Windows, or a leading "/" -- neither matches the
  // repo-id regex) makes it correctly treat this as a local directory.
  tokenizer = await AutoTokenizer.from_pretrained(resolve(artifactDir));
}

function meanPool(
  lastHiddenState: Float32Array,
  seqLen: number,
  hiddenDim: number,
  attentionMask: BigInt64Array,
): Float32Array {
  const out = new Float32Array(hiddenDim);
  let count = 0;
  for (let t = 0; t < seqLen; t++) {
    if (attentionMask[t] === 0n) continue;
    count++;
    const base = t * hiddenDim;
    for (let d = 0; d < hiddenDim; d++) {
      out[d] += lastHiddenState[base + d];
    }
  }
  const denom = count > 0 ? count : 1;
  for (let d = 0; d < hiddenDim; d++) out[d] /= denom;
  return out;
}

function l2normalize(v: Float32Array): Float32Array {
  let sumSq = 0;
  for (let i = 0; i < v.length; i++) sumSq += v[i] * v[i];
  const norm = Math.sqrt(sumSq) || 1e-9;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / norm;
  return out;
}

const MAX_QUERY_TOKENS = 64; // ARCHITECTURE.md §9: truncate query tokens to 64
// Passages are not queries: 64 tokens is a deliberate latency choice for a
// short spoken query, but it would silently clip the tail off a real corpus
// passage. ingest/04_embed.py embedded the corpus via sentence-transformers'
// .encode(), i.e. at the model's own baked-in 512-token limit -- match that
// here so a passage embedded through this session is comparable to one from
// the production artifacts.
const MAX_PASSAGE_TOKENS = 512;

function tokenize(text: string, maxTokens: number): { inputIds: BigInt64Array; attentionMask: BigInt64Array } {
  if (!tokenizer) throw new Error("embed.boot() must be called before embed()");
  const encoded = tokenizer(text, {
    padding: false,
    truncation: true,
    max_length: maxTokens,
  });
  return {
    inputIds: encoded.input_ids.data as BigInt64Array,
    attentionMask: encoded.attention_mask.data as BigInt64Array,
  };
}

export async function embed(
  text: string,
  kind: "query" | "passage" = "query",
): Promise<Float32Array> {
  if (!session) throw new Error("embed.boot() must be called before embed()");

  const prefixed = kind === "passage" ? addPassagePrefix(text) : addQueryPrefix(text);
  const { inputIds, attentionMask } = tokenize(
    prefixed,
    kind === "passage" ? MAX_PASSAGE_TOKENS : MAX_QUERY_TOKENS,
  );
  const seqLen = inputIds.length;

  const tokenTypeIds = new BigInt64Array(seqLen); // all zeros, see module docstring

  const feeds: Record<string, ort.Tensor> = {
    input_ids: new ort.Tensor("int64", inputIds, [1, seqLen]),
    attention_mask: new ort.Tensor("int64", attentionMask, [1, seqLen]),
    token_type_ids: new ort.Tensor("int64", tokenTypeIds, [1, seqLen]),
  };

  const results = await session.run(feeds);
  const lastHiddenState = results.last_hidden_state ?? Object.values(results)[0];
  const hiddenDim = (lastHiddenState.dims.at(-1) as number) ?? 384;

  const pooled = meanPool(
    lastHiddenState.data as Float32Array,
    seqLen,
    hiddenDim,
    attentionMask,
  );
  return l2normalize(pooled);
}
