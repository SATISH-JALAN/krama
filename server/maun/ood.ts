/**
 * maun -- guardrails. L2: out-of-domain detection (ARCHITECTURE.md §8.1).
 * Refuse if EITHER the max top-k retrieval score is too low, OR the query
 * embedding is too far from the corpus centroid -- either signal alone can
 * indicate "this question isn't about anything in our indexed corpus."
 *
 * Calibrated (PLAN.md E5.5, eval/calibrate_guardrails.ts): joint grid search
 * with L1's safety threshold against 500 real in-domain queries + 199
 * hand-written OOD queries (6 categories per ARCHITECTURE.md §8.2: personal/
 * news/arithmetic/chitchat/other-domain/injection), for <=5% COMBINED
 * false-refusal rate on in-domain. Real, measured, honest findings, not
 * flattering ones smoothed over:
 *   - `minCentroidCosine` calibrated to 0 -- i.e. it contributes NOTHING at
 *     the chosen operating point. The corpus-centroid signal did not
 *     separate in-domain from OOD queries better than top-score alone on
 *     this data. Left in the interface/check logic (cheap, harmless, and
 *     may help at a different corpus scale) but don't assume it's doing
 *     real work -- see eval/results/guardrail_calibration.json.
 *   - OOD recall at this operating point is 43.7% combined with L1 (up from
 *     33.7% for L2 alone) -- real numbers, not the >90% one might hope for.
 *     Personal-question and chit-chat OOD categories are hardest to catch
 *     (~18% each) since short generic phrasings embed close to legitimate
 *     short queries by chance; arithmetic/news are easier (~58-64%).
 * See eval/results/ood_roc.png and guardrail_calibration.json for the full
 * ROC curves and category breakdown.
 */

export interface OodThresholds {
  minTopScore: number; // tau_1 -- calibrated, see docstring
  minCentroidCosine: number; // tau_2 -- calibrated to 0, see docstring
}

// Calibrated -- see module docstring and artifacts/thresholds.json.
export const DEFAULT_OOD_THRESHOLDS: OodThresholds = {
  minTopScore: 0.84,
  minCentroidCosine: 0,
};

export interface L2Result {
  refused: boolean;
  reason?: "off_topic";
  detail?: string;
}

export function cosineSim(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) throw new Error("vector length mismatch");
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB) || 1e-9;
  return dot / denom;
}

export function computeCentroid(embeddings: Float32Array[]): Float32Array {
  if (embeddings.length === 0) throw new Error("cannot compute centroid of zero embeddings");
  const dim = embeddings[0].length;
  const sum = new Float32Array(dim);
  for (const emb of embeddings) {
    for (let i = 0; i < dim; i++) sum[i] += emb[i];
  }
  for (let i = 0; i < dim; i++) sum[i] /= embeddings.length;
  // re-normalize -- the mean of unit vectors is not itself unit length,
  // and cosineSim already normalizes internally, but an explicit unit
  // centroid is easier to reason about / serialize as an artifact.
  let norm = 0;
  for (let i = 0; i < dim; i++) norm += sum[i] * sum[i];
  norm = Math.sqrt(norm) || 1e-9;
  for (let i = 0; i < dim; i++) sum[i] /= norm;
  return sum;
}

export class OodGuard {
  constructor(
    private readonly centroid: Float32Array,
    private readonly thresholds: OodThresholds = DEFAULT_OOD_THRESHOLDS,
  ) {}

  check(queryEmbedding: Float32Array, topRetrievalScore: number): L2Result {
    if (topRetrievalScore < this.thresholds.minTopScore) {
      return {
        refused: true,
        reason: "off_topic",
        detail: `top retrieval score ${topRetrievalScore.toFixed(3)} < tau_1 ${this.thresholds.minTopScore}`,
      };
    }

    const centroidCos = cosineSim(queryEmbedding, this.centroid);
    if (centroidCos < this.thresholds.minCentroidCosine) {
      return {
        refused: true,
        reason: "off_topic",
        detail: `centroid cosine ${centroidCos.toFixed(3)} < tau_2 ${this.thresholds.minCentroidCosine}`,
      };
    }

    return { refused: false };
  }
}
