/**
 * maun -- guardrails. L2: out-of-domain detection (ARCHITECTURE.md §8.1).
 * Refuse if EITHER the max top-k retrieval score is too low, OR the query
 * embedding is too far from the corpus centroid -- either signal alone can
 * indicate "this question isn't about anything in our indexed corpus."
 *
 * Thresholds here are PLACEHOLDERS, explicitly not calibrated yet
 * (CLAUDE.md #6: never invent a benchmark number -- these are marked
 * ESTIMATED, not measured). Real calibration is Sprint 4 / PLAN.md E5.5:
 * sweep against 500 in-domain + 200 hand-written OOD queries, pick the
 * operating point at <=5% false-refusal rate on in-domain, replace these
 * defaults with the calibrated values in thresholds.json.
 */

export interface OodThresholds {
  minTopScore: number; // tau_1 -- ESTIMATED placeholder, not calibrated
  minCentroidCosine: number; // tau_2 -- ESTIMATED placeholder, not calibrated
}

// Placeholder only -- see docstring. Do not treat as tuned.
export const DEFAULT_OOD_THRESHOLDS: OodThresholds = {
  minTopScore: 0.5,
  minCentroidCosine: 0.3,
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
