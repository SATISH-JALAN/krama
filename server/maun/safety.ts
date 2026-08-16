/**
 * maun -- L1: safety / injection detection (ARCHITECTURE.md §8.1).
 * Embedding-vs-exemplar-set check, reusing the query embedding already
 * computed for retrieval (no extra embedding call needed, ~0.3ms per the
 * architecture doc). Exemplar embeddings are precomputed once at boot from
 * exemplars.ts and passed in here, not recomputed per-request.
 */
import { cosineSim } from "./ood";

export interface L1Result {
  refused: boolean;
  reason?: "unsafe";
  detail?: string;
}

// Placeholder, not calibrated -- see exemplars.ts docstring and
// PLAN.md E5.5 for the real calibration step.
export const DEFAULT_SAFETY_THRESHOLD = 0.75;

export class SafetyGuard {
  constructor(
    private readonly exemplarEmbeddings: Float32Array[],
    private readonly threshold: number = DEFAULT_SAFETY_THRESHOLD,
  ) {
    if (exemplarEmbeddings.length === 0) {
      throw new Error("SafetyGuard requires at least one exemplar embedding");
    }
  }

  check(queryEmbedding: Float32Array): L1Result {
    let maxSim = -Infinity;
    for (const exemplar of this.exemplarEmbeddings) {
      const sim = cosineSim(queryEmbedding, exemplar);
      if (sim > maxSim) maxSim = sim;
    }

    if (maxSim >= this.threshold) {
      return {
        refused: true,
        reason: "unsafe",
        detail: `max exemplar similarity ${maxSim.toFixed(3)} >= threshold ${this.threshold}`,
      };
    }

    return { refused: false };
  }
}
