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

// Calibrated (PLAN.md E5.5, eval/calibrate_guardrails.ts): joint grid search
// over (safetyThreshold, minTopScore, minCentroidCosine) against 500 real
// in-domain queries + 199 hand-written OOD queries, for <=5% COMBINED
// (L1 OR L2) false-refusal rate on in-domain -- see
// eval/results/guardrail_calibration.json's `jointOperatingPoint` and
// artifacts/thresholds.json. Marginal (L1-alone) calibration would have
// allowed 0.82; this is 0.84 because L1+L2 calibrated independently at 5%
// each measurably compounds to 8.6% combined (real measured number, not
// assumed) -- the joint search is what actually respects the spec.
// At this value: 100% of hand-written prompt injections caught, in EVERY
// language tested (en/hi/bn/ta all 100%) -- multilingual-e5's cross-lingual
// generalization for unsafe-query similarity, previously flagged UNTESTED,
// is now a real measured result, not an open question.
export const DEFAULT_SAFETY_THRESHOLD = 0.84;

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
