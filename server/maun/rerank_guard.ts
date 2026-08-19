/**
 * maun -- L3: cross-encoder relevance gate (server/ghana/rerank.ts owns the
 * model itself, this is the guard/threshold logic around it, same split as
 * ood.ts vs the dense index).
 *
 * Runs AFTER retrieval+fusion (unlike L2, which checks the raw dense score
 * before fusion), on the small top-N fused candidates only -- this is what
 * actually stops a completely irrelevant passage from being returned as the
 * "grounded" answer just because it won the RRF ranking. Found live: a
 * Hindi query about the Taj Mahal returned a passage about video-game
 * characters, because the bi-encoder cosine score and RRF rank aren't a
 * reliable enough relevance signal on their own -- L2's dense-only check
 * had already passed it. A cross-encoder that jointly attends over
 * (query, passage) is a much better-calibrated judge of "does this
 * actually answer that."
 *
 * Threshold calibrated the same way L1/L2 were (eval/calibrate_reranker.ts,
 * mirroring eval/calibrate_guardrails.ts's methodology) -- see that script's
 * own docstring for the real measured operating point before trusting
 * DEFAULT_MIN_RERANK_SCORE blindly.
 */

export const DEFAULT_MIN_RERANK_SCORE = -2.0;

export interface RerankGuardResult {
  refused: boolean;
  reason?: "off_topic";
  detail?: string;
}

export function checkRerankScore(
  bestScore: number,
  threshold: number = DEFAULT_MIN_RERANK_SCORE,
): RerankGuardResult {
  if (bestScore < threshold) {
    return {
      refused: true,
      reason: "off_topic",
      detail: `cross-encoder relevance score ${bestScore.toFixed(3)} < threshold ${threshold}`,
    };
  }
  return { refused: false };
}
