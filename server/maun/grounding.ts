/**
 * maun -- L4: per-sentence grounding check (ARCHITECTURE.md §8.1). Runs on
 * the LLM-synthesized (slow-path) answer before it's shown: each answer
 * sentence must be embedding-similar enough to its cited source chunk, or
 * it gets stripped from the answer. If EVERY sentence fails, the whole
 * answer is refused rather than shown empty or fabricated.
 *
 * This is what makes "grounded answer" a checked property, not an assumed
 * one -- an LLM can still produce a fluent sentence that isn't actually
 * supported by the retrieved passage, and this is the check that catches
 * that before the user sees it.
 */
import { cosineSim } from "./ood";

export interface AnswerSentence {
  text: string;
  embedding: Float32Array;
  citedChunkId: string;
}

export interface L4Result {
  refused: boolean;
  reason?: "no_grounding";
  groundedSentences: string[];
  strippedSentences: string[];
}

// Placeholder, not calibrated -- see ood.ts/safety.ts for the same caveat.
// Real calibration is Sprint 4 / PLAN.md E5.5.
export const DEFAULT_GROUNDING_THRESHOLD = 0.6;

export function checkL4(
  sentences: AnswerSentence[],
  chunkEmbeddings: Map<string, Float32Array>,
  threshold: number = DEFAULT_GROUNDING_THRESHOLD,
): L4Result {
  const grounded: string[] = [];
  const stripped: string[] = [];

  for (const sentence of sentences) {
    const chunkEmbedding = chunkEmbeddings.get(sentence.citedChunkId);
    if (!chunkEmbedding) {
      // cited a chunk we have no embedding for -- can't verify grounding,
      // treat as ungrounded rather than silently trusting an unverifiable
      // citation.
      stripped.push(sentence.text);
      continue;
    }
    const sim = cosineSim(sentence.embedding, chunkEmbedding);
    if (sim >= threshold) {
      grounded.push(sentence.text);
    } else {
      stripped.push(sentence.text);
    }
  }

  return {
    refused: grounded.length === 0,
    reason: grounded.length === 0 ? "no_grounding" : undefined,
    groundedSentences: grounded,
    strippedSentences: stripped,
  };
}
