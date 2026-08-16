/**
 * maun -- guardrails. L0: input quality (ARCHITECTURE.md §8.1).
 *
 * CORRECTED scope (CLAUDE.md): the original L0 design included "ASR
 * confidence < τ0" -- Sarvam returns no confidence score on either STT
 * endpoint (verified via docs.sarvam.ai), so that sub-check is dropped
 * entirely, not just left unimplemented. L0 here only ever looks at the
 * transcript text itself: empty, too short to be a real question, or no
 * word-like content at all (gibberish/noise artifacts from STT).
 *
 * Word-like detection reuses the same Intl.Segmenter approach as bm25.ts's
 * tokenizer, for methodological consistency -- "does this look like real
 * text" and "how do we tokenize real text" are the same underlying
 * question, so they should use the same segmentation logic rather than two
 * different, potentially-inconsistent heuristics.
 */

export interface L0Result {
  refused: boolean;
  reason?: "empty_or_gibberish";
  detail?: string;
}

const MIN_CHARS = 2;

function hasWordLikeContent(text: string, lang: string): boolean {
  const segmenter = new Intl.Segmenter(lang, { granularity: "word" });
  for (const { isWordLike } of segmenter.segment(text)) {
    if (isWordLike) return true;
  }
  return false;
}

export function checkL0(text: string, lang: string = "en"): L0Result {
  const trimmed = text.trim();

  if (trimmed.length === 0) {
    return { refused: true, reason: "empty_or_gibberish", detail: "empty transcript" };
  }

  if (trimmed.length < MIN_CHARS) {
    return {
      refused: true,
      reason: "empty_or_gibberish",
      detail: `transcript too short (${trimmed.length} chars)`,
    };
  }

  if (!hasWordLikeContent(trimmed, lang)) {
    return {
      refused: true,
      reason: "empty_or_gibberish",
      detail: "no word-like content -- likely STT noise/garbage",
    };
  }

  return { refused: false };
}
