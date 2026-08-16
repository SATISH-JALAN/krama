/**
 * Typed contracts at every stage boundary (ARCHITECTURE.md §7).
 * These are the harness requirement's backbone -- every stage's output gets
 * parsed against one of these before the next stage sees it.
 */
import { z } from "zod";

export const Transcript = z.object({
  text: z.string().min(1),
  lang: z.string().length(2),
  isFinal: z.boolean(),
  // CLAUDE.md: Sarvam returns no confidence score on either STT endpoint --
  // do not add a `confidence` field here expecting to gate on it. L0 uses
  // empty/gibberish/short-transcript heuristics instead (see maun/input.ts).
});
export type Transcript = z.infer<typeof Transcript>;

export const RetrievedChunk = z.object({
  chunkId: z.string(),
  passageId: z.string(),
  text: z.string(),
  lang: z.string().length(2),
  score: z.number(),
});
export type RetrievedChunk = z.infer<typeof RetrievedChunk>;

export const GroundedAnswer = z.object({
  answer: z.string(),
  citations: z.array(z.string()).min(1), // chunk_ids -- enforced non-empty when not refused
  confidence: z.number().min(0).max(1),
  refused: z.boolean(),
  refusalReason: z
    .enum(["off_topic", "unsafe", "empty_or_gibberish", "no_grounding"])
    .optional(),
  // "empty_or_gibberish" replaces ARCHITECTURE.md's original "low_asr" --
  // there is no ASR confidence signal to gate on (see Transcript above).
});
export type GroundedAnswer = z.infer<typeof GroundedAnswer>;

export const Span = z.object({
  name: z.string(),
  ms: z.number(),
  ok: z.boolean(),
  err: z.string().optional(),
});
export type Span = z.infer<typeof Span>;
