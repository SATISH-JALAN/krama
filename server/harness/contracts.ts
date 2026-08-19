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

export const Span = z.object({
  name: z.string(),
  ms: z.number(),
  ok: z.boolean(),
  err: z.string().optional(),
});
export type Span = z.infer<typeof Span>;

const RefusalReason = z.enum(["off_topic", "unsafe", "empty_or_gibberish", "no_grounding"]);

// The async synthesis path's (POST /query/synthesize) result -- distinct
// from the fast extractive path. `declined` is true when a provider
// answered successfully but its own judgment was "these passages don't
// actually answer this question" (llm/synthesize.ts's prompt explicitly
// instructs an empty answer in that case) -- the frontend needs this to
// show "not confident enough" rather than a blank card, same "knows when
// not to answer" requirement as the guardrails, just enforced by the LLM's
// own reasoning instead of a cosine threshold.
export const SynthesizedAnswer = z.object({
  answer: z.string(),
  streaming: z.boolean(),
  declined: z.boolean(),
  provider: z.string(),
  citedChunkIds: z.array(z.string()),
  // false = answered from the model's own knowledge because retrieval found
  // no usable passage (llm/synthesize.ts's answerFromGeneralKnowledge()).
  // Such an answer has passed NO grounding check and carries no citations,
  // so every consumer must label it as unverified rather than rendering it
  // like a grounded one -- presenting an ungrounded answer as grounded is
  // the exact failure mode maun/ exists to prevent.
  grounded: z.boolean(),
});
export type SynthesizedAnswer = z.infer<typeof SynthesizedAnswer>;

export const SynthesisResponse = z.object({
  // `refused` describes the RETRIEVAL verdict and stays true even when a
  // `synthesized` answer is present: the corpus really did decline, and the
  // trace/telemetry should keep saying so. A non-null `synthesized` with
  // grounded=false alongside refused=true is the general-knowledge fallback.
  refused: z.boolean(),
  refusalReason: RefusalReason.optional(),
  synthesized: SynthesizedAnswer.nullable(),
});
export type SynthesisResponse = z.infer<typeof SynthesisResponse>;

export const GroundedAnswer = z.object({
  answer: z.string(),
  citations: z.array(z.string()).min(1), // chunk_ids -- enforced non-empty when not refused
  confidence: z.number().min(0).max(1),
  refused: z.boolean(),
  refusalReason: RefusalReason.optional(),
  // "empty_or_gibberish" replaces ARCHITECTURE.md's original "low_asr" --
  // there is no ASR confidence signal to gate on (see Transcript above).
  // Optional so a stage that never populates it (or an older server build)
  // degrades honestly on the frontend instead of the UI inventing timings
  // -- see web/src/components/Waterfall.tsx.
  trace: z.array(Span).optional(),
  // Whether this answer was served from harness/cache.ts's semantic cache
  // (cos>0.97 against a prior query) instead of a fresh retrieval -- lets
  // callers (bench/latency.ts) report cached/uncached latency separately,
  // CLAUDE.md invariant #4 -- never conflate the two.
  cached: z.boolean().optional(),
  // Roman-script transliteration of `answer` for hi/bn/ta (krama/romanize.ts)
  // -- same words, not a translation, matching how users already type
  // "Hinglish"/"Banglish"/"Tanglish". Absent for English answers (nothing to
  // romanize) or refusals (no answer text to transliterate).
  answerRomanized: z.string().optional(),
});
export type GroundedAnswer = z.infer<typeof GroundedAnswer>;
