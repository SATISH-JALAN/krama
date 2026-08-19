// Mirrors server/harness/contracts.ts (Zod schemas). Kept as plain TS types
// here rather than importing zod client-side — the frontend only needs the
// shape, not runtime validation of its own outgoing requests.

export type RefusalReason = "off_topic" | "unsafe" | "empty_or_gibberish" | "no_grounding";

export interface GroundedAnswer {
  answer: string;
  citations: string[];
  confidence: number;
  refused: boolean;
  refusalReason?: RefusalReason;
  // Not yet emitted by server/index.ts's handleQuery (it computes spans but
  // doesn't return them — see MEMORY.md). Optional so a real server response
  // degrades honestly instead of the UI inventing timings.
  trace?: Span[];
  // Populated by a separate, later call to /query/synthesize (lib/api.ts's
  // querySynthesize), not by /query itself -- the slow LLM path stays off
  // the t0->t1 core budget (CLAUDE.md #4), so App.tsx fetches it after the
  // fast answer already landed and merges it in here.
  synthesized?: SynthesizedAnswer;
  // Mirrors server/harness/contracts.ts's GroundedAnswer.cached -- whether
  // this answer came from the semantic cache (cos>0.97) instead of a fresh
  // retrieval. CLAUDE.md invariant #4: never conflate cached/uncached
  // latency, so the frontend badge needs this to label which one it's
  // showing rather than presenting a cache hit's ~9ms as the general number.
  cached?: boolean;
  // Roman-script transliteration of `answer` (server/krama/romanize.ts) --
  // same words, not a translation. Present for hi/bn/ta answers, absent for
  // English (nothing to romanize) or refusals.
  answerRomanized?: string;
}

const ROMANIZATION_LABEL: Record<string, string> = {
  hi: "Hinglish",
  bn: "Banglish",
  ta: "Tanglish",
};

export function romanizationLabel(lang: string): string | null {
  return ROMANIZATION_LABEL[lang] ?? null;
}

export interface Span {
  name: string;
  ms: number;
  ok: boolean;
  err?: string;
}

// Mirrors server/harness/contracts.ts's SynthesizedAnswer/SynthesisResponse.
export interface SynthesizedAnswer {
  answer: string;
  streaming: boolean;
  // True when a provider answered successfully but judged the retrieved
  // passage didn't actually answer the question (llm/synthesize.ts's own
  // prompt instructs this) -- distinct from the request failing outright.
  declined: boolean;
  provider: string;
  citedChunkIds: string[];
  // false = answered from the model's own knowledge because retrieval found
  // nothing usable. Carries no citations and passed no grounding check, so
  // it MUST be rendered as unverified rather than styled like a grounded
  // answer. Optional only so an older server build (which omits the field)
  // degrades to the cautious reading rather than the confident one.
  grounded?: boolean;
}

export interface SynthesisResponse {
  refused: boolean;
  refusalReason?: RefusalReason;
  synthesized: SynthesizedAnswer | null;
}

// t0->t1 core latency (CLAUDE.md #4: transcript-in -> grounded-answer-out,
// STT and LLM synthesis excluded -- consistent with what bench/latency.ts
// measures, since STT already happened before handleQuery() built this
// trace). Shared by Header's live latency badge and Waterfall's total so the
// two numbers can never drift apart.
export function totalTraceMs(trace?: Span[]): number | null {
  if (!trace || trace.length === 0) return null;
  return trace.reduce((sum, s) => sum + s.ms, 0);
}

export const MODULE_GLOSSARY: Record<string, string> = {
  shruti: "that which is heard — voice ingress / STT",
  krama: "step-by-step recitation — chunking + extractive ranking",
  ghana: "dense recitation — the ANN vector index",
  jata: "braided recitation — hybrid rank fusion (RRF)",
  maun: "silence — the guardrails that decide not to answer",
};

// Maps a trace span's stage name to the module that owns it, for the
// waterfall's secondary caption. Falls back to no module tag if unmapped.
export function moduleForSpan(name: string): string | null {
  if (name.startsWith("l0") || name.startsWith("l1") || name.startsWith("l2") || name.startsWith("l4"))
    return "maun";
  if (name.includes("embed")) return "ghana";
  if (name.includes("dense_search")) return "ghana";
  if (name.includes("bm25")) return "krama";
  if (name.includes("fuse")) return "jata";
  if (name.includes("extract")) return "krama";
  return null;
}
