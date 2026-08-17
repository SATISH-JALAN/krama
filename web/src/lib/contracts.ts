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
  // Same gap: the slow LLM-synthesized path isn't wired into handleQuery yet.
  synthesized?: {
    answer: string;
    streaming: boolean;
  };
}

export interface Span {
  name: string;
  ms: number;
  ok: boolean;
  err?: string;
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
