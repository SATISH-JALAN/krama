/**
 * llm -- structured output + repair (ARCHITECTURE.md §7: "force JSON from
 * the LLM, validate with Zod, one repair attempt with the validation error
 * fed back, then fall back to extractive"). This is what turns chain.ts's
 * raw provider text into KRAMA's actual answer shape, and is the only place
 * that knows the synthesis prompt/schema -- chain.ts stays generic.
 */
import { z } from "zod";
import type { CircuitBreaker } from "../harness/breaker";
import type { RetryOptions } from "../harness/retry";
import { generateWithFallback, type LlmProvider } from "./chain";

export const LlmAnswer = z.object({
  answer: z.string(),
  citedChunkIds: z.array(z.string()),
});
export type LlmAnswer = z.infer<typeof LlmAnswer>;

export interface SynthesisCandidate {
  chunkId: string;
  text: string;
}

export interface SynthesisResult {
  answer: string;
  citedChunkIds: string[];
  provider: string;
}

function buildPrompt(query: string, lang: string, candidates: SynthesisCandidate[]): string {
  const context = candidates.map((c) => `[${c.chunkId}] ${c.text}`).join("\n\n");
  return `You are answering a question using ONLY the passages below. Do not use outside knowledge, even if you know the answer.
Respond in the same language as the question (language code: ${lang}).
Respond with ONLY a single JSON object, no markdown code fences, no commentary, matching exactly this shape:
{"answer": string, "citedChunkIds": string[]}

"citedChunkIds" must be a subset of the bracketed passage ids below, and must be non-empty whenever "answer" is non-empty.
If none of the passages actually answer the question, respond with {"answer": "", "citedChunkIds": []}.

Passages:
${context}

Question: ${query}`;
}

// Language names the model actually recognizes -- passing a bare ISO code
// ("bn") into a natural-language instruction is much weaker steering than
// naming the language, and register instructions below need the model to be
// confident about which language it's writing.
const LANG_NAMES: Record<string, string> = {
  hi: "Hindi",
  bn: "Bengali",
  ta: "Tamil",
  en: "English",
};

/**
 * The ungrounded path's prompt. Deliberately NOT the JSON/citation shape
 * buildPrompt() uses: there are no passages here, so there is nothing to
 * cite, and demanding a citation array the model cannot legitimately fill
 * is how you teach it to invent ids.
 *
 * The register instructions are the fix for a real, user-reported problem:
 * the extractive path can only ever return the corpus's own wording, which
 * for MS MARCO-XI is stiff machine-translated prose ("pure Hindi which
 * rarely people knows about", in the report). Here the wording is ours to
 * choose, so it asks explicitly for the everyday spoken register -- the
 * common English loanwords Indian speakers actually use out loud, rather
 * than the Sanskritised formal equivalents a model defaults to when it is
 * told only "answer in Hindi".
 */
function buildGeneralKnowledgePrompt(query: string, lang: string): string {
  const langName = LANG_NAMES[lang] ?? lang;
  return `Answer the question below from your own general knowledge, in ${langName}.

Rules:
- Write the way people actually speak ${langName} in everyday conversation, NOT in a formal, literary, or textbook register. Keep the common English loanwords that ordinary speakers use out loud instead of substituting rare formal equivalents.
- Keep it to 2-3 short sentences. Be direct: answer the question first, then at most one sentence of context.
- If you are genuinely unsure of the answer, say so plainly in ${langName} rather than guessing.
- Output ONLY the answer text itself. No preamble, no JSON, no markdown, no quotes around it.

Question: ${query}`;
}

/**
 * Answers WITHOUT corpus grounding, from the model's own knowledge.
 *
 * Exists because a refusal used to be a dead end: real testing found the
 * corpus simply has no passage on many perfectly reasonable questions (it
 * is an MS MARCO web-passage subset, not an encyclopedia -- "who built the
 * Taj Mahal" has zero matching passages in all 79,540), so the honest
 * retrieval answer is "I can't", which is correct but useless to a user.
 *
 * Routing those to this function is what makes it safe to TIGHTEN the L3
 * relevance gate rather than loosen it. The alternative -- relaxing
 * thresholds until something always comes back -- is exactly how the
 * reference implementation ends up serving an unrelated passage under a
 * "GROUNDED ANSWER / HALLUCINATION UNDETECTED" banner. The caller MUST
 * surface the resulting answer as ungrounded (SynthesizedAnswer.grounded =
 * false); an answer from here carries no citations and has not passed any
 * grounding check, and presenting it as if it had is the precise failure
 * this whole module exists to avoid.
 *
 * Returns null if every provider failed (network/quota/timeout) -- callers
 * degrade to the plain refusal, same convention as synthesizeAnswer().
 */
export async function answerFromGeneralKnowledge(
  query: string,
  lang: string,
  providers: LlmProvider[],
  breakers: Map<string, CircuitBreaker>,
  retryOpts: RetryOptions,
): Promise<SynthesisResult | null> {
  const prompt = buildGeneralKnowledgePrompt(query, lang);
  const result = await generateWithFallback(providers, breakers, prompt, retryOpts);
  if (!result.text) return null;

  // Plain text, so there is no JSON contract to validate or repair -- the
  // only cleanup needed is stripping a code fence the model may have added
  // anyway (same defensive reason stripCodeFence() exists for the JSON path).
  const answer = stripCodeFence(result.text).trim();
  if (answer.length === 0) return null;

  return { answer, citedChunkIds: [], provider: result.provider };
}

function buildRepairPrompt(originalPrompt: string, badOutput: string, error: string): string {
  return `${originalPrompt}

Your previous response was:
${badOutput}

That response was INVALID: ${error}
Respond again with ONLY the corrected JSON object -- no markdown code fences, no commentary.`;
}

type ParsedLlmJson = { ok: true; value: LlmAnswer } | { ok: false; error: string };

// Real models add a ```json fence despite being told not to often enough
// that stripping one defensively is worth it before treating it as invalid.
function stripCodeFence(raw: string): string {
  return raw
    .trim()
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/```\s*$/, "");
}

export function parseLlmJson(raw: string): ParsedLlmJson {
  let parsedJson: unknown;
  try {
    parsedJson = JSON.parse(stripCodeFence(raw));
  } catch (e) {
    return { ok: false, error: `not valid JSON: ${String(e)}` };
  }
  const result = LlmAnswer.safeParse(parsedJson);
  if (!result.success) return { ok: false, error: result.error.message };
  return { ok: true, value: result.data };
}

// A structurally-valid LlmAnswer can still be wrong: citing a chunk id that
// was never offered as context (a model hallucinating its own citation),
// or a non-empty answer with no citations at all. Both are the same class
// of failure as invalid JSON -- worth a repair attempt, not silent accept.
function validateAgainstCandidates(
  value: LlmAnswer,
  candidates: SynthesisCandidate[],
): { ok: true } | { ok: false; error: string } {
  const validIds = new Set(candidates.map((c) => c.chunkId));
  const unknownIds = value.citedChunkIds.filter((id) => !validIds.has(id));
  if (unknownIds.length > 0) {
    return { ok: false, error: `citedChunkIds references passages not offered as context: ${JSON.stringify(unknownIds)}` };
  }
  if (value.answer.trim().length > 0 && value.citedChunkIds.length === 0) {
    return { ok: false, error: "answer is non-empty but citedChunkIds is empty" };
  }
  return { ok: true };
}

/**
 * Returns null if every provider failed/skipped, or the repair attempt
 * still didn't produce valid+grounded JSON -- callers should fall back to
 * the extractive fast-path answer in either case, exactly like chain.ts's
 * own `text: null` / "extractive_fallback" convention.
 */
export async function synthesizeAnswer(
  candidates: SynthesisCandidate[],
  query: string,
  lang: string,
  providers: LlmProvider[],
  breakers: Map<string, CircuitBreaker>,
  retryOpts: RetryOptions,
): Promise<SynthesisResult | null> {
  const prompt = buildPrompt(query, lang, candidates);
  const first = await generateWithFallback(providers, breakers, prompt, retryOpts);
  if (!first.text) return null;

  const firstParsed = parseLlmJson(first.text);
  if (firstParsed.ok) {
    const validation = validateAgainstCandidates(firstParsed.value, candidates);
    if (validation.ok) return { ...firstParsed.value, provider: first.provider };
  }

  // one repair attempt (ARCHITECTURE.md §7) -- feed the actual validation
  // error back and re-run the fallback chain once more. Re-running the
  // whole chain (not pinning to whichever provider answered first) keeps
  // this simple and still respects breaker state carried over from the
  // first attempt.
  const errorForRepair = firstParsed.ok
    ? (validateAgainstCandidates(firstParsed.value, candidates) as { ok: false; error: string }).error
    : firstParsed.error;
  const repairPrompt = buildRepairPrompt(prompt, first.text, errorForRepair);
  const second = await generateWithFallback(providers, breakers, repairPrompt, retryOpts);
  if (!second.text) return null;

  const secondParsed = parseLlmJson(second.text);
  if (!secondParsed.ok) return null;
  const secondValidation = validateAgainstCandidates(secondParsed.value, candidates);
  if (!secondValidation.ok) return null;

  return { ...secondParsed.value, provider: second.provider };
}
