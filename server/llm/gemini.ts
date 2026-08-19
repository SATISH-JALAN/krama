/**
 * llm -- real Gemini provider. No `@google/genai` SDK dependency -- same
 * "a plain fetch is the whole client" decision as cerebras.ts, avoiding an
 * extra dependency for a single REST call.
 *
 * Endpoint/auth/request-response shape verified live against
 * ai.google.dev's own API reference before writing this (CLAUDE.md #6:
 * never invent an API fact), not assumed from memory:
 *   POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}
 *   body: {"contents":[{"role":"user","parts":[{"text": prompt}]}], "generationConfig":{...}}
 *   response text at: candidates[0].content.parts[0].text
 * Model default `gemini-3.6-flash` -- also verified live (ai.google.dev's
 * models page, 2026-08-20): Google's current general-purpose free-tier
 * Flash model, balancing speed and capability for everyday tasks like this
 * one (short grounded-QA JSON generation, not complex agentic reasoning),
 * rather than `3.5-flash-lite` (cheaper but weaker at judgment calls like
 * "do these passages actually answer this question") or `3.7-flash`
 * (overkill, and a tighter free-tier quota for a task this simple).
 *
 * Real bug found and fixed before shipping, not assumed correct: Gemini 3.x
 * models default to "dynamic thinking" (medium/high extended reasoning)
 * even for simple prompts -- confirmed live, the actual synthesis prompt
 * (structured JSON extraction over one short passage) took >20s and timed
 * out entirely under the default config, while a trivial one-word prompt
 * returned instantly. `generationConfig.thinkingConfig.thinkingLevel:
 * "minimal"` (the REST field for Gemini 3.x models specifically --
 * `thinkingBudget` is the equivalent for 2.5 models, verified against
 * ai.google.dev's own generateContent thinking-config docs) fixes this;
 * Google's own docs recommend exactly this setting for "high-volume
 * extraction, routing, or classification tasks", which is what this is.
 *
 * `fetchImpl` is injectable, same DI pattern as cerebras.ts, so the
 * request-shape logic is unit-testable without a real GEMINI_API_KEY.
 */
import { withTimeout } from "../harness/pipeline";
import type { LlmProvider } from "./chain";

export const DEFAULT_GEMINI_MODEL = "gemini-3.6-flash";

/**
 * Tried in order, each as its OWN provider entry with its own circuit
 * breaker (see createGeminiProvider's `name`), so chain.ts falls through to
 * the next model when one is exhausted rather than giving up on Gemini.
 *
 * This exists because of a real, measured failure, not defensiveness: the
 * free tier meters requests-per-day PER MODEL, and gemini-3.6-flash's cap is
 * 20/day (quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier,
 * read off a live 429 response). Twenty requests is a single testing
 * session, after which every synthesis and every general-knowledge fallback
 * silently returned nothing. Because the cap is per-model, a second model on
 * the SAME key has its own separate budget -- so falling through actually
 * buys capacity, which a retry against the same model never could.
 *
 * Ordered best-first: 3.6-flash leads on answer quality, the flash-lite
 * models are the backstop. Both fallbacks were verified live against this
 * project's real general-knowledge prompt (correct Hindi answer, and they
 * accept the thinkingConfig field below) rather than assumed compatible.
 * Once 3.6-flash's daily cap trips, its breaker opens and later requests
 * skip straight to a lite model instead of paying a doomed call first.
 */
export const GEMINI_MODEL_CHAIN = [
  DEFAULT_GEMINI_MODEL,
  "gemini-3.5-flash-lite",
  "gemini-3.1-flash-lite",
];
const GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models";
// Off the t0->t1 core budget (CLAUDE.md #4), but not unbounded. 25s, not
// 20s: measured live even with thinkingLevel:"minimal", single-call latency
// ranged 1.9s-19.3s across 5 real requests for the same short prompt --
// "minimal" still allows some reasoning tokens, plus real free-tier
// queueing variance. 20s clipped a real request mid-flight during testing.
const GENERATE_TIMEOUT_MS = 25_000;

interface GeminiGenerateResponse {
  candidates?: { content?: { parts?: { text?: string }[] } }[];
}

export function createGeminiProvider(
  apiKey: string,
  opts: { model?: string; fetchImpl?: typeof fetch } = {},
): LlmProvider {
  const model = opts.model ?? DEFAULT_GEMINI_MODEL;
  const fetchImpl = opts.fetchImpl ?? fetch;
  const url = `${GEMINI_BASE_URL}/${model}:generateContent?key=${apiKey}`;

  return {
    // Model-qualified, because harness/breaker.ts state is keyed by this
    // name: several Gemini models share one API key but NOT one quota, so a
    // bare "gemini" would make one model's daily-cap 429 open the breaker
    // for every other model too -- defeating the fallthrough entirely.
    name: `gemini:${model}`,
    async generate(prompt: string): Promise<string> {
      const res = await withTimeout(
        fetchImpl(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            contents: [{ role: "user", parts: [{ text: prompt }] }],
            generationConfig: {
              temperature: 0.2,
              maxOutputTokens: 800,
              thinkingConfig: { thinkingLevel: "minimal" },
            },
          }),
        }),
        GENERATE_TIMEOUT_MS,
      );

      if (!res.ok) {
        const body = await res.text().catch(() => "<unreadable body>");
        // Never let a raw error body reach logs/traces verbatim -- it can
        // echo the request URL, which carries the API key as a query param
        // (the officially documented auth method for this endpoint).
        throw new Error(`gemini: HTTP ${res.status} ${body.slice(0, 300)}`);
      }

      const data = (await res.json()) as GeminiGenerateResponse;
      const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
      if (!text) throw new Error("gemini: empty completion");
      return text;
    },
  };
}
