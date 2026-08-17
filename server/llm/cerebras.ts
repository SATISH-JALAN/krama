/**
 * llm -- real Cerebras provider, fallback for Groq (CLAUDE.md stack table:
 * "Cerebras via OpenAI-compatible fetch (avoid `zai-glm-4.7`)"). No
 * `cerebras-sdk` dependency -- their API is OpenAI-compatible, a plain
 * `fetch` is the whole client, per the original architecture decision.
 *
 * Model default verified via live search 2026-08-17 (re-verify if stale,
 * same discipline as CLAUDE.md's other provider facts): `gpt-oss-120b` is
 * currently the ONLY model in Cerebras' "production" tier
 * (inference-docs.cerebras.ai/models/overview). `gemma-4-31b` is
 * preview-only (evaluation, not for reliance); `zai-glm-4.7` deprecates
 * 2026-08-17 -- today, as of this writing -- matching CLAUDE.md's existing
 * warning, not a new finding, just confirms it.
 *
 * `fetchImpl` is injectable so the request/response wiring is
 * unit-testable without a real CEREBRAS_API_KEY, same reasoning as
 * groq.ts's injectable client.
 */
import { withTimeout } from "../harness/pipeline";
import type { LlmProvider } from "./chain";

export const DEFAULT_CEREBRAS_MODEL = "gpt-oss-120b";
const CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions";
const GENERATE_TIMEOUT_MS = 20_000;

interface CerebrasChatResponse {
  choices?: { message?: { content?: string } }[];
}

export function createCerebrasProvider(
  apiKey: string,
  opts: { model?: string; fetchImpl?: typeof fetch } = {},
): LlmProvider {
  const model = opts.model ?? DEFAULT_CEREBRAS_MODEL;
  const fetchImpl = opts.fetchImpl ?? fetch;

  return {
    name: "cerebras",
    async generate(prompt: string): Promise<string> {
      const res = await withTimeout(
        fetchImpl(CEREBRAS_URL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${apiKey}`,
          },
          body: JSON.stringify({
            model,
            messages: [{ role: "user", content: prompt }],
            temperature: 0.2,
            max_tokens: 800,
          }),
        }),
        GENERATE_TIMEOUT_MS,
      );

      if (!res.ok) {
        const body = await res.text().catch(() => "<unreadable body>");
        throw new Error(`cerebras: HTTP ${res.status} ${body}`);
      }

      const data = (await res.json()) as CerebrasChatResponse;
      const text = data.choices?.[0]?.message?.content;
      if (!text) throw new Error("cerebras: empty completion");
      return text;
    },
  };
}
