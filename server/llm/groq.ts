/**
 * llm -- real Groq provider (ARCHITECTURE.md §2.1, CLAUDE.md stack table:
 * `openai/gpt-oss-20b`/`120b` or `qwen/qwen3.6-27b` -- the older llama
 * models were deprecated 2026-08-16). Implements chain.ts's LlmProvider
 * interface; chain.ts owns retry/breaker/fallback-ordering, this file only
 * knows how to make ONE call to Groq -- do not add retry logic here, it
 * would double up with generateWithFallback's withRetry.
 *
 * `client` is injectable so the request/response wiring (model name,
 * message shape, empty-completion handling) can be unit-tested without a
 * real GROQ_API_KEY -- unlike ghana/hnsw.ts, this code path IS testable
 * without live credentials, only the actual network round-trip to Groq's
 * servers isn't. That round-trip itself remains genuinely unverified until
 * a real key is supplied and this is exercised for real.
 */
import Groq from "groq-sdk";
import { withTimeout } from "../harness/pipeline";
import type { LlmProvider } from "./chain";

export const DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b";
const GENERATE_TIMEOUT_MS = 20_000; // off the t0->t1 core budget (CLAUDE.md #4), but not unbounded

type ChatClient = Pick<Groq, "chat">;

export function createGroqProvider(
  apiKey: string,
  opts: { model?: string; client?: ChatClient } = {},
): LlmProvider {
  const client = opts.client ?? new Groq({ apiKey });
  const model = opts.model ?? DEFAULT_GROQ_MODEL;

  return {
    name: "groq",
    async generate(prompt: string): Promise<string> {
      const completion = await withTimeout(
        client.chat.completions.create({
          model,
          messages: [{ role: "user", content: prompt }],
          temperature: 0.2,
          max_tokens: 800,
        }),
        GENERATE_TIMEOUT_MS,
      );
      const text = completion.choices[0]?.message?.content;
      if (!text) throw new Error("groq: empty completion");
      return text;
    },
  };
}
