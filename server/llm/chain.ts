/**
 * llm -- fallback chain: Groq -> Cerebras -> extractive-only
 * (ARCHITECTURE.md §7). Ties together retry.ts (per-provider retry with a
 * deadline) and breaker.ts (per-provider circuit breaker) -- a provider
 * that's currently open gets skipped without even attempting a call, so a
 * known-down provider never eats into the latency budget on every request.
 *
 * Provider clients (groq.ts, cerebras.ts) are not implemented yet -- this
 * module only implements the fallback/routing logic and is tested with
 * mock providers, which is enough to verify the chain behavior itself
 * (order, breaker skipping, eventual extractive fallback) independent of
 * any real API integration.
 */
import type { CircuitBreaker } from "../harness/breaker";
import { withRetry, type RetryOptions } from "../harness/retry";

export interface LlmProvider {
  name: string;
  generate: (prompt: string) => Promise<string>;
}

export interface GenerateResult {
  text: string | null;
  provider: string; // a provider's name, or "extractive_fallback" if all failed/skipped
}

export async function generateWithFallback(
  providers: LlmProvider[],
  breakers: Map<string, CircuitBreaker>,
  prompt: string,
  retryOpts: RetryOptions,
): Promise<GenerateResult> {
  for (const provider of providers) {
    const breaker = breakers.get(provider.name);
    if (breaker && !breaker.canAttempt()) continue; // skip a known-down provider entirely

    try {
      const text = await withRetry(() => provider.generate(prompt), retryOpts);
      breaker?.onSuccess();
      return { text, provider: provider.name };
    } catch {
      breaker?.onFailure();
      // fall through to the next provider
    }
  }

  return { text: null, provider: "extractive_fallback" };
}
