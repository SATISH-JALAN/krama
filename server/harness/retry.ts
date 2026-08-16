/**
 * Retry with exponential backoff + jitter, hard-capped by remaining budget
 * (ARCHITECTURE.md §7: "a retry that blows the deadline is worse than a
 * degraded answer"). Used for STT/LLM calls, never for local/in-process
 * stages -- those should just fail fast into the fallback chain instead.
 */
import { withTimeout } from "./pipeline";

export interface RetryOptions {
  maxAttempts: number;
  baseDelayMs: number;
  deadlineMs: number; // total time budget across all attempts, from first call
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function withRetry<T>(fn: () => Promise<T>, opts: RetryOptions): Promise<T> {
  const start = performance.now();
  let lastErr: unknown = new Error("withRetry called with maxAttempts <= 0");

  for (let attempt = 0; attempt < opts.maxAttempts; attempt++) {
    const remaining = opts.deadlineMs - (performance.now() - start);
    if (remaining <= 0) break;

    try {
      return await withTimeout(fn(), remaining);
    } catch (e) {
      lastErr = e;
      const isLastAttempt = attempt === opts.maxAttempts - 1;
      if (isLastAttempt) break;

      const backoff = opts.baseDelayMs * 2 ** attempt;
      const jitter = Math.random() * backoff * 0.3;
      const stillRemaining = opts.deadlineMs - (performance.now() - start);
      const delay = Math.min(backoff + jitter, stillRemaining);
      if (delay <= 0) break;
      await sleep(delay);
    }
  }

  throw lastErr;
}
