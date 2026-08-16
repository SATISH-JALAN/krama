/**
 * Stage runner + tracing (ARCHITECTURE.md §7). Every pipeline stage goes
 * through runStage: typed input/output (Zod), a timeout, and a trace entry
 * pushed regardless of success or failure. The trace array is what gets
 * returned to the client and rendered as the waterfall UI (Sprint 4) --
 * this one function is what satisfies "structured I/O + error recovery"
 * (requirement #5) and "guardrails show their work" (requirement #6) at once.
 */
import type { z } from "zod";
import type { Span } from "./contracts";

export class StageError extends Error {
  constructor(
    public readonly stage: string,
    public readonly cause: unknown,
  ) {
    super(`stage "${stage}" failed: ${String(cause)}`);
    this.name = "StageError";
  }
}

export function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`timed out after ${ms}ms`)), ms);
    promise.then(
      (v) => {
        clearTimeout(timer);
        resolve(v);
      },
      (e) => {
        clearTimeout(timer);
        reject(e);
      },
    );
  });
}

export async function runStage<I, O>(
  name: string,
  input: I,
  schema: z.ZodType<O>,
  fn: (input: I) => Promise<O>,
  trace: Span[],
  opts?: { timeoutMs?: number },
): Promise<O> {
  const t0 = performance.now();
  try {
    const raw = await withTimeout(fn(input), opts?.timeoutMs ?? 5000);
    const parsed = schema.parse(raw);
    trace.push({ name, ms: performance.now() - t0, ok: true });
    return parsed;
  } catch (e) {
    trace.push({ name, ms: performance.now() - t0, ok: false, err: String(e) });
    throw new StageError(name, e);
  }
}
