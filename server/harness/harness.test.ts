import { describe, expect, test } from "bun:test";
import { z } from "zod";
import { runStage, withTimeout, StageError } from "./pipeline";
import { withRetry } from "./retry";
import { CircuitBreaker } from "./breaker";

describe("pipeline.runStage", () => {
  test("returns parsed output and records a successful span", async () => {
    const trace: import("./contracts").Span[] = [];
    const out = await runStage(
      "double",
      21,
      z.number(),
      async (n) => n * 2,
      trace,
    );
    expect(out).toBe(42);
    expect(trace).toHaveLength(1);
    expect(trace[0].name).toBe("double");
    expect(trace[0].ok).toBe(true);
    expect(trace[0].ms).toBeGreaterThanOrEqual(0);
  });

  test("throws StageError and records a failed span when fn rejects", async () => {
    const trace: import("./contracts").Span[] = [];
    await expect(
      runStage("boom", null, z.unknown(), async () => {
        throw new Error("kaboom");
      }, trace),
    ).rejects.toThrow(StageError);
    expect(trace).toHaveLength(1);
    expect(trace[0].ok).toBe(false);
    expect(trace[0].err).toContain("kaboom");
  });

  test("throws and records a failed span when schema validation fails", async () => {
    const trace: import("./contracts").Span[] = [];
    await expect(
      runStage("badShape", null, z.string(), async () => 123 as unknown as string, trace),
    ).rejects.toThrow(StageError);
    expect(trace[0].ok).toBe(false);
  });

  test("withTimeout rejects slow promises", async () => {
    const slow = new Promise((resolve) => setTimeout(resolve, 200));
    await expect(withTimeout(slow, 20)).rejects.toThrow(/timed out/);
  });
});

describe("retry.withRetry", () => {
  test("returns on first success without retrying", async () => {
    let calls = 0;
    const result = await withRetry(
      async () => {
        calls++;
        return "ok";
      },
      { maxAttempts: 3, baseDelayMs: 10, deadlineMs: 1000 },
    );
    expect(result).toBe("ok");
    expect(calls).toBe(1);
  });

  test("retries on failure and eventually succeeds", async () => {
    let calls = 0;
    const result = await withRetry(
      async () => {
        calls++;
        if (calls < 3) throw new Error("transient");
        return "ok";
      },
      { maxAttempts: 5, baseDelayMs: 5, deadlineMs: 2000 },
    );
    expect(result).toBe("ok");
    expect(calls).toBe(3);
  });

  test("gives up after maxAttempts and throws the last error", async () => {
    let calls = 0;
    await expect(
      withRetry(
        async () => {
          calls++;
          throw new Error(`fail ${calls}`);
        },
        { maxAttempts: 3, baseDelayMs: 5, deadlineMs: 2000 },
      ),
    ).rejects.toThrow("fail 3");
    expect(calls).toBe(3);
  });

  test("stops retrying once the deadline is exhausted, not after maxAttempts", async () => {
    let calls = 0;
    const start = performance.now();
    await expect(
      withRetry(
        async () => {
          calls++;
          throw new Error("always fails");
        },
        { maxAttempts: 100, baseDelayMs: 50, deadlineMs: 120 },
      ),
    ).rejects.toThrow();
    const elapsed = performance.now() - start;
    expect(calls).toBeLessThan(100); // deadline cut it off long before 100 attempts
    expect(elapsed).toBeLessThan(500); // generous margin, but proves it didn't run to maxAttempts
  });
});

describe("breaker.CircuitBreaker", () => {
  test("starts closed and allows attempts", () => {
    const b = new CircuitBreaker();
    expect(b.getState()).toBe("closed");
    expect(b.canAttempt()).toBe(true);
  });

  test("opens after failureThreshold failures within the window", () => {
    const b = new CircuitBreaker({ failureThreshold: 3, windowMs: 30_000, openMs: 60_000 });
    const t = 1_000_000;
    b.onFailure(t);
    b.onFailure(t + 100);
    expect(b.getState()).toBe("closed");
    b.onFailure(t + 200);
    expect(b.getState()).toBe("open");
    expect(b.canAttempt(t + 300)).toBe(false);
  });

  test("old failures outside the window don't count toward the threshold", () => {
    const b = new CircuitBreaker({ failureThreshold: 3, windowMs: 1000, openMs: 60_000 });
    const t = 1_000_000;
    b.onFailure(t);
    b.onFailure(t + 5000); // outside the 1000ms window relative to the first failure
    b.onFailure(t + 5100);
    // only 2 failures within any 1000ms window at this point
    expect(b.getState()).toBe("closed");
  });

  test("transitions to half-open after openMs, then closed on a successful probe", () => {
    const b = new CircuitBreaker({ failureThreshold: 1, windowMs: 30_000, openMs: 1000 });
    const t = 1_000_000;
    b.onFailure(t);
    expect(b.getState()).toBe("open");
    expect(b.canAttempt(t + 500)).toBe(false); // still within openMs
    expect(b.canAttempt(t + 1500)).toBe(true); // openMs elapsed -> half-open probe allowed
    expect(b.getState()).toBe("half-open");
    b.onSuccess();
    expect(b.getState()).toBe("closed");
  });

  test("a failed half-open probe reopens immediately, not after threshold failures again", () => {
    const b = new CircuitBreaker({ failureThreshold: 5, windowMs: 30_000, openMs: 1000 });
    const t = 1_000_000;
    for (let i = 0; i < 5; i++) b.onFailure(t + i);
    expect(b.getState()).toBe("open");
    b.canAttempt(t + 2000); // triggers half-open transition
    expect(b.getState()).toBe("half-open");
    b.onFailure(t + 2001); // single failure while half-open
    expect(b.getState()).toBe("open"); // back open immediately, not waiting for 5 more failures
  });
});
