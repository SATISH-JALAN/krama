import { describe, expect, test } from "bun:test";
import { CircuitBreaker } from "../harness/breaker";
import { generateWithFallback, type LlmProvider } from "./chain";

const FAST_RETRY = { maxAttempts: 2, baseDelayMs: 5, deadlineMs: 500 };

function mockBreakers(names: string[]): Map<string, CircuitBreaker> {
  const m = new Map<string, CircuitBreaker>();
  for (const name of names) m.set(name, new CircuitBreaker());
  return m;
}

describe("generateWithFallback", () => {
  test("returns the first provider's result when it succeeds", async () => {
    const providers: LlmProvider[] = [
      { name: "groq", generate: async () => "groq answer" },
      { name: "cerebras", generate: async () => "cerebras answer" },
    ];
    const result = await generateWithFallback(
      providers,
      mockBreakers(["groq", "cerebras"]),
      "prompt",
      FAST_RETRY,
    );
    expect(result.provider).toBe("groq");
    expect(result.text).toBe("groq answer");
  });

  test("falls through to the second provider when the first fails", async () => {
    const providers: LlmProvider[] = [
      { name: "groq", generate: async () => { throw new Error("rate limited"); } },
      { name: "cerebras", generate: async () => "cerebras answer" },
    ];
    const result = await generateWithFallback(
      providers,
      mockBreakers(["groq", "cerebras"]),
      "prompt",
      FAST_RETRY,
    );
    expect(result.provider).toBe("cerebras");
    expect(result.text).toBe("cerebras answer");
  });

  test("falls back to extractive_fallback when every provider fails", async () => {
    const providers: LlmProvider[] = [
      { name: "groq", generate: async () => { throw new Error("down"); } },
      { name: "cerebras", generate: async () => { throw new Error("down"); } },
    ];
    const result = await generateWithFallback(
      providers,
      mockBreakers(["groq", "cerebras"]),
      "prompt",
      FAST_RETRY,
    );
    expect(result.provider).toBe("extractive_fallback");
    expect(result.text).toBeNull();
  });

  test("skips a provider whose breaker is open without calling it", async () => {
    let groqCalls = 0;
    const providers: LlmProvider[] = [
      { name: "groq", generate: async () => { groqCalls++; return "groq answer"; } },
      { name: "cerebras", generate: async () => "cerebras answer" },
    ];
    const breakers = mockBreakers(["groq", "cerebras"]);
    // force groq's breaker open
    const groqBreaker = breakers.get("groq")!;
    for (let i = 0; i < 5; i++) groqBreaker.onFailure();
    expect(groqBreaker.getState()).toBe("open");

    const result = await generateWithFallback(providers, breakers, "prompt", FAST_RETRY);
    expect(groqCalls).toBe(0); // never attempted
    expect(result.provider).toBe("cerebras");
  });

  test("a successful call resets that provider's breaker", async () => {
    const providers: LlmProvider[] = [
      { name: "groq", generate: async () => "ok" },
    ];
    const breakers = mockBreakers(["groq"]);
    const breaker = breakers.get("groq")!;
    breaker.onFailure();
    breaker.onFailure();

    await generateWithFallback(providers, breakers, "prompt", FAST_RETRY);
    expect(breaker.getState()).toBe("closed");
  });
});
