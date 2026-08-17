import { describe, expect, test } from "bun:test";
import { parseLlmJson, synthesizeAnswer } from "./synthesize";
import type { LlmProvider } from "./chain";

const CANDIDATES = [
  { chunkId: "chunk_a", text: "A corporation is a legal entity separate from its owners." },
  { chunkId: "chunk_b", text: "Corporations can issue stock." },
];

function providerReturning(...responses: string[]): LlmProvider {
  let i = 0;
  return {
    name: "test-provider",
    generate: async () => {
      const r = responses[Math.min(i, responses.length - 1)];
      i++;
      return r;
    },
  };
}

const RETRY_OPTS = { maxAttempts: 1, baseDelayMs: 0, deadlineMs: 1000 };

describe("parseLlmJson", () => {
  test("parses a clean JSON object", () => {
    const r = parseLlmJson('{"answer": "hi", "citedChunkIds": ["chunk_a"]}');
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.value.answer).toBe("hi");
  });

  test("strips a markdown code fence models add despite instructions not to", () => {
    const r = parseLlmJson('```json\n{"answer": "hi", "citedChunkIds": ["chunk_a"]}\n```');
    expect(r.ok).toBe(true);
  });

  test("reports invalid JSON as a failure, not a throw", () => {
    const r = parseLlmJson("not json at all");
    expect(r.ok).toBe(false);
  });

  test("reports schema-mismatched JSON as a failure", () => {
    const r = parseLlmJson('{"totally": "wrong shape"}');
    expect(r.ok).toBe(false);
  });
});

describe("synthesizeAnswer", () => {
  test("returns the answer on a clean first try, no repair needed", async () => {
    const provider = providerReturning('{"answer": "A corporation is a legal entity.", "citedChunkIds": ["chunk_a"]}');
    const result = await synthesizeAnswer(CANDIDATES, "what is a corporation?", "en", [provider], new Map(), RETRY_OPTS);
    expect(result?.answer).toBe("A corporation is a legal entity.");
    expect(result?.citedChunkIds).toEqual(["chunk_a"]);
  });

  test("repairs once when the first response is invalid JSON, then succeeds", async () => {
    const provider = providerReturning(
      "this is not json",
      '{"answer": "A corporation is a legal entity.", "citedChunkIds": ["chunk_a"]}',
    );
    const result = await synthesizeAnswer(CANDIDATES, "what is a corporation?", "en", [provider], new Map(), RETRY_OPTS);
    expect(result?.answer).toBe("A corporation is a legal entity.");
  });

  test("repairs once when citedChunkIds references a chunk never offered as context", async () => {
    const provider = providerReturning(
      '{"answer": "hi", "citedChunkIds": ["chunk_does_not_exist"]}',
      '{"answer": "hi", "citedChunkIds": ["chunk_b"]}',
    );
    const result = await synthesizeAnswer(CANDIDATES, "q", "en", [provider], new Map(), RETRY_OPTS);
    expect(result?.citedChunkIds).toEqual(["chunk_b"]);
  });

  test("repairs once when answer is non-empty but citedChunkIds is empty", async () => {
    const provider = providerReturning(
      '{"answer": "an answer with no citation", "citedChunkIds": []}',
      '{"answer": "an answer with no citation", "citedChunkIds": ["chunk_a"]}',
    );
    const result = await synthesizeAnswer(CANDIDATES, "q", "en", [provider], new Map(), RETRY_OPTS);
    expect(result?.citedChunkIds).toEqual(["chunk_a"]);
  });

  test("returns null (extractive fallback) when the repair attempt ALSO fails", async () => {
    const provider = providerReturning("not json", "still not json");
    const result = await synthesizeAnswer(CANDIDATES, "q", "en", [provider], new Map(), RETRY_OPTS);
    expect(result).toBeNull();
  });

  test("returns null when every provider fails outright (chain.ts's own extractive_fallback path)", async () => {
    const failing: LlmProvider = {
      name: "failing",
      generate: async () => {
        throw new Error("provider down");
      },
    };
    const result = await synthesizeAnswer(CANDIDATES, "q", "en", [failing], new Map(), RETRY_OPTS);
    expect(result).toBeNull();
  });

  test("accepts an explicit empty answer as valid (model correctly declining)", async () => {
    const provider = providerReturning('{"answer": "", "citedChunkIds": []}');
    const result = await synthesizeAnswer(CANDIDATES, "unrelated question", "en", [provider], new Map(), RETRY_OPTS);
    expect(result?.answer).toBe("");
    expect(result?.citedChunkIds).toEqual([]);
  });
});
