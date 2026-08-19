import { describe, expect, test } from "bun:test";
import { answerFromGeneralKnowledge, parseLlmJson, synthesizeAnswer } from "./synthesize";
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

describe("answerFromGeneralKnowledge", () => {
  test("returns the model's plain text, with no citations", async () => {
    const r = await answerFromGeneralKnowledge(
      "ताज महल किसने बनाया",
      "hi",
      [providerReturning("ताज महल शाहजहाँ ने बनवाया था।")],
      new Map(),
      RETRY_OPTS,
    );
    expect(r).not.toBeNull();
    expect(r!.answer).toBe("ताज महल शाहजहाँ ने बनवाया था।");
    // Nothing was retrieved, so there is nothing legitimate to cite -- an id
    // here would be the model inventing one, which is exactly what the
    // grounded path's validateAgainstCandidates() exists to reject.
    expect(r!.citedChunkIds).toEqual([]);
  });

  test("takes prose -- output that would fail the grounded path's JSON contract is valid here", async () => {
    const r = await answerFromGeneralKnowledge(
      "what is the capital of India",
      "en",
      [providerReturning("New Delhi is the capital of India.")],
      new Map(),
      RETRY_OPTS,
    );
    expect(r?.answer).toBe("New Delhi is the capital of India.");
  });

  test("strips a code fence the model adds anyway", async () => {
    const r = await answerFromGeneralKnowledge(
      "q",
      "en",
      [providerReturning("```\nNew Delhi.\n```")],
      new Map(),
      RETRY_OPTS,
    );
    expect(r?.answer).toBe("New Delhi.");
  });

  test("returns null on empty output, so the caller degrades to the plain refusal", async () => {
    const r = await answerFromGeneralKnowledge("q", "en", [providerReturning("   ")], new Map(), RETRY_OPTS);
    expect(r).toBeNull();
  });

  test("names the language and asks for a spoken register, rather than passing a bare ISO code", async () => {
    let seenPrompt = "";
    const spy: LlmProvider = {
      name: "spy",
      generate: async (prompt: string) => {
        seenPrompt = prompt;
        return "ok";
      },
    };
    await answerFromGeneralKnowledge("q", "bn", [spy], new Map(), RETRY_OPTS);
    expect(seenPrompt).toContain("Bengali");
    // The register instruction is the fix for real user-reported unreadable
    // formal output -- regression-guard it rather than trusting the prose.
    expect(seenPrompt).toContain("everyday conversation");
  });
});
