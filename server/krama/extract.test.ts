import { describe, expect, test } from "bun:test";
import { extractSpan, scoreCandidate, type ExtractCandidate } from "./extract";

function mkCandidate(overrides: Partial<ExtractCandidate>): ExtractCandidate {
  return {
    chunkId: "c1",
    parentPassageId: "p1",
    text: "placeholder text",
    embedding: new Float32Array([1, 0]),
    lang: "en",
    ...overrides,
  };
}

describe("scoreCandidate", () => {
  test("higher cosine similarity yields a higher score, all else equal", () => {
    const queryEmb = new Float32Array([1, 0]);
    const close = mkCandidate({ text: "irrelevant filler", embedding: new Float32Array([0.99, 0.14]) });
    const far = mkCandidate({ text: "irrelevant filler", embedding: new Float32Array([0, 1]) });
    const tokens: string[] = [];
    const closeScore = scoreCandidate(tokens, queryEmb, "DESCRIPTION", close);
    const farScore = scoreCandidate(tokens, queryEmb, "DESCRIPTION", far);
    expect(closeScore).toBeGreaterThan(farScore);
  });

  test("lexical overlap contributes to the score independent of embedding", () => {
    const queryEmb = new Float32Array([1, 0]);
    const sameEmb = new Float32Array([1, 0]); // identical embedding for both, isolates overlap term
    const overlapping = mkCandidate({ text: "the corporation is a legal entity", embedding: sameEmb });
    const disjoint = mkCandidate({ text: "bananas are yellow", embedding: sameEmb });
    const queryTokens = ["corporation", "legal", "entity"];
    const overlapScore = scoreCandidate(queryTokens, queryEmb, "DESCRIPTION", overlapping);
    const disjointScore = scoreCandidate(queryTokens, queryEmb, "DESCRIPTION", disjoint);
    expect(overlapScore).toBeGreaterThan(disjointScore);
  });

  test("NUMERIC query type prior rewards a sentence containing digits", () => {
    const emb = new Float32Array([1, 0]);
    const withNumber = mkCandidate({ text: "it was founded in 1998", embedding: emb });
    const withoutNumber = mkCandidate({ text: "it was founded long ago", embedding: emb });
    const s1 = scoreCandidate([], emb, "NUMERIC", withNumber);
    const s2 = scoreCandidate([], emb, "NUMERIC", withoutNumber);
    expect(s1).toBeGreaterThan(s2);
    // same candidates score identically under a query type with no prior defined
    const s3 = scoreCandidate([], emb, "DESCRIPTION", withNumber);
    const s4 = scoreCandidate([], emb, "DESCRIPTION", withoutNumber);
    expect(s3).toBeCloseTo(s4, 10);
  });
});

describe("extractSpan", () => {
  test("returns null for an empty candidate pool", () => {
    expect(extractSpan("query", "DESCRIPTION", new Float32Array([1, 0]), [], "en")).toBeNull();
  });

  test("picks the highest-scoring candidate", () => {
    const queryEmb = new Float32Array([1, 0]);
    const candidates = [
      mkCandidate({ chunkId: "a", text: "unrelated", embedding: new Float32Array([0, 1]) }),
      mkCandidate({ chunkId: "b", text: "relevant", embedding: new Float32Array([0.99, 0.14]) }),
    ];
    const result = extractSpan("query", "DESCRIPTION", queryEmb, candidates, "en");
    expect(result?.chunkId).toBe("b");
  });

  test("attaches the following sentence as neighborText when siblings are present", () => {
    const queryEmb = new Float32Array([1, 0]);
    const candidates = [
      mkCandidate({
        chunkId: "p1_S0",
        parentPassageId: "p1",
        sentenceIdx: 0,
        text: "first sentence, highly relevant",
        embedding: new Float32Array([1, 0]),
      }),
      mkCandidate({
        chunkId: "p1_S1",
        parentPassageId: "p1",
        sentenceIdx: 1,
        text: "second sentence, the neighbor",
        embedding: new Float32Array([0, 1]), // deliberately low-scoring so S0 wins outright
      }),
    ];
    const result = extractSpan("query", "DESCRIPTION", queryEmb, candidates, "en");
    expect(result?.chunkId).toBe("p1_S0");
    expect(result?.neighborText).toBe("second sentence, the neighbor");
  });

  test("neighborText is undefined when there is no next sentence", () => {
    const queryEmb = new Float32Array([1, 0]);
    const candidates = [
      mkCandidate({
        chunkId: "p1_S0",
        parentPassageId: "p1",
        sentenceIdx: 0,
        embedding: new Float32Array([1, 0]),
      }),
    ];
    const result = extractSpan("query", "DESCRIPTION", queryEmb, candidates, "en");
    expect(result?.neighborText).toBeUndefined();
  });
});
