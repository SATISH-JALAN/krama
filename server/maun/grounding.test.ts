import { describe, expect, test } from "bun:test";
import { checkL4, type AnswerSentence } from "./grounding";

function mkChunkMap(entries: [string, Float32Array][]): Map<string, Float32Array> {
  return new Map(entries);
}

describe("checkL4", () => {
  test("passes a sentence closely aligned with its cited chunk", () => {
    const chunks = mkChunkMap([["c1", new Float32Array([1, 0])]]);
    const sentences: AnswerSentence[] = [
      { text: "grounded sentence", embedding: new Float32Array([0.99, 0.14]), citedChunkId: "c1" },
    ];
    const result = checkL4(sentences, chunks, 0.9);
    expect(result.refused).toBe(false);
    expect(result.groundedSentences).toEqual(["grounded sentence"]);
    expect(result.strippedSentences).toEqual([]);
  });

  test("strips a sentence that drifts from its cited chunk", () => {
    const chunks = mkChunkMap([["c1", new Float32Array([1, 0])]]);
    const sentences: AnswerSentence[] = [
      { text: "hallucinated sentence", embedding: new Float32Array([0, 1]), citedChunkId: "c1" },
    ];
    const result = checkL4(sentences, chunks, 0.9);
    expect(result.strippedSentences).toEqual(["hallucinated sentence"]);
    expect(result.groundedSentences).toEqual([]);
  });

  test("refuses entirely when every sentence fails grounding", () => {
    const chunks = mkChunkMap([["c1", new Float32Array([1, 0])]]);
    const sentences: AnswerSentence[] = [
      { text: "bad 1", embedding: new Float32Array([0, 1]), citedChunkId: "c1" },
      { text: "bad 2", embedding: new Float32Array([-1, 0]), citedChunkId: "c1" },
    ];
    const result = checkL4(sentences, chunks, 0.9);
    expect(result.refused).toBe(true);
    expect(result.reason).toBe("no_grounding");
  });

  test("partial grounding: keeps grounded sentences, strips the rest, doesn't refuse", () => {
    const chunks = mkChunkMap([["c1", new Float32Array([1, 0])]]);
    const sentences: AnswerSentence[] = [
      { text: "good", embedding: new Float32Array([0.99, 0.14]), citedChunkId: "c1" },
      { text: "bad", embedding: new Float32Array([0, 1]), citedChunkId: "c1" },
    ];
    const result = checkL4(sentences, chunks, 0.9);
    expect(result.refused).toBe(false);
    expect(result.groundedSentences).toEqual(["good"]);
    expect(result.strippedSentences).toEqual(["bad"]);
  });

  test("a citation to a chunk with no known embedding is treated as ungrounded, not trusted", () => {
    const chunks = mkChunkMap([["c1", new Float32Array([1, 0])]]);
    const sentences: AnswerSentence[] = [
      { text: "unverifiable citation", embedding: new Float32Array([1, 0]), citedChunkId: "unknown-chunk" },
    ];
    const result = checkL4(sentences, chunks, 0.9);
    expect(result.refused).toBe(true);
    expect(result.strippedSentences).toEqual(["unverifiable citation"]);
  });

  test("empty sentence list refuses (nothing grounded, vacuously)", () => {
    const result = checkL4([], mkChunkMap([]), 0.9);
    expect(result.refused).toBe(true);
  });
});
