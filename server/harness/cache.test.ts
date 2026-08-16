import { describe, expect, test } from "bun:test";
import { SemanticCache } from "./cache";

describe("SemanticCache", () => {
  test("misses on an empty cache", () => {
    const cache = new SemanticCache<string>();
    expect(cache.get(new Float32Array([1, 0])).hit).toBe(false);
  });

  test("hits for a near-identical query embedding above threshold", () => {
    const cache = new SemanticCache<string>(0.97);
    cache.set(new Float32Array([1, 0]), "cached answer");
    const result = cache.get(new Float32Array([0.999, 0.045])); // cos ~0.999
    expect(result.hit).toBe(true);
    expect(result.value).toBe("cached answer");
  });

  test("misses for a dissimilar query embedding below threshold", () => {
    const cache = new SemanticCache<string>(0.97);
    cache.set(new Float32Array([1, 0]), "cached answer");
    const result = cache.get(new Float32Array([0.7, 0.7])); // cos ~0.707, well below 0.97
    expect(result.hit).toBe(false);
    expect(result.value).toBeUndefined();
  });

  test("returns the closest match's similarity, not just a boolean", () => {
    const cache = new SemanticCache<string>(0.9);
    cache.set(new Float32Array([1, 0]), "a");
    const result = cache.get(new Float32Array([1, 0]));
    expect(result.similarity).toBeCloseTo(1, 5);
  });

  test("picks the best match among multiple entries", () => {
    const cache = new SemanticCache<string>(0.5);
    cache.set(new Float32Array([1, 0]), "far-ish");
    cache.set(new Float32Array([0.99, 0.14]), "closest");
    const result = cache.get(new Float32Array([0.98, 0.2]));
    expect(result.value).toBe("closest");
  });

  test("evicts the oldest entry once maxEntries is exceeded", () => {
    const cache = new SemanticCache<string>(0.99, 2);
    cache.set(new Float32Array([1, 0]), "first");
    cache.set(new Float32Array([0, 1]), "second");
    expect(cache.size).toBe(2);
    cache.set(new Float32Array([-1, 0]), "third");
    expect(cache.size).toBe(2);
    // "first"'s embedding was evicted -- a query for it should no longer hit
    const result = cache.get(new Float32Array([1, 0]));
    expect(result.hit).toBe(false);
  });
});
