import { describe, expect, test } from "bun:test";
import { checkL0 } from "./input";
import { cosineSim, computeCentroid, OodGuard } from "./ood";
import { SafetyGuard } from "./safety";

describe("checkL0", () => {
  test("passes a real question through", () => {
    expect(checkL0("what is a corporation?", "en").refused).toBe(false);
  });

  test("refuses an empty transcript", () => {
    const r = checkL0("", "en");
    expect(r.refused).toBe(true);
    expect(r.reason).toBe("empty_or_gibberish");
  });

  test("refuses a whitespace-only transcript", () => {
    expect(checkL0("   \n  ", "en").refused).toBe(true);
  });

  test("refuses a transcript below the minimum length", () => {
    expect(checkL0("a", "en").refused).toBe(true);
  });

  test("refuses punctuation-only noise with no word-like content", () => {
    expect(checkL0("...??!!", "en").refused).toBe(true);
  });

  test("passes real Hindi text", () => {
    expect(checkL0("कॉर्पोरेशन क्या है?", "hi").refused).toBe(false);
  });

  test("passes a short but legitimate word (not just noise)", () => {
    // "yes" is short but is real word-like content -- L0 shouldn't reject
    // legitimate brief answers, only empty/sub-minimum/no-word-content cases.
    expect(checkL0("yes", "en").refused).toBe(false);
  });
});

describe("ood: cosineSim / computeCentroid", () => {
  test("cosineSim of identical vectors is 1", () => {
    const v = new Float32Array([0.6, 0.8]);
    expect(cosineSim(v, v)).toBeCloseTo(1, 5);
  });

  test("cosineSim of orthogonal vectors is 0", () => {
    expect(cosineSim(new Float32Array([1, 0]), new Float32Array([0, 1]))).toBeCloseTo(0, 5);
  });

  test("computeCentroid of a single vector returns that vector (normalized)", () => {
    const v = new Float32Array([1, 0]);
    const c = computeCentroid([v]);
    expect(cosineSim(c, v)).toBeCloseTo(1, 5);
  });

  test("computeCentroid of opposite vectors is degenerate but doesn't throw", () => {
    const a = new Float32Array([1, 0]);
    const b = new Float32Array([-1, 0]);
    expect(() => computeCentroid([a, b])).not.toThrow();
  });

  test("computeCentroid throws on empty input", () => {
    expect(() => computeCentroid([])).toThrow();
  });
});

describe("OodGuard", () => {
  const centroid = new Float32Array([1, 0]);

  test("passes when both score and centroid similarity are high", () => {
    const guard = new OodGuard(centroid, { minTopScore: 0.5, minCentroidCosine: 0.3 });
    const inDomainQuery = new Float32Array([0.99, 0.14]); // close to centroid direction
    expect(guard.check(inDomainQuery, 0.8).refused).toBe(false);
  });

  test("refuses when top retrieval score is too low", () => {
    const guard = new OodGuard(centroid, { minTopScore: 0.5, minCentroidCosine: 0.3 });
    const q = new Float32Array([1, 0]);
    const result = guard.check(q, 0.1);
    expect(result.refused).toBe(true);
    expect(result.reason).toBe("off_topic");
  });

  test("refuses when query is far from the corpus centroid even with a high score", () => {
    const guard = new OodGuard(centroid, { minTopScore: 0.5, minCentroidCosine: 0.3 });
    const farQuery = new Float32Array([0, 1]); // orthogonal to centroid
    const result = guard.check(farQuery, 0.9); // score alone would pass
    expect(result.refused).toBe(true);
    expect(result.reason).toBe("off_topic");
  });
});

describe("SafetyGuard", () => {
  // "unsafe cluster" points near [1,0], "benign" points near [0,1] --
  // synthetic but directly exercises the same nearest-exemplar logic that
  // real embeddings would go through.
  const unsafeExemplars = [
    new Float32Array([1, 0.05]),
    new Float32Array([0.98, -0.05]),
    new Float32Array([0.95, 0.1]),
  ];

  test("refuses a query embedding close to an unsafe exemplar", () => {
    const guard = new SafetyGuard(unsafeExemplars, 0.9);
    const nearUnsafe = new Float32Array([0.99, 0.02]);
    const result = guard.check(nearUnsafe);
    expect(result.refused).toBe(true);
    expect(result.reason).toBe("unsafe");
  });

  test("passes a query embedding far from all unsafe exemplars", () => {
    const guard = new SafetyGuard(unsafeExemplars, 0.9);
    const benign = new Float32Array([0.05, 0.99]); // near-orthogonal to the unsafe cluster
    expect(guard.check(benign).refused).toBe(false);
  });

  test("threshold controls sensitivity", () => {
    const borderline = new Float32Array([0.8, 0.6]);
    const strict = new SafetyGuard(unsafeExemplars, 0.95);
    const lenient = new SafetyGuard(unsafeExemplars, 0.5);
    expect(strict.check(borderline).refused).toBe(false);
    expect(lenient.check(borderline).refused).toBe(true);
  });

  test("throws if constructed with zero exemplars rather than silently never refusing", () => {
    expect(() => new SafetyGuard([], 0.9)).toThrow();
  });
});
