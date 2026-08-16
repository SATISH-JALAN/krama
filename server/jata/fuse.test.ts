import { describe, expect, test } from "bun:test";
import { rrf, rrfRanked } from "./fuse";

describe("rrf", () => {
  test("a doc ranked first in both lists scores higher than one in only one list", () => {
    const scores = rrf([
      { ids: ["a", "b", "c"] },
      { ids: ["a", "c", "b"] },
    ]);
    expect(scores.get("a")!).toBeGreaterThan(scores.get("b")!);
    expect(scores.get("a")!).toBeGreaterThan(scores.get("c")!);
  });

  test("a doc appearing in only one list still gets a score", () => {
    const scores = rrf([{ ids: ["a", "b"] }, { ids: ["c"] }]);
    expect(scores.has("c")).toBe(true);
    expect(scores.get("c")).toBeGreaterThan(0);
  });

  test("weight scales a list's contribution (dense:BM25 tuning sweep)", () => {
    const equal = rrf([{ ids: ["a"], weight: 1 }, { ids: ["b"], weight: 1 }]);
    expect(equal.get("a")).toBeCloseTo(equal.get("b")!, 10);

    const denseFavoured = rrf([{ ids: ["a"], weight: 2 }, { ids: ["b"], weight: 1 }]);
    expect(denseFavoured.get("a")!).toBeGreaterThan(denseFavoured.get("b")!);
  });

  test("rrfRanked returns sorted, capped results", () => {
    const results = rrfRanked(
      [{ ids: ["a", "b", "c", "d"] }, { ids: ["b", "a", "d", "c"] }],
      2,
    );
    expect(results).toHaveLength(2);
    expect(results[0].score).toBeGreaterThanOrEqual(results[1].score);
    // a and b are top-2-ish in both lists, should beat c/d
    expect(["a", "b"]).toContain(results[0].id);
  });

  test("empty lists produce empty results", () => {
    expect(rrfRanked([], 10)).toEqual([]);
    expect(rrfRanked([{ ids: [] }], 10)).toEqual([]);
  });
});
