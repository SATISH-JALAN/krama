import { describe, expect, test } from "bun:test";
import { checkRerankScore, DEFAULT_MIN_RERANK_SCORE } from "./rerank_guard";

describe("checkRerankScore", () => {
  test("passes a score above the threshold", () => {
    const result = checkRerankScore(5.0, 0);
    expect(result.refused).toBe(false);
  });

  test("refuses a score below the threshold, with a reason and detail", () => {
    const result = checkRerankScore(-6.5, 0);
    expect(result.refused).toBe(true);
    expect(result.reason).toBe("off_topic");
    expect(result.detail).toContain("-6.500");
  });

  test("uses DEFAULT_MIN_RERANK_SCORE when no threshold given", () => {
    expect(checkRerankScore(DEFAULT_MIN_RERANK_SCORE - 0.01).refused).toBe(true);
    expect(checkRerankScore(DEFAULT_MIN_RERANK_SCORE + 0.01).refused).toBe(false);
  });

  test("a score exactly at the threshold is not refused (strict less-than)", () => {
    expect(checkRerankScore(1.5, 1.5).refused).toBe(false);
  });
});
