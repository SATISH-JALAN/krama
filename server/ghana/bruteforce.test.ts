import { describe, expect, test } from "bun:test";
import { writeFileSync, unlinkSync } from "fs";
import { boot, search, size } from "./bruteforce";

// bruteforce.ts hardcodes DIM=384 (CLAUDE.md invariant #2 -- same model,
// same dimensionality, everywhere), so fixtures must actually be 384-wide,
// not a toy small dim -- these embed the same "known relative similarity"
// idea as a real e5-small vector would, just with only the first 3 of 384
// components nonzero for readability. Not L2-normalized, doesn't matter for
// a plain dot-product correctness check.
const DIM = 384;
function pad(head: number[]): number[] {
  return [...head, ...new Array(DIM - head.length).fill(0)];
}
const ROWS = [pad([1, 0, 0]), pad([0, 1, 0]), pad([0.9, 0.1, 0]), pad([-1, 0, 0])];
const IDS = ["a", "b", "c", "d"];
const QUERY = new Float32Array(pad([1, 0, 0]));

function writeFixtures(embPath: string, idsPath: string): void {
  const flat = new Float32Array(ROWS.flat());
  writeFileSync(embPath, Buffer.from(flat.buffer));
  writeFileSync(idsPath, JSON.stringify(IDS));
}

describe("bruteforce", () => {
  test("search throws before boot() is called", () => {
    // Must run before any other test in this file boots the module-level
    // singleton -- bun test runs a file's tests in source order, so this
    // sits first deliberately rather than fighting for a fresh module
    // instance.
    expect(() => search(QUERY, 1)).toThrow();
  });

  test("boots from real files and reports the correct size", async () => {
    const embPath = "server/ghana/_test_bf_embeddings.f32bin";
    const idsPath = "server/ghana/_test_bf_ids.json";
    writeFixtures(embPath, idsPath);
    try {
      await boot(embPath, idsPath);
      expect(size()).toBe(4);
    } finally {
      unlinkSync(embPath);
      unlinkSync(idsPath);
    }
  });

  test("search ranks by inner product, closest first", async () => {
    const embPath = "server/ghana/_test_bf_embeddings2.f32bin";
    const idsPath = "server/ghana/_test_bf_ids2.json";
    writeFixtures(embPath, idsPath);
    try {
      await boot(embPath, idsPath);
      const results = search(QUERY, 3);
      expect(results.map((r) => r.passageId)).toEqual(["a", "c", "b"]);
      expect(results[0].score).toBeCloseTo(1, 5);
      expect(results[1].score).toBeCloseTo(0.9, 5);
      expect(results[2].score).toBeCloseTo(0, 5);
    } finally {
      unlinkSync(embPath);
      unlinkSync(idsPath);
    }
  });

  test("boot rejects a mismatched id count", async () => {
    const embPath = "server/ghana/_test_bf_embeddings3.f32bin";
    const idsPath = "server/ghana/_test_bf_ids3.json";
    const flat = new Float32Array(ROWS.flat());
    writeFileSync(embPath, Buffer.from(flat.buffer));
    writeFileSync(idsPath, JSON.stringify(["only", "two"]));
    try {
      await expect(boot(embPath, idsPath)).rejects.toThrow();
    } finally {
      unlinkSync(embPath);
      unlinkSync(idsPath);
    }
  });
});
