import { describe, expect, test } from "bun:test";
import { writeFileSync, unlinkSync, mkdirSync } from "fs";
import { createApp, loadThresholds } from "./index";
import { DEFAULT_SAFETY_THRESHOLD } from "./maun/safety";
import { DEFAULT_OOD_THRESHOLDS } from "./maun/ood";

/**
 * Only covers what's testable without boot() -- boot() needs a real HNSW
 * index (hnswlib-node can't build on this machine) and full-corpus
 * artifacts that don't exist yet. This still catches real regressions in
 * the Hono wiring/validation layer independent of that blocker -- e.g. this
 * test suite is what caught that importing index.ts crashed entirely
 * before ghana/hnsw.ts's import was made lazy.
 */
describe("createApp", () => {
  test("/health reports not-booted before boot() is called", async () => {
    const app = createApp();
    const res = await app.request("/health");
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: false });
  });

  test("/query rejects a malformed request body with 400", async () => {
    const app = createApp();
    const res = await app.request("/query", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ bad: "shape" }),
    });
    expect(res.status).toBe(400);
  });

  test("/query rejects a lang code that isn't exactly 2 characters", async () => {
    const app = createApp();
    const res = await app.request("/query", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text: "what is a corporation?", lang: "hindi" }),
    });
    expect(res.status).toBe(400);
  });

  test("/query/voice 503s before boot() (STT never configured)", async () => {
    const app = createApp();
    const res = await app.request("/query/voice", {
      method: "POST",
      headers: { "content-type": "application/octet-stream" },
      body: new Int16Array([1, 2, 3]).buffer,
    });
    expect(res.status).toBe(503);
  });
});

describe("loadThresholds", () => {
  test("falls back to the calibrated DEFAULT_* constants when no path given", () => {
    const t = loadThresholds(undefined);
    expect(t.safetyThreshold).toBe(DEFAULT_SAFETY_THRESHOLD);
    expect(t.oodThresholds).toEqual(DEFAULT_OOD_THRESHOLDS);
  });

  test("loads a real calibration-artifact-shaped file", () => {
    // artifacts/ is gitignored (built, not committed -- ARCHITECTURE.md §3),
    // so this test can't depend on the real thresholds.json existing after
    // a fresh clone; it writes a fixture in exactly the shape
    // eval/calibrate_guardrails.ts actually produces instead. That real
    // file was separately confirmed to round-trip through this same
    // function during PLAN.md E5.5's calibration run.
    const path = "artifacts/_test_thresholds_fixture.json";
    mkdirSync("artifacts", { recursive: true });
    writeFileSync(
      path,
      JSON.stringify({
        generatedAt: new Date().toISOString(),
        calibrationSource: "eval/results/guardrail_calibration.json",
        safetyThreshold: 0.84,
        oodThresholds: { minTopScore: 0.84, minCentroidCosine: 0 },
      }),
    );
    try {
      const t = loadThresholds(path);
      expect(t.safetyThreshold).toBe(0.84);
      expect(t.oodThresholds).toEqual({ minTopScore: 0.84, minCentroidCosine: 0 });
    } finally {
      unlinkSync(path);
    }
  });

  test("falls back without throwing when the artifact is missing", () => {
    const t = loadThresholds("artifacts/does_not_exist.json");
    expect(t.safetyThreshold).toBe(DEFAULT_SAFETY_THRESHOLD);
  });

  test("falls back without throwing when the artifact is malformed", () => {
    const path = "artifacts/_test_malformed_thresholds.json";
    mkdirSync("artifacts", { recursive: true });
    writeFileSync(path, JSON.stringify({ wrong: "shape" }));
    try {
      const t = loadThresholds(path);
      expect(t.oodThresholds).toEqual(DEFAULT_OOD_THRESHOLDS);
    } finally {
      unlinkSync(path);
    }
  });
});
