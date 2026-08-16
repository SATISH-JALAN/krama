import { describe, expect, test } from "bun:test";
import { createApp } from "./index";

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
});
