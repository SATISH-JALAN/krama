import { describe, expect, test } from "bun:test";
import { createGeminiProvider, DEFAULT_GEMINI_MODEL, GEMINI_MODEL_CHAIN } from "./gemini";

function fakeFetch(response: { ok: boolean; status?: number; body: unknown }): typeof fetch {
  return (async () => ({
    ok: response.ok,
    status: response.status ?? 200,
    json: async () => response.body,
    text: async () => JSON.stringify(response.body),
  })) as unknown as typeof fetch;
}

describe("createGeminiProvider", () => {
  test("returns the completion text on success", async () => {
    const provider = createGeminiProvider("fake-key", {
      fetchImpl: fakeFetch({
        ok: true,
        body: { candidates: [{ content: { parts: [{ text: "the answer" }] } }] },
      }),
    });
    expect(await provider.generate("a prompt")).toBe("the answer");
    expect(provider.name).toBe(`gemini:${DEFAULT_GEMINI_MODEL}`);
  });

  test("sends the documented generateContent request shape to the right URL", async () => {
    let capturedUrl: unknown;
    let capturedInit: any;
    const fetchImpl = (async (url: unknown, init: any) => {
      capturedUrl = url;
      capturedInit = init;
      return {
        ok: true,
        status: 200,
        json: async () => ({ candidates: [{ content: { parts: [{ text: "ok" }] } }] }),
        text: async () => "",
      };
    }) as unknown as typeof fetch;

    const provider = createGeminiProvider("fake-key", { fetchImpl, model: "custom-model" });
    await provider.generate("what is a corporation?");

    expect(capturedUrl).toBe(
      "https://generativelanguage.googleapis.com/v1beta/models/custom-model:generateContent?key=fake-key",
    );
    expect(capturedInit.method).toBe("POST");
    const body = JSON.parse(capturedInit.body);
    expect(body.contents).toEqual([
      { role: "user", parts: [{ text: "what is a corporation?" }] },
    ]);
  });

  test("disables extended thinking -- regression test for a real bug where Gemini 3.x's default thinking made the actual synthesis prompt time out at >20s", async () => {
    let capturedInit: any;
    const fetchImpl = (async (_url: unknown, init: any) => {
      capturedInit = init;
      return {
        ok: true,
        status: 200,
        json: async () => ({ candidates: [{ content: { parts: [{ text: "ok" }] } }] }),
        text: async () => "",
      };
    }) as unknown as typeof fetch;

    await createGeminiProvider("fake-key", { fetchImpl }).generate("x");
    const body = JSON.parse(capturedInit.body);
    expect(body.generationConfig.thinkingConfig).toEqual({ thinkingLevel: "minimal" });
  });

  test("defaults to DEFAULT_GEMINI_MODEL when no model given", async () => {
    let capturedUrl: string | null = null;
    const fetchImpl = (async (url: unknown) => {
      capturedUrl = String(url);
      return {
        ok: true,
        status: 200,
        json: async () => ({ candidates: [{ content: { parts: [{ text: "ok" }] } }] }),
        text: async () => "",
      };
    }) as unknown as typeof fetch;

    await createGeminiProvider("fake-key", { fetchImpl }).generate("x");
    expect(capturedUrl).toContain(`/${DEFAULT_GEMINI_MODEL}:generateContent`);
  });

  test("gives each model a distinct provider name, so they get separate circuit breakers", () => {
    // The free tier meters requests-per-day per MODEL, so these are separate
    // quotas. harness/breaker.ts keys state by provider name -- if two models
    // shared a name they would share a breaker, and one model's daily-cap 429
    // would open the breaker for the others too, skipping the very fallbacks
    // GEMINI_MODEL_CHAIN exists to provide.
    const names = GEMINI_MODEL_CHAIN.map((model) => createGeminiProvider("fake-key", { model }).name);
    expect(new Set(names).size).toBe(GEMINI_MODEL_CHAIN.length);
  });

  test("the model chain leads with the default model and offers real fallbacks", () => {
    expect(GEMINI_MODEL_CHAIN[0]).toBe(DEFAULT_GEMINI_MODEL);
    expect(GEMINI_MODEL_CHAIN.length).toBeGreaterThan(1);
  });

  test("throws with the status code on a non-ok HTTP response", async () => {
    const provider = createGeminiProvider("fake-key", {
      fetchImpl: fakeFetch({ ok: false, status: 429, body: { error: "rate limited" } }),
    });
    await expect(provider.generate("x")).rejects.toThrow(/429/);
  });

  test("throws on an empty completion", async () => {
    const provider = createGeminiProvider("fake-key", {
      fetchImpl: fakeFetch({ ok: true, body: { candidates: [{ content: { parts: [{}] } }] } }),
    });
    await expect(provider.generate("x")).rejects.toThrow(/empty completion/);
  });

  test("throws on candidates being entirely absent (e.g. safety block)", async () => {
    const provider = createGeminiProvider("fake-key", {
      fetchImpl: fakeFetch({ ok: true, body: {} }),
    });
    await expect(provider.generate("x")).rejects.toThrow(/empty completion/);
  });
});
