import { describe, expect, test } from "bun:test";
import { createCerebrasProvider, DEFAULT_CEREBRAS_MODEL } from "./cerebras";

function fakeFetch(response: { ok: boolean; status?: number; body: unknown }): typeof fetch {
  return (async () => ({
    ok: response.ok,
    status: response.status ?? 200,
    json: async () => response.body,
    text: async () => JSON.stringify(response.body),
  })) as unknown as typeof fetch;
}

describe("createCerebrasProvider", () => {
  test("returns the completion text on success", async () => {
    const provider = createCerebrasProvider("fake-key", {
      fetchImpl: fakeFetch({ ok: true, body: { choices: [{ message: { content: "the answer" } }] } }),
    });
    expect(await provider.generate("a prompt")).toBe("the answer");
    expect(provider.name).toBe("cerebras");
  });

  test("sends the OpenAI-compatible request shape to the right URL", async () => {
    let capturedUrl: unknown;
    let capturedInit: any;
    const fetchImpl = (async (url: unknown, init: any) => {
      capturedUrl = url;
      capturedInit = init;
      return {
        ok: true,
        status: 200,
        json: async () => ({ choices: [{ message: { content: "ok" } }] }),
        text: async () => "",
      };
    }) as unknown as typeof fetch;

    const provider = createCerebrasProvider("fake-key", { fetchImpl, model: "custom-model" });
    await provider.generate("what is a corporation?");

    expect(capturedUrl).toBe("https://api.cerebras.ai/v1/chat/completions");
    expect(capturedInit.method).toBe("POST");
    expect(capturedInit.headers["Authorization"]).toBe("Bearer fake-key");
    const body = JSON.parse(capturedInit.body);
    expect(body.model).toBe("custom-model");
    expect(body.messages).toEqual([{ role: "user", content: "what is a corporation?" }]);
  });

  test("defaults to DEFAULT_CEREBRAS_MODEL when no model given", async () => {
    let capturedModel: string | null = null;
    const fetchImpl = (async (_url: unknown, init: any) => {
      capturedModel = JSON.parse(init.body).model;
      return { ok: true, status: 200, json: async () => ({ choices: [{ message: { content: "ok" } }] }), text: async () => "" };
    }) as unknown as typeof fetch;

    await createCerebrasProvider("fake-key", { fetchImpl }).generate("x");
    expect(capturedModel).toBe(DEFAULT_CEREBRAS_MODEL);
  });

  test("throws with the status code on a non-ok HTTP response", async () => {
    const provider = createCerebrasProvider("fake-key", {
      fetchImpl: fakeFetch({ ok: false, status: 429, body: { error: "rate limited" } }),
    });
    await expect(provider.generate("x")).rejects.toThrow(/429/);
  });

  test("throws on an empty completion", async () => {
    const provider = createCerebrasProvider("fake-key", {
      fetchImpl: fakeFetch({ ok: true, body: { choices: [{ message: {} }] } }),
    });
    await expect(provider.generate("x")).rejects.toThrow(/empty completion/);
  });
});
