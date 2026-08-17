import { describe, expect, test } from "bun:test";
import { createSarvamClient, pcmToWav, DEFAULT_SARVAM_MODEL } from "./sarvam";

describe("pcmToWav", () => {
  test("produces a correct 44-byte RIFF/WAVE header for 16kHz mono PCM16", () => {
    const pcm = new Int16Array([100, -200, 300, -400]);
    const wav = pcmToWav(pcm, 16_000);
    const view = new DataView(wav.buffer);

    expect(wav.length).toBe(44 + pcm.length * 2);
    expect(String.fromCharCode(...wav.slice(0, 4))).toBe("RIFF");
    expect(String.fromCharCode(...wav.slice(8, 12))).toBe("WAVE");
    expect(String.fromCharCode(...wav.slice(12, 16))).toBe("fmt ");
    expect(view.getUint16(20, true)).toBe(1); // PCM format tag
    expect(view.getUint16(22, true)).toBe(1); // mono
    expect(view.getUint32(24, true)).toBe(16_000); // sample rate
    expect(view.getUint16(34, true)).toBe(16); // bits per sample
    expect(String.fromCharCode(...wav.slice(36, 40))).toBe("data");
    expect(view.getUint32(40, true)).toBe(pcm.length * 2);
  });

  test("round-trips the exact PCM samples into the data chunk", () => {
    const pcm = new Int16Array([32767, -32768, 0, 12345]);
    const wav = pcmToWav(pcm);
    const view = new DataView(wav.buffer);
    for (let i = 0; i < pcm.length; i++) {
      expect(view.getInt16(44 + i * 2, true)).toBe(pcm[i]);
    }
  });
});

describe("createSarvamClient", () => {
  function mockFetch(status: number, body: unknown) {
    const calls: { url: string; init: RequestInit }[] = [];
    const fetchImpl = (async (url: string, init: RequestInit) => {
      calls.push({ url, init });
      return new Response(JSON.stringify(body), { status });
    }) as unknown as typeof fetch;
    return { fetchImpl, calls };
  }

  test("sends the request with the correct URL, auth header, and multipart fields", async () => {
    const { fetchImpl, calls } = mockFetch(200, {
      request_id: "req-1",
      transcript: "कॉर्पोरेशन क्या है?",
      language_code: "hi-IN",
    });
    const client = createSarvamClient("test-key", { fetchImpl });

    const result = await client.transcribe(new Int16Array([1, 2, 3]), "hi");

    expect(calls.length).toBe(1);
    expect(calls[0].url).toBe("https://api.sarvam.ai/speech-to-text");
    expect(calls[0].init.method).toBe("POST");
    expect((calls[0].init.headers as Record<string, string>)["api-subscription-key"]).toBe("test-key");

    const form = calls[0].init.body as FormData;
    expect(form.get("model")).toBe(DEFAULT_SARVAM_MODEL);
    expect(form.get("language_code")).toBe("hi-IN");
    expect(form.get("mode")).toBe("transcribe");
    expect(form.get("file")).toBeInstanceOf(Blob);

    expect(result).toEqual({ text: "कॉर्पोरेशन क्या है?", lang: "hi", isFinal: true });
  });

  test("sends language_code=unknown when no lang is given, and derives lang from the response", async () => {
    const { fetchImpl, calls } = mockFetch(200, {
      request_id: "req-2",
      transcript: "hello",
      language_code: "en-IN",
    });
    const client = createSarvamClient("test-key", { fetchImpl });

    const result = await client.transcribe(new Int16Array([1, 2, 3]));

    const form = calls[0].init.body as FormData;
    expect(form.get("language_code")).toBe("unknown");
    expect(result.lang).toBe("en");
  });

  test("throws with status and body text on a non-2xx response", async () => {
    const fetchImpl = (async () => new Response("bad key", { status: 401 })) as unknown as typeof fetch;
    const client = createSarvamClient("bad-key", { fetchImpl });

    await expect(client.transcribe(new Int16Array([1, 2, 3]), "hi")).rejects.toThrow(/401/);
  });
});
