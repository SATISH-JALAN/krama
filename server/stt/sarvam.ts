/**
 * shruti -- real Sarvam batch STT client.
 *
 * ARCHITECTURE.md's original design used the WS realtime endpoint
 * (`wss://api.sarvam.ai/speech-to-text-realtime/ws`). The approved
 * 2026-08-18/19 scope simplification (MEMORY.md "RAGINGOA reference
 * analyzed" entry) switched to batch transcription instead -- far less
 * code, still a real Sarvam call, and matches the reference repo's own
 * approach (also batch-only in practice, despite its README claiming
 * streaming).
 *
 * REST contract verified live against docs.sarvam.ai on 2026-08-19, not
 * assumed -- CLAUDE.md's existing Sarvam research only covered the two WS
 * endpoints, never this one:
 *   POST https://api.sarvam.ai/speech-to-text
 *   multipart/form-data fields: file, model ("saaras:v3"), language_code
 *   (BCP-47, e.g. "hi-IN", or "unknown" for auto-detect), mode
 *   ("transcribe"). Auth header: `api-subscription-key`. JSON response:
 *   { request_id, transcript, language_code, timestamps? }.
 *
 * Input is raw 16kHz mono Int16 PCM (CLAUDE.md invariant #8, the same
 * AudioWorklet output the frontend already captures) -- wrapped in a
 * minimal WAV container before upload rather than relying on the
 * under-documented `input_audio_codec` raw-PCM parameter, since WAV is
 * unambiguous and universally accepted by the endpoint's documented format
 * list.
 *
 * `fetchImpl` is injectable, same DI pattern as `llm/cerebras.ts`, so the
 * request-shape logic (WAV framing, form fields, response parsing) is
 * unit-testable without a live key -- only the actual network round-trip to
 * Sarvam's servers stays genuinely unverified until a real SARVAM_API_KEY
 * is exercised against it for real.
 */
import type { Transcript } from "../harness/contracts";

export const DEFAULT_SARVAM_MODEL = "saaras:v3";
const SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text";
const SAMPLE_RATE_HZ = 16_000;

// KRAMA's internal 2-letter codes <-> Sarvam's BCP-47 codes. Only the
// languages this project actually ingests/serves (CLAUDE.md's hi/bn/ta
// scope) plus en, since the frontend's language selector offers it too.
const LANG_TO_BCP47: Record<string, string> = {
  hi: "hi-IN",
  bn: "bn-IN",
  ta: "ta-IN",
  en: "en-IN",
};

function bcp47ToLang(bcp47: string): string {
  const twoLetter = bcp47.split("-")[0]?.toLowerCase();
  return twoLetter && twoLetter.length === 2 ? twoLetter : "en";
}

/**
 * Wraps raw 16-bit signed PCM samples in a minimal 44-byte canonical WAV
 * (RIFF/WAVE, PCM format tag) header -- mono, 16kHz, matching the
 * AudioWorklet's fixed output format, not a general-purpose encoder.
 */
export function pcmToWav(pcm: Int16Array, sampleRateHz: number = SAMPLE_RATE_HZ): Uint8Array {
  const bytesPerSample = 2;
  const dataSize = pcm.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeStr = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeStr(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM format tag
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRateHz, true);
  view.setUint32(28, sampleRateHz * bytesPerSample, true); // byte rate
  view.setUint16(32, bytesPerSample, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeStr(36, "data");
  view.setUint32(40, dataSize, true);

  for (let i = 0; i < pcm.length; i++) {
    view.setInt16(44 + i * bytesPerSample, pcm[i], true);
  }

  return new Uint8Array(buffer);
}

export interface SarvamClient {
  /** `lang`, if given, is a 2-letter KRAMA-internal code (hi/bn/ta/en); omit for Sarvam auto-detect. */
  transcribe(pcm: Int16Array, lang?: string): Promise<Transcript>;
}

export function createSarvamClient(
  apiKey: string,
  opts: { model?: string; fetchImpl?: typeof fetch } = {},
): SarvamClient {
  const fetchFn = opts.fetchImpl ?? fetch;
  const model = opts.model ?? DEFAULT_SARVAM_MODEL;

  return {
    async transcribe(pcm: Int16Array, lang?: string): Promise<Transcript> {
      const wav = pcmToWav(pcm);
      const form = new FormData();
      form.append("file", new Blob([wav], { type: "audio/wav" }), "audio.wav");
      form.append("model", model);
      form.append("language_code", lang ? (LANG_TO_BCP47[lang] ?? "unknown") : "unknown");
      form.append("mode", "transcribe");

      const res = await fetchFn(SARVAM_STT_URL, {
        method: "POST",
        headers: { "api-subscription-key": apiKey },
        body: form,
      });

      if (!res.ok) {
        throw new Error(`sarvam stt failed: ${res.status} ${await res.text()}`);
      }

      const body = (await res.json()) as { transcript: string; language_code: string };

      return {
        text: body.transcript,
        lang: lang ?? bcp47ToLang(body.language_code),
        isFinal: true, // batch transcription has no partial-result concept
      };
    },
  };
}
