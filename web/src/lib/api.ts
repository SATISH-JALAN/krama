import type { GroundedAnswer } from "./contracts";
import { mockQuery } from "./mock";

const API_URL = import.meta.env.VITE_API_URL as string | undefined;
const LIVE_TIMEOUT_MS = 3000;
// STT (real Sarvam batch transcription, server/stt/sarvam.ts) adds real
// network + inference time on top of the fast-path budget CLAUDE.md #4
// scopes handleQuery() to -- this endpoint is genuinely slower than /query,
// not a bug, so it gets a longer timeout rather than the same one.
const VOICE_TIMEOUT_MS = 12_000;

export interface QueryResult {
  data: GroundedAnswer;
  source: "live" | "mock";
}

// Real path: POST to the deployed server/index.ts /query route (works today
// for typed text once a server is booted with real artifacts; the WS voice
// route doesn't exist server-side yet, see server/index.ts's docstring).
// Falls back to the local mock responder when VITE_API_URL isn't set, the
// server isn't reachable, or it times out -- so the UI stays usable for
// design/demo work independent of deployment status, and is honest about
// which one produced any given answer (see the "live"/"mock" badge in App).
export async function queryBackend(text: string, lang: string, queryType = "DESCRIPTION"): Promise<QueryResult> {
  if (API_URL) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), LIVE_TIMEOUT_MS);
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, lang, queryType }),
        signal: controller.signal,
      });
      clearTimeout(timeout);
      if (res.ok) {
        const data = (await res.json()) as GroundedAnswer;
        return { data, source: "live" };
      }
    } catch {
      // fall through to mock
    }
  }
  return { data: await mockQuery(text, lang), source: "mock" };
}

export interface VoiceQueryResult extends GroundedAnswer {
  transcript: string;
  detectedLang: string;
}

// Real path: POST raw 16kHz mono Int16 PCM to server/index.ts's
// /query/voice route (real Sarvam batch STT -> handleQuery(), see
// server/stt/sarvam.ts). Returns null if no live server is configured or
// the request fails/times out -- there's no audio-capable mock to fall
// back to, so the caller (App.tsx) is responsible for falling back to the
// browser's own Web Speech transcript through queryBackend() instead.
export async function queryVoice(
  pcm: Int16Array,
  lang: string,
  queryType = "DESCRIPTION",
): Promise<VoiceQueryResult | null> {
  if (!API_URL) return null;
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), VOICE_TIMEOUT_MS);
    const params = new URLSearchParams({ lang, queryType });
    const res = await fetch(`${API_URL}/query/voice?${params}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: pcm.buffer as ArrayBuffer,
      signal: controller.signal,
    });
    clearTimeout(timeout);
    if (!res.ok) return null;
    return (await res.json()) as VoiceQueryResult;
  } catch {
    return null;
  }
}
