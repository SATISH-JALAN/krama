import type { GroundedAnswer } from "./contracts";
import { mockQuery } from "./mock";

const API_URL = import.meta.env.VITE_API_URL as string | undefined;
const LIVE_TIMEOUT_MS = 3000;

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
