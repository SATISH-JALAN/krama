import LanguageSelect from "./LanguageSelect";
import "./Header.css";

const LATENCY_BUDGET_MS = 200;

interface Props {
  lang: string;
  onLangChange: (code: string) => void;
  source: "live" | "mock" | null;
  latencyMs: number | null;
  cached?: boolean;
}

export default function Header({ lang, onLangChange, source, latencyMs, cached }: Props) {
  return (
    <header className="header">
      <div className="header-brand">
        <span className="header-wordmark mono">KRAMA</span>
        <span className="header-tagline">voice → grounded answer, &lt;200ms core</span>
      </div>
      <div className="header-controls">
        <LatencyBadge ms={latencyMs} cached={cached} />
        <span className={`source-badge mono ${source ?? "unset"}`}>
          <span className="source-dot" aria-hidden />
          {source === "live" ? "live server" : source === "mock" ? "mock demo" : "not yet queried"}
        </span>
        <LanguageSelect value={lang} onChange={onLangChange} />
      </div>
    </header>
  );
}

// Always-visible core-latency readout (t0->t1, CLAUDE.md #4 -- STT and any
// LLM synthesis are outside this number by design, same boundary bench/
// latency.ts measures). Lives in the header, not just the bottom-of-page
// waterfall, so a judge sees it on every single query without scrolling.
function LatencyBadge({ ms, cached }: { ms: number | null; cached?: boolean }) {
  const state = ms === null ? "unset" : ms <= LATENCY_BUDGET_MS ? "ok" : "over";
  return (
    <span className={`latency-badge mono ${state}`} title={`core latency, transcript-in → grounded-answer-out (budget: ${LATENCY_BUDGET_MS}ms)`}>
      <span className="latency-dot" aria-hidden />
      {ms === null ? (
        <>—ms</>
      ) : (
        <>
          {ms.toFixed(1)}ms
          {cached && <span className="latency-cached">cached</span>}
        </>
      )}
    </span>
  );
}
