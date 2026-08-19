import { Fragment } from "react";
import type { Span } from "../lib/contracts";
import { moduleForSpan, totalTraceMs } from "../lib/contracts";
import "./Waterfall.css";

interface Props {
  trace?: Span[];
}

// The signature element: krama-patha, the Vedic recitation technique the
// chunking module is named for, recites text as overlapping word pairs
// (word1-word2, word2-word3, word3-word4...) so a listener can verify each
// step against the one before it. A pipeline trace is the same shape --
// each stage's output is verified by the next -- so the waterfall renders
// spans the same way: bars whose joints are labelled with the adjacent
// stage pair, not a generic timeline.
export default function Waterfall({ trace }: Props) {
  if (!trace || trace.length === 0) {
    return (
      <div className="waterfall-empty mono">
        {trace ? "no spans recorded" : "trace not returned by this server build"}
      </div>
    );
  }

  const maxMs = Math.max(...trace.map((s) => s.ms), 1);
  const totalMs = totalTraceMs(trace) ?? 0;

  return (
    <div className="waterfall">
      <div className="waterfall-header">
        <span className="eyebrow">trace</span>
        <span className="waterfall-total mono">{totalMs.toFixed(1)}ms total</span>
      </div>
      <div className="waterfall-track">
        {trace.map((span, i) => {
          const heightPct = 18 + (span.ms / maxMs) * 82;
          const mod = moduleForSpan(span.name);
          const next = trace[i + 1];
          return (
            <Fragment key={span.name + i}>
              <div className="waterfall-step">
                <div className="waterfall-bar-wrap">
                  <div
                    className={`waterfall-bar ${span.ok ? "ok" : "fail"}`}
                    style={{ height: `${heightPct}%` }}
                    title={`${span.name}: ${span.ms.toFixed(2)}ms${span.err ? ` — ${span.err}` : ""}`}
                  />
                </div>
                <span className="waterfall-ms mono">{span.ms.toFixed(1)}</span>
                <span className="waterfall-name mono">{span.name}</span>
                {mod && <span className="waterfall-module mono">{mod}</span>}
              </div>
              {next && (
                <div className="waterfall-joint" aria-hidden>
                  <span className="waterfall-joint-line" />
                  <span className="waterfall-joint-label mono">
                    {span.name.split("_")[0]}→{next.name.split("_")[0]}
                  </span>
                </div>
              )}
            </Fragment>
          );
        })}
      </div>
    </div>
  );
}
