import type { ReactNode } from "react";
import type { GroundedAnswer, RefusalReason } from "../lib/contracts";
import { romanizationLabel } from "../lib/contracts";
import Citations from "./Citations";
import "./Answer.css";

interface Props {
  result: GroundedAnswer | null;
  busy: boolean;
  synthesizing: boolean;
  synthesizeStatus: "not_configured" | "failed" | null;
  lang: string;
}

const SYNTHESIZE_UNAVAILABLE_COPY: Record<"not_configured" | "failed" | "unknown", string> = {
  not_configured: "no LLM provider is set up on this server (no API key configured).",
  failed: "the request didn't come back this time (provider error, rate limit, or timeout) — try again in a moment.",
  unknown: "hasn't been requested yet.",
};

const REFUSAL_COPY: Record<RefusalReason, string> = {
  empty_or_gibberish: "Didn't catch a question there — try speaking again or typing it in.",
  unsafe: "This falls outside what the guardrails allow an answer for.",
  off_topic: "That's not covered by what's in the knowledge base.",
  no_grounding: "No passage in the knowledge base supports an answer — refusing rather than guessing.",
};

export default function Answer({ result, busy, synthesizing, synthesizeStatus, lang }: Props) {
  if (busy) {
    return (
      <div className="answer-grid">
        <AnswerCard label="instant · grounded" state="pending" />
        <AnswerCard label="synthesized · llm" state="pending" />
      </div>
    );
  }

  if (!result) return null;

  // A refusal is the RETRIEVAL verdict and is always shown as such -- but it
  // is no longer the end of the interaction. For off_topic/no_grounding the
  // server answers from the model's own general knowledge (grounded=false),
  // so the refusal notice and that answer appear together, visibly separated:
  // the corpus really did decline, and this answer really is unverified.
  if (result.refused) {
    const eligible =
      result.refusalReason === "off_topic" || result.refusalReason === "no_grounding";
    return (
      <>
        <div className="answer-refused">
          <span className="eyebrow refused-eyebrow">refused</span>
          <p>{REFUSAL_COPY[result.refusalReason ?? "no_grounding"]}</p>
        </div>
        {eligible && (
          <div className="answer-grid single">
            <AnswerCard
              label="general knowledge · unverified"
              state={synthesizing ? "pending" : result.synthesized ? "ready" : "unavailable"}
              tone="ungrounded"
            >
              {/* No disclaimer paragraph and no provider row: the card's own
                  "general knowledge · unverified" label plus the refusal
                  notice above it already say both things. */}
              {result.synthesized && <p className="answer-text">{result.synthesized.answer}</p>}
              {/* Rendered even when the attempt yields nothing. Letting the
                  card disappear was a real reported confusion: the user saw a
                  bare "refused" and reasonably read it as the fallback never
                  having run, when in fact it ran and every provider 429'd. */}
              {!synthesizing && !result.synthesized && (
                <p className="answer-unavailable">
                  unavailable — {SYNTHESIZE_UNAVAILABLE_COPY[synthesizeStatus ?? "unknown"]}
                </p>
              )}
            </AnswerCard>
          </div>
        )}
      </>
    );
  }

  return (
    <div className="answer-grid">
      <AnswerCard label="instant · grounded" state="ready">
        <p className="answer-text">{result.answer}</p>
        {result.answerRomanized && (
          <p className="answer-romanized mono">
            <span className="answer-romanized-label">{romanizationLabel(lang) ?? "romanized"}</span>
            {result.answerRomanized}
          </p>
        )}
        <div className="answer-meta">
          <Citations ids={result.citations} />
          <span className="confidence mono">conf {result.confidence.toFixed(2)}</span>
        </div>
      </AnswerCard>
      <AnswerCard
        label="synthesized · llm"
        state={synthesizing ? "pending" : result.synthesized ? "ready" : "unavailable"}
      >
        {result.synthesized &&
          (result.synthesized.declined ? (
            <p className="answer-declined">
              The retrieved passage doesn't actually answer this — declining rather than reciting it anyway.
            </p>
          ) : (
            <>
              <p className="answer-text">{result.synthesized.answer}</p>
              <div className="answer-meta">
                <span className="confidence mono">via {result.synthesized.provider}</span>
              </div>
            </>
          ))}
        {!synthesizing && !result.synthesized && (
          <p className="answer-unavailable">
            unavailable — {SYNTHESIZE_UNAVAILABLE_COPY[synthesizeStatus ?? "unknown"]} The fast extractive
            path above is still the grounded answer.
          </p>
        )}
      </AnswerCard>
    </div>
  );
}

function AnswerCard({
  label,
  state,
  tone,
  children,
}: {
  label: string;
  state: "pending" | "ready" | "unavailable";
  tone?: "ungrounded";
  children?: ReactNode;
}) {
  return (
    <div className={`answer-card ${state}${tone ? ` ${tone}` : ""}`}>
      <span className="eyebrow">{label}</span>
      {state === "pending" ? <div className="answer-skeleton" /> : children}
    </div>
  );
}
