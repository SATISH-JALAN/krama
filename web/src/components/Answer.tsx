import type { ReactNode } from "react";
import type { GroundedAnswer, RefusalReason } from "../lib/contracts";
import Citations from "./Citations";
import "./Answer.css";

interface Props {
  result: GroundedAnswer | null;
  busy: boolean;
}

const REFUSAL_COPY: Record<RefusalReason, string> = {
  empty_or_gibberish: "Didn't catch a question there — try speaking again or typing it in.",
  unsafe: "This falls outside what the guardrails allow an answer for.",
  off_topic: "That's not covered by what's in the knowledge base.",
  no_grounding: "No passage in the knowledge base supports an answer — refusing rather than guessing.",
};

export default function Answer({ result, busy }: Props) {
  if (busy) {
    return (
      <div className="answer-grid">
        <AnswerCard label="instant · grounded" state="pending" />
        <AnswerCard label="synthesized · streaming" state="pending" />
      </div>
    );
  }

  if (!result) return null;

  if (result.refused) {
    return (
      <div className="answer-refused">
        <span className="eyebrow refused-eyebrow">refused</span>
        <p>{REFUSAL_COPY[result.refusalReason ?? "no_grounding"]}</p>
      </div>
    );
  }

  return (
    <div className="answer-grid">
      <AnswerCard label="instant · grounded" state="ready">
        <p className="answer-text">{result.answer}</p>
        <div className="answer-meta">
          <Citations ids={result.citations} />
          <span className="confidence mono">conf {result.confidence.toFixed(2)}</span>
        </div>
      </AnswerCard>
      <AnswerCard label="synthesized · streaming" state={result.synthesized ? "ready" : "unavailable"}>
        {result.synthesized ? (
          <p className="answer-text">
            {result.synthesized.answer}
            {result.synthesized.streaming && <span className="cursor" aria-hidden />}
          </p>
        ) : (
          <p className="answer-unavailable">
            not wired into this server build yet — the fast extractive path above is the grounded answer.
          </p>
        )}
      </AnswerCard>
    </div>
  );
}

function AnswerCard({
  label,
  state,
  children,
}: {
  label: string;
  state: "pending" | "ready" | "unavailable";
  children?: ReactNode;
}) {
  return (
    <div className={`answer-card ${state}`}>
      <span className="eyebrow">{label}</span>
      {state === "pending" ? <div className="answer-skeleton" /> : children}
    </div>
  );
}
