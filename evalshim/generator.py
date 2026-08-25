"""generate_answer, backed by Krama's real L3 gate + LLM synthesis chain.

`grounded` is the field the suite's reliability check ("lying factor") keys
off, and it is the one place a target can flatter itself for free: a generator
that hardcodes grounded=True can never be caught fabricating on an unanswerable
query. So it is wired to Krama's actual refusal signal, not to "did we produce
a string" -- the adapter returns grounded=False whenever the cross-encoder
relevance gate rejects the best passage, whenever the LLM itself declines, and
whenever no provider answered. See eval-adapter/serve.ts for which of Krama's
guardrails are replayed here and, more importantly, which are not and why.
"""

from dataclasses import dataclass

from . import _client


@dataclass
class Answer:
    text: str
    grounded: bool
    generation_ms: float
    model: str
    # Not part of the suite's required surface -- it reads .text/.grounded/
    # .generation_ms/.model and ignores the rest. Kept because when a run
    # comes back with a surprising reliability number, "which gate stopped
    # this one" is the first question, and re-running to find out is slow.
    gate: str = ""


def generate_answer(query: str, results: list) -> Answer:
    passages = [
        {"text": getattr(r, "text", ""), "source": str(getattr(r, "source", ""))}
        for r in results
    ]
    payload = _client.generate(query, passages)
    return Answer(
        text=payload.get("text", ""),
        grounded=bool(payload.get("grounded", False)),
        generation_ms=float(payload.get("generation_ms", 0.0)),
        model=str(payload.get("model", "krama")),
        gate=str(payload.get("gate", "")),
    )
