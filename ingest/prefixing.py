"""
e5 asymmetric prefixing — shared logic for index-time (Python) and referenced by
the query-time (TypeScript, server/ghana/embed.ts) implementation.

Invariant (CLAUDE.md #1): every passage embedded as `passage: {text}`, every
query as `query: {text}`. This is the single most common silent RAG bug —
missing or swapped prefixes costs 5-15 recall points with no visible error.
Both sides MUST stay in sync; if this file's prefix strings change, the
TypeScript equivalent must change too.
"""

PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


def add_passage_prefix(text: str) -> str:
    return f"{PASSAGE_PREFIX}{text}"


def add_query_prefix(text: str) -> str:
    return f"{QUERY_PREFIX}{text}"
