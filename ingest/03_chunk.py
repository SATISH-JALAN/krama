"""
Chunk variants B (fixed 256/64 word window) and D/E (sentence-level, shared
index -- see below) from deduped passages.

Every chunk row carries `parent_passage_id` (the SHA1 hash 02_dedupe.py
already assigns to the whole passage) regardless of strategy, so
eval/retrieval_eval.py can resolve relevance the same way for every
strategy: is_selected is a passage-level label, so a chunk counts as
relevant iff its parent_passage_id matches a query's known-positive passage
hash. Strategy A's own "chunks" are just the deduped passages themselves
(parent_passage_id == passage_id, self-referential) -- not regenerated here.

Strategy D vs E (ARCHITECTURE.md §4.3): both retrieve at SENTENCE
granularity -- they share the exact same index and therefore the exact same
ranking/recall numbers by construction. They differ only in what gets
returned to the user/LLM after retrieval: D expands to the sentence's +/-2
neighbors, E expands to the whole parent passage. That expansion is a
retrieval-time (server-side) concern, not an indexing-time one, so this
script produces one "chunks_DE" sentence-level index, not two.

Run: python ingest/03_chunk.py --in data/bakeoff/passages_dedup.jsonl --out data/bakeoff
"""

import argparse
import io
import json
import sys

from indicnlp.tokenize import sentence_tokenize

WINDOW_WORDS = 256
OVERLAP_WORDS = 64

# indic_nlp_library language codes differ from our 2-letter dataset codes for
# some languages; hi/bn/ta (our actual scope) map directly.
INDIC_NLP_LANG = {"hi": "hi", "bn": "bn", "ta": "ta"}


def chunk_fixed_window(text: str, window: int = WINDOW_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if len(words) <= window:
        return [text]
    chunks = []
    step = window - overlap
    for start in range(0, len(words), step):
        piece = words[start : start + window]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + window >= len(words):
            break
    return chunks


def chunk_sentences(text: str, lang: str) -> list[str]:
    indic_lang = INDIC_NLP_LANG.get(lang)
    if indic_lang is None:
        # fall back to whole-text-as-one-sentence for unsupported languages
        # rather than guessing at punctuation rules we haven't verified.
        return [text]
    sents = sentence_tokenize.sentence_split(text, lang=indic_lang)
    return [s for s in sents if s.strip()] or [text]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/bakeoff/passages_dedup.jsonl")
    ap.add_argument("--out", default="data/bakeoff")
    args = ap.parse_args()

    with io.open(args.in_path, "r", encoding="utf-8") as f:
        passages = [json.loads(line) for line in f]

    b_rows = []
    de_rows = []
    b_multi_chunk_count = 0
    de_sentence_counts = []

    for p in passages:
        parent_id = p["passage_id"]
        lang = p["lang"]
        text = p["text"]

        b_chunks = chunk_fixed_window(text)
        if len(b_chunks) > 1:
            b_multi_chunk_count += 1
        for i, chunk_text in enumerate(b_chunks):
            b_rows.append({
                "passage_id": f"{parent_id}_B{i}",
                "parent_passage_id": parent_id,
                "lang": lang,
                "text": chunk_text,
            })

        sents = chunk_sentences(text, lang)
        de_sentence_counts.append(len(sents))
        for i, sent_text in enumerate(sents):
            de_rows.append({
                "passage_id": f"{parent_id}_S{i}",
                "parent_passage_id": parent_id,
                "lang": lang,
                "text": sent_text,
                "sentence_idx": i,
                "num_sentences": len(sents),
            })

    b_path = f"{args.out}/chunks_B.jsonl"
    de_path = f"{args.out}/chunks_DE.jsonl"
    with io.open(b_path, "w", encoding="utf-8") as f:
        for row in b_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with io.open(de_path, "w", encoding="utf-8") as f:
        for row in de_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    avg_sents = sum(de_sentence_counts) / len(de_sentence_counts) if de_sentence_counts else 0
    sys.stderr.write(
        f"strategy B: {len(passages)} passages -> {len(b_rows)} chunks "
        f"({b_multi_chunk_count} passages split into >1 window) -> {b_path}\n"
    )
    sys.stderr.write(
        f"strategy D/E: {len(passages)} passages -> {len(de_rows)} sentence-chunks "
        f"(avg {avg_sents:.2f} sentences/passage) -> {de_path}\n"
    )


if __name__ == "__main__":
    main()
