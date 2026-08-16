"""
Dedupe passages across all subset JSONL files by SHA-1 of NFC-normalized text.

Invariant (CLAUDE.md #3): Unicode NFC normalization on all Indic text, offline
and online, before hashing or tokenizing — Devanagari nukta characters have two
valid encodings that never match otherwise.

MS MARCO reuses passages heavily across queries; skipping dedupe roughly doubles
the index for zero recall gain (ARCHITECTURE.md §4.1).

Run: python ingest/02_dedupe.py --in data --out data/passages_dedup.jsonl
"""

import argparse
import hashlib
import json
import os
import sys
import unicodedata


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def passage_hash(text: str) -> str:
    return hashlib.sha1(nfc(text).encode("utf-8")).hexdigest()


def dedupe(in_dir: str, out_path: str) -> tuple[int, int]:
    seen: dict[str, dict] = {}
    total_instances = 0

    for fname in sorted(os.listdir(in_dir)):
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(in_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                lang = row["lang"]
                for translated, english, is_sel in zip(
                    row["passages"], row["eng_passages"], row["is_selected"]
                ):
                    total_instances += 1
                    translated_nfc = nfc(translated)
                    h = passage_hash(translated_nfc)
                    if h not in seen:
                        seen[h] = {
                            "passage_id": h,
                            "lang": lang,
                            "text": translated_nfc,
                            "eng_text": nfc(english),
                            # is_selected is NOT a reliable per-passage label once
                            # deduped across queries (CLAUDE.md #7) — kept only as
                            # a same-query provenance signal, never trust it alone.
                            "seen_is_selected": [is_sel],
                            "source_queries": [row["qid"]],
                        }
                    else:
                        seen[h]["seen_is_selected"].append(is_sel)
                        seen[h]["source_queries"].append(row["qid"])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in seen.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return total_instances, len(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", default="data")
    ap.add_argument("--out", default="data/passages_dedup.jsonl")
    args = ap.parse_args()

    total, unique = dedupe(args.in_dir, args.out)
    ratio = (unique / total * 100) if total else 0
    print(
        f"instances={total} unique={unique} ({ratio:.1f}% kept) -> {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
