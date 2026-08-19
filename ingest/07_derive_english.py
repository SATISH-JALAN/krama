"""
Derive an English passage subset from the already-ingested hi/bn/ta corpus,
instead of streaming a fourth language split from ai4bharat/MSMARCO-XI.

Why this works: MSMARCO-XI is a *translation* dataset -- every row 01_subset.py
already wrote carries `eng_passages` (`English_passages`) alongside the
translated `passages`, and 02_dedupe.py already carried that through as
`eng_text` on every deduped row (see its `seen[h] = {..., "eng_text": ...}`).
The original English MS MARCO passages are therefore already sitting in
data/medium/passages_dedup.jsonl -- no new network stream, no new
translation, nothing to re-verify against dataset schema drift (01_subset.py's
own docstring already lived through that pain once).

This script:
  1. Extracts every unique `eng_text` value from an existing deduped corpus
     (same SHA-1-of-NFC-normalized-text dedup as 02_dedupe.py, applied to
     `eng_text` instead of `text`), producing new rows with the same schema
     server/index.ts's bootFromDisk() already reads (passage_id/lang/text).
  2. Skips any hash already present in the target corpus, so it's safe to
     re-run.
  3. Aggregates `seen_is_selected`/`source_queries` across every hi/bn/ta row
     that shares the same underlying English passage, mirroring 02_dedupe.py's
     own aggregation (CLAUDE.md #7: is_selected is corpus-provenance, not a
     reliable per-passage label -- kept for the same reason 02_dedupe.py keeps
     it, not used at serve time).

Run: python ingest/07_derive_english.py --in data/medium/passages_dedup.jsonl --out data/medium/_new_en.jsonl
"""

import argparse
import hashlib
import json
import sys
import unicodedata


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def passage_hash(text: str) -> str:
    return hashlib.sha1(nfc(text).encode("utf-8")).hexdigest()


def derive_english(in_path: str, out_path: str) -> tuple[int, int, int]:
    existing_hashes: set[str] = set()
    en_rows: dict[str, dict] = {}
    total_source_rows = 0

    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            total_source_rows += 1
            if row["lang"] == "en":
                existing_hashes.add(row["passage_id"])
                continue

            eng_text_nfc = nfc(row["eng_text"])
            if not eng_text_nfc.strip():
                continue
            h = passage_hash(eng_text_nfc)
            if h in existing_hashes:
                continue

            if h not in en_rows:
                en_rows[h] = {
                    "passage_id": h,
                    "lang": "en",
                    "text": eng_text_nfc,
                    "eng_text": eng_text_nfc,
                    "seen_is_selected": list(row.get("seen_is_selected", [])),
                    "source_queries": list(row.get("source_queries", [])),
                }
            else:
                en_rows[h]["seen_is_selected"].extend(row.get("seen_is_selected", []))
                en_rows[h]["source_queries"].extend(row.get("source_queries", []))

    with open(out_path, "w", encoding="utf-8") as f:
        for row in en_rows.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return total_source_rows, len(existing_hashes), len(en_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/medium/passages_dedup.jsonl")
    ap.add_argument("--out", default="data/medium/_new_en.jsonl")
    args = ap.parse_args()

    total, already_en, new_en = derive_english(args.in_path, args.out)
    print(
        f"scanned {total} rows, {already_en} already lang=en, "
        f"derived {new_en} new unique English passages -> {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
