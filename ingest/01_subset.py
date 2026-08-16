"""
Stream a per-language subset of ai4bharat/MSMARCO-XI to local JSONL.

Invariant (CLAUDE.md #9): never load_dataset without streaming=True -- the full
dataset is 55.6GB and does not fit in memory.

CORRECTED from the first version of this script (see MEMORY.md for the full
story): ARCHITECTURE.md/RESEARCH-SPEC.md assumed
`load_dataset("ai4bharat/MSMARCO-XI", lang, split="validation")` with `lang` as
a 2-letter config name. That does not work -- verified by running it. There is
exactly one builder config ("default"), and the `target_lang` field uses
FLORES-200-style codes (e.g. "hin_Deva"), not "hi". Worse, the merged
validation split is language-grouped, not interleaved (the first ~800 rows
sampled were all Assamese), so a naive stream+filter would waste huge amounts
of time/bandwidth scanning past irrelevant languages.

The actual fix, found via `HfApi().list_repo_files`: the dataset ships as
separate per-language parquet files (train/hintrain.parquet,
validation/hinval.parquet, etc.), targeted directly via `data_files` --
verified this loads correctly and every row reports the expected target_lang.

Run: python ingest/01_subset.py --langs hi bn ta --n-per-lang 25000
Smoke test: python ingest/01_subset.py --langs hi --n-per-lang 50 --out data/smoke
"""

import argparse
import io
import json
import os
import sys

from datasets import load_dataset

# 2-letter code (used throughout CLAUDE.md/PLAN.md) -> 3-letter file-name
# prefix (used by the dataset's actual parquet file names). Only hi/bn/ta are
# exercised by the MVP corpus target; the rest are filled in from the file
# listing for completeness but unverified beyond "the file exists".
LANG_FILE_PREFIX = {
    "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan",
    "ml": "mal", "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan",
    "sa": "san", "ta": "tam", "te": "tel", "ur": "urd",
}

EXPECTED_FIELDS = {
    "source_lang", "target_lang", "meta", "query", "query_id", "query_type",
    "Answer", "Eng_Query", "Eng_Answer", "passages",
}
EXPECTED_PASSAGE_FIELDS = {"is_selected", "English_passages", "Translated_passages"}


def subset_language(lang: str, n: int, out_dir: str, split: str = "validation") -> int:
    if lang not in LANG_FILE_PREFIX:
        raise ValueError(f"unknown language code {lang!r}, known: {sorted(LANG_FILE_PREFIX)}")
    prefix = LANG_FILE_PREFIX[lang]
    remote_path = f"{split}/{prefix}{'val' if split == 'validation' else 'train'}.parquet"

    ds = load_dataset(
        "ai4bharat/MSMARCO-XI",
        data_files={split: remote_path},
        split=split,
        streaming=True,
    )

    written = 0
    out_path = os.path.join(out_dir, f"{lang}.jsonl")
    os.makedirs(out_dir, exist_ok=True)

    with io.open(out_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if i == 0:
                _validate_schema(ex, lang)
            if i >= n:
                break
            row = {
                "qid": ex["query_id"],
                "lang": lang,
                "query": ex["query"],
                "eng_query": ex["Eng_Query"],
                "answer": ex["Answer"],
                "eng_answer": ex["Eng_Answer"],
                "qtype": ex["query_type"],
                "passages": ex["passages"]["Translated_passages"],
                "eng_passages": ex["passages"]["English_passages"],
                "is_selected": ex["passages"]["is_selected"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    return written


def _validate_schema(example: dict, lang: str) -> None:
    """Fail loudly on the first row if the dataset card has changed again."""
    missing = EXPECTED_FIELDS - set(example.keys())
    if missing:
        raise RuntimeError(
            f"[{lang}] dataset schema drift: missing fields {missing}. "
            f"Actual keys: {sorted(example.keys())}. "
            "The HF dataset card has a history of being wrong (see RESEARCH-SPEC.md) "
            "-- fix this script before trusting any output from it."
        )
    missing_passage = EXPECTED_PASSAGE_FIELDS - set(example["passages"].keys())
    if missing_passage:
        raise RuntimeError(
            f"[{lang}] passages sub-schema drift: missing {missing_passage}. "
            f"Actual: {sorted(example['passages'].keys())}"
        )
    sys.stderr.write(
        f"[{lang}] schema OK, target_lang={example['target_lang']!r}, "
        f"query_type={example['query_type']!r}, "
        f"#passages={len(example['passages']['is_selected'])}\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["hi", "bn", "ta"])
    ap.add_argument("--n-per-lang", type=int, default=25_000)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    total = 0
    for lang in args.langs:
        n = subset_language(lang, args.n_per_lang, args.out, args.split)
        sys.stderr.write(f"[{lang}] wrote {n} rows -> {args.out}/{lang}.jsonl\n")
        total += n
    sys.stderr.write(f"TOTAL rows written: {total}\n")


if __name__ == "__main__":
    main()
