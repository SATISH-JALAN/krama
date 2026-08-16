"""
Retrieval evaluation: Recall@{1,5,10}, MRR@10, nDCG@10 against is_selected.

Sparse-label caveat (CLAUDE.md #7): is_selected==1 passages ARE valid known
positives -- these metrics are safe to compute and standard MS MARCO practice.
What is NOT valid is treating is_selected==0 as a confirmed negative (MS MARCO
is sparsely labelled; an unselected passage may still be relevant). This
script never does that -- it only ever checks "did a known positive appear in
the top-k", never "did a known negative correctly stay out of the top-k".
That asymmetry is why these numbers should be read as a floor on true recall,
not an exact figure -- true recall can only be equal or higher.

Uses brute-force (exact) nearest-neighbor via a single matmul rather than
HNSW, because the Python `hnswlib` package cannot currently build on this
Windows dev machine (no MSVC Build Tools -- see MEMORY.md). This is not a
worse substitute for this purpose: brute-force is exact, so it actually
upper-bounds what an approximate HNSW index could ever achieve on the same
embeddings -- if brute-force recall is bad, no amount of HNSW tuning fixes
it, so this is still a valid way to judge the embedding/prefix/dedup
pipeline itself, just not a substitute for measuring HNSW's own recall loss
(a separate, later question once HNSW can actually run).

Relevance resolution is via `parent_passage_id` (CORRECTED for the Sprint 3 chunking bake-off): a
chunk's own text hash generally does NOT match the original passage's hash (it's a fragment, not the
whole passage), so relevance can't be resolved by re-hashing chunk text the way strategy A's
whole-passage chunks allowed. Every chunk row from ingest/03_chunk.py carries `parent_passage_id`
(the hash 02_dedupe.py assigned to the ORIGINAL passage) -- a chunk counts as relevant iff its
parent_passage_id matches a query's known-positive passage hash. Falls back to treating `passage_id`
as its own parent when `parent_passage_id` is absent, for backward compatibility with strategy A's
files (passages_dedup.jsonl), which predate this field and don't need it (whole passage == its own
chunk). Multiple chunk rows can share one parent_passage_id (e.g. several sentences from one relevant
passage) -- all of them count as relevant, any one appearing in top-k is a hit.

Run: python eval/retrieval_eval.py --queries data/medium --passages data/medium/passages_dedup.jsonl \
    --embeddings data/medium/embeddings.npy --ids data/medium/embeddings_ids.json
"""

import argparse
import hashlib
import io
import json
import math
import os
import sys
import unicodedata

import numpy as np


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def passage_hash(text: str) -> str:
    return hashlib.sha1(nfc(text).encode("utf-8")).hexdigest()


def load_queries(query_dir: str, langs: list[str]) -> list[dict]:
    rows = []
    for lang in langs:
        path = os.path.join(query_dir, f"{lang}.jsonl")
        if not os.path.exists(path):
            continue
        with io.open(path, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def dcg_at_k(relevances: list[int], k: int) -> float:
    return sum(
        rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k])
    )


def build_parent_pid_to_rows(chunk_rows: list[dict], row_ids: list[str]) -> dict[str, list[int]]:
    """Map parent_passage_id -> [row indices into the embedding matrix].

    row_ids[i] is the chunk's own passage_id (per embeddings_ids.json, i.e.
    the order actually embedded). chunk_rows carries parent_passage_id per
    chunk. Join on passage_id, not list order -- 04_embed.py's output order
    matches its input order today, but this makes that assumption explicit
    and safe to break later rather than a silent positional dependency.
    """
    id_to_parent = {
        r["passage_id"]: r.get("parent_passage_id", r["passage_id"]) for r in chunk_rows
    }
    out: dict[str, list[int]] = {}
    for row_index, pid in enumerate(row_ids):
        parent = id_to_parent.get(pid, pid)  # fallback: strategy A has no chunk file at all
        out.setdefault(parent, []).append(row_index)
    return out


def evaluate(
    queries: list[dict],
    passage_embeddings: np.ndarray,
    parent_pid_to_rows: dict[str, list[int]],
    embed_query_fn,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> dict:
    recalls = {k: [] for k in k_values}
    reciprocal_ranks = []
    ndcgs = []
    skipped_no_positive = 0
    skipped_no_row = 0

    for row in queries:
        positive_texts = [
            p for p, sel in zip(row["passages"], row["is_selected"]) if sel == 1
        ]
        if not positive_texts:
            skipped_no_positive += 1
            continue

        positive_rows = set()
        for text in positive_texts:
            pid = passage_hash(text)
            positive_rows |= set(parent_pid_to_rows.get(pid, []))
        if not positive_rows:
            skipped_no_row += 1
            continue

        q_emb = embed_query_fn(row["query"])
        scores = passage_embeddings @ q_emb
        ranking = np.argsort(-scores)

        ranked_relevance = [1 if r in positive_rows else 0 for r in ranking]

        for k in k_values:
            recalls[k].append(1 if any(ranked_relevance[:k]) else 0)

        first_hit = next((i for i, rel in enumerate(ranked_relevance) if rel), None)
        reciprocal_ranks.append(1.0 / (first_hit + 1) if first_hit is not None and first_hit < 10 else 0.0)

        ideal = sorted(ranked_relevance, reverse=True)
        idcg = dcg_at_k(ideal, 10)
        ndcgs.append(dcg_at_k(ranked_relevance, 10) / idcg if idcg > 0 else 0.0)

    n = len(reciprocal_ranks)
    return {
        "n_evaluated": n,
        "skipped_no_positive_label": skipped_no_positive,
        "skipped_positive_not_in_corpus": skipped_no_row,
        **{f"recall@{k}": (sum(v) / n if n else None) for k, v in recalls.items()},
        "mrr@10": (sum(reciprocal_ranks) / n if n else None),
        "ndcg@10": (sum(ndcgs) / n if n else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="data/medium")
    ap.add_argument("--passages", default="data/medium/passages_dedup.jsonl")
    ap.add_argument("--embeddings", default="data/medium/embeddings.npy")
    ap.add_argument("--ids", default="data/medium/embeddings_ids.json")
    ap.add_argument("--langs", nargs="+", default=["hi", "bn", "ta"])
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--out", default=None, help="write JSON results here, e.g. eval/results/A.json")
    ap.add_argument("--strategy", default=None, help="label for this run, included in the JSON output")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ingest"))
    from prefixing import add_query_prefix  # noqa: E402

    passage_embeddings = np.load(args.embeddings)
    with open(args.ids, "r", encoding="utf-8") as f:
        row_ids = json.load(f)

    with io.open(args.passages, "r", encoding="utf-8") as f:
        chunk_rows = [json.loads(line) for line in f]
    parent_pid_to_rows = build_parent_pid_to_rows(chunk_rows, row_ids)

    queries = load_queries(args.queries, args.langs)
    print(f"loaded {len(queries)} queries, {passage_embeddings.shape[0]} chunks", file=sys.stderr)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    def embed_query_fn(text: str) -> np.ndarray:
        return model.encode(
            [add_query_prefix(text)], normalize_embeddings=True, convert_to_numpy=True
        )[0]

    results = evaluate(queries, passage_embeddings, parent_pid_to_rows, embed_query_fn)
    if args.strategy:
        results = {"strategy": args.strategy, **results}
    print(json.dumps(results, indent=2), file=sys.stderr)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
