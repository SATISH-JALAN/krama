"""
Build the HNSW dense index from passage embeddings.

Design note (correction to ARCHITECTURE.md §3's listed artifacts): BM25
postings are NOT built here. Building lexical postings in Python and loading
them in the TS server would mean the Indic tokenizer has to match exactly
across two languages -- the same class of cross-language consistency risk as
the e5 prefix invariant, but for tokenization instead of prefixing, and with
no unit test able to catch a drift between the two implementations. Since
BM25's postings only need to match between build-time and query-time (not
between two different runtimes), building them in server/maun/../bm25.ts at
server boot, in the same language they're queried from, removes that risk
entirely for free. artifacts/bm25.bin is dropped; server/bm25.ts builds its
postings in-memory from passages_dedup.jsonl on boot instead.

hnswlib's on-disk index format is produced by the same underlying C++ library
that both the Python `hnswlib` package and the server's `hnswlib-node` binding
wrap, so an index built here loads directly via hnswlib-node's readIndexSync
with no conversion step.

Run: python ingest/05_build_index.py \
    --embeddings artifacts/passage_embeddings.npy \
    --ids artifacts/passage_embeddings_ids.json \
    --out artifacts/hnsw.bin
"""

import argparse
import json
import sys

import numpy as np


def build_hnsw(embeddings: np.ndarray, ef_construction: int = 200, m: int = 16):
    # Imported lazily -- not needed by any other script, keeps import-time
    # dependency surface small for scripts that don't need it.
    import hnswlib

    dim = embeddings.shape[1]
    n = embeddings.shape[0]

    index = hnswlib.Index(space="ip", dim=dim)  # inner product -- CLAUDE.md #2
    index.init_index(max_elements=n, ef_construction=ef_construction, M=m)
    index.add_items(embeddings, np.arange(n))
    index.set_ef(64)  # query-time default; server re-sets this after readIndexSync too
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="artifacts/passage_embeddings.npy")
    ap.add_argument("--ids", default="artifacts/passage_embeddings_ids.json")
    ap.add_argument("--out", default="artifacts/hnsw.bin")
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--m", type=int, default=16)
    args = ap.parse_args()

    embeddings = np.load(args.embeddings)
    with open(args.ids, "r", encoding="utf-8") as f:
        ids = json.load(f)
    assert len(ids) == embeddings.shape[0], "embedding/id count mismatch -- rebuild both together"

    # Sanity check: CLAUDE.md #2 requires L2-normalized vectors for IP-space
    # HNSW to behave as cosine similarity. Catch a normalization bug here,
    # loudly, instead of silently degrading recall.
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        bad = int((~np.isclose(norms, 1.0, atol=1e-3)).sum())
        raise RuntimeError(
            f"{bad}/{len(norms)} embeddings are not L2-normalized "
            f"(norm range [{norms.min():.4f}, {norms.max():.4f}]) -- "
            "fix 04_embed.py before building the index, do not build on bad vectors."
        )

    index = build_hnsw(embeddings, args.ef_construction, args.m)
    index.save_index(args.out)

    # id_map.json: HNSW integer label (row index) -> real passage_id string,
    # needed by the server to translate ANN results back to passages.
    with open(args.out.replace(".bin", "_id_map.json"), "w", encoding="utf-8") as f:
        json.dump(ids, f)

    print(
        f"built HNSW index: {embeddings.shape[0]} vectors, {embeddings.shape[1]}d, "
        f"ip space -> {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
