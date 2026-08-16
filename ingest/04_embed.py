"""
Embed deduped passages with multilingual-e5-small.

Invariants enforced here (CLAUDE.md):
  #1 e5 prefixes    -- every passage embedded as `passage: {text}` (prefixing.py)
  #2 same model both sides -- this is the ONE place passage embeddings are
     produced; the server's query-time embedding (server/ghana/embed.ts, not
     yet written) must use the identical model + normalization or retrieval
     silently degrades.

Pooling: intfloat/multilingual-e5-small ships as a sentence-transformers model
with its own pooling config (mean pooling over token embeddings weighted by
attention mask) baked into the model files -- calling .encode() uses that
config directly, so this script does not hand-roll pooling. The ONNX export
path used for the server (server-side, not this script) DOES need pooling
implemented by hand and must match this exactly -- confirmed via research
that mean-pooling-with-attention-mask is correct for this model.

Run: python ingest/04_embed.py --in data/passages_dedup.jsonl --out artifacts/passage_embeddings
"""

import argparse
import json
import os
import sys

import numpy as np

from prefixing import add_passage_prefix

MODEL_NAME = "intfloat/multilingual-e5-small"
BATCH_SIZE = 256


def load_passages(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def embed_passages(rows: list[dict], model_name: str = MODEL_NAME) -> np.ndarray:
    # Imported lazily -- sentence-transformers/torch are a heavy, slow install
    # and not needed for any other script in this directory.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = [add_passage_prefix(r["text"]) for r in rows]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,   # L2-normalize -- CLAUDE.md #2, required for IP-space HNSW
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    return embeddings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/passages_dedup.jsonl")
    ap.add_argument("--out", default="artifacts/passage_embeddings")
    ap.add_argument("--model", default=MODEL_NAME)
    args = ap.parse_args()

    rows = load_passages(args.in_path)
    print(f"loaded {len(rows)} passages from {args.in_path}", file=sys.stderr)

    embeddings = embed_passages(rows, args.model)
    assert embeddings.shape[0] == len(rows)
    assert embeddings.shape[1] == 384, f"expected 384-d, got {embeddings.shape[1]}"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out + ".npy", embeddings)

    # Sidecar mapping row index -> passage_id, needed to attach HNSW's integer
    # labels back to real passage_ids at query time.
    with open(args.out + "_ids.json", "w", encoding="utf-8") as f:
        json.dump([r["passage_id"] for r in rows], f)

    print(
        f"wrote {embeddings.shape} float32 embeddings -> {args.out}.npy "
        f"(+ id mapping -> {args.out}_ids.json)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
