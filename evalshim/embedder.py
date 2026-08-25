"""embed / embed_one / get_model, backed by Krama's real ONNX embedder.

THE ONE THING THAT MATTERS HERE is that `embed` and `embed_one` are NOT the
same call with different arity. Krama uses intfloat/multilingual-e5-small,
which is asymmetric: ingest/04_embed.py prefixed every corpus passage with
"passage: " while server/ghana/embed.ts prefixes every query with "query: ".

The suite uses these two functions for different things:
  * eval/index_build.py calls embed(texts) on CHUNKS  -> those are passages
  * eval/pipeline.py  calls embed_one(query)          -> that is a query

Routing both through the "query: " path (the obvious shim, and what Krama's
runtime embed() did before it took an explicit kind) puts passage vectors in
the wrong half of the asymmetric space. Recall@k would come out low and the
number would be an artifact of this file, not a fact about Krama.
"""

import numpy as np

from . import _client


def get_model():
    """Called once by the suite before timing anything.

    Krama's model lives in the Bun process, so there is nothing to load here --
    but this is the suite's first contact with the target, which makes it the
    right place to fail loudly if the adapter is not running. Boot the ONNX
    session too, so the first real embed() call is not paying load time that
    then shows up in the latency report.
    """
    info = _client.health()
    if not info.get("rerankerEnabled"):
        print(
            "[evalshim] WARNING: adapter reports the reranker is NOT loaded. "
            "Krama's L3 relevance gate is off, so the reliability check cannot "
            "catch fabrication and its number will be meaningless."
        )
    _client.embed_texts(["warmup"], "query")
    return info


def embed(texts):
    """Passages. Shape (len(texts), dim), float32."""
    vectors = _client.embed_texts(list(texts), "passage")
    return np.asarray(vectors, dtype=np.float32)


def embed_one(text):
    """A single query. Shape (dim,), float32.

    The suite calls .reshape(1, -1) and .shape[-1] on this, both of which a
    numpy array satisfies. Vectors arrive L2-normalised from ghana/embed.ts,
    which is what the suite's faiss.METRIC_INNER_PRODUCT index expects for
    inner product to equal cosine.
    """
    vectors = _client.embed_texts([text], "query")
    return np.asarray(vectors[0], dtype=np.float32)
