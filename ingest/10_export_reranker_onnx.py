"""
Export a cross-encoder reranker to ONNX, for real-time query-passage
relevance scoring inside handleQuery()'s fast path (server/ghana/rerank.ts).

Why this exists: real voice-testing found the bi-encoder cosine score
(dense_search's topScore) doesn't reliably separate a genuinely relevant
passage from an irrelevant one -- a completely unrelated corpus passage (a
video-game-character discussion) scored high enough to pass as the
"grounded" answer to a Taj Mahal question. A cross-encoder that jointly
encodes (query, passage) together is a much better-calibrated relevance
signal than comparing two independently-computed embedding vectors, and is
the same technique the reference implementation studied for this task
(RAGINGOA) uses for its own reranking stage.

Model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 -- trained on mMARCO
(machine-translated MS MARCO, 14 languages including Hindi, verified live
against the mMARCO paper before picking this, not assumed) -- a direct
fit since this project's own corpus IS MS MARCO-XI. Bengali and Tamil are
NOT in mMARCO's 14 languages; the base "multilingual MiniLMv2" model is
still broadly multilingual from its own pretraining, so it likely carries
*some* transferable signal for bn/ta, but this is exactly the kind of claim
that must be verified against real corpus pairs, not assumed by lore --
see eval/verify_reranker.py, run after this script, before trusting bn/ta
scores in production.

Unlike the embedding model (06_export_onnx.py), this is a SEQUENCE
CLASSIFICATION model (a single relevance logit per (query, passage) pair,
not a pooled sentence embedding) -- different AutoModel class, different
ONNX graph shape (a real single `input_ids` sequence containing BOTH
segments joined by [SEP], with token_type_ids meaningfully distinguishing
segment A (query) from segment B (passage), not all-zeros like embed.ts).

Run: python ingest/10_export_reranker_onnx.py --out artifacts/onnx_reranker
"""

import argparse
import os
import sys

import numpy as np


def export_onnx(model_name: str, out_dir: str) -> str:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    os.makedirs(out_dir, exist_ok=True)
    model = ORTModelForSequenceClassification.from_pretrained(model_name, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    fp32_path = os.path.join(out_dir, "model.onnx")
    assert os.path.exists(fp32_path), f"expected {fp32_path} after export"
    return fp32_path


def quantize_int8(fp32_path: str, out_dir: str) -> str:
    from onnxruntime.quantization import quantize_dynamic, QuantType

    int8_path = os.path.join(out_dir, "model_int8.onnx")
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
    return int8_path


def score_onnx(onnx_path: str, tokenizer_dir: str, pairs: list[tuple[str, str]]) -> np.ndarray:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    queries = [p[0] for p in pairs]
    passages = [p[1] for p in pairs]
    # Sentence-PAIR tokenization -- tokenizer joins both with [SEP] and sets
    # token_type_ids itself (0 for query tokens, 1 for passage tokens); this
    # is exactly what makes a cross-encoder different from the bi-encoder in
    # embed.ts, which only ever sees one segment.
    enc = tok(queries, passages, padding=True, truncation=True, max_length=256, return_tensors="np")
    session_inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    expected_inputs = {inp.name for inp in session.get_inputs()}
    if "token_type_ids" in expected_inputs:
        session_inputs["token_type_ids"] = enc["token_type_ids"].astype(np.int64)
    outputs = session.run(None, session_inputs)
    logits = outputs[0]
    return logits.reshape(-1)


def score_pytorch(model_name: str, pairs: list[tuple[str, str]]) -> np.ndarray:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)
    return np.array(model.predict(pairs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    ap.add_argument("--out", default="artifacts/onnx_reranker")
    args = ap.parse_args()

    fp32_path = export_onnx(args.model, args.out)
    sys.stderr.write(f"exported fp32 ONNX -> {fp32_path}\n")

    int8_path = quantize_int8(fp32_path, args.out)
    sys.stderr.write(f"quantized int8 -> {int8_path}\n")

    # Sanity pairs spanning clearly-relevant and clearly-irrelevant, English
    # only here (the real cross-lingual validation is eval/verify_reranker.py,
    # which uses this exact ONNX artifact against real corpus passages).
    pairs = [
        ("what is a corporation", "A corporation is a company or group of people authorized to act as a single entity and recognized as such in law."),
        ("what is a corporation", "Bananas are a good source of potassium and dietary fiber."),
    ]
    pt_scores = score_pytorch(args.model, pairs)
    onnx_fp32_scores = score_onnx(fp32_path, args.out, pairs)
    onnx_int8_scores = score_onnx(int8_path, args.out, pairs)
    sys.stderr.write(f"PyTorch scores:     {pt_scores}\n")
    sys.stderr.write(f"ONNX fp32 scores:   {onnx_fp32_scores}\n")
    sys.stderr.write(f"ONNX int8 scores:   {onnx_int8_scores}\n")

    fp32_diff = float(np.abs(pt_scores - onnx_fp32_scores).max())
    int8_diff = float(np.abs(pt_scores - onnx_int8_scores).max())
    sys.stderr.write(f"max abs diff PyTorch vs fp32 ONNX: {fp32_diff:.5f}\n")
    sys.stderr.write(f"max abs diff PyTorch vs int8 ONNX: {int8_diff:.5f}\n")

    relevant_scores_higher_fp32 = onnx_fp32_scores[0] > onnx_fp32_scores[1]
    relevant_scores_higher_int8 = onnx_int8_scores[0] > onnx_int8_scores[1]
    sys.stderr.write(f"fp32 correctly ranks relevant > irrelevant: {relevant_scores_higher_fp32}\n")
    sys.stderr.write(f"int8 correctly ranks relevant > irrelevant: {relevant_scores_higher_int8}\n")

    if not (relevant_scores_higher_fp32 and relevant_scores_higher_int8):
        sys.stderr.write("FAIL -- exported model doesn't even separate an obviously relevant pair from an obviously irrelevant one\n")
        raise SystemExit(1)

    sys.stderr.write(f"\nwrote {args.out}/model.onnx (fp32) and {args.out}/model_int8.onnx\n")


if __name__ == "__main__":
    main()
