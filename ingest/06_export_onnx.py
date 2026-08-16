"""
Export multilingual-e5-small to ONNX + int8 quantize, for the Bun server's
runtime query embedding (onnxruntime-node). This is separate from
04_embed.py's use of the PyTorch model -- that one does one-time offline
batch embedding of the whole passage corpus (speed doesn't matter much,
correctness/simplicity does); this script produces the artifact the actual
server loads for real-time per-query embedding, where speed matters a lot.

Critical validation step (ARCHITECTURE.md §4.5): quantization occasionally
destroys a model's embedding quality. Embed a set of real queries with both
the PyTorch model and the exported int8 ONNX model and assert mean cosine
similarity > 0.995 before trusting the ONNX artifact. Falls back to fp32 ONNX
(no quantization) automatically if int8 fails this check.

Run: python ingest/06_export_onnx.py --out artifacts/onnx
"""

import argparse
import os
import sys

import numpy as np


def export_onnx(model_name: str, out_dir: str) -> str:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    os.makedirs(out_dir, exist_ok=True)
    model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
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


def mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., None].astype(np.float32)
    summed = (last_hidden_state * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def l2norm(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, a_min=1e-9, a_max=None)


def embed_onnx(onnx_path: str, tokenizer_dir: str, texts: list[str]) -> np.ndarray:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    enc = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="np")
    session_inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    # XLM-R/BERT-style models require token_type_ids as an explicit graph
    # input even for single-segment text -- the exported ONNX graph doesn't
    # default it internally. All-zeros is correct here (single segment).
    # This bit the first run of this script for real: onnxruntime raised
    # "Required inputs (['token_type_ids']) are missing" -- the TS server's
    # equivalent embed function (server/ghana/embed.ts) must pass this too,
    # or it will hit the exact same error at boot.
    expected_inputs = {inp.name for inp in session.get_inputs()}
    if "token_type_ids" in expected_inputs:
        session_inputs["token_type_ids"] = np.zeros_like(enc["input_ids"], dtype=np.int64)
    outputs = session.run(None, session_inputs)
    last_hidden_state = outputs[0]
    pooled = mean_pool(last_hidden_state, enc["attention_mask"])
    return l2norm(pooled)


def embed_pytorch(model_name: str, texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def validate_cosine_agreement(
    model_name: str, onnx_path: str, tokenizer_dir: str, n_samples: int = 50
) -> float:
    from prefixing import add_query_prefix

    sample_texts = [
        add_query_prefix(f"sample validation query number {i} about a random topic")
        for i in range(n_samples)
    ]

    pt_emb = embed_pytorch(model_name, sample_texts)
    onnx_emb = embed_onnx(onnx_path, tokenizer_dir, sample_texts)

    cos_sim = (pt_emb * onnx_emb).sum(axis=1)  # both already L2-normalized
    return float(cos_sim.mean())


def main():
    sys.path.insert(0, os.path.dirname(__file__))

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--out", default="artifacts/onnx")
    ap.add_argument("--threshold", type=float, default=0.995)
    args = ap.parse_args()

    fp32_path = export_onnx(args.model, args.out)
    sys.stderr.write(f"exported fp32 ONNX -> {fp32_path}\n")

    int8_path = quantize_int8(fp32_path, args.out)
    sys.stderr.write(f"quantized int8 -> {int8_path}\n")

    agreement = validate_cosine_agreement(args.model, int8_path, args.out)
    sys.stderr.write(f"int8 vs PyTorch mean cosine similarity: {agreement:.5f}\n")

    if agreement >= args.threshold:
        sys.stderr.write(f"PASS (>= {args.threshold}) -- server should load model_int8.onnx\n")
    else:
        sys.stderr.write(
            f"FAIL (< {args.threshold}) -- int8 quantization degraded this model. "
            f"Server should load model.onnx (fp32) instead, ~2x slower but correct, "
            f"per ARCHITECTURE.md §4.5's documented fallback.\n"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
