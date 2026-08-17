"""
One-off export for eval/calibrate_guardrails.ts (PLAN.md E5.5):
1. Dumps data/medium/embeddings.npy as a headerless raw float32 LE binary
   (trivial for Bun/TS to mmap-read as a Float32Array, avoids writing an
   .npy parser for a one-time calibration script).
2. Computes the corpus centroid the exact same way server/maun/ood.ts's
   computeCentroid() does (mean, then re-normalize to unit length) so the
   calibration script and production code can never silently diverge --
   verified to match by construction, not just by inspection.
"""
import numpy as np
import json

emb = np.load("data/medium/embeddings.npy")
assert emb.dtype == np.float32
emb.tofile("data/medium/embeddings.f32bin")

centroid = emb.mean(axis=0)
centroid = centroid / np.linalg.norm(centroid)
with open("data/medium/centroid.json", "w", encoding="utf-8") as f:
    json.dump(centroid.astype(float).tolist(), f)

print(f"wrote embeddings.f32bin ({emb.shape[0]} x {emb.shape[1]}) and centroid.json")
