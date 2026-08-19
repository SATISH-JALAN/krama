"""
One-off merge step (run once, after 07_derive_english.py + 04_embed.py on its
output): folds the new English passage rows + embeddings into the live
data/medium/ corpus, in matching row order, then regenerates embeddings.f32bin
+ centroid.json the same way eval/export_centroid_and_raw.py does.

Row-order invariant this script exists to preserve: passages_dedup.jsonl's
row order must exactly match embeddings_ids.json's order, since
server/index.ts's bootFromDisk() zips them together positionally when
building passageEmbeddingsForCentroid. Appending both in the same order
(existing rows first, then new English rows in the same order they were
embedded) keeps that invariant intact.
"""
import json
import numpy as np

DATA_DIR = "data/medium"

with open(f"{DATA_DIR}/embeddings_ids.json", encoding="utf-8") as f:
    existing_ids = json.load(f)
existing_emb = np.load(f"{DATA_DIR}/embeddings.npy")
assert existing_emb.shape[0] == len(existing_ids)

with open(f"{DATA_DIR}/_new_en_embeddings_ids.json", encoding="utf-8") as f:
    new_ids_raw = json.load(f)
new_emb_raw = np.load(f"{DATA_DIR}/_new_en_embeddings.npy")
assert new_emb_raw.shape[0] == len(new_ids_raw)

# A handful of passages (untranslated boilerplate/formulas/URLs -- the
# translation left them character-for-character identical to the English
# source) hash-collide with a row already in the corpus under a different
# `lang` tag. They're exact-text duplicates by construction (same NFC text ->
# same SHA-1), so drop them from the new English batch rather than double-add
# the same passage_id.
existing_id_set = set(existing_ids)
keep_mask = [pid not in existing_id_set for pid in new_ids_raw]
dropped = len(new_ids_raw) - sum(keep_mask)
if dropped:
    print(f"dropping {dropped} passage(s) already present under a different lang tag: "
          f"{[pid for pid, keep in zip(new_ids_raw, keep_mask) if not keep]}")
new_ids = [pid for pid, keep in zip(new_ids_raw, keep_mask) if keep]
new_emb = new_emb_raw[keep_mask]

assert set(existing_ids).isdisjoint(new_ids), "id collision between existing and new corpus"

combined_emb = np.vstack([existing_emb, new_emb]).astype("float32")
combined_ids = existing_ids + new_ids
dropped_id_set = set(pid for pid, keep in zip(new_ids_raw, keep_mask) if not keep)

np.save(f"{DATA_DIR}/embeddings.npy", combined_emb)
with open(f"{DATA_DIR}/embeddings_ids.json", "w", encoding="utf-8") as f:
    json.dump(combined_ids, f)

combined_emb.tofile(f"{DATA_DIR}/embeddings.f32bin")

centroid = combined_emb.mean(axis=0)
centroid = centroid / np.linalg.norm(centroid)
with open(f"{DATA_DIR}/centroid.json", "w", encoding="utf-8") as f:
    json.dump(centroid.astype(float).tolist(), f)

with open(f"{DATA_DIR}/passages_dedup.jsonl", "a", encoding="utf-8") as out_f:
    with open(f"{DATA_DIR}/_new_en.jsonl", encoding="utf-8") as in_f:
        for line in in_f:
            row = json.loads(line)
            if row["passage_id"] in dropped_id_set:
                continue
            out_f.write(line)

print(
    f"merged: {len(existing_ids)} existing + {len(new_ids)} new english "
    f"= {len(combined_ids)} total passages, embeddings {combined_emb.shape}"
)
