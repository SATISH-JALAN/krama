"""
One-off: derive data/medium/en.jsonl (a query file matching the shape
bench/latency.ts already expects for hi/bn/ta) from the `eng_query` field
already present in data/medium/{hi,bn,ta}.jsonl -- same "it's already sitting
in the translation-paired data" logic as ingest/07_derive_english.py, applied
to queries instead of passages.
"""
import json

SRC_LANGS = ["hi", "bn", "ta"]
OUT_PATH = "data/medium/en.jsonl"

seen_text = set()
rows = []
per_lang_rows = {lang: [] for lang in SRC_LANGS}

for lang in SRC_LANGS:
    with open(f"data/medium/{lang}.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            per_lang_rows[lang].append(row)

# Round-robin across the three source files so the derived English query set
# isn't skewed toward whichever language file happens to be read first.
max_len = max(len(v) for v in per_lang_rows.values())
for i in range(max_len):
    for lang in SRC_LANGS:
        src = per_lang_rows[lang]
        if i >= len(src):
            continue
        row = src[i]
        q = row["eng_query"].strip()
        if not q or q in seen_text:
            continue
        seen_text.add(q)
        rows.append({"qid": row["qid"], "lang": "en", "query": q, "qtype": row.get("qtype", "DESCRIPTION")})

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"wrote {len(rows)} unique English queries -> {OUT_PATH}")
