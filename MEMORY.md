# KRAMA — Live Project Memory

**Read this first in any new session, before ARCHITECTURE.md/RESEARCH-SPEC.md.** CLAUDE.md has the
static invariants/stack/corrections. PLAN.md has the sprint structure. This file is the dynamic
"what's actually true right now" — update it after every significant step, don't let it drift stale.

Last updated: 2026-08-16 (Sprint 1 GO/NO-GO gate cleared with real numbers: Recall@10=0.666)

## Where things actually stand
- **Git**: initialized, on `main`, 2 commits (`3463163` scaffold, plus the Sprint 1 ingestion-code
  commit — check `git log` for the current hash, don't hardcode it here). Repo lives at
  `c:\projects\Krama` on native Windows (`win32`), **not** WSL2 — WSL2 is not installed on this
  machine (`wsl --install -d Ubuntu` needs admin elevation + reboot, both blocked in this sandbox,
  handed to the user as a manual step, not yet confirmed done).
- **Local toolchain installed this session** (via winget/pip, all user-scope, no admin needed):
  Python 3.12.10 at `C:\Users\satish\AppData\Local\Programs\Python\Python312\python.exe` (the bare
  `python`/`py` commands on PATH are Windows Store alias stubs and resolve to nothing — always
  invoke the full path or the venv's python), `uv` 0.12.5, a project venv at `.venv/` with
  `datasets`/`pyarrow`/`polars`/`torch`(CPU)/`sentence-transformers` installed — no lockfile yet,
  should add one once the full dependency list is settled. `hnswlib` (Python) does NOT install here
  (no MSVC Build Tools) — expected to work fine once run on Linux (WSL2/Oracle VM), not fixed here.
- **Bun**: not installed anywhere yet. Server code (Sprint 2+) hasn't started.
- **Not done, blocked on the user**: WSL2 install, Oracle Cloud account + Ampere A1 VM, GCP fallback
  signup, Sarvam/Groq/Cerebras API keys. None of Sprint -1 or the ARM-specific half of Sprint 0 can
  proceed until at least one of Oracle/GCP exists.
- **Sprint 1 ingestion pipeline**: all 5 scripts written (`01_subset.py` through `05_build_index.py`,
  plus `prefixing.py`/`test_prefixing.py`), and everything runnable on this machine has actually been
  run for real at smoke scale (50 rows/lang) and verified correct — see "Sprint 1 status" section
  below for exactly what's proven vs. still blocked.

## Real finding that corrects ARCHITECTURE.md / RESEARCH-SPEC.md
Both docs assume `load_dataset("ai4bharat/MSMARCO-XI", lang, split="validation", streaming=True)`
where `lang` is a per-language config name (`"hi"`, `"bn"`, `"ta"`). **This is wrong on two separate
counts, both verified by actually running it, not by reading docs:**

1. There is exactly one builder config, `"default"`, with only `train` (10,080,140 rows) and
   `validation` (1,371,174 rows) splits — no per-language configs exist (confirmed via
   `load_dataset_builder`).
2. The `target_lang` field (the actual language marker) uses **FLORES-200-style codes**, not ISO
   `"hi"/"bn"/"ta"` — e.g. Assamese is `"asm_Beng"`. Sampling the first 800 rows of the merged
   `validation` split found them **all** the same language (`asm_Beng`) — the split is
   language-grouped, not interleaved, so a naive stream-and-filter approach for a language later in
   the file order would mean scanning through huge irrelevant blocks first.

**The actual fix (found via `HfApi().list_repo_files`, not assumed): the dataset ships as separate
per-language parquet files**, not a merged split at all:
```
train/{asm,ben,guj,hin,kan,mal,mar,nep,ori,pan,san,tam,urd}train.parquet
validation/{asm,ben,guj,hin,kan,mal,mar,nep,ori,pan,san,tam,tel,urd}val.parquet
```
So the correct load is targeting the specific file directly, e.g.:
```python
load_dataset("ai4bharat/MSMARCO-XI",
             data_files={"validation": "validation/hinval.parquet"},
             split="validation", streaming=True)
```
This is both correct *and* far more efficient than streaming the merged split and filtering client-side
(no wasted bandwidth/time scanning other languages). **Confirmed**: `validation/hinval.parquet` loaded
directly gives 300/300 rows with `target_lang == "hin_Deva"`, real Hindi query text, matching English
gloss. `ingest/01_subset.py` has been **rewritten** to use `data_files` per language via a
`LANG_FILE_PREFIX` table (2-letter code -> 3-letter file prefix: hi->hin, bn->ben, ta->tam, +11 more
for completeness, only hi/bn/ta actually exercised). A 50-rows/lang smoke test across hi/bn/ta is
running now (background id `bbyouiqss`) — check its output before assuming this is fully verified for
bn/ta too (only hi has been directly confirmed so far).

Aside, not yet acted on: the train split is missing a Telugu parquet file (`train/teltrain.parquet`
absent while `validation/telval.parquet` exists) — irrelevant to hi/bn/ta, noted in case Telugu is
ever added to scope.

## Code written so far (all in this session)
- `ingest/prefixing.py` — shared `add_passage_prefix`/`add_query_prefix` helpers (CLAUDE.md #1).
- `ingest/test_prefixing.py` — **ran, 5/5 pass.** Covers prefix correctness + NFC/nukta normalization
  (built with `chr(codepoint)` fixtures, not literal Devanagari characters — every attempt to type the
  decomposed nukta form as a literal got silently NFC-normalized by the editing pipeline itself before
  reaching the file, which is a live example of the exact bug the test exists to catch; `chr()` avoids
  the problem entirely instead of fighting it).
- `ingest/01_subset.py` — per-language subset via direct parquet `data_files` targeting (see finding
  above). Schema-validates row 0 against expected fields before writing. **hi confirmed working
  end-to-end via direct parquet load (300 rows); hi/bn/ta 50-row smoke test running now, id
  `bbyouiqss`, not yet confirmed for bn/ta.**
- `ingest/02_dedupe.py` — SHA-1 of NFC-normalized passage text, dedupes across all subset JSONLs.
  **Written, not yet run** (needs `01_subset.py` output to exist first).
- `ingest/04_embed.py` — e5-small passage embedding via sentence-transformers (uses the model's own
  baked-in mean-pooling config, not hand-rolled). **Written, not yet run** — needs sentence-transformers
  +torch installed (not done yet, heavy download) and deduped passages to exist first.
- `ingest/05_build_index.py` — HNSW index build via the Python `hnswlib` package (same underlying C++
  lib as the server's `hnswlib-node`, so the on-disk format loads directly, no conversion). Includes a
  hard assertion that embeddings are L2-normalized before building (CLAUDE.md #2). **Written, not yet
  run.** Also documents a design correction: BM25 postings are NOT built here — building them in Python
  and querying in TS would risk the same cross-language-consistency class of bug as e5 prefixes, but
  for tokenization, with no unit test able to catch drift. BM25 build+query both move into
  `server/bm25.ts`, built at server boot from the same passages file — eliminates the risk for free.
  **This changes ARCHITECTURE.md §3's artifact list** (no `bm25.bin`) — not yet reflected in
  CLAUDE.md/PLAN.md, do that next time either file is touched.

## Local toolchain state (Windows, no WSL2)
- Python 3.12.10, venv at `.venv/`. Installed: `datasets`, `pyarrow`, `polars`, `huggingface_hub`,
  `uv`. **Not yet installed**: `sentence-transformers`, `torch`, `optimum[onnxruntime]`, `hnswlib`
  (Python package), `accelerate`, `indic-nlp-library`, `ir-measures` — needed before `04_embed.py`/
  `05_build_index.py` can actually run, not installed yet because they're large/slow downloads and
  weren't needed for what's been verified so far.
- **Console encoding gotcha discovered**: printing Devanagari/Bengali/Tamil text directly to stdout in
  this Windows/Git-Bash environment crashes with `UnicodeEncodeError` (console defaults to cp1252).
  Always set `PYTHONIOENCODING=utf-8` and/or write Unicode output to a file with `io.open(...,
  encoding="utf-8")` instead of printing it — don't rediscover this the hard way again.
- **Downloads are slow enough in this sandbox that most dataset-touching commands exceed the 180s
  foreground timeout and move to background** — expect this, don't shrink scope to avoid it, just use
  background execution and check the notification.

## Next immediate action (pick this up here)
1. sentence-transformers install running in background (id `bw1vofz21`) — check it, then run
   `04_embed.py` against `data/smoke/passages_dedup.jsonl` (1497 rows, already verified good) and
   confirm output shape is (1497, 384) and every row L2-normalizes to ~1.0.
2. `05_build_index.py` **cannot run on this machine** — the Python `hnswlib` package needs a native
   C++ compile and this Windows box has no MSVC Build Tools (`pip install hnswlib` fails with
   "Microsoft Visual C++ 14.0 or greater is required"). Not worth fixing here: the real HNSW build/
   load target is Linux (WSL2 or the Oracle VM), where research already confirmed the build story
   works fine (manylinux wheels / routine `node-gyp` compile). Defer index-build verification to when
   WSL2 exists. If local Windows verification is ever wanted anyway, the fix is installing "Microsoft
   C++ Build Tools" (a multi-GB, slow install) — not recommended given the above.
3. Once embeddings are verified at smoke scale: reflect the BM25-moves-to-TS correction in
   CLAUDE.md/PLAN.md (Epic 1/ARCHITECTURE cross-reference) — still not done as of this update.
4. Decide with the user before attempting the real 25k-queries/lang full-scale run — it's a genuinely
   long CPU-bound job on this machine, don't launch it unprompted.
5. Update this file again once that's done — don't let it go stale.

## Verified so far, small-scale, real runs (not assumed)
- `01_subset.py`: hi/bn/ta, 50 rows each, schema-valid, content spot-checked and coherent (same
  underlying query correctly translated across all three languages).
- `02_dedupe.py`: 1500 instances -> 1497 unique (99.8% kept at this tiny scale — expect much higher
  dedup rate at full 25k/lang scale per ARCHITECTURE.md's ~40-50% estimate; 99.8% here is not
  representative, sample is too small and homogeneous to draw a real ratio from).
- `test_prefixing.py`: 5/5 pass.
- `04_embed.py`: ran on the 1497 deduped passages -> (1497, 384) float32, all norms == 1.0 within
  1e-3, no NaNs. Confirms CLAUDE.md invariant #2 in practice, not just in code review.
- Brute-force retrieval sanity check (plain numpy dot product against all 1497 vectors, no HNSW
  needed at this scale): embeds real Hindi queries with the `query:` prefix, checks whether the
  passage actually marked `is_selected=1` for each query ranks near the top by inner product against
  the `passage:`-prefixed corpus embeddings. **Result: 4/4 queries had their true passage in the top
  3** (of 4 queries checked out of 10 sampled — the other 6 had no `is_selected=1` passage or a hash
  edge case, not investigated further, low priority at this sample size), scores 0.83-0.91. This is
  real evidence the whole embed+prefix+retrieval chain works mechanically, not just that scripts run
  without crashing. **Do not read this as a real recall number** (n=4, one language, a 1497-passage
  corpus with only 50 queries' worth of distractors) — it's a sanity check, not an eval. The actual
  Recall@10 gate needs the full pipeline at real scale, which needs WSL2 (for `hnswlib`) and a much
  larger corpus.

## Sprint 1 status: medium-scale run in progress (2026-08-16, session 5)
User said "continue" without specifying scope. Chose a deliberate middle ground instead of either
extreme: not the full 25k-queries/lang run (estimated 4+ hours on this CPU at the smoke test's
measured ~27 passages/sec, too long to launch unprompted), and not staying at the 50-row smoke sample
(too small to mean anything). Landed on **2,000 queries/lang x hi/bn/ta**:
- `01_subset.py` run for real: 2000 rows/lang, 6000 total, schema-valid, written to `data/medium/`.
- `02_dedupe.py` run for real: 59,961 instances -> 59,666 unique (**99.5% kept — lower duplication
  than ARCHITECTURE.md's ~40-50% estimate**. Read as a real measured data point, not a sign the
  original estimate was wrong at full scale — cross-language passages never collide by hash since
  translations differ per language, and within-language reuse may only show up at much higher query
  counts than 2000. Don't extrapolate this ratio to the full 25k run without re-measuring).
- `04_embed.py` running now in the background (id `b51bwdo3m`), ~35-40 min estimated for ~59.7k
  passages at the measured throughput. **Not complete as of this memory update.**
- `eval/retrieval_eval.py` written (brute-force Recall@{1,5,10}/MRR@10/nDCG@10 against `is_selected`,
  pulled forward from Sprint 3 scope because it's needed to actually judge Sprint 1's gate without
  HNSW). Explicitly only ever checks known positives against top-k, never treats `is_selected==0` as
  a negative (CLAUDE.md #7) -- reads these numbers as a floor on true recall, not exact.
- **Not yet run**: the actual eval, waiting on embeddings to finish.

## Sprint 1 GATE CLEARED (2026-08-16, real measurement, not projected)
`eval/results/strategy_A_medium.json`, brute-force exact search, 59,666-passage medium-scale corpus
(2000 queries/lang x hi/bn/ta), strategy A (passage-as-is) only:
```
n_evaluated: 3657 (of 6000 sampled -- 2343 had no is_selected=1 label at all, a real measurement of
             the MS MARCO sparse-label problem, invariant #7 -- not a bug, don't "fix" this number)
recall@1:  0.2472
recall@5:  0.5649
recall@10: 0.6658   <- PLAN.md Sprint 1 gate is >0.6. CLEARED.
mrr@10:    0.3795   (IndicRAGSuite e5-small baselines: hi 0.44 / bn 0.39 / ta 0.38 -- same ballpark,
                     not strictly comparable: different corpus size/methodology, theirs per-language,
                     this is hi+bn+ta combined)
ndcg@10:   0.4444
```
This is the first time this gate has been evaluated with real numbers instead of being an open
question. Caveats that still apply: brute-force exact search, not HNSW (still blocked on WSL2 --
brute-force is an upper bound on what approximate HNSW could achieve on these same embeddings, so
this doesn't overstate what production retrieval will do, if anything HNSW will be slightly lower);
medium-scale corpus (2000/lang), not the full 25k/lang MVP target; strategy A only, no chunking
variants compared yet (that's Sprint 3).

**Operational note for future long runs on this machine**: the embed job's log shows a ~58-minute
stall mid-run (batch 150->151, timestamps jump from 39:49 to 1:37:34) then resumes at normal speed --
almost certainly Windows sleep/suspend during a long background job. Didn't corrupt anything this
time (job completed fine), but worth disabling sleep (`powercfg /change standby-timeout-ac 0` or via
Settings) before launching the full 25k/lang run, or it could silently extend a multi-hour job by
hours more, or worse, get killed if the user closes the lid rather than just idling.

## Next immediate action (pick this up here)
1. Report the gate-clearing result to the user (done, this session).
2. Decide with the user: run the full 25k/lang corpus now (est. 4+ hrs based on measured throughput,
   scaled from the medium run's actual 104 min for ~60k passages -> roughly proportional to corpus
   size, i.e. full corpus at ~300-450k passages could be 8-12x this run), or move to Sprint 2 (Bun
   server) using the medium-scale artifacts as a stand-in until the real corpus is built, or wait for
   WSL2/Oracle before going further. **Not decided as of this update -- genuinely the user's call**,
   flagged as a real resource/time tradeoff, not something to default silently.
3. Whichever is chosen: if Sprint 2 starts, note Bun is not installed anywhere yet either.

## Decisions/corrections already locked in (full detail in CLAUDE.md, summarized here for speed)
- Hosting: Oracle Always Free Ampere A1 (2 OCPU/12GB as of the 2026-06-15 cut), GCP $300 trial fallback.
- Ingress: Caddy + Let's Encrypt on-VM via `<ip>.sslip.io` — **not** is-a.dev/Cloudflare Tunnel
  (is-a.dev blocks CNAMEs to `*.cfargotunnel.com` by design, confirmed via their repo's validation
  rules).
- STT: Sarvam `saaras:v3-realtime`, `wss://api.sarvam.ai/speech-to-text-realtime/ws`, `linear16`
  encoding, base64 JSON, auth header `API-SUBSCRIPTION-KEY`. No confidence score returned — L0
  guardrail redesigned around empty/gibberish/short-transcript detection instead.
- LLM: Groq (`openai/gpt-oss-20b`/`120b`, `qwen/qwen3.6-27b` — old llama models deprecated 8/16) →
  Cerebras (avoid `zai-glm-4.7`, deprecates 8/17) → extractive-only. Gemini as a possible 3rd fallback
  raised by the user but not yet researched/added — **open decision, not yet resolved**.
- Groundedness model: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, ONNX pre-published, no export needed.
- `hnswlib-node`: always compiles from source (no prebuilds on any platform) — needs build-essential
  +python3 on the target VM. `onnxruntime-node`: ships prebuilt linux-arm64 binaries — confirmed.
- Sarvam free credit is ₹100 (not ₹1,000) — be economical with STT calls during dev.

## Open questions not yet answered by the user
- Add Gemini as a 3rd LLM fallback? (proposed, awaiting go-ahead)
- Status of WSL2 install / Oracle account / provider keys — last asked, no answer yet as of this update.
