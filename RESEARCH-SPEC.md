# Voice-Enabled RAG on MSMARCO-XI — Project Specification for HH Goa 2026 Shortlisting

**Bottom line:** Build a voice→answer RAG pipeline over AI4Bharat's MSMARCO-XI that wins on three defensible differentiators — (1) exploiting the dataset's `is_selected` labels as free relevance ground truth to run a rigorous multi-strategy chunking bake-off, (2) an *honest, instrumented* latency story that reinterprets the impossible "200ms end-to-end" as a bounded **retrieval-core** budget while showing a dual-path (instant extractive + streamed LLM) UX, and (3) a real harness plus layered guardrails that visibly *refuse to answer* when ungrounded or off-topic. The 200ms full-voice target is physically impossible (Sarvam STT alone is <150ms TTFT and Groq LLM TTFT is ~0.22–0.3s), so the entire submission strategy hinges on defining and defending the measurement boundary rather than faking the number.

## TL;DR
- **Use Sarvam Saaras v3 (not ElevenLabs) for STT, multilingual-e5-small (ONNX int8) for embeddings, hnswlib-node + Indic-tokenized BM25 + Reciprocal Rank Fusion for retrieval, and Groq (Cerebras fallback) for the off-critical-path LLM.** Deploy backend always-on on Fly.io Mumbai (`bom`), frontend on Cloudflare Pages.
- **Reinterpret "under 200ms" as the retrieval core** (transcript-in → grounded extractive-answer-out: embed + ANN + BM25 + RRF + span-select), which is genuinely achievable (~15–40ms typical, well under 200ms), and report STT + LLM separately as clearly-labeled figures. Ship a dual-path answer so users see a grounded answer instantly and the LLM synthesis streams in after.
- **Win the "vast chunking" and guardrails asks with rigor, not volume:** MS MARCO passages are already ~50–100 words, so classic chunking is nearly a no-op — say so explicitly, then bake off 6 strategies scored with `is_selected` via Recall@k/MRR/nDCG, and demonstrate calibrated OOD + groundedness refusals live in the demo.

## Key Findings

**The hackathon.** HH Goa 2026 (Hacker House Goa 2026) is a 4-day AI×Crypto builder residency, 28–31 October 2026 at a private beach resort, organized by 2:47PM Studio, selecting 247 builders via a five-stage funnel (Open Trials Aug 2026 → Alpha–Delta Selections Sep 2026 → Partner Trials → RSVP & Stake → Residency). This voice-RAG task is a shortlisting/Open-Trials challenge; per the site: "Speak a question, get a grounded answer. Build a full voice-to-answer RAG pipeline — transcription, engineered chunking, vector retrieval, and generation — wired together end to end, fast and guardrailed." All code must be written during the event window; existing libraries/APIs/frameworks are allowed. There is a photo-frame side task ("#FrameInGoa"). No specific numeric judging rubric is published, so optimize for the six stated technical requirements plus repo/demo craft.

**Dataset (validated directly on the HF dataset card).** `ai4bharat/MSMARCO-XI` = **11,451,314 rows, 55.6 GB Parquet, splits train 10.1M + validation 1.37M**, 14 Indic languages (as, bn, gu, hi, kn, ml, mr, ne, or, pa, sa, ta, te, ur — includes Hindi, Bengali, Tamil). Columns confirmed: `source_lang`, `target_lang`, `meta` (translation-model dict), `query`, `Answer`, `query_id`, `query_type`, `passages{is_selected[], English_passages[], Translated_passages[]}`, `Eng_Query`, `Eng_Answer`. Load per-language: `load_dataset("ai4bharat/MSMARCO-XI","hi",split="validation")`. (Note: the card's usage snippet references `example['answers']`, a typo — the real field is `Answer`. Dataset Viewer currently throws JobManagerCrashedError, so verify a couple of raw rows programmatically.) Provenance: **IndicRAGSuite**, arXiv:2506.01615 (Prasanjith, More, Kunchukuttan, Dabre; AI4Bharat / IIT Madras).

**IndicRAGSuite baselines you should align to.** The paper evaluates dense retrievers on IndicMSMarco (1,000 manually-verified queries) using **MRR**. Reported MRR (Table 2): **BGE-M3 leads in 8/13 languages** — Hindi 0.52, Bengali 0.49, Tamil 0.49, Telugu 0.50, Malayalam 0.49; **multilingual-e5-large** wins 4 (Hindi 0.52, Gujarati 0.48, Urdu 0.49); **e5-small** is materially weaker (Hindi 0.44, Bengali 0.39, Tamil 0.38); LLM2Vec (LLaMA-3.1-8B) competitive (Hindi/Marathi 0.49). Assamese/Odia are the hardest. This gives you exact numbers to benchmark against and cite.

## Details

### 1. Architecture
**Data flow:** Browser mic (`getUserMedia`/MediaRecorder, HTTPS-only) → 16 kHz WAV/PCM frames → **Sarvam Saaras streaming WebSocket** (`saaras:v3-realtime`) for live capture, or REST `/speech-to-text` (`saaras:v3`, `mode=transcribe`) for clips <30 s → transcript + language ID → **local ONNX int8 query embedding** → parallel **{HNSW dense ANN, BM25 lexical}** → **Reciprocal Rank Fusion** → top-k passages → **FAST PATH**: extractive span selection → grounded answer returned instantly (inside the core budget); **SLOW PATH**: guardrail checks → Groq/Cerebras LLM synthesis streamed after. Everything wrapped in a typed harness emitting OpenTelemetry spans to Langfuse.

**Components:** (1) an offline ingestion/indexing job (Python `datasets` or Bun) that subsets, dedupes passages, embeds, and builds HNSW + BM25 artifacts; (2) a **Bun + Hono** API server that holds the index in memory; (3) a Cloudflare Pages **React** frontend; (4) observability via OTel GenAI semantic conventions + Langfuse.

### 2. Technology Decisions

| Component | Chosen | Alternatives considered | Reason |
|---|---|---|---|
| **STT** | **Sarvam Saaras v3** (`saaras:v3`, `mode=transcribe`; streaming `saaras:v3-realtime`) | ElevenLabs Scribe v2 Realtime | Dataset is Indic; per Sarvam's Saaras V3 page, the model is "trained on 1M+ hours of real Indian audio" with a 4-stage pipeline, supports all 22 scheduled Indian languages + English, handles code-mixing, and "processes over 100 million minutes of audio monthly." Fast mode guarantees **<150 ms TTFT**; streaming WS supports true partial transcripts + VAD tuning. The task permits either; Sarvam is the defensible Indic-native pick. |
| **STT alt (coded fallback)** | ElevenLabs Scribe v2 Realtime | — | Per ElevenLabs' "Introducing Scribe v2 Realtime," it "delivers state-of-the-art accuracy in over 90 languages with an ultra-low 150 ms of latency" and "achieves 93.5% accuracy across 30 commonly used European and Asian languages" (independent Coval notes p95 ~250 ms, p99 ~400 ms). Weaker on low-resource Indic than Sarvam; keep as a redundancy path only. |
| **Embedding (primary)** | intfloat/multilingual-e5-small (384-d, ONNX int8) | bge-m3, e5-base/large, jina-v3, EmbeddingGemma-300m, Granite R2, potion static | Small enough for ~sub-15 ms CPU inference; 384-d keeps the index in RAM; IndicRAGSuite lists it as a viable baseline (though the weakest of its tier — plan to show headroom vs bge-m3). |
| **Embedding (quality ceiling)** | BAAI/bge-m3 (offline eval + optional async rerank family) | — | IndicRAGSuite: best/near-best MRR in 8/13 Indic langs; use it to quantify how much recall you leave on the table by choosing speed. |
| **Static fast-path candidate** | potion-multilingual-128M (Model2Vec, 256-d) — **BENCHMARK ONLY, do not ship blind** | static-similarity-mrl-multilingual-v1 | Orders of magnitude faster, but retrieval is the weakest task type for static embeddings. MMTEB "Ret" for potion-multilingual = **37.86** (aggregate, biased to high-resource langs, **no Indic validation**); static-similarity-mrl scores higher on retrieval (41.21) but covers only 50 languages. On English MTEB, model2vec's own numbers show static `potion-base-32M` reaches only ~76% of a *weak* transformer (`all-MiniLM-L6-v2`) on Retrieval (32.67 vs 42.92) while *beating* it on Classification (71.70 vs 69.25) — i.e. retrieval is exactly where static loses most. Treat as an emergency latency lever, validated on MSMARCO-XI first. |
| **Vector index** | hnswlib-node (in-process) | usearch, Voyager (Spotify), LanceDB, Qdrant embedded, FAISS-node, cuVS (GPU) | Mature Node binding, no network hop; HNSW at 1M×768-d shows ~3 ms P50 in published benchmarks (you're at 384-d, so faster). Keep M/efConstruction sane and tune efSearch. |
| **Lexical** | In-memory BM25 with **Indic-aware normalization + tokenization** (indic_nlp_library / Unicode NFC) | Whitespace BM25, Lucene analyzers | Whitespace BM25 on Devanagari/Bengali/Tamil matches poorly; script-aware tokenization is mandatory. Cross-lingual BM25 (English query ↔ Hindi corpus) barely works — lean on dense there. |
| **Fusion** | Reciprocal Rank Fusion | Weighted sum, learned fusion | Robust, parameter-light, standard; Anthropic's own stack combines embeddings + BM25 via rank fusion. |
| **Reranker** | bge-reranker-v2-m3 (async / slow-path only) | Cohere rerank, jina-reranker-v3 | Only if it earns its latency; keep off the 200 ms core path. |
| **LLM (synthesis)** | Groq (Llama 3.x/4) primary, Cerebras fallback | OpenAI, Gemini Flash | Lowest TTFT among providers — Groq's own Llama-2-70B benchmark hit **0.22 s TTFT**; general figures ~0.22–0.3 s. Both open-weight, OpenAI-compatible → trivial A/B and fallback. Off the core budget. |
| **Harness** | BAML (typed LLM I/O, schema-aligned parsing, retries, model-swap) + Zod validation + custom TS orchestration | LangGraph, DSPy, Instructor, Pydantic AI, XGrammar | TS-native, low overhead; BAML's benchmarks report ~98% valid type-correct objects and lean schemas that cut prompt tokens. Instructor/Pydantic AI are Python-first; LangGraph adds weight. |
| **Guardrails** | Layered: embedding OOD threshold (sync) + input safety (sync) + NLI/HHEM groundedness (async) | NeMo Guardrails, Guardrails AI, Llama Guard, RAGAS-online | Lightweight checks stay sub-budget; groundedness runs async. Fast NLI/HHEM models run ~50–200 ms at 85–90% accuracy; LLM-as-judge (RAGAS faithfulness) is 2–5 s → offline only. |
| **Deployment** | Fly.io Mumbai (`bom`), **always-on** machine (no scale-to-zero) | Railway (SG), Render, Hetzner, DO Bangalore, AWS Mumbai, Cloudflare Containers | India-region sub-20 ms to users; Fly Machines cold-start ~150 ms but scale-to-zero adds seconds → keep always-on for a reliable "live link." Shared-CPU VMs start ~$2/mo; right-size RAM for the index. |
| **Index artifact** | Baked into Docker image (fallback: Cloudflare R2 pull on boot) | Git LFS, HF Hub | Baked = fastest boot, no runtime fetch; use R2 if the image gets too large. |

### 3. Latency Strategy (the crux of the submission)
**Interpretation.** The full voice→answer path cannot hit 200 ms: Sarvam Fast mode is **<150 ms TTFT for STT alone**, and Groq **LLM TTFT is ~0.22–0.3 s**. Define the measurement boundary as the **retrieval core** — from *transcript-in* to *grounded extractive-answer-out* — explicitly **excluding STT and LLM generation**. Report that against the 200 ms target, and separately report end-to-end wall-clock (including STT + LLM) as a second, clearly-labeled figure. Serve a **dual-path answer**: the FAST path (no LLM) returns a grounded extractive answer instantly inside the core budget; the SLOW path streams the LLM-synthesized answer after guardrails.

**Stage-by-stage budget (core target < 200 ms; MEASURED vs ESTIMATED to be labeled in the repo):**

| Stage | Est. latency | Basis |
|---|---|---|
| Query embedding (e5-small int8, CPU, short query) | ~5–15 ms | ONNX int8 small-transformer CPU inference figures |
| HNSW ANN search @ ~500k–1M vectors (384-d) | ~1–5 ms P50 | HNSW ~3 ms P50 @ 1M (768-d); faster at 384-d |
| BM25 in-memory (subset corpus) | ~2–10 ms | in-memory JS BM25 |
| RRF fusion + dedupe | <1 ms | trivial |
| Extractive span selection | ~1–5 ms | string ops / small scoring |
| **Core total (fast path)** | **~15–40 ms typical; budget < 200 ms** | sum |
| STT (Saaras Fast mode TTFT) | <150 ms | Sarvam STT page |
| LLM synthesis TTFT (Groq) | ~0.22–0.3 s | Groq LPU benchmarks |
| Reranker (if used, async) | +tens of ms | off core path |

**Analytics methodology.** Report **P50/P70/P100** for the core across a reasonable number of queries (≥ several hundred; ideally ≥5,000 for a stable tail), after warmup, using an **open-loop** harness (constant arrival rate) to avoid **coordinated omission** (Gil Tene), recording into an **HDR histogram**. Explain Bun/Node tail causes (GC pauses, JIT warmup). This methodology maturity is itself a scoring signal.

### 4. Chunking Strategy Design
**Critical, honest insight:** MS MARCO passages are *already short* (~50–100 words), so classic document chunking is largely a no-op — state this explicitly and reframe "chunking" as an *index-granularity/enrichment* problem. The 2024–2026 evidence supports restraint: a NAACL 2025 Findings paper concluded semantic chunking's compute isn't justified, with **fixed ~200-word chunks matching or beating semantic chunking** (corroborated by Qu et al. 2025); recursive 512-token is the benchmark-validated default; semantic chunking gets the highest *retrieval recall* (~91.9% in Chroma's tests) but *lower end-to-end accuracy* (~54% vs ~69% for recursive in the Vecta/FloTorch benchmark) and is ~14× slower. **Anthropic's Contextual Retrieval** (Sept 2024) is the high-value enrichment: per Anthropic, Contextual Embeddings alone reduced top-20-chunk retrieval failure rate by **35% (5.7% → 3.7%)**, Contextual Embeddings + Contextual BM25 by **49% (5.7% → 2.9%)**, and **Reranked Contextual Embedding + Contextual BM25 by 67% (5.7% → 1.9%)**. Jina's late chunking helps *long* docs → irrelevant here.

**Six strategies to bake off** (scored on `is_selected` via Recall@{1,5,10}, MRR@10, nDCG@10):
- **(A) Passage-as-is** — baseline; likely strong given short passages.
- **(B) Fixed 256/64-overlap re-chunk** — deliberately show it *barely moves the needle* (a rigorous negative result is a differentiator).
- **(C) Semantic percentile splitting** — quantify the recall-vs-latency/complexity tradeoff.
- **(D) Sentence-window** — retrieve sentence, expand to neighbors.
- **(E) Small-to-big / parent-document** — retrieve passage, return grouped context.
- **(F) Contextual retrieval** — Anthropic-style: prepend `query_type` + Eng_Query-derived context before embedding/indexing.
- **(G) Cross-lingual dual index** — index both `English_passages` and `Translated_passages`; exploit the parallel data.

Also compare **hybrid (dense+BM25+RRF) vs dense-only vs BM25-only**, and **with/without bge-reranker-v2-m3**. Expected reportable finding: for short MS MARCO passages, *passage-as-is + hybrid + contextual enrichment* wins, and naive re-chunking hurts.

### 5. Harness Design
A typed TS pipeline where each stage (transcribe, embed, retrieve, fuse, guard, synthesize) is a function with **Zod-typed I/O, timeouts, exponential-backoff retries, and circuit breakers**. LLM calls go through **BAML** for schema-aligned parsing and typed outputs with a **fallback chain (Groq → Cerebras → extractive-only)**. Every stage emits OpenTelemetry spans (GenAI conventions) → Langfuse. Error recovery: STT failure → prompt re-record; empty retrieval → refuse; LLM failure → serve fast-path extractive answer. Structured JSON I/O throughout — never raw prompt-in/text-out (this is the literal "harness" rubric requirement).

### 6. Guardrails Design (layered; cheap checks sync, expensive async)
- **Layer 0 (input):** language ID + basic safety/inappropriate-input classifier; reject empty/garbage transcripts.
- **Layer 1 (OOD / off-topic):** embedding-based out-of-domain detection — compare the query embedding to the corpus centroid and/or the max top-k retrieval score; below a **calibrated threshold**, refuse ("I can't answer that from this dataset"). Calibrate on held-out in-domain vs out-of-domain queries, picking the threshold at a target false-positive rate from the score distribution.
- **Layer 2 (groundedness/hallucination):** an NLI/HHEM-style entailment check that the answer is supported by retrieved context, run on the **slow path** before showing the synthesized answer; on failure, fall back to the extractive answer or refuse. Reserve RAGAS faithfulness / LLM-as-judge (2–5 s) for offline eval.
- **Demo the refusals** — the rubric explicitly wants a system that "knows when *not* to answer."

### 7. Evaluation Plan
**Retrieval:** Recall@{1,5,10}, MRR@10, nDCG@10 using `is_selected`, computed with ir_measures/TREC-eval conventions. **Generation:** RAGAS faithfulness, answer relevancy, context precision/recall, using gold `Answer`/`Eng_Answer` as reference. **Splits:** build the index on a subset (hi/bn/ta + English, ~60–100k queries, deduped to ~400k–800k unique passages); evaluate on held-out validation queries. **Produce five tables:** (1) chunking bake-off (6 strategies × metrics), (2) hybrid vs dense vs BM25, (3) rerank on/off, (4) latency percentiles, (5) guardrail confusion matrix.

**Mandatory caveat to state:** MS MARCO is *sparsely labeled* — per Arabzadeh et al. (2022), "94% of the nearly seven thousand queries in the MS MARCO passage ranking development set have only a single known relevant passage, and no query has more than four," and `is_selected=0` is **not a true negative** (false negatives abound; the dataset's own README warns negatives "may not be a true negative"). So absolute Recall/MRR *underestimate* quality — use them for *relative* comparison between strategies and align to IndicRAGSuite's published MRR (e5-small hi 0.44 / bn 0.39 / ta 0.38; bge-m3 hi 0.52 / bn 0.49 / ta 0.49).

## Recommendations

**Staged plan with GO/NO-GO gates:**

- **Day 0 — Setup.** In WSL2 Ubuntu: install Bun (`curl -fsSL https://bun.sh/install | bash`), Node LTS, Python 3.11 + `uv`, `pip install datasets huggingface_hub`. Get Sarvam key (free credits on signup — note ₹100 default per the changelog, verify current amount on the dashboard), Groq key, Cloudflare + Fly.io accounts (`flyctl`). Scaffold repo (Hono+Bun API, Vite React web). **GO/NO-GO:** `bun run dev` serves and the Sarvam key transcribes a test WAV.
- **Day 1 — Data + index.** Stream hi/bn/ta + English subsets via `load_dataset("ai4bharat/MSMARCO-XI","hi",split="validation")`; dedupe passages to ~400–800k. Export ONNX int8 e5-small (mind the `query:`/`passage:` prefixes); embed all passages; build hnswlib + Indic-tokenized BM25; persist artifacts. **GO/NO-GO:** a sample query returns sane top-k and Recall@10 beats a random baseline.
- **Day 2 — Retrieval core + latency.** Wire embed→HNSW→BM25→RRF→extractive in Bun; instrument spans; build the open-loop HDR-histogram latency harness; produce P50/P70/P100. **GO/NO-GO:** core P50 < 200 ms.
- **Day 3 — Chunking bake-off + eval.** Implement the 6 strategies; compute Recall@k/MRR/nDCG with `is_selected`; add hybrid/dense/BM25 and rerank on/off tables. **GO/NO-GO:** at least one clear, defensible winning configuration.
- **Day 4 — Harness + guardrails + LLM + frontend + deploy.** BAML LLM synthesis via Groq with Cerebras fallback; OOD threshold + async groundedness NLI; React mic UI (record → stream → dual-path display with grounding highlights and refusal states); deploy backend to Fly.io `bom` (always-on) + frontend to Cloudflare Pages. **GO/NO-GO:** end-to-end voice demo works on the live link.
- **Day 5 — Polish + videos + submit.** RAGAS generation metrics; finalize README with all tables; record Video 1 (process, 90 s) and Video 2 (demo); final test of live link + repo; submit form. **NO-GO to submit** if any of {live link, repo, both videos} is missing — **there are no resubmissions.**

**Thresholds that change the plan:** if core P50 > 200 ms → drop to 384-d int8, lower efSearch, shrink the corpus subset, then (last resort) benchmark potion-multilingual static fast-path on MSMARCO-XI before shipping it. If Fly.io OOMs → int8 vectors + right-size RAM, or serve the index from R2/LanceDB on disk. If Indic recall is poor → verify e5 prefixes/normalization first, then add contextual enrichment (strategy F), then bge-reranker on the slow path.

**Repo + README.** Structure: `/apps/api` (Bun+Hono, harness, guardrails), `/apps/web` (Vite React → CF Pages), `/packages/rag` (embedding, hnsw, bm25, rrf, chunking), `/scripts` (ingest, dedupe, build-index, eval, latency-bench), `/baml_src`, `/artifacts`, `/eval`, plus `README.md`, `Dockerfile`, `fly.toml`. README must contain: one-paragraph pitch; architecture diagram; the **honest latency-boundary definition + P50/P70/P100 table**; the chunking bake-off tables; guardrail description + refusal examples; how-to-run; deployment notes; the **MS MARCO sparse-label caveat**; STT-choice justification; and links to the live demo + both videos.

**Video plan.** *Video 1 (90 s, process — solo, reframed as a build log, not a fake team):* terminal build log, the latency histogram generating, the chunking bake-off table filling in, a git commit graph, an architecture sketch; narrate the *decisions* (why Sarvam, why bound the 200 ms honestly, why `is_selected` as ground truth). *Video 2 (demo, end-to-end):* speak a Hindi question → live transcript → instant grounded extractive answer (core-latency badge on screen) → streamed LLM answer with grounding highlights → then a deliberately out-of-domain question → **system refuses** ("not in dataset") → show a groundedness-failure fallback. Put the P50/P70/P100 numbers on screen.

## WHAT TO AVOID (prioritized)

**Technical:**
1. **Embedding mismatch** — index and query MUST use the same model, the same e5 `query:`/`passage:` prefixes, and the same L2 normalization. Missing prefixes silently tank recall (the single most common RAG bug).
2. **Static-embedding trap** — potion-multilingual-128M is tempting for sub-10 ms embedding, but retrieval is its weakest task type (static models retain far less of transformer retrieval quality than of classification quality — model2vec's own English numbers show ~76% of a weak MiniLM on Retrieval vs *beating* it on Classification), and its MMTEB retrieval (37.86) is an aggregate biased to high-resource languages with **zero Indic validation**. Never ship it as primary without benchmarking on MSMARCO-XI.
3. **BM25 on raw Indic text** — whitespace tokenization on Devanagari/Bengali/Tamil matches poorly; apply Unicode NFC normalization + nukta handling + script-aware tokenization. Cross-lingual BM25 (English query vs Hindi corpus) barely works — rely on dense there.
4. **HNSW parameter mistakes** — too-low efSearch/M kills recall; too-high efConstruction blows build time. Tune efSearch for recall@target-latency.
5. **Memory blowup** — 1M×1024-d fp32 ≈ 4 GB; use 384-d + int8 to fit a modest Fly.io VM.
6. **Scale-to-zero cold starts** — keep the Fly machine always-on; a cold start means judges hit a dead demo.
7. **Audio format** — Sarvam streaming WS accepts only WAV/PCM (`pcm_s16le`/`l16`/`raw` at 16 kHz); wrong sample rate/codec = garbage transcripts. Handle mic permission, the HTTPS requirement for `getUserMedia`, and API CORS.
8. **WebSocket** — reconnect with exponential backoff; do NOT auto-retry on 4xxx auth/quota closes; idle connections close (~1 min).
9. **Parquet ingestion** — 55.6 GB won't fit in memory; stream per-language configs, dedupe passages, never load the whole thing.
10. **Coordinated omission** — don't report one warm best-case run; open-loop, many samples, HDR histogram, P50/P70/P100.

**Submission-strategic:**
11. **No resubmissions** — full dry-run of live link, repo, and both videos before submitting.
12. **Don't claim end-to-end 200 ms** — you can't, and technical judges will tear it apart; the bounded, honest claim is stronger.
13. **Solo "team video"** — reframe as a process/build-log video.
14. **Don't overbuild** — a working guardrailed hybrid pipeline with honest metrics beats a half-broken kitchen-sink.
15. **Don't skip the refusal demo** — "knows when not to answer" is explicitly in the ask.

## Risk Register

| Risk | Mitigation / fallback |
|---|---|
| Core P50 exceeds 200 ms | 384-d int8, lower efSearch, shrink subset, precompute; static fast-path as last resort (benchmarked first). |
| Sarvam credits/quota exhausted mid-demo | Cache transcripts; coded ElevenLabs Scribe fallback; pre-record demo audio. |
| Fly.io machine OOM from index | Right-size RAM, int8 vectors, or serve index from R2/LanceDB on disk. |
| Cold start kills live link | Always-on machine + health-check pings. |
| Indic BM25 poor matching | Indic normalization + tokenization; lean on dense for cross-lingual. |
| LLM TTFT too slow / provider down | Dual-path already shows extractive answer instantly; Groq→Cerebras fallback. |
| Sparse labels make metrics look weak | State the caveat; use metrics for relative comparison; align to IndicRAGSuite baselines. |
| Guardrail false-refusals | Calibrate OOD threshold on held-out set at a target FPR. |
| Sarvam pricing ambiguity | Sources conflict (₹30/hour vs ₹1.5/min ≈ ₹90/hour, plus a ₹100-vs-₹1,000 free-credit change) — verify current pricing/credits on the Sarvam dashboard before relying on them. |
| No resubmission | Full dry-run of link + repo + videos before submit. |

## Caveats
- **No published numeric judging rubric** for HH Goa 2026 was found; guidance here optimizes for the six stated technical requirements plus repo/demo craft. Confirm details against the official HH Goa timeline and the challenge brief before submitting.
- **Vendor/self-reported latency:** Sarvam's <150 ms, Groq's ~0.22 s TTFT, and ElevenLabs' 150 ms are provider claims; independent measurement (Coval) shows Scribe p95 ~250 ms / p99 ~400 ms. Measure your own pipeline. ElevenLabs Scribe v2 Realtime launch dates conflict across sources (Nov 2025 vs Jan 2026) — immaterial to the recommendation.
- **Some embedding latency figures are ESTIMATED**, extrapolated from ONNX int8 CPU benchmarks and HNSW papers rather than measured on your exact WSL2/Fly hardware; label them as such and replace with measured numbers before submission.
- **Chunking end-to-end accuracy numbers** (e.g., recursive 69% vs semantic 54%) come from vendor/third-party benchmarks (Vecta/FloTorch/Chroma) on non-MS-MARCO corpora and won't transfer directly to already-short MS MARCO passages — that mismatch is itself part of your thesis; validate on MSMARCO-XI.
- **potion-multilingual retrieval quality for Indic specifically is unvalidated** in any public source; the one figure I could not independently pin to a precise citation is the exact "% of transformer retrieval retained" for the multilingual variant (the ~76% figure is from model2vec's English tables). Benchmark before use.
- The **Dataset Viewer is currently broken**; row/size/split/column/language values were confirmed from the dataset-card metadata, not by inspecting raw rows — spot-check programmatically after loading.