# Voice-Enabled RAG — System Architecture & Build Spec

**Project codename:** `dhwani` (ध्वनि — "sound")
**Target:** HH Goa 2026 shortlisting Task 2
**Dataset:** `ai4bharat/MSMARCO-XI`
**Builder:** solo, WSL2 + Bun/TS

---

## 0. First, the thing nobody tells you

There is **no model training** in a standard RAG build. When the brief says "build a RAG model," what you actually build is four separate things that people lazily call "training":

| What people call "training" | What it actually is | Where it happens |
|---|---|---|
| "Training the RAG" | Building the vector index (embed + HNSW graph) | Offline, once |
| "Tuning the RAG" | Grid-searching retrieval hyperparameters on a dev set | Offline, ~1 hour |
| "Training the guardrails" | Calibrating scalar thresholds on labelled positives/negatives | Offline, ~20 min |
| **Actual training** | **Contrastive fine-tuning of the embedding model on `is_selected` pairs** | Optional, 1 GPU-hour |

That last one is real gradient descent, and it's your biggest differentiator — this dataset hands you 10M labelled (query, relevant-passage) pairs, which is exactly what a retriever fine-tune needs. Section 6 covers it. Almost nobody in a hackathon does this. Do it and say so loudly.

---

## 1. System overview

```
┌──────────────────────────── BROWSER (Cloudflare Pages) ────────────────────────────┐
│  AudioWorklet → Int16 PCM @16kHz ──WS──┐                                            │
│  React UI: waveform · transcript · instant answer · streamed answer · trace waterfall│
└────────────────────────────────────────┼───────────────────────────────────────────┘
                                         │ wss://api.../v1/stream
┌────────────────────────── BUN + HONO SERVER (Fly.io bom) ──────────────────────────┐
│                                        ▼                                            │
│  ┌──────────────┐   PCM frames    ┌──────────────────┐   partial+final transcript    │
│  │ WS ingress   │ ──────────────► │ Sarvam Saaras WS │ ─────────────┐                │
│  └──────────────┘                 │ (key stays here) │              │                │
│                                   └──────────────────┘              ▼                │
│                                                        ╔═══════════════════════════╗ │
│                                            t₀ ────────►║   RETRIEVAL CORE          ║ │
│                                                        ║  ① input guard    <1ms    ║ │
│  IN MEMORY (loaded at boot):                           ║  ② embed ONNX     6-12ms  ║ │
│   • ONNX e5-small int8       (~35 MB)                  ║  ③ OOD gate       <0.5ms  ║ │
│   • HNSW graph, 400k × 384d  (~700 MB)                 ║  ④ HNSW ANN       2-5ms   ║ │
│   • BM25 postings (Indic)    (~180 MB)                 ║  ⑤ BM25           3-8ms   ║ │
│   • passage store (mmap)     (~250 MB)                 ║  ⑥ RRF + MMR      <1ms    ║ │
│   • calibrated thresholds    (~2 KB)                   ║  ⑦ parent expand  1-2ms   ║ │
│                                                        ║  ⑧ extractive span 5-15ms ║ │
│                                                        ╚═══════════╤═══════════════╝ │
│                                          t₁ ◄──────────────────────┘  FAST ANSWER    │
│                                          (t₁ - t₀ < 200ms — the claim)               │
│                                                        ┌───────────▼───────────────┐ │
│                                                        │ SLOW PATH (async)         │ │
│                                                        │  Groq → Cerebras → extract│ │
│                                                        │  + groundedness NLI check │ │
│                                                        └───────────┬───────────────┘ │
│                                          t₂ ◄──────────────────────┘  RICH ANSWER    │
│  Harness wraps everything: Zod contracts · retries · circuit breakers · span trace   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**The core idea:** `t₀` is transcript-in, `t₁` is grounded-answer-out. That window is what you optimise and report against 200 ms. STT (before `t₀`) and LLM synthesis (after `t₁`) are measured and reported separately with an explicit boundary definition in the README. The user sees a real, grounded, cited answer at `t₁` — it is not a placeholder or a loading spinner.

---

## 2. Stack

### 2.1 Runtime

| Layer | Choice | Exact package | Why |
|---|---|---|---|
| Server runtime | Bun 1.1+ | — | Native WS, fast startup, TS with no build step |
| HTTP/WS framework | Hono | `hono` | ~3 µs routing overhead, first-class Bun WS |
| Embeddings | ONNX Runtime | `onnxruntime-node` (**not** transformers.js) | Direct session control, no tokenizer re-init per call |
| Tokenizer | HF tokenizers | `@huggingface/jinja` + `tokenizers` (native) | XLM-R SentencePiece; JS tokenizers mishandle Indic conjuncts |
| ANN index | HNSW | `hnswlib-node` | Mature, in-process, no network hop |
| ANN (memory-tight alt) | usearch | `usearch` | Supports `i8`/`f16` quantization → 4× memory cut |
| Lexical | custom BM25 | — (write it, ~150 LOC) | You need control of the Indic tokenizer; no JS lib does this right |
| Validation | Zod | `zod` | Stage contracts |
| LLM | Groq → Cerebras | `groq-sdk`, OpenAI-compatible fetch | Lowest TTFT |
| Tracing | custom spans | — | OTel is overkill; you need ~40 LOC and a JSON array |
| Bench | HDR histogram | `hdr-histogram-js` | Correct percentiles, no coordinated omission |

### 2.2 Offline (Python, WSL2)

```
uv venv && source .venv/bin/activate
uv pip install datasets polars sentence-transformers optimum[onnxruntime] \
               onnxruntime accelerate indic-nlp-library ir-measures pyarrow
```

### 2.3 Frontend

Vite + React + TypeScript. `AudioWorklet` for PCM capture (**not** `MediaRecorder` — that gives you WebM/Opus, and Sarvam's streaming endpoint wants raw 16-bit PCM at 16 kHz). Deploy to Cloudflare Pages.

---

## 3. Repository layout

```
dhwani/
├── ingest/                        # Python — offline, run once
│   ├── 01_subset.py               # stream parquet → jsonl subset
│   ├── 02_dedupe.py               # passage dedupe by hash
│   ├── 03_chunk.py                # emit 7 chunk variants
│   ├── 04_embed.py                # GPU batch embed → .npy
│   ├── 05_build_index.py          # HNSW graph + BM25 postings → artifacts/
│   ├── 06_finetune.py             # OPTIONAL: contrastive fine-tune
│   └── 07_calibrate.py            # guardrail thresholds → thresholds.json
├── eval/
│   ├── retrieval_eval.py          # Recall@k / MRR@10 / nDCG@10 per strategy
│   ├── guardrail_eval.py          # OOD ROC, hallucination P/R
│   └── results/                   # committed tables + plots
├── server/                        # Bun + Hono
│   ├── index.ts                   # entrypoint, boot loader
│   ├── harness/
│   │   ├── pipeline.ts            # stage runner + tracing
│   │   ├── contracts.ts           # Zod schemas
│   │   ├── breaker.ts             # circuit breaker
│   │   └── retry.ts               # backoff + jitter
│   ├── core/
│   │   ├── embed.ts               # ONNX session
│   │   ├── hnsw.ts
│   │   ├── bm25.ts
│   │   ├── fuse.ts                # RRF + MMR
│   │   └── extract.ts             # span selection
│   ├── guards/
│   │   ├── input.ts
│   │   ├── ood.ts
│   │   └── grounded.ts
│   ├── stt/sarvam.ts              # WS proxy
│   └── llm/                       # groq.ts, cerebras.ts, chain.ts
├── web/                           # Vite + React
│   ├── src/worklet/pcm.ts
│   ├── src/components/{Mic,Answer,Waterfall,Citations}.tsx
├── bench/
│   ├── latency.ts                 # open-loop harness
│   └── results/LATENCY.md
├── artifacts/                     # gitignored; built or pulled from R2
│   ├── model.onnx  tokenizer.json
│   ├── hnsw.bin  bm25.bin  passages.bin  meta.json  thresholds.json
├── Dockerfile  fly.toml  README.md
```

---

## 4. Data pipeline (the "training" of the RAG, part 1)

### 4.1 Subset — do not touch all 55.6 GB

```python
# ingest/01_subset.py
from datasets import load_dataset

LANGS = ["hi", "bn", "ta"]
N_PER_LANG = 25_000          # queries, not rows of passages

for lang in LANGS:
    ds = load_dataset("ai4bharat/MSMARCO-XI", lang,
                      split="validation", streaming=True)   # stream! never load()
    out = []
    for i, ex in enumerate(ds):
        if i >= N_PER_LANG: break
        out.append({
            "qid": ex["query_id"], "lang": lang,
            "query": ex["query"], "eng_query": ex["Eng_Query"],
            "answer": ex["Answer"], "eng_answer": ex["Eng_Answer"],
            "qtype": ex["query_type"],
            "passages": ex["passages"]["Translated_passages"],
            "eng_passages": ex["passages"]["English_passages"],
            "is_selected": ex["passages"]["is_selected"],
        })
    write_jsonl(f"data/{lang}.jsonl", out)
```

**Sanity-check the first row before you build anything.** The dataset card's usage snippet says `example['answers']`; the actual field is `Answer`. The HF Dataset Viewer is currently broken, so the card is unverified against the real files. Print `ex.keys()` on row 0 and confirm.

75k queries × 10 passages = 750k passage instances → **dedupe by SHA-1 of normalised text** → expect ~300–450k unique passages. MS MARCO reuses passages heavily across queries; skipping dedupe roughly doubles your index for zero recall gain.

### 4.2 Memory maths — decide corpus size from RAM, not vibes

`hnswlib-node` stores vectors as **fp32**. There is no int8 option. So:

```
bytes ≈ N × (dim × 4  +  M × 2 × 4  +  ~40 overhead)
400k × (384×4 + 16×2×4 + 40) = 400k × 1,704 ≈ 682 MB
```

Plus passage store (~250 MB), BM25 postings (~180 MB), ONNX model (~35 MB), Bun heap (~150 MB) ≈ **1.3 GB**. Provision a **2 GB Fly machine**; 4 GB if you can spare the cost.

If you want more passages, switch to `usearch` with `quantization: 'i8'` → 384 bytes/vector instead of 1,536, and 1M passages fits in ~500 MB with roughly 1–2% recall loss. Benchmark before switching.

### 4.3 Chunk variants

Generate seven parallel corpora from the same deduped passage pool. Each writes `chunks_{variant}.jsonl` with `{chunk_id, passage_id, text, lang, qtype_hint, parent_span}`.

| ID | Variant | Implementation note |
|---|---|---|
| A | passage-as-is | Baseline. Expect this to win or tie — MS MARCO passages are already ~60–100 words. |
| B | fixed 256/64 | Deliberate strawman. Publish the negative result. |
| C | semantic percentile | Sentence-embed, split where adjacent cosine < 5th percentile. |
| D | sentence-window | Index sentences; return ±2 neighbours at retrieval time. |
| E | small-to-big | Index sentences (D), return the whole parent passage. |
| F | contextual | Prepend an LLM-written one-line situating summary before embedding. Run on a 50k subset only — this costs real money and time. |
| G | cross-lingual dual | Index both `Translated_passages` and `English_passages` in one space; RRF across both. |

**Indic sentence splitting:** do not use `.split('.')`. Devanagari and Bengali use `।` (U+0964, danda) and `॥`. Tamil uses `.` but with different spacing conventions. Use `indicnlp.tokenize.sentence_tokenize`.

### 4.4 Embedding

```python
# ingest/04_embed.py
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("intfloat/multilingual-e5-small")

# NON-NEGOTIABLE: e5 requires asymmetric prefixes
emb = m.encode([f"passage: {t}" for t in texts],
               batch_size=256, normalize_embeddings=True,
               convert_to_numpy=True).astype("float32")
```

At query time you must use `query: {text}`. **Mismatched or missing prefixes is the single most common RAG bug and silently costs you 5–15 points of recall.** Write a unit test that asserts the prefix is applied on both sides.

Normalise to unit length and use **inner-product** space in HNSW — cosine on pre-normalised vectors is identical and IP is faster.

### 4.5 ONNX export for the server

```bash
optimum-cli export onnx \
  --model intfloat/multilingual-e5-small \
  --task feature-extraction \
  --optimize O2 artifacts/onnx/

python -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('artifacts/onnx/model.onnx',
                 'artifacts/onnx/model_int8.onnx',
                 weight_type=QuantType.QInt8)"
```

**Critical validation step:** embed 1,000 queries with both the PyTorch model and the int8 ONNX model, and assert mean cosine similarity > 0.995. Quantization occasionally destroys a model. If it does, fall back to fp32 ONNX (~2× slower, still ~15 ms — survivable).

---

## 5. Retrieval core (the 200 ms path)

### 5.1 Stage code sketch

```ts
// server/core/embed.ts
import * as ort from "onnxruntime-node";

let session: ort.InferenceSession;
export async function boot() {
  session = await ort.InferenceSession.create("artifacts/onnx/model_int8.onnx", {
    executionProviders: ["cpu"],
    intraOpNumThreads: 2,          // >2 hurts for batch-size-1 short seqs
    interOpNumThreads: 1,
    graphOptimizationLevel: "all",
  });
  // WARMUP — first inference is 10× slower (allocator + JIT)
  for (let i = 0; i < 20; i++) await embed("query: warmup");
}

const outBuf = new Float32Array(384);   // preallocate, never allocate in hot path

export async function embed(text: string): Promise<Float32Array> {
  const { ids, mask } = tokenize(`query: ${text}`, 64);   // truncate hard at 64
  const r = await session.run({
    input_ids:      new ort.Tensor("int64", ids,  [1, ids.length]),
    attention_mask: new ort.Tensor("int64", mask, [1, mask.length]),
  });
  return l2norm(meanPool(r.last_hidden_state, mask, outBuf));
}
```

```ts
// server/core/fuse.ts — Reciprocal Rank Fusion
const K = 60;
export function rrf(...lists: string[][]): Map<string, number> {
  const s = new Map<string, number>();
  for (const list of lists)
    list.forEach((id, i) => s.set(id, (s.get(id) ?? 0) + 1 / (K + i + 1)));
  return s;
}
```

```ts
// server/core/hnsw.ts
import { HierarchicalNSW } from "hnswlib-node";
export const index = new HierarchicalNSW("ip", 384);
index.readIndexSync("artifacts/hnsw.bin");
index.setEf(64);                 // tune this — see §5.3
```

### 5.2 BM25 for Indic scripts

Whitespace tokenisation on Devanagari/Bengali/Tamil is nearly useless. Minimum viable Indic tokenizer:

1. **Unicode NFC normalise** — Devanagari nukta characters have two encodings (`क़` as U+0958 vs U+0915+U+093C). Without NFC these never match.
2. **Grapheme-cluster segmentation** via `Intl.Segmenter('hi', {granularity:'word'})` — built into Bun, handles conjuncts correctly.
3. Strip ZWJ/ZWNJ (U+200D/U+200C) — inconsistently present in translated text.
4. **No stemming.** Indic stemmers are unreliable; light suffix stripping for Hindi (`ों`, `ाओं`, `ियों`) is the most you should attempt, and measure whether it helps before keeping it.
5. BM25 params: `k1=0.9, b=0.4` (MS MARCO-tuned defaults, better than the classic 1.2/0.75 for short passages).

**Do not expect cross-lingual BM25 to work.** A Hindi query will not lexically match an English passage. BM25 covers same-language exact matches (numbers, names, entities); dense covers cross-lingual. That division of labour is the argument for hybrid, and it's worth one paragraph in your README.

### 5.3 Hyperparameter tuning (the "training" of the RAG, part 2)

Grid-search on a held-out dev set of 2,000 queries, scoring MRR@10 against `is_selected`:

| Parameter | Grid | Note |
|---|---|---|
| `efSearch` | 32, 48, 64, 96, 128 | Pick the knee: highest recall where P99 latency still fits budget |
| `M` (build) | 16, 32 | 32 = better recall, 2× memory. Probably not worth it |
| `efConstruction` | 200 | Build-time only, set once |
| RRF `k` | 20, 40, 60, 80 | 60 is standard; verify |
| dense:BM25 weight | 1:1, 2:1, 3:1 | Weighted RRF; expect dense-favoured for Indic |
| top-k retrieved | 10, 20, 50 | Feeds reranker/LLM |
| MMR λ | 0.5, 0.7, 1.0 | 1.0 = pure relevance |

Commit `eval/results/tuning.csv`. Showing the sweep is worth more than showing the winner.

### 5.4 Extractive answering (what makes the fast path a real answer)

MS MARCO answers are usually a span of the selected passage. Score each sentence of the top-3 passages:

```
score(sent) = 0.5·cos(q_emb, sent_emb)          // pre-computed at index time
            + 0.3·bm25_overlap(query, sent)
            + 0.2·qtype_prior(qtype, sent)      // NUMERIC → sentence has digits
                                                // PERSON  → sentence has a capitalised NE
```

Return the top sentence plus its neighbour, with the source `chunk_id` as citation. Sentence embeddings are precomputed offline, so this is pure arithmetic at runtime — ~5 ms.

---

## 6. Actual model training (optional, high-payoff)

This is the section that separates you from every other submission. You have 10M labelled pairs; use them.

```python
# ingest/06_finetune.py
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

model = SentenceTransformer("intfloat/multilingual-e5-small")

# positives from is_selected == 1
# hard negatives = top-20 BM25/dense hits for the query that are NOT selected
train = [InputExample(texts=[f"query: {q}",
                             f"passage: {pos}",
                             f"passage: {hard_neg}"])
         for q, pos, hard_neg in mined_triplets]

loss = losses.MultipleNegativesRankingLoss(model)   # in-batch negatives
model.fit(train_objectives=[(DataLoader(train, shuffle=True, batch_size=64), loss)],
          epochs=1, warmup_steps=500, use_amp=True)
model.save("artifacts/e5-small-msmarco-xi")
```

**Hard negative mining is the whole trick.** Random negatives teach the model nothing — it already separates "Manhattan Project" from "banana bread." Negatives mined from the top-20 retrieved-but-not-selected set are what actually move MRR.

**Caveat you must state:** `is_selected == 0` is *not* a reliable negative in MS MARCO. Sparse labelling means many unselected passages are actually relevant. Mitigate by skipping the top-3 retrieved when mining negatives (they're most likely false negatives), and mine from ranks 5–30 instead.

Expected outcome: +3 to +8 MRR@10 points on 1 epoch over ~200k triplets, roughly 30–60 min on a Colab T4. Report before/after in the README against IndicRAGSuite's published baselines (e5-small: hi 0.44 / bn 0.39 / ta 0.38; bge-m3: hi 0.52 / bn 0.49 / ta 0.49). If your fine-tuned e5-small closes part of the gap to bge-m3 *while staying 5× faster*, that is a genuinely strong result and the headline of your submission.

**Budget guard:** this is a day-5 stretch goal. If day 4 is not fully green, skip it. A working pipeline beats a fine-tuned broken one.

---

## 7. Harness

```ts
// server/harness/contracts.ts
export const Transcript = z.object({
  text: z.string().min(1), lang: z.string().length(2),
  confidence: z.number().min(0).max(1), isFinal: z.boolean(),
});
export const GroundedAnswer = z.object({
  answer: z.string(),
  citations: z.array(z.string()).min(1),      // chunk_ids — enforced non-empty
  confidence: z.number(),
  refused: z.boolean(),
  refusalReason: z.enum(["off_topic","unsafe","low_asr","no_grounding"]).optional(),
});
```

```ts
// server/harness/pipeline.ts
export async function runStage<I, O>(
  name: string, input: I, schema: z.ZodType<O>,
  fn: (i: I) => Promise<O>, trace: Span[], opts?: { timeoutMs?: number }
): Promise<O> {
  const t = performance.now();
  try {
    const out = await withTimeout(fn(input), opts?.timeoutMs ?? 5000);
    const parsed = schema.parse(out);
    trace.push({ name, ms: performance.now() - t, ok: true });
    return parsed;
  } catch (e) {
    trace.push({ name, ms: performance.now() - t, ok: false, err: String(e) });
    throw new StageError(name, e);
  }
}
```

Required harness behaviours, all demonstrable:

- **Typed contracts** at every boundary (above).
- **Retries** — exponential backoff with jitter on STT/LLM, but **capped by remaining budget**. A retry that blows the deadline is worse than a degraded answer.
- **Circuit breaker** per provider: 5 failures in 30 s → open for 60 s → half-open probe.
- **Fallback chain:** Sarvam → browser Web Speech API; Groq → Cerebras → extractive-only.
- **Structured output + repair:** force JSON from the LLM, validate with Zod, one repair attempt with the validation error fed back, then fall back to extractive.
- **Trace array returned in every response** and rendered as a live waterfall in the UI. This single feature satisfies requirements 4 and 5 simultaneously and is your best demo visual.
- **Semantic cache:** if `cos(q_emb, cached_q_emb) > 0.97`, serve cached. Report cached and uncached percentiles separately — conflating them is dishonest and a judge will ask.

---

## 8. Guardrails

### 8.1 Layers

| Layer | Check | Cost | Action on fail |
|---|---|---|---|
| L0 input | empty / gibberish / single-token | <0.1 ms | ask to repeat |
| L0 input | ASR confidence < τ₀ | free (Sarvam returns it) | "I didn't catch that" |
| L1 safety | unsafe/injection — keyword + embedding vs exemplar set | ~0.3 ms (reuses q_emb) | refuse |
| L2 OOD | max top-k score < τ₁ **or** cos(q_emb, corpus_centroid) < τ₂ | ~0.2 ms | "not in my knowledge base" |
| L3 retrieval | rank1−rank5 score gap flat → no clear winner | free | refuse |
| L4 grounding | each answer sentence must cos > τ₃ with its cited chunk | ~2 ms | strip sentence / refuse |
| L5 async | NLI entailment check on slow path | ~50–150 ms | retract to extractive |

### 8.2 Calibration (the "training", part 3)

Build two labelled query sets:
- **In-domain:** 500 held-out validation queries.
- **Out-of-domain:** 200 hand-written — personal questions ("what's my name"), post-cutoff news, arithmetic, chit-chat, other-domain factual, prompt injections.

Sweep each threshold, plot ROC, pick the operating point at **≤5% false-refusal rate on in-domain**. Commit `eval/results/ood_roc.png` and `thresholds.json`. Report the confusion matrix.

This is why the guardrails are defensible rather than vibes: you have numbers.

### 8.3 Demo requirement

The brief says "show that your system knows when *not* to answer." Your demo video **must** show at least three refusals: off-topic, unsafe, and low-retrieval-confidence. Hardcode three known-good demo queries and rehearse them.

---

## 9. Latency engineering — concrete techniques

**Things that will blow your budget if you don't do them:**

| Technique | Saves | Why |
|---|---|---|
| Local ONNX embedding, never an API | 150–300 ms | One HTTPS round-trip to an embedding provider is game over |
| Boot warmup: 50 dummy queries | 200+ ms on P100 | First ONNX run allocates arenas; first HNSW query faults pages in |
| Preallocated `Float32Array` buffers | tail only | Removes GC pressure from the hot path |
| Truncate query tokens to 64 | 5–10 ms | Attention is quadratic; queries are short anyway |
| `intraOpNumThreads: 2` | 3–8 ms | Thread contention dominates for batch-1 short sequences |
| Always-on Fly machine, no scale-to-zero | seconds | A cold start means a dead demo when judges click |
| Index baked into Docker image | seconds at boot | No runtime S3/R2 fetch |
| `--max-old-space-size` tuned, index in a long-lived object | P99/P100 | Prevents major GC during a request |
| Fly region `bom` (Mumbai) | 40–150 ms network | Judges are in India |

**P100 is brutal** — it's the max, so a single GC pause or page fault ruins it. Mitigations: warm up, preallocate, keep the index pinned, and run the benchmark long enough that you're reporting steady-state behaviour rather than one unlucky sample.

**Benchmark harness — open loop, not closed loop:**

```ts
// bench/latency.ts
import hdr from "hdr-histogram-js";
const h = hdr.build({ numberOfSignificantValueDigits: 3 });

for (const q of warmupQueries.slice(0, 50)) await core(q);   // discarded

const arrivalMs = 20;                    // constant rate — avoids coordinated omission
const started = performance.now();
await Promise.all(queries.map((q, i) => (async () => {
  const due = started + i * arrivalMs;
  await sleepUntil(due);
  const t = performance.now();
  await core(q);
  h.recordValue(performance.now() - due);   // measure from DUE time, not start time
})()));

console.log({ p50: h.getValueAtPercentile(50), p70: h.getValueAtPercentile(70),
              p90: h.getValueAtPercentile(90), p99: h.getValueAtPercentile(99),
              p100: h.maxValue });
```

Recording from *due* time rather than *start* time is what corrects for coordinated omission. Mention this in the README — it signals you know what you're doing.

Run ≥1,000 queries, stratified across languages and `query_type`. Emit `results.json`, `LATENCY.md`, and a histogram PNG.

---

## 10. Deployment

**Backend — Fly.io:**

```toml
# fly.toml
app = "dhwani-api"
primary_region = "bom"

[build]
  dockerfile = "Dockerfile"

[[vm]]
  size = "shared-cpu-2x"
  memory = "2gb"

[http_service]
  internal_port = 3000
  auto_stop_machines = false      # CRITICAL — never scale to zero
  min_machines_running = 1
```

```dockerfile
FROM oven/bun:1
WORKDIR /app
COPY package.json bun.lockb ./
RUN bun install --frozen-lockfile
COPY . .
COPY artifacts/ /app/artifacts/       # bake the index in
EXPOSE 3000
CMD ["bun", "run", "server/index.ts"]
```

If the image exceeds ~2 GB, push `hnsw.bin` to Cloudflare R2 and pull on boot instead — but then add a `/health` endpoint that reports index-loaded status, and don't route traffic until it's true.

**Frontend — Cloudflare Pages**, your usual flow. Set `VITE_API_URL` to the Fly hostname. CORS: allow your Pages origin explicitly, and remember the WS upgrade needs `wss://`.

**Keep-alive:** a cron pinging `/health` every 5 minutes so the machine and its caches stay warm through judging.

---

## 11. Execution order with GO/NO-GO gates

| Day | Work | Gate |
|---|---|---|
| 0 | WSL2 env, Bun, uv, Sarvam + Groq keys, repo scaffold, Fly + Pages accounts | Test WAV transcribes via Sarvam |
| 1 | Subset → dedupe → embed → HNSW + BM25 built; Recall@10 sane | **Recall@10 > 0.6 on strategy A** — if not, debug prefixes/normalisation before anything else |
| 2 | Bun core wired, ONNX warm, tracing in, benchmark harness running | **Core P50 < 200 ms** — if not, cut dims/corpus |
| 3 | All 7 chunk variants indexed + evaluated; tuning sweep; tables committed | **One clear winning config with a defensible table** |
| 4 | Harness, guardrails + calibration, LLM slow path, React UI, deploy | **Live URL works end-to-end from a phone on mobile data** |
| 5 | Optional fine-tune; README; both videos; final dry run; submit | **All four artifacts present** — repo, live link, 2 videos |

**Day 1's gate is the one that kills projects.** If Recall@10 is bad, it is almost always (a) missing `query:`/`passage:` prefixes, (b) unnormalised vectors with cosine space, or (c) an index/query embedding model mismatch. Check those three before touching anything else.

---

## 12. Things that will bite you

1. **e5 prefix mismatch** — the #1 silent recall killer. Unit-test it.
2. **Devanagari nukta double-encoding** — NFC normalise both index and query, offline and online.
3. **`MediaRecorder` gives you Opus, not PCM** — use AudioWorklet, resample to 16 kHz mono Int16.
4. **`getUserMedia` requires HTTPS** — works on `localhost`, breaks on a LAN IP. Test on the deployed URL early.
5. **WebSocket idle timeout** (~60 s on Sarvam) — send keepalives; don't auto-retry on 4xxx auth closes, you'll burn credits.
6. **`load_dataset` without `streaming=True`** will try to materialise 55.6 GB.
7. **Cross-filesystem I/O in WSL2** — keep the repo in `~/`, never `/mnt/c/`. 10–20× slower otherwise.
8. **hnswlib `setEf` is per-process, not persisted** — set it after every `readIndex`.
9. **Quantization can silently degrade the model** — validate cosine agreement vs fp32.
10. **Reporting cached and uncached latency together** — separate them or a judge will call it.
11. **Claiming 200 ms end-to-end including STT and LLM** — physically impossible, and technical judges will notice. The bounded honest claim is the stronger play.
12. **No resubmissions** — dry-run the live link, the repo clone, and both video links from an incognito window before you hit submit.

---

## 13. What the README must contain

In this order:
1. One-paragraph pitch + live link + both video links at the very top.
2. Architecture diagram.
3. **The latency boundary definition, stated before any number**, then the P50/P70/P90/P99/P100 table with per-stage breakdown and cached/uncached split.
4. Chunking bake-off table (7 strategies × Recall@1/5/10, MRR@10, nDCG@10) + the tuning sweep.
5. Guardrail design + ROC plot + confusion matrix + example refusals.
6. Fine-tuning results vs IndicRAGSuite baselines, if you did it.
7. The MS MARCO sparse-label caveat — stating it yourself is a credibility signal, not a weakness.
8. Reproduction instructions (`bun run bench`, `python eval/retrieval_eval.py`).

The README is graded whether the organisers admit it or not. Judges spend two minutes; make the first screen carry the whole argument.
