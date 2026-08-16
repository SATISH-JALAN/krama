# KRAMA — persistent context

## Project
Voice-enabled RAG for HH Goa 2026 shortlisting Task 2. Speak a question (Indic language) →
Sarvam STT → local ONNX retrieval core → grounded extractive answer (fast path, <200ms) →
LLM-synthesized answer (slow path, streamed, off critical path). Dataset: `ai4bharat/MSMARCO-XI`
(11.4M rows, 55.6GB parquet, 14 Indic languages, streaming only — never `load_dataset` without
`streaming=True`). Solo build, ~5 days, zero budget (free tier everything).

Full design: `ARCHITECTURE.md` (wins on conflict) and `RESEARCH-SPEC.md` (underlying research/evidence).
This file is the dense summary + the corrections that supersede both docs. If either doc disagrees
with this file, this file wins — it's newer.

**Naming note:** the project is called KRAMA. The module *also* called `krama` below is just the
chunking/ranking submodule — the name is reused deliberately (fits both "the method" and the Vedic
recitation scheme) but don't let it cause confusion when reading code.

## Module naming
| Module | Name | Meaning |
|---|---|---|
| Voice ingress / STT | `shruti` | "that which is heard" |
| Chunking + ranking core | `krama` | Vedic overlapping-window recitation scheme |
| Dense index | `ghana` | "dense" — also the densest recitation pattern |
| Hybrid fusion (RRF) | `jata` | "braided" |
| Guardrails | `maun` | "silence" — decides not to answer |

## Non-negotiable invariants
1. **e5 prefixes** — every passage embedded as `passage: {text}`, every query as `query: {text}`. Unit test both.
2. **Same model both sides** — index-time and query-time embeddings from the same model + normalization. L2-normalize everything; inner-product space in HNSW.
3. **Unicode NFC normalization** on all Indic text, offline and online, before tokenizing/hashing. Devanagari nukta has two encodings that won't match otherwise.
4. **Latency boundary**: t₀ = transcript-in, t₁ = grounded-answer-out. STT and LLM generation are outside it, reported separately. Never conflate. Never report cached and uncached latency together.
5. **No embedding API calls, ever.** Query embedding is local ONNX. A network round-trip is 150-300ms and ends the project.
6. **Never invent a benchmark number.** Unmeasured → mark ESTIMATED or leave blank. Applies to code comments, README, and anything said to the user.
7. **`is_selected == 0` is not a true negative.** MS MARCO is sparsely labelled. Mine hard negatives from ranks 5-30, skip top 3.
8. **PCM audio, not MediaRecorder.** AudioWorklet → Int16 PCM @ 16kHz mono. MediaRecorder gives WebM/Opus — Sarvam's streaming endpoint rejects it.
9. **Never load the full dataset.** 55.6GB. Always `streaming=True`.
10. **Secrets never touch the client.** Sarvam/Groq keys stay server-side; browser talks only to our WS proxy.

## Corrections (supersede ARCHITECTURE.md / RESEARCH-SPEC.md)
- **Hosting**: Fly.io is out (no free tier for new accounts). Primary: **Oracle Cloud Always Free**,
  Ampere A1, aarch64. Fallback: **GCP $300 trial**, x86, Mumbai. Ignore `fly.toml`
  and all Fly-specific deployment text in ARCHITECTURE §10.
  - Oracle Ampere A1 capacity is frequently unavailable for new tenancies in most regions (confirmed
    via live search, not just the original doc's caution) and is locked to your signup home region.
    Use a retry-script tool (`oci-arm-host-capacity` or `oracle-freetier-instance-creation` on GitHub)
    rather than manual retrying. Attempt Oracle **and** start a GCP trial signup in parallel on
    Day -1 — don't wait on Oracle sequentially before trying the fallback.
  - **Ingress/hostname — REVISED, is-a.dev is blocked**: is-a.dev's own validation rules explicitly
    reject any CNAME pointing at `*.cfargotunnel.com` (verified against their repo's
    `disallowed-cnames.json`), so it cannot front a Cloudflare Tunnel. Default plan is now:
    **Caddy + Let's Encrypt directly on the VM, hostname via sslip.io** (`<public-ip>.sslip.io`
    resolves with zero registration, Caddy auto-acquires a real cert against it). Requires opening
    ports 80/443 in Oracle's security list **and** the VM's local firewall (Oracle images commonly
    block both layers independently — check `iptables`/`firewalld` too, not just the console security
    list). If a real Cloudflare-zone domain becomes available some other way, Cloudflare Tunnel is
    still preferable (origin-hiding, no inbound ports) — swap back to it opportunistically, not as a
    blocking dependency.
- **ARM risk, re-graded down after verification**: `onnxruntime-node` ships official prebuilt
  linux-arm64 binaries (confirmed in the npm tarball) — no build step needed. `hnswlib-node` never
  ships prebuilds on *any* platform (always `node-gyp rebuild` at install), so ARM isn't a special
  case for it — just make sure `build-essential`+python3 are on the box. The one real unknown left is
  whether `onnxruntime-node` runs correctly **inside Bun** specifically (not Node) on aarch64 —
  documented Bun+onnxruntime-node crashes so far are Windows-specific or from 2023, nothing
  linux-arm64-specific found, so this is now LIKELY-fine rather than a known gap. Still smoke-test it
  in Sprint 0 before trusting it. Fallback if it fails: run ONNX inference as a plain-Node sidecar,
  keep Bun for HTTP/WS only.
- **RAM**: Oracle Always Free Ampere A1 is **2 OCPU / 12GB**, not 24GB — Oracle silently halved this
  tier on 2026-06-15 with no announcement; people found out when running instances got shut down.
  Verify the actual allocation at signup, it can change again without warning. fp32 HNSW at 1M×384d ≈
  1.7GB still trivially fits in 12GB. This *removes* scope either way: drop the `usearch`/
  int8-quantization fallback path from ARCHITECTURE §2.1/§4.2 entirely, it's unnecessary at this corpus
  size. Corpus target stays ~400k unique passages for MVP (25k queries × hi/bn/ta, deduped).
- **Chunking strategy F (contextual retrieval)**: 5,000-chunk subset only (not 50k — ARCHITECTURE
  §4.3 is wrong on this number). Ships as a clearly-labelled ablation, not full-corpus.
- **Budget is zero.** Every dependency/service must be free-tier, no credit card on file where
  avoidable. If something requires payment, stop and say so — don't proceed and don't substitute a
  paid alternative silently.
- **Harness**: no BAML, no OTel/Langfuse (ARCHITECTURE wins the conflict with RESEARCH-SPEC on both —
  hand-rolled Zod contracts + a ~40-LOC span tracer is the whole harness).

## Free-tier facts & provider specifics (verified via live search 2026-08-16 — re-verify if stale)
- **Sarvam**: **₹100** free credit on signup, confirmed authoritative from two independent
  docs.sarvam.ai pages (the ₹1,000 figure is stale marketing copy on the main site — don't trust it).
  STT ~₹30/hr → **~3.3 free hours only**. Be economical: reuse recorded test audio, don't hammer the
  API during dev.
  - Use the **realtime** WS endpoint for streaming: `wss://api.sarvam.ai/speech-to-text-realtime/ws`,
    model `saaras:v3-realtime`. Audio encoding param is **`linear16`** (not `pcm_s16le` — that literal
    string only exists on the separate *legacy* endpoint `wss://api.sarvam.ai/speech-to-text/ws`).
    16000Hz or 8000Hz only, mono only, audio sent **base64-encoded** in a JSON `"audio"` field.
    Partial vs final distinguished by `"type": "transcript.partial"` / `"transcript.final"`.
  - Auth header: `API-SUBSCRIPTION-KEY` (also usable as WS subprotocol `api-subscription-key.<key>`).
  - **No confidence score is returned by either endpoint.** The ARCHITECTURE §8.1 L0 guardrail "ASR
    confidence < τ₀" is not implementable as written — drop that sub-check, keep only
    empty/gibberish/very-short-transcript detection for L0. `language_confidence` exists but only for
    language-ID when `language_code="auto"`, not transcript confidence.
- **Groq**: no card required. 30 requests/min, 14,400 requests/day org-wide (not per-key); per-model
  TPM/TPD varies 6K-15K TPM / 100K-500K TPD. `llama-3.1-8b-instant` and `llama-3.3-70b-versatile`
  were **deprecated 2026-08-16** — use `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, or
  `qwen/qwen3.6-27b` instead. Check `console.groq.com/docs/deprecations` before picking a model.
- **Cerebras** (fallback LLM): 1M tokens/day free, 30 RPM. The Aug 17 2026 "deprecation date" on
  their pricing page is scoped to the **`zai-glm-4.7` model only** — not the free tier. Avoid that
  model, everything else on the free tier is unaffected. One open item: some sources suggest a
  payment method may now be required for the $5 signup credit (unconfirmed on Cerebras' own pages) —
  check at signup; if a card is demanded, drop Cerebras and rely on Groq → extractive-only fallback.
- **Oracle Always Free Ampere A1**: 2 OCPU / 12GB aarch64 as of the 2026-06-15 cut (see above).
- **Groundedness/hallucination model** (fills the E5.6 "model not yet chosen" gap):
  `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` — 279M params, **ONNX weights already published on the
  model repo** (no export step needed), trained on XNLI covering 15 languages including Hindi/Urdu, so
  it runs directly on Indic text. CPU latency unbenchmarked — test early since it's still meant to
  fit well under the async slow-path budget.
- **is-a.dev**: free subdomain registrar, but explicitly blocks CNAMEs to `*.cfargotunnel.com` —
  cannot be used to front a Cloudflare Tunnel. See hosting section above for the current plan
  (sslip.io + Caddy).

## WSL2 / environment constraints
- Repo lives in `~/`, **never** `/mnt/c/` — 9p filesystem is 10-20× slower.
- Bun/Vite tooling preferred over npm/webpack equivalents.
- Offline ingestion (Python/uv) runs on the WSL2 dev box (x86) — only the **inference runtime**
  (`onnxruntime-node` on the Oracle ARM server) needs arch verification. ONNX export itself is portable.
- `getUserMedia` requires HTTPS — works on `localhost`, breaks on a bare LAN IP. Test the deployed URL early.

## Stack (exact packages)
| Layer | Package |
|---|---|
| Server runtime | Bun 1.1+ |
| HTTP/WS | `hono` |
| Embeddings | `onnxruntime-node` (not transformers.js) — arm64 prebuilt binaries confirmed shipped |
| Tokenizer | native `tokenizers` (XLM-R SentencePiece) |
| ANN index | `hnswlib-node` — always compiles from source (`node-gyp rebuild`), needs build-essential+python3 |
| Lexical | hand-rolled BM25 (~150 LOC, Indic-aware tokenizer), built AND queried in TS at server boot — not pre-built offline in Python (avoids a cross-language tokenizer-consistency risk, see MEMORY.md) |
| Validation | `zod` |
| STT | Sarvam `saaras:v3-realtime`, `wss://api.sarvam.ai/speech-to-text-realtime/ws`, `linear16` encoding |
| LLM | `groq-sdk` (`openai/gpt-oss-20b`/`120b` or `qwen/qwen3.6-27b`), Cerebras via OpenAI-compatible fetch (avoid `zai-glm-4.7`) |
| Groundedness (stretch) | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, ONNX weights pre-published |
| Ingress/TLS | Caddy + Let's Encrypt on-VM, hostname via sslip.io (Cloudflare Tunnel blocked — see corrections) |
| Bench | `hdr-histogram-js` |
| Offline (Python/uv) | `datasets polars sentence-transformers optimum[onnxruntime] onnxruntime accelerate indic-nlp-library ir-measures pyarrow` |
| Frontend | Vite + React + TS, AudioWorklet PCM capture, Cloudflare Pages |
| Embedding model | `intfloat/multilingual-e5-small`, ONNX int8, mean-pooling w/ attention-mask (confirmed correct) |

## Current phase
**See `MEMORY.md` for the live, frequently-updated state** (what's actually running, what code exists,
what's blocked, the immediate next action) — this section only gets refreshed at major milestones, so
treat MEMORY.md as authoritative over this paragraph if they disagree. As of last edit: repo scaffolded
and committed, local Python toolchain installed on Windows (WSL2 not available in this environment),
Sprint 1 data-pipeline code in progress. Sprint -1 (Oracle capacity) and the ARM-specific half of
Sprint 0 are blocked on human account/install steps not yet done.
