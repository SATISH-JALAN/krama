# KRAMA — Agile Execution Plan

Solo build, zero budget, ~5-6 days. This adapts Scrum to a one-person team: 1-day sprints, written
async stand-ups (they double as raw material for the process video), a GO/NO-GO gate as the sprint
review, and a one-line retro that feeds the risk register. See `CLAUDE.md` for invariants/stack/
corrections — this file is process + backlog, not technical spec.

## 1. Framework

**Sprint length**: 1 calendar day. 6 sprints: Day -1 (spike), Day 0 (spike), Day 1-5 (build).

**Ceremonies (solo-adapted, keep them short or they become overhead)**:
- *Sprint Planning* (5 min, start of day): read that day's Sprint Backlog below, confirm nothing from
  yesterday's Blocked column carries a hidden dependency into today.
- *Daily stand-up* (written, not spoken — log 3 bullets in a `STANDUP.md` or as a commit message
  prefix): Done yesterday / Doing today / Blocked by. This is also your Video 1 raw material —
  screen-record yourself writing it once or twice, don't stage it separately.
- *Sprint Review* = the day's GO/NO-GO gate. Binary, testable, no vibes.
- *Sprint Retro* (2 min, end of day): one line — "what would I change tomorrow." Append to the Risk
  Register (§5) if it's risk-relevant, otherwise discard.

**Board**: GitHub Projects (free, zero setup, lives next to the repo you already need for submission).
Columns: `Backlog / In Progress / Blocked / Done`. One issue per story below; label with epic + MoSCoW.

**Definition of Done** (every story, no exceptions):
1. Code merged to `main` (small repo, trunk-based — no long-lived branches, there's no time for them).
2. The story's stated verification step actually run, not assumed.
3. No invariant from `CLAUDE.md` violated (spot-check against the list, especially #1/#2/#4/#6 which are
   the easiest to silently break).
4. No silent TODO — if it's not done, it's a new backlog item, not a comment.

**Prioritization**: MoSCoW (Must/Should/Could/Won't), then ordered by risk within each day — de-risk
unknowns before building on top of them. This is why Day -1/0 are pure spikes with no feature work.

## 2. Product Backlog

Estimates are T-shirt sizes for solo work, not story points: **S** <2h, **M** 2-4h, **L** 4-8h,
**XL** unpredictable / spans the sprint. MoSCoW: **Mu**st, **Sh**ould, **Co**uld, **Wo**n't-this-round.

### Epic 0 — Environment & Hosting De-risking
| ID | Story | Pri | Est |
|---|---|---|---|
| E0.1 | Secure Oracle Always Free Ampere A1 instance via retry-script tool (`oci-arm-host-capacity` or `oracle-freetier-instance-creation`), GCP trial in parallel as fallback | Mu | XL |
| E0.2 | Verify `hnswlib-node` compiles from source on the VM (build-essential+python3 present) under plain Node | Mu | S |
| E0.3 | Verify `onnxruntime-node`'s prebuilt arm64 binary runs correctly **inside Bun** specifically | Mu | S |
| E0.4 | Caddy + Let's Encrypt on the VM, hostname via `<public-ip>.sslip.io`; open ports 80/443 in Oracle security list **and** local firewall (both layers) | Mu | M |
| E0.5 | Sarvam/Groq/Cerebras keys obtained; confirm at signup whether Cerebras now requires a card — drop it as fallback if so | Mu | S |
| E0.6 | ONNX export pipeline runs on WSL2 (portable, no arch risk) | Mu | S |

### Epic 1 — Data Pipeline & Index
| ID | Story | Pri | Est |
|---|---|---|---|
| E1.1 | Stream + subset hi/bn/ta, 25k queries/lang, verify `ex.keys()` against real schema first | Mu | M |
| E1.2 | Dedupe passages: SHA-1 of NFC-normalized text | Mu | S |
| E1.3 | Embed passages, e5-small, `passage:` prefix, L2-normalized fp32 | Mu | M |
| E1.4 | Unit test: e5 prefix applied correctly index-side and query-side | Mu | S |
| E1.5 | Build HNSW (inner-product space) + Indic-tokenized BM25 postings | Mu | L |
| E1.6 | ONNX int8 quantize + validate cosine agreement >0.995 vs PyTorch, fp32 fallback if it fails | Mu | M |
| E1.7 | Strategy A (passage-as-is) indexed as baseline | Mu | S |

### Epic 2 — Retrieval Core & Latency
| ID | Story | Pri | Est |
|---|---|---|---|
| E2.1 | Bun+Hono server skeleton, ONNX session boot + 20-query warmup | Mu | M |
| E2.2 | `embed`/`hnsw`/`bm25`/`fuse` (RRF) wired end-to-end | Mu | L |
| E2.3 | Extractive span selection (§5.4 scoring) | Mu | M |
| E2.4 | Span tracer (~40 LOC JSON array, no OTel) | Mu | S |
| E2.5 | Open-loop HDR-histogram bench harness, arrival-time-based recording | Mu | M |
| E2.6 | ≥1,000 queries run, stratified by lang/qtype, P50/P70/P90/P100 reported | Mu | S |

### Epic 3 — Chunking Bake-off
| ID | Story | Pri | Est |
|---|---|---|---|
| E3.1 | Strategy B — fixed 256/64 overlap | Mu | S |
| E3.2 | Strategy D — sentence-window | Mu | M |
| E3.3 | Strategy E — small-to-big / parent-document | Mu | M |
| E3.4 | Strategy G — cross-lingual dual index | Sh | M |
| E3.5 | Retrieval eval: Recall@{1,5,10}/MRR@10/nDCG@10 via `is_selected`, sparse-label caveat stated | Mu | M |
| E3.6 | Hyperparameter sweep (efSearch, RRF k, dense:BM25 weight, top-k, MMR λ), `tuning.csv` committed | Sh | M |
| E3.7 | Strategy F — contextual retrieval, 5,000-chunk ablation, clearly labelled | Co | M |
| E3.8 | Strategy C — semantic percentile splitting | Co | M |
| E3.9 | bge-reranker-v2-m3 on/off comparison | Co | M |

### Epic 4 — Harness
| ID | Story | Pri | Est |
|---|---|---|---|
| E4.1 | Zod contracts at every stage boundary | Mu | S |
| E4.2 | Retries, backoff+jitter, capped by remaining latency budget | Mu | S |
| E4.3 | Circuit breaker per provider (5 fail/30s → open 60s → half-open) | Mu | S |
| E4.4 | Fallback chains: Sarvam→WebSpeech, Groq→Cerebras→extractive | Mu | M |
| E4.5 | Semantic cache (cos>0.97), cached/uncached latency reported separately | Sh | S |

### Epic 5 — Guardrails
| ID | Story | Pri | Est |
|---|---|---|---|
| E5.1 | L0 — empty/gibberish/very-short-transcript detection (**not** ASR confidence — Sarvam doesn't return one, confirmed) | Mu | S |
| E5.2 | L1 — safety exemplar set (write ~50-100 examples) + embedding check | Mu | M |
| E5.3 | L2 — OOD via corpus-centroid / top-k-score threshold | Mu | M |
| E5.4 | L4 — per-sentence grounding cosine check | Mu | M |
| E5.5 | Calibration: 500 in-domain + 200 hand-written OOD, ROC sweep, `thresholds.json`, confusion matrix | Mu | M |
| E5.6 | L5 — async NLI groundedness using `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (ONNX weights pre-published, no export step) — benchmark CPU latency first, unverified | Co | M |

### Epic 6 — Frontend & Voice UX
| ID | Story | Pri | Est |
|---|---|---|---|
| E6.1 | AudioWorklet → Int16 PCM @16kHz capture (not MediaRecorder) | Mu | M |
| E6.2 | Mic/waveform/transcript UI | Mu | M |
| E6.3 | Dual-path answer display + trace waterfall | Mu | M |
| E6.4 | Deploy to Cloudflare Pages, `VITE_API_URL` wired | Mu | S |

### Epic 7 — Deployment
| ID | Story | Pri | Est |
|---|---|---|---|
| E7.1 | Backend deployed on Oracle/GCP VM behind Caddy (TLS termination, reverse proxy to Bun) | Mu | M |
| E7.2 | Index baked into image/VM, no runtime fetch | Mu | S |
| E7.3 | CORS locked to Pages origin, WSS path verified | Mu | S |
| E7.4 | Free uptime monitor (cron-job.org or UptimeRobot) pinging `/health` | Sh | S |

### Epic 8 — Fine-tune (stretch only)
| ID | Story | Pri | Est |
|---|---|---|---|
| E8.1 | Hard-negative mining, ranks 5-30, skip top 3 | Co/Wo | L |
| E8.2 | Contrastive fine-tune on Colab T4 (Kaggle GPU as backup if Colab session-limited) | Co/Wo | L |
| E8.3 | Before/after MRR@10 vs IndicRAGSuite baselines, measured not estimated | Co/Wo | S |

### Epic 9 — Submission
| ID | Story | Pri | Est |
|---|---|---|---|
| E9.1 | README assembled in ARCHITECTURE §13 order | Mu | M |
| E9.2 | Video 1 — 90s process/build-log | Mu | S |
| E9.3 | Video 2 — demo, incl. ≥3 refusal cases | Mu | M |
| E9.4 | Full dry-run from incognito window: link, repo clone, both videos | Mu | S |
| E9.5 | Submit form | Mu | S |

## 3. Sprint Plan

### Sprint -1 — Spike: Hosting capacity
**Sprint goal**: a running VM, reachable over SSH, before the 5-day clock starts.
**Sprint backlog**: E0.1
**Review (GO/NO-GO)**: VM up with a public IP. If Oracle capacity never clears, GCP trial is the
committed fallback — don't keep retrying Oracle past this sprint, it costs no compute progress.
**Retro prompt**: which region/provider actually worked, so it's not re-discovered next time.

### Sprint 0 — Spike: Runtime verification
**Sprint goal**: prove the ARM+Bun runtime story works, or know the fallback, before writing app code.
**Sprint backlog**: E0.2, E0.3, E0.4, E0.5, E0.6
**Review (GO/NO-GO)**: `hnswlib-node` compiles on the VM, `onnxruntime-node` runs correctly inside Bun
(or the Node-sidecar fallback decision is made and recorded in CLAUDE.md), a test WAV transcribes via
Sarvam's realtime endpoint, `<ip>.sslip.io` serves a valid Let's Encrypt cert through Caddy. This gate
is allowed to run long — nothing downstream is trustworthy until it clears, though research already
de-risked most of these (onnxruntime-node ships arm64 prebuilds, hnswlib-node's from-source build is
routine) so this sprint should be quick confirmation, not discovery.
**Retro prompt**: log the actual working Node/Bun/ORT version combo so it isn't re-derived later.

### Sprint 1 — Data + index
**Sprint goal**: a queryable, recall-validated index exists.
**Sprint backlog**: E1.1–E1.7
**Review (GO/NO-GO)**: Recall@10 > 0.6 on strategy A. If not, check in order: (a) missing/mismatched
`query:`/`passage:` prefixes, (b) cosine space instead of inner-product on normalized vectors, (c)
index/query embedding model mismatch — before touching anything else.
**Retro prompt**: actual unique-passage count after dedupe vs the ~300-450k estimate.

### Sprint 2 — Retrieval core + latency
**Sprint goal**: the t₀→t₁ path is wired and measured honestly.
**Sprint backlog**: E2.1–E2.6
**Review (GO/NO-GO)**: core P50 < 200ms, measured transcript-in→grounded-answer-out only, no STT/LLM
in the number. If not: cut `efSearch`, cut corpus size, check `intraOpNumThreads` before anything
more drastic.
**Retro prompt**: which stage actually dominated the P100 tail — informs Sprint 3's tuning sweep.

### Sprint 3 — Chunking bake-off + eval
**Sprint goal**: a defensible, evidenced chunking recommendation.
**Sprint backlog**: E3.1, E3.2, E3.3, E3.5 (must) → E3.4, E3.6 (should, if must-items land by midday) →
E3.7, E3.8, E3.9 (could, only if the day is otherwise green)
**Review (GO/NO-GO)**: one clear winning configuration with a table showing the recall/latency
tradeoff, not just the winner.
**Retro prompt**: did any "should" or "could" story get pulled in — recalibrate tomorrow's estimate.

### Sprint 4 — Harness + guardrails + LLM + frontend + deploy
**Sprint goal**: a judge can use the live link end-to-end on a phone.
**Sprint backlog**: E4.1–E4.4 (must), E4.5 (should) → E5.1–E5.5 (must), E5.6 (could, only if ahead) →
E6.1–E6.4 (must) → E7.1–E7.3 (must), E7.4 (should)
**Review (GO/NO-GO)**: live URL works end-to-end from a phone on mobile data. Specifically test the
`getUserMedia`-requires-HTTPS trap on the deployed URL — localhost hides this bug.
**Retro prompt**: which guardrail layer actually triggered false-refusals in manual testing.

### Sprint 5 — Fine-tune (stretch) + polish + videos + submit
**Sprint goal**: all four submission artifacts exist and have been verified cold.
**Sprint backlog**: E8.1–E8.3 (only if Sprint 4 closed with room to spare) → E9.1–E9.5 (must, always)
**Review (GO/NO-GO, final, no resubmits)**: repo, live link, both videos — all checked from a fresh
incognito session, not "it worked when I built it."
**Retro**: none needed, this is the last sprint.

## 4. Cut List (priority order if a sprint runs over)
1. Fine-tune (Epic 8, already stretch-only)
2. Chunking strategies C and F (E3.7, E3.8)
3. bge-reranker comparison (E3.9)
4. bn/ta languages — fall back to hi-only if Sprint 1's gate isn't comfortably cleared
5. Guardrail L5 async NLI (E5.6) — L0-L2 + L4 already satisfy "knows when not to answer"

**Never cut**: the harness (E4.1-E4.4), guardrails L0-L2+L4 (E5.1-E5.4), the honest P50/P70/P90/P100
benchmark (E2.5-E2.6), the sparse-label caveat in the README (part of E9.1).

## 5. Risk Register
| Risk | Mitigation | Status |
|---|---|---|
| Oracle Ampere A1 capacity unavailable | Retry-script tool + GCP trial in parallel from Sprint -1 | open — resolve Sprint -1 |
| `onnxruntime-node` broken under Bun on aarch64 | Node-sidecar fallback, decide in Sprint 0 | downgraded — no linux-arm64-specific crash reports found; still smoke-test Sprint 0 |
| Cerebras free-tier terms change | **Resolved** — only `zai-glm-4.7` model deprecates 8/17, free tier itself unaffected. Avoid that model. | resolved, verify no-card-required at signup |
| Groq 30 RPM too tight under load | LLM synthesis is off the critical path; only matters live during judging, low volume | monitor |
| Groq model names stale | `llama-3.1-8b-instant`/`llama-3.3-70b-versatile` deprecated 8/16 — use `gpt-oss-20b`/`120b` or `qwen3.6-27b` | resolved |
| No stable HTTPS hostname without a domain | **is-a.dev blocked by design** (rejects CNAME to `*.cfargotunnel.com`) — switched to Caddy+Let's Encrypt on-VM via sslip.io | resolved (E0.4), adds port-opening step |
| Sarvam free credit is ₹100 not ₹1,000 | Budget dev/test STT calls carefully (~3.3 free hours); reuse recorded audio | open — track usage from Sprint 0 |
| ASR-confidence guardrail (L0) unimplementable | Sarvam returns no confidence score on either endpoint — L0 uses empty/gibberish/short-transcript detection instead | resolved (E5.1 updated) |
| L5 groundedness model unchosen | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, ONNX pre-published, multilingual incl. Hindi/Urdu — CPU latency still unverified | resolved (E5.6), benchmark in Sprint 4 |
| Guardrail false-refusals | Calibration sweep at ≤5% FPR on in-domain (E5.5) | planned Sprint 4 |
| No resubmission | Full incognito dry-run before submit (E9.4) | planned Sprint 5 |
