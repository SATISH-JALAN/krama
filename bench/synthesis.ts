/**
 * Real P50/P70/P100 latency for the SLOW synthesis path -- POST
 * /query/synthesize / `handleSynthesisQuery()` -- i.e. transcript-in to
 * LLM-*generated*-answer-out.
 *
 * Why this file exists, separately from bench/latency.ts: the brief's
 * pipeline shape ends at "Answer generation", and this project answers on
 * two tiers (README "Latency"). bench/latency.ts measures the fast
 * extractive tier (t0->t1, CLAUDE.md invariant #4) and is the tier the
 * <200ms budget is claimed against. That left the generative tier with no
 * published numbers at all, which is the weakest possible answer to "how
 * long does generation take" -- silence reads as an unmeasured gap.
 * CLAUDE.md #6 forbids inventing a number, not measuring an inconvenient
 * one, so this measures it. The result is expected to be SECONDS, not
 * milliseconds (llm/gemini.ts already documents 1.9s-19.3s single-call
 * latency measured live), and is reported as its own labelled tier -- never
 * averaged into the fast path's numbers (CLAUDE.md #4, never conflate two
 * latency classes).
 *
 * What the measured wall-clock includes: the whole of
 * `handleSynthesisQuery()` -- guardrails, embedding, dense+BM25 retrieval,
 * RRF fusion, rerank, AND the LLM round-trip. It therefore contains one
 * redundant retrieval pass (~60ms, index.ts documents this as a deliberate
 * simplification), which is noise next to a multi-second LLM call and is
 * NOT subtracted out -- an honest end-to-end number should not quietly
 * discount work the real endpoint actually does.
 *
 * Query sample: real hi/bn/ta/en queries from data/medium/{hi,bn,ta,en}.jsonl,
 * first N in file order -- the same source and the same deterministic
 * selection bench/latency.ts uses, so the two tiers are measured over
 * comparable input, not different query sets (CLAUDE.md #6 extends to
 * benchmark *inputs*).
 *
 * Outcomes are bucketed, not pooled, because they are different amounts of
 * work and pooling them would report a blended number describing nothing:
 *   - `grounded`   -- retrieval found a passage, LLM rewrote it.
 *   - `ungrounded` -- retrieval refused (off_topic/no_grounding), LLM
 *                     answered from general knowledge instead. Still one
 *                     full LLM round-trip, so still a generation number.
 *   - `refusedNoLlm` -- L0/L1 refusal, deliberately never reaches an LLM.
 *                     Fast by construction; a latency sample from this
 *                     bucket says nothing about generation speed.
 *   - `providerFailure` -- every provider errored/timed out. Recorded as a
 *                     count, not as a latency sample: timing a failure
 *                     would measure the retry deadline, not generation.
 *
 * Run: bun run bench:synthesis  (or `bun run bench/synthesis.ts`)
 * Optional: `--n <count>` queries per language (default 5 -- deliberately
 * small: free-tier LLM quotas are per-minute and per-day, and at up to ~19s
 * per call a large N buys variance you can get more cheaply by re-running).
 *           `--delay-ms <ms>` pause between calls (default 1000) to stay
 * inside free-tier per-minute request caps.
 *
 * Requires GEMINI_API_KEY and/or CEREBRAS_API_KEY (Bun loads .env
 * automatically). With neither set this exits non-zero WITHOUT writing a
 * results file, rather than emitting placeholder numbers.
 */
import { readFileSync, mkdirSync, writeFileSync } from "fs";
import * as hdr from "hdr-histogram-js";
import { bootFromDisk, handleSynthesisQuery } from "../server/index";

const LANGS = ["hi", "bn", "ta", "en"];

type Hist = ReturnType<typeof hdr.build>;

interface QueryRow {
  qid: number;
  lang: string;
  query: string;
  qtype?: string;
}

function loadQueries(lang: string, n: number): QueryRow[] {
  const rows = readFileSync(`data/medium/${lang}.jsonl`, "utf-8")
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line) as QueryRow);
  return rows.slice(0, n);
}

function numArg(flag: string, fallback: number): number {
  const i = process.argv.indexOf(flag);
  if (i === -1) return fallback;
  const v = Number(process.argv[i + 1]);
  return Number.isFinite(v) ? v : fallback;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

type Bucket = "grounded" | "ungrounded" | "refusedNoLlm";

function summarize(h: Hist) {
  if (h.totalCount === 0) return { n: 0 };
  return {
    p50: h.getValueAtPercentile(50),
    p70: h.getValueAtPercentile(70),
    p90: h.getValueAtPercentile(90),
    p99: h.getValueAtPercentile(99),
    p100: h.getValueAtPercentile(100),
    min: h.minNonZeroValue,
    mean: Math.round(h.mean * 100) / 100,
    n: h.totalCount,
  };
}

async function main(): Promise<void> {
  const n = numArg("--n", 5);
  const delayMs = numArg("--delay-ms", 1000);

  console.log("booting from data/medium/ + artifacts/ ...");
  await bootFromDisk();

  const queries = LANGS.flatMap((lang) => loadQueries(lang, n));
  console.log(
    `running ${queries.length} real queries (${n}/lang x ${LANGS.join("/")}) ` +
      `through handleSynthesisQuery() -- expect seconds per call, not ms ...`,
  );

  // One histogram per bucket, plus a combined one over every call that
  // actually made an LLM round-trip (grounded + ungrounded). The combined
  // figure is the one that answers "how long does generation take",
  // reported alongside the split rather than instead of it.
  const histograms: Record<Bucket, Hist> = {
    grounded: hdr.build({ useWebAssembly: false }),
    ungrounded: hdr.build({ useWebAssembly: false }),
    refusedNoLlm: hdr.build({ useWebAssembly: false }),
  };
  const llmBacked = hdr.build({ useWebAssembly: false });

  let providerFailures = 0;
  let declinedCount = 0;
  const providerCounts: Record<string, number> = {};

  for (const [i, row] of queries.entries()) {
    const start = performance.now();
    let result;
    try {
      result = await handleSynthesisQuery(row.query, row.lang, row.qtype ?? "DESCRIPTION");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes("no LLM provider configured")) {
        console.error(
          "\nNo LLM provider configured -- set GEMINI_API_KEY and/or CEREBRAS_API_KEY in .env.\n" +
            "Refusing to write a results file rather than publish placeholder numbers (CLAUDE.md #6).",
        );
        process.exit(1);
      }
      throw err;
    }
    const elapsedMs = Math.max(1, Math.round(performance.now() - start));

    let bucket: Bucket;
    if (result.synthesized === null) {
      if (
        result.refused &&
        result.refusalReason !== "off_topic" &&
        result.refusalReason !== "no_grounding"
      ) {
        // L0/L1 refusal -- never reached an LLM, by design.
        bucket = "refusedNoLlm";
      } else {
        // Every provider failed. Not a generation-latency sample.
        providerFailures++;
        console.log(`  [${i + 1}/${queries.length}] ${row.lang} provider failure after ${elapsedMs}ms`);
        if (delayMs > 0) await sleep(delayMs);
        continue;
      }
    } else {
      bucket = result.synthesized.grounded ? "grounded" : "ungrounded";
      if (result.synthesized.declined) declinedCount++;
      const p = result.synthesized.provider;
      providerCounts[p] = (providerCounts[p] ?? 0) + 1;
    }

    histograms[bucket].recordValue(elapsedMs);
    if (bucket !== "refusedNoLlm") llmBacked.recordValue(elapsedMs);
    console.log(`  [${i + 1}/${queries.length}] ${row.lang} ${bucket} ${elapsedMs}ms`);

    if (delayMs > 0) await sleep(delayMs);
  }

  const summary = {
    generatedAt: new Date().toISOString(),
    boundary:
      "transcript-in -> LLM-generated-answer-out (includes retrieval + guardrails + LLM round-trip)",
    nQueries: queries.length,
    perLang: n,
    delayMs,
    providerFailures,
    declinedCount,
    providerCounts,
    latencyMs: {
      llmBacked: summarize(llmBacked),
      grounded: summarize(histograms.grounded),
      ungrounded: summarize(histograms.ungrounded),
      refusedNoLlm: summarize(histograms.refusedNoLlm),
    },
  };

  console.log(`\n${JSON.stringify(summary, null, 2)}`);

  mkdirSync("bench/results", { recursive: true });
  writeFileSync("bench/results/synthesis.json", JSON.stringify(summary, null, 2));
  console.log("wrote bench/results/synthesis.json");
}

if (import.meta.main) {
  await main();
}
