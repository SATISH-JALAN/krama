/**
 * Real P50/P70/P100 latency for the t0->t1 fast-path boundary (CLAUDE.md
 * invariant #4: transcript-in to grounded-answer-out, STT and LLM
 * generation excluded/out of band). Runs against the actual
 * `server/index.ts`'s `handleQuery()` after a real `bootFromDisk()` -- the
 * same boot path the live server uses, not a mock or a separate offline
 * script (that gap is exactly what this project's honest audit found and
 * this file exists to close, see MEMORY.md's "RAGINGOA reference analyzed"
 * entry).
 *
 * Query sample: real hi/bn/ta/en queries from data/medium/{hi,bn,ta,en}.jsonl
 * (the same medium-scale data server/index.ts boots its corpus from), not
 * synthetic text (CLAUDE.md invariant #6 -- never invent a benchmark
 * number, and that extends to never inventing benchmark *inputs* either).
 * en.jsonl is derived (ingest/_derive_english_queries.py) from the `eng_query`
 * field MSMARCO-XI already carries on every hi/bn/ta row -- real original
 * MS MARCO queries, not synthetic English text.
 *
 * Run: bun run bench (package.json script) or `bun run bench/latency.ts`.
 * Optional: `--n <count>` queries per language (default 100).
 *
 * Reports cached and uncached latency SEPARATELY (CLAUDE.md #4 -- never
 * conflate the two): the main pass is all-novel queries (first N in file
 * order, each asked once, so every one is a genuine cache miss -- the
 * semantic cache, harness/cache.ts, is empty at the start of each of these
 * for that exact query); a second pass re-asks a real subset of the SAME
 * queries again, which now hit the cache for real (not simulated) since
 * the exact same text embeds to ~cosine-1.0 similarity against its own
 * cached entry, comfortably over the 0.97 threshold.
 */
import { readFileSync, mkdirSync, writeFileSync } from "fs";
import * as hdr from "hdr-histogram-js";
import { bootFromDisk, handleQuery } from "../server/index";

const CACHE_REPLAY_COUNT = 30;

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
  // First N in file order -- deterministic, reproducible across runs,
  // not randomly sampled (a fixed bench set is easier to compare run over
  // run than a fresh random draw every time).
  return rows.slice(0, n);
}

async function main(): Promise<void> {
  const nArgIdx = process.argv.indexOf("--n");
  const n = nArgIdx !== -1 ? Number(process.argv[nArgIdx + 1]) : 100;

  console.log(`booting from data/medium/ + artifacts/ ...`);
  await bootFromDisk();

  const LANGS = ["hi", "bn", "ta", "en"];
  const queries = LANGS.flatMap((lang) => loadQueries(lang, n));
  console.log(`running ${queries.length} real queries (${n}/lang x ${LANGS.join("/")}) through handleQuery()...`);

  const uncachedHistogram = hdr.build({ useWebAssembly: false });
  const cachedHistogram = hdr.build({ useWebAssembly: false });
  let refusedCount = 0;
  let cacheHits = 0;

  for (const row of queries) {
    const start = performance.now();
    const result = await handleQuery(row.query, row.lang, row.qtype ?? "DESCRIPTION");
    const elapsedMs = performance.now() - start;
    uncachedHistogram.recordValue(Math.max(1, Math.round(elapsedMs)));
    if (result.refused) refusedCount++;
  }

  console.log(`replaying ${CACHE_REPLAY_COUNT} of the same queries to measure real cache-hit latency...`);
  for (const row of queries.slice(0, CACHE_REPLAY_COUNT)) {
    const start = performance.now();
    const result = await handleQuery(row.query, row.lang, row.qtype ?? "DESCRIPTION");
    const elapsedMs = performance.now() - start;
    if (result.cached) {
      cachedHistogram.recordValue(Math.max(1, Math.round(elapsedMs)));
      cacheHits++;
    } else {
      // Guardrail refusals bypass the cache lookup path entirely (see
      // index.ts), so a refused query on replay is a genuine, expected
      // uncached repeat, not a cache-wiring bug -- don't count it as a hit.
      uncachedHistogram.recordValue(Math.max(1, Math.round(elapsedMs)));
    }
  }

  function summarize(h: typeof uncachedHistogram) {
    return {
      p50: h.getValueAtPercentile(50),
      p70: h.getValueAtPercentile(70),
      p90: h.getValueAtPercentile(90),
      p99: h.getValueAtPercentile(99),
      p100: h.getValueAtPercentile(100),
      min: h.totalCount > 0 ? h.minNonZeroValue : null,
      mean: h.totalCount > 0 ? Math.round(h.mean * 100) / 100 : null,
      n: h.totalCount,
    };
  }

  const summary = {
    generatedAt: new Date().toISOString(),
    nQueries: queries.length,
    refusedCount,
    cacheHits,
    cacheReplayAttempted: CACHE_REPLAY_COUNT,
    latencyMs: {
      uncached: summarize(uncachedHistogram),
      cached: summarize(cachedHistogram),
    },
  };

  console.log(JSON.stringify(summary, null, 2));

  mkdirSync("bench/results", { recursive: true });
  writeFileSync("bench/results/latency.json", JSON.stringify(summary, null, 2));
  console.log("wrote bench/results/latency.json");
}

if (import.meta.main) {
  await main();
}
