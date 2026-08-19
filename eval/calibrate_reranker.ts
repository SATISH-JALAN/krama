/**
 * Calibrates maun/rerank_guard.ts's threshold (L3, the cross-encoder
 * relevance gate added after real voice-testing found the bi-encoder/RRF
 * signal alone wasn't reliable enough -- see that module's own docstring)
 * against the same real calibration query sets L1/L2 used
 * (eval/calibration/in_domain_queries.jsonl, 666 real queries hi/bn/ta/en;
 * ood_queries.jsonl, 199 hand-written across 6 categories).
 *
 * Reuses the REAL production path (server/index.ts's bootFromDisk() +
 * handleQuery()) rather than reimplementing retrieval -- the whole point of
 * L3 is what it does inside the actual pipeline, and duplicating that logic
 * here would risk the same "calibration script quietly drifts from what's
 * actually served" bug already found once this session (the language-
 * scoping mismatch in the original calibrate_guardrails.ts). To sweep a
 * threshold without the gate refusing queries mid-measurement, boot() is
 * called once with rerankMinScore set to -1000 (never refuses on L3),
 * and each query's real cross-encoder score is recovered from the returned
 * `confidence` field via inverse-sigmoid (handleQuery() applies
 * sigmoid(bestScore) when the reranker is enabled -- see its own comment).
 * A query L0/L1/L2 refuses before reaching L3 has no rerank score to
 * recover; that's fine, it's already correctly handled by an earlier layer
 * and isn't what L3's own threshold needs to account for.
 *
 * Run: bun run eval/calibrate_reranker.ts
 */
import { readFileSync, writeFileSync } from "fs";
import { bootFromDisk, handleQuery } from "../server/index";

interface QueryRow {
  lang: string;
  query: string;
  qtype?: string;
  category?: string;
}

function loadJsonl(path: string): QueryRow[] {
  return readFileSync(path, "utf-8")
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));
}

function inverseSigmoid(p: number): number {
  const clamped = Math.min(Math.max(p, 1e-9), 1 - 1e-9);
  return Math.log(clamped / (1 - clamped));
}

interface ScoredQuery {
  lang: string;
  query: string;
  category: string;
  refusedBeforeL3: boolean;
  rerankScore: number | null; // null if refused before reaching L3
}

async function scoreQueries(rows: QueryRow[], category: string): Promise<ScoredQuery[]> {
  const out: ScoredQuery[] = [];
  for (const row of rows) {
    const result = await handleQuery(row.query, row.lang, row.qtype ?? "DESCRIPTION");
    if (result.refused) {
      out.push({ lang: row.lang, query: row.query, category: row.category ?? category, refusedBeforeL3: true, rerankScore: null });
    } else {
      out.push({
        lang: row.lang,
        query: row.query,
        category: row.category ?? category,
        refusedBeforeL3: false,
        rerankScore: inverseSigmoid(result.confidence),
      });
    }
  }
  return out;
}

function sweepRange(lo: number, hi: number, step: number): number[] {
  const out: number[] = [];
  for (let v = lo; v <= hi + 1e-9; v += step) out.push(Math.round(v * 100) / 100);
  return out;
}

async function main() {
  console.log("booting real production path with rerankMinScore=-1000 (L3 never refuses during measurement)...");
  // A temp thresholds artifact -1000 so L3 lets every query through to
  // be scored; L0/L1/L2 remain at their own real calibrated thresholds
  // unchanged, since only L3's own operating point is being calibrated here.
  const tmpThresholdsPath = "eval/calibration/_tmp_rerank_calibration_thresholds.json";
  const currentThresholds = JSON.parse(readFileSync("artifacts/thresholds.json", "utf-8"));
  writeFileSync(tmpThresholdsPath, JSON.stringify({ ...currentThresholds, rerankMinScore: -1000 }));

  await bootFromDisk({ thresholdsPath: tmpThresholdsPath });

  const inDomainRows = loadJsonl("eval/calibration/in_domain_queries.jsonl");
  const oodRows = loadJsonl("eval/calibration/ood_queries.jsonl");
  console.log(`scoring ${inDomainRows.length} in-domain + ${oodRows.length} OOD queries through the real pipeline...`);

  const t0 = performance.now();
  const inDomain = await scoreQueries(inDomainRows, "in_domain");
  const ood = await scoreQueries(oodRows, "ood");
  console.log(`  done in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

  const inDomainAlreadyRefused = inDomain.filter((q) => q.refusedBeforeL3).length;
  const oodAlreadyRefused = ood.filter((q) => q.refusedBeforeL3).length;
  console.log(`in-domain already refused by L0/L1/L2 (before L3 even runs): ${inDomainAlreadyRefused}/${inDomain.length}`);
  console.log(`OOD already refused by L0/L1/L2: ${oodAlreadyRefused}/${ood.length}`);

  // Sweep: for each candidate threshold, false-refusal rate is measured
  // over the WHOLE in-domain set (L0/L1/L2 refusals count against the
  // budget same as before -- L3 can only ever add refusals on top, not
  // remove them), and OOD catch rate likewise over the whole OOD set.
  const scoreRange = sweepRange(-8, 4, 0.1);
  let best: { tau: number; fpr: number; tpr: number } | null = null;
  const roc: { tau: number; fpr: number; tpr: number }[] = [];
  for (const tau of scoreRange) {
    const fpCount = inDomain.filter((q) => q.refusedBeforeL3 || q.rerankScore! < tau).length;
    const fpr = fpCount / inDomain.length;
    const tpCount = ood.filter((q) => q.refusedBeforeL3 || q.rerankScore! < tau).length;
    const tpr = tpCount / ood.length;
    roc.push({ tau, fpr, tpr });
    if (fpr > 0.05) continue;
    if (!best || tpr > best.tpr || (tpr === best.tpr && fpr < best.fpr)) {
      best = { tau, fpr, tpr };
    }
  }
  if (!best) {
    best = roc.reduce((a, b) => (b.fpr < a.fpr ? b : a));
    console.warn(`no threshold hit <=5% combined FPR -- using the lowest-FPR point found: ${JSON.stringify(best)}`);
  }

  console.log("\n=== L3 (reranker) calibration result ===");
  console.log(`chosen rerankMinScore = ${best.tau}`);
  console.log(`combined in-domain FPR (L0/L1/L2/L3 together) = ${(best.fpr * 100).toFixed(1)}%`);
  console.log(`combined OOD TPR (L0/L1/L2/L3 together) = ${(best.tpr * 100).toFixed(1)}%`);

  const report = {
    generatedAt: new Date().toISOString(),
    sampleSizes: { inDomain: inDomain.length, ood: ood.length },
    inDomainAlreadyRefusedByL0L1L2: inDomainAlreadyRefused,
    oodAlreadyRefusedByL0L1L2: oodAlreadyRefused,
    chosenRerankMinScore: best.tau,
    combinedFprInDomain: best.fpr,
    combinedTprOod: best.tpr,
    roc,
  };
  writeFileSync("eval/results/reranker_calibration.json", JSON.stringify(report, null, 2));

  // Merge into the real thresholds.json -- this IS what ships.
  writeFileSync(
    "artifacts/thresholds.json",
    JSON.stringify({ ...currentThresholds, rerankMinScore: best.tau, generatedAt: report.generatedAt }, null, 2),
  );
  console.log("\nwrote eval/results/reranker_calibration.json and updated artifacts/thresholds.json");
}

main();
