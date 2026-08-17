/**
 * PLAN.md E5.5 — real guardrail calibration, not placeholder guesses.
 * Embeds 500 real in-domain queries (sampled from data/medium/{hi,bn,ta}.jsonl)
 * and 199 hand-written OOD queries (eval/calibration/ood_queries.jsonl, 6
 * categories per ARCHITECTURE.md §8.2) through the SAME production embed.ts
 * path the server uses, then sweeps L1 (safety) and L2 (OOD) thresholds to
 * find the operating point with <=5% false-refusal rate on in-domain,
 * per ARCHITECTURE.md §8.2.
 *
 * L4/L5 (grounding/NLI) are NOT calibrated here -- both need real LLM-
 * generated answer sentences to check groundedness against, and there is no
 * real Groq/Cerebras client yet (chain.ts is mock-tested only). Calibrating
 * against fabricated sentences would violate CLAUDE.md #6. Left as an open
 * gap, same as before this script.
 *
 * L2's "top retrieval score" (tau_1) can't be calibrated against real HNSW
 * either (hnswlib-node won't build on this machine) -- brute-force cosine
 * against the real 59,666 medium-scale passage embeddings stands in, same
 * substitution eval/retrieval_eval.py already uses, and for the same reason
 * (brute-force is an upper bound on what approximate HNSW achieves, so this
 * doesn't overstate real guardrail performance).
 */
import { readFileSync, writeFileSync, mkdirSync } from "fs";
import * as embed from "../server/ghana/embed";
import { cosineSim } from "../server/maun/ood";
import { SAFETY_EXEMPLARS } from "../server/maun/exemplars";

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

function loadPassageEmbeddings(path: string, dim: number): Float32Array {
  const buf = readFileSync(path);
  const arr = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  if (arr.length % dim !== 0) throw new Error("embeddings file not a multiple of dim");
  return arr;
}

// Brute-force max cosine of `q` against every row of `passages` (flat,
// row-major, N x dim). Passages are already L2-normalized (verified at
// export time, see export_centroid_and_raw.py / CLAUDE.md invariant #2), so
// this is a plain dot product, not a full cosine recompute.
function maxDotAgainstCorpus(q: Float32Array, passages: Float32Array, dim: number): number {
  let best = -Infinity;
  const n = passages.length / dim;
  for (let i = 0; i < n; i++) {
    const base = i * dim;
    let dot = 0;
    for (let d = 0; d < dim; d++) dot += q[d] * passages[base + d];
    if (dot > best) best = dot;
  }
  return best;
}

interface ScoredQuery {
  lang: string;
  query: string;
  category: string;
  maxExemplarSim: number;
  topScore: number;
  centroidCos: number;
}

async function scoreQueries(
  rows: QueryRow[],
  category: string,
  exemplarEmb: Float32Array[],
  passages: Float32Array,
  centroid: Float32Array,
  dim: number,
): Promise<ScoredQuery[]> {
  const out: ScoredQuery[] = [];
  for (const row of rows) {
    const qEmb = await embed.embed(row.query);
    let maxExemplarSim = -Infinity;
    for (const ex of exemplarEmb) {
      const s = cosineSim(qEmb, ex);
      if (s > maxExemplarSim) maxExemplarSim = s;
    }
    const topScore = maxDotAgainstCorpus(qEmb, passages, dim);
    const centroidCos = cosineSim(qEmb, centroid);
    out.push({
      lang: row.lang,
      query: row.query,
      category: row.category ?? category,
      maxExemplarSim,
      topScore,
      centroidCos,
    });
  }
  return out;
}

function sweepRange(lo: number, hi: number, step: number): number[] {
  const out: number[] = [];
  for (let v = lo; v <= hi + 1e-9; v += step) out.push(Math.round(v * 1000) / 1000);
  return out;
}

function main() {
  return (async () => {
    console.log("booting embed model...");
    await embed.boot("artifacts/onnx");

    console.log("loading corpus embeddings + centroid...");
    const DIM = 384;
    const passages = loadPassageEmbeddings("data/medium/embeddings.f32bin", DIM);
    const centroid = new Float32Array(JSON.parse(readFileSync("data/medium/centroid.json", "utf-8")));
    console.log(`  ${passages.length / DIM} passages loaded`);

    console.log("embedding safety exemplars...");
    const exemplarEmb: Float32Array[] = [];
    for (const text of SAFETY_EXEMPLARS) exemplarEmb.push(await embed.embed(text));

    const inDomainRows = loadJsonl("eval/calibration/in_domain_queries.jsonl");
    const oodRows = loadJsonl("eval/calibration/ood_queries.jsonl");
    console.log(`scoring ${inDomainRows.length} in-domain + ${oodRows.length} OOD queries...`);

    const t0 = performance.now();
    const inDomain = await scoreQueries(inDomainRows, "in_domain", exemplarEmb, passages, centroid, DIM);
    const ood = await scoreQueries(oodRows, "ood", exemplarEmb, passages, centroid, DIM);
    console.log(`  done in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

    // ---------- L1 safety: sweep a single threshold ----------
    const safetyRoc = sweepRange(0.3, 0.95, 0.01).map((tau) => {
      const fp = inDomain.filter((q) => q.maxExemplarSim >= tau).length;
      const injections = ood.filter((q) => q.category === "injection");
      const tp = injections.filter((q) => q.maxExemplarSim >= tau).length;
      return {
        tau,
        fpr: fp / inDomain.length,
        tpr: injections.length ? tp / injections.length : 0,
      };
    });
    // pick the highest tau (most conservative / fewest false refusals) whose
    // FPR is <=5%, then take the one among ties with best TPR (== lowest tau
    // satisfying the constraint, since TPR is monotonically non-increasing
    // as tau rises).
    const safetyCandidates = safetyRoc.filter((p) => p.fpr <= 0.05);
    const safetyOperating = safetyCandidates.length
      ? safetyCandidates.reduce((best, p) => (p.tau < best.tau ? p : best))
      : safetyRoc[safetyRoc.length - 1];

    // ---------- L2 OOD: joint grid sweep over (tau1, tau2) ----------
    const tau1Range = sweepRange(0.1, 0.9, 0.02);
    const tau2Range = sweepRange(0.0, 0.7, 0.02);
    let best: { tau1: number; tau2: number; fpr: number; tpr: number } | null = null;
    const oodRocByTau1: { tau1: number; fpr: number; tpr: number }[] = [];
    for (const tau1 of tau1Range) {
      // for the single-parameter ROC report, hold tau2 at 0 (score-only)
      const fpr1 = inDomain.filter((q) => q.topScore < tau1).length / inDomain.length;
      const tpr1 = ood.filter((q) => q.topScore < tau1).length / ood.length;
      oodRocByTau1.push({ tau1, fpr: fpr1, tpr: tpr1 });

      for (const tau2 of tau2Range) {
        const fpCount = inDomain.filter((q) => q.topScore < tau1 || q.centroidCos < tau2).length;
        const fpr = fpCount / inDomain.length;
        if (fpr > 0.05) continue;
        const tpCount = ood.filter((q) => q.topScore < tau1 || q.centroidCos < tau2).length;
        const tpr = tpCount / ood.length;
        if (!best || tpr > best.tpr || (tpr === best.tpr && fpr < best.fpr)) {
          best = { tau1, tau2, fpr, tpr };
        }
      }
    }
    if (!best) {
      // no combination hit <=5% FPR -- fall back to the most conservative
      // corner (highest thresholds) and report the true FPR honestly rather
      // than silently picking something that doesn't meet the bar.
      const tau1 = tau1Range[tau1Range.length - 1];
      const tau2 = tau2Range[tau2Range.length - 1];
      const fpr = inDomain.filter((q) => q.topScore < tau1 || q.centroidCos < tau2).length / inDomain.length;
      const tpr = ood.filter((q) => q.topScore < tau1 || q.centroidCos < tau2).length / ood.length;
      best = { tau1, tau2, fpr, tpr };
    }

    // ---------- joint 3-parameter search for the TRUE system-level operating point ----------
    // The two guardrails were each independently calibrated to <=5% FPR
    // above, but since a query is refused if EITHER fires, the combined FPR
    // can (and, measured for real below, does) compound past 5% -- a judge
    // only experiences the combined behaviour, so that's what must actually
    // respect ARCHITECTURE.md's <=5% bar, not each guardrail's own budget.
    // Precompute into flat typed arrays -- this grid is ~65k-100k
    // candidates x 699 queries, plain array .filter() per candidate would
    // reallocate far too much.
    const idExemplar = Float64Array.from(inDomain.map((q) => q.maxExemplarSim));
    const idTop = Float64Array.from(inDomain.map((q) => q.topScore));
    const idCentroid = Float64Array.from(inDomain.map((q) => q.centroidCos));
    const oodExemplar = Float64Array.from(ood.map((q) => q.maxExemplarSim));
    const oodTop = Float64Array.from(ood.map((q) => q.topScore));
    const oodCentroid = Float64Array.from(ood.map((q) => q.centroidCos));

    function combinedFprTpr(tauS: number, tau1: number, tau2: number) {
      let fp = 0;
      for (let i = 0; i < idExemplar.length; i++) {
        if (idExemplar[i] >= tauS || idTop[i] < tau1 || idCentroid[i] < tau2) fp++;
      }
      let tp = 0;
      for (let i = 0; i < oodExemplar.length; i++) {
        if (oodExemplar[i] >= tauS || oodTop[i] < tau1 || oodCentroid[i] < tau2) tp++;
      }
      return { fpr: fp / idExemplar.length, tpr: tp / oodExemplar.length };
    }

    const jointSafetyRange = sweepRange(0.5, 0.95, 0.01);
    const jointTau1Range = sweepRange(0.3, 0.9, 0.02);
    const jointTau2Range = sweepRange(0.0, 0.5, 0.02);
    let joint: { tauS: number; tau1: number; tau2: number; fpr: number; tpr: number } | null = null;
    for (const tauS of jointSafetyRange) {
      for (const tau1 of jointTau1Range) {
        for (const tau2 of jointTau2Range) {
          const { fpr, tpr } = combinedFprTpr(tauS, tau1, tau2);
          if (fpr > 0.05) continue;
          if (!joint || tpr > joint.tpr || (tpr === joint.tpr && fpr < joint.fpr)) {
            joint = { tauS, tau1, tau2, fpr, tpr };
          }
        }
      }
    }
    if (!joint) {
      const tauS = jointSafetyRange[jointSafetyRange.length - 1];
      const tau1 = jointTau1Range[jointTau1Range.length - 1];
      const tau2 = jointTau2Range[jointTau2Range.length - 1];
      const { fpr, tpr } = combinedFprTpr(tauS, tau1, tau2);
      joint = { tauS, tau1, tau2, fpr, tpr };
    }

    // These joint values are what actually ships in thresholds.json -- the
    // per-guardrail `safetyOperating`/`best` above are kept only as
    // diagnostic marginal operating points (useful to see each guardrail's
    // own ROC shape), not what's used at runtime.
    const combinedFprInDomain = joint.fpr;
    const combinedTprOod = joint.tpr;
    const combinedRefused = (q: ScoredQuery) =>
      q.maxExemplarSim >= joint!.tauS || q.topScore < joint!.tau1 || q.centroidCos < joint!.tau2;

    // per-category / per-language breakdown, for the honest writeup
    const categories = [...new Set(ood.map((q) => q.category))];
    const categoryBreakdown = Object.fromEntries(
      categories.map((cat) => {
        const subset = ood.filter((q) => q.category === cat);
        const caught = subset.filter(combinedRefused).length;
        return [cat, { n: subset.length, caught, rate: caught / subset.length }];
      }),
    );
    const injectionByLang = Object.fromEntries(
      [...new Set(ood.filter((q) => q.category === "injection").map((q) => q.lang))].map((lang) => {
        const subset = ood.filter((q) => q.category === "injection" && q.lang === lang);
        const caughtBySafety = subset.filter((q) => q.maxExemplarSim >= safetyOperating.tau).length;
        return [lang, { n: subset.length, caughtBySafetyGuard: caughtBySafety, rate: caughtBySafety / subset.length }];
      }),
    );

    const report = {
      generatedAt: new Date().toISOString(),
      sampleSizes: { inDomain: inDomain.length, ood: ood.length, safetyExemplars: exemplarEmb.length },
      // Marginal (independently-chosen) operating points -- diagnostic only,
      // NOT what ships in thresholds.json. Kept because each guardrail's own
      // ROC shape is useful to see in isolation.
      safetyMarginal: {
        chosenThreshold: safetyOperating.tau,
        fprInDomain: safetyOperating.fpr,
        tprInjectionsOnly: safetyOperating.tpr,
        roc: safetyRoc,
      },
      oodMarginal: {
        chosenThresholds: { minTopScore: best.tau1, minCentroidCosine: best.tau2 },
        fprInDomain: best.fpr,
        tprOodAll: best.tpr,
        rocScoreOnly: oodRocByTau1,
      },
      // The real, ships-to-production operating point: jointly searched so
      // that the COMBINED (L1 OR L2) false-refusal rate on in-domain is
      // <=5%, per ARCHITECTURE.md §8.2 -- calibrating each guardrail
      // separately to 5% measurably compounds past that bound (a query
      // refused by either counts), so this is the number that actually
      // matters and the marginal ones above do not, by themselves, satisfy
      // the spec.
      jointOperatingPoint: {
        safetyThreshold: joint.tauS,
        oodThresholds: { minTopScore: joint.tau1, minCentroidCosine: joint.tau2 },
        fprInDomain: combinedFprInDomain,
        tprOodAll: combinedTprOod,
      },
      categoryBreakdown,
      crossLingualInjectionGeneralization: injectionByLang,
    };

    mkdirSync("eval/results", { recursive: true });
    writeFileSync("eval/results/guardrail_calibration.json", JSON.stringify(report, null, 2));

    mkdirSync("artifacts", { recursive: true });
    const thresholdsArtifact = {
      generatedAt: report.generatedAt,
      calibrationSource: "eval/results/guardrail_calibration.json",
      safetyThreshold: joint.tauS,
      oodThresholds: { minTopScore: joint.tau1, minCentroidCosine: joint.tau2 },
    };
    writeFileSync("artifacts/thresholds.json", JSON.stringify(thresholdsArtifact, null, 2));

    console.log("\n=== marginal (diagnostic only) ===");
    console.log(`  L1 alone: threshold=${safetyOperating.tau}  in-domain FPR=${(safetyOperating.fpr * 100).toFixed(1)}%  injection TPR=${(safetyOperating.tpr * 100).toFixed(1)}%`);
    console.log(`  L2 alone: minTopScore=${best.tau1}  minCentroidCosine=${best.tau2}  in-domain FPR=${(best.fpr * 100).toFixed(1)}%  OOD TPR=${(best.tpr * 100).toFixed(1)}%`);
    console.log("=== joint operating point (this is what ships) ===");
    console.log(`  safetyThreshold=${joint.tauS}  minTopScore=${joint.tau1}  minCentroidCosine=${joint.tau2}`);
    console.log(`  combined in-domain FPR=${(combinedFprInDomain * 100).toFixed(1)}%  combined OOD TPR=${(combinedTprOod * 100).toFixed(1)}%`);
    console.log("\nwrote eval/results/guardrail_calibration.json and artifacts/thresholds.json");
  })();
}

main();
