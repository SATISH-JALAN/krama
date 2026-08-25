/**
 * ghana -- dense index, brute-force inner-product search.
 *
 * Replaces hnsw.ts on the live critical path (see MEMORY.md's "RAGINGOA
 * reference analyzed" entry): the reference repo for this same task measures
 * 0.73ms/query over 148k vectors with brute force, and this project's own
 * guardrail-calibration run already measured ~27ms/query for embed+60k-
 * passage brute force combined -- comfortably under the 200ms core budget at
 * this corpus's actual MVP scale. This removes the hnswlib-node/WSL2/Oracle
 * blocker by making it unnecessary for MVP scale, not by resolving it --
 * hnsw.ts stays in the repo, undeleted, as a documented option for a future
 * larger-scale index.
 *
 * Same boot()/search()/size() interface as hnsw.ts by design, so index.ts's
 * wiring barely changes. `embeddingsPath` is a flat row-major Float32
 * binary (N x 384, same layout `eval/export_centroid_and_raw.py` writes and
 * `eval/calibrate_guardrails.ts` reads); `idsPath` is a JSON array of
 * passage_id strings in the same row order.
 */
const DIM = 384;

let embeddings: Float32Array | null = null;
let ids: string[] = [];
// Parallel to `ids` -- which language each row's passage is in, so search()
// can restrict candidates to the query's own language. Without this, dense
// search ranks across the whole (now 4-language: hi/bn/ta/en) corpus at
// once, and a Hindi query can end up "grounded" in a Tamil passage the user
// never asked for and can't read -- a real bug found by voice-testing the
// live app, not a hypothetical.
let langs: string[] = [];
// Reused across search() calls -- see the comment in search() for why.
let scratchScores: Float32Array | null = null;
let scratchCandidates: Int32Array | null = null;

export async function boot(
  embeddingsPath: string,
  idsPath: string,
  passageIdToLang?: Map<string, string>,
): Promise<void> {
  const buf = await Bun.file(embeddingsPath).arrayBuffer();
  const arr = new Float32Array(buf);
  if (arr.length % DIM !== 0) {
    throw new Error(`bruteforce.boot: embeddings file length not a multiple of dim=${DIM}`);
  }

  const raw = await Bun.file(idsPath).text();
  const parsedIds: string[] = JSON.parse(raw);
  if (parsedIds.length !== arr.length / DIM) {
    throw new Error(
      `bruteforce.boot: id count (${parsedIds.length}) doesn't match embedding row count (${arr.length / DIM})`,
    );
  }

  embeddings = arr;
  ids = parsedIds;
  langs = passageIdToLang ? parsedIds.map((id) => passageIdToLang.get(id) ?? "") : [];
}

export function search(
  queryVector: Float32Array,
  topK: number,
  filterLang?: string,
): { passageId: string; score: number }[] {
  if (!embeddings) throw new Error("bruteforce.boot() must be called first");

  const n = ids.length;
  const useFilter = !!filterLang && langs.length === n;
  // Vectors are pre-normalized (CLAUDE.md #2), so inner product == cosine --
  // a plain dot product against every row, same computation as
  // eval/calibrate_guardrails.ts's maxDotAgainstCorpus, generalized to
  // top-k via a full sort. n is tens of thousands at MVP scale, not
  // millions -- an O(n log n) sort is cheap here and simpler than a bounded
  // top-k structure, matching the calibration run's own measured latency.
  // Scratch buffers are allocated once and reused across every search()
  // call, not freshly per query. search() is fully synchronous -- no await
  // inside it -- so a call always runs to completion before the next one
  // starts, and reuse is safe on JS's single thread.
  //
  // Why bother: the previous version allocated a 239KB Float32Array plus a
  // growing candidates array on EVERY query, ~100MB of garbage across a
  // 400-query benchmark. That is exactly the shape that produces the
  // occasional multi-hundred-millisecond GC pause showing up in P100.
  if (!scratchScores || scratchScores.length < n) scratchScores = new Float32Array(n);
  if (!scratchCandidates || scratchCandidates.length < n) scratchCandidates = new Int32Array(n);
  const scores = scratchScores;
  let candidateCount = 0;
  // Only rows that survive the language filter are collected here, so the
  // sort below never touches the ~45k rows of the other three languages.
  // The previous version scored into a full-length array, wrote -Infinity
  // for filtered-out rows, then sorted ALL 59,666 indices and discarded the
  // -Infinity entries three lines later -- roughly 950k comparator calls,
  // most of them ordering values that could never be returned. Results are
  // byte-identical (those rows were unreachable either way); this is purely
  // the same answer computed without the wasted work.
  for (let i = 0; i < n; i++) {
    if (useFilter && langs[i] !== filterLang) continue;
    const base = i * DIM;
    let dot = 0;
    for (let d = 0; d < DIM; d++) dot += queryVector[d] * embeddings[base + d];
    scores[i] = dot;
    scratchCandidates[candidateCount++] = i;
  }

  const candidates = scratchCandidates.subarray(0, candidateCount);
  candidates.sort((a, b) => scores[b]! - scores[a]!);
  const out: { passageId: string; score: number }[] = [];
  for (let k = 0; k < Math.min(topK, candidateCount); k++) {
    const i = candidates[k]!;
    out.push({ passageId: ids[i]!, score: scores[i]! });
  }
  return out;
}

export function size(): number {
  return ids.length;
}
