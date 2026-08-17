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

export async function boot(embeddingsPath: string, idsPath: string): Promise<void> {
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
}

export function search(
  queryVector: Float32Array,
  topK: number,
): { passageId: string; score: number }[] {
  if (!embeddings) throw new Error("bruteforce.boot() must be called first");

  const n = ids.length;
  // Vectors are pre-normalized (CLAUDE.md #2), so inner product == cosine --
  // a plain dot product against every row, same computation as
  // eval/calibrate_guardrails.ts's maxDotAgainstCorpus, generalized to
  // top-k via a full sort. n is tens of thousands at MVP scale, not
  // millions -- an O(n log n) sort is cheap here and simpler than a bounded
  // top-k structure, matching the calibration run's own measured latency.
  const scores = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const base = i * DIM;
    let dot = 0;
    for (let d = 0; d < DIM; d++) dot += queryVector[d] * embeddings[base + d];
    scores[i] = dot;
  }

  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[b] - scores[a]);
  return order.slice(0, topK).map((i) => ({ passageId: ids[i], score: scores[i] }));
}

export function size(): number {
  return ids.length;
}
