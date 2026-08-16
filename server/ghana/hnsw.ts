/**
 * ghana -- dense index, HNSW wrapper over hnswlib-node.
 *
 * UNVERIFIED ON THIS MACHINE: hnswlib-node cannot build here (no MSVC Build
 * Tools on this Windows dev box, confirmed -- see MEMORY.md). Written
 * against the documented API (ARCHITECTURE.md §5.1) and hnswlib-node's
 * published interface, but has not been run. Smoke-test this for real the
 * moment WSL2/the Oracle VM exists -- don't assume it's correct just because
 * it type-checks, the same way embed.ts's tokenizer bug (missing
 * token_type_ids) only surfaced by actually running it.
 *
 * Import is deliberately LAZY (dynamic `await import(...)` inside boot(),
 * not a top-level static import): hnswlib-node throws at require-time on
 * this machine ("Could not locate the bindings file"), not just at first
 * use -- a static import would crash anything that imports this module
 * transitively (server/index.ts included) the moment it's loaded, even
 * code paths that never call boot(). Found this the hard way: server/
 * index.ts couldn't even be imported for a basic smoke check until this
 * was fixed. Change back to a static import once hnswlib-node actually
 * works here (WSL2/Oracle) if the dynamic import ever feels unnecessary --
 * but there's no cost to leaving it lazy either.
 *
 * Index file compatibility: the .bin file loaded here is produced by
 * ingest/05_build_index.py using the Python `hnswlib` package, which wraps
 * the same underlying C++ library as hnswlib-node -- the on-disk format is
 * shared, no conversion step (confirmed via research, not yet via an actual
 * cross-load test since neither binding builds on this machine).
 */
type HierarchicalNSWInstance = {
  readIndexSync(path: string): void;
  setEf(ef: number): void;
  searchKnn(vector: number[], k: number): { neighbors: number[]; distances: number[] };
  getCurrentCount(): number;
};

const SPACE = "ip"; // inner product -- CLAUDE.md #2, vectors are pre-normalized
const DIM = 384;
const DEFAULT_EF_SEARCH = 64; // ARCHITECTURE.md §5.3 tuning default, sweep before trusting

let index: HierarchicalNSWInstance | null = null;
let idMap: string[] = []; // HNSW integer label -> passage_id, from *_id_map.json

export async function boot(indexPath: string, idMapPath: string): Promise<void> {
  const { HierarchicalNSW } = await import("hnswlib-node");
  index = new HierarchicalNSW(SPACE, DIM) as unknown as HierarchicalNSWInstance;
  index.readIndexSync(indexPath);
  index.setEf(DEFAULT_EF_SEARCH); // must be re-set after every readIndexSync -- CLAUDE.md/ARCHITECTURE.md §12 pitfall #8, it is NOT persisted in the index file

  const raw = await Bun.file(idMapPath).text();
  idMap = JSON.parse(raw);
}

export function setEfSearch(ef: number): void {
  if (!index) throw new Error("hnsw.boot() must be called first");
  index.setEf(ef);
}

export function search(
  queryVector: Float32Array,
  topK: number,
): { passageId: string; score: number }[] {
  if (!index) throw new Error("hnsw.boot() must be called first");

  const result = index.searchKnn(Array.from(queryVector), topK);
  // hnswlib-node returns cosine/IP *distance*, not similarity -- for "ip"
  // space, distance = 1 - inner_product for normalized vectors, so score
  // (higher = better) = 1 - distance. Re-verify this sign convention the
  // moment this actually runs -- unverified on this machine, see docstring.
  return result.neighbors.map((label, i) => ({
    passageId: idMap[label],
    score: 1 - result.distances[i],
  }));
}

export function size(): number {
  return index?.getCurrentCount() ?? 0;
}
