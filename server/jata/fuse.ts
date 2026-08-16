/**
 * jata -- hybrid fusion. Reciprocal Rank Fusion combining dense (HNSW) and
 * lexical (BM25) rankings. ARCHITECTURE.md §5.1/§5.3: k=60 is the standard
 * default; weight supports the dense:BM25 tuning sweep (1:1, 2:1, 3:1) --
 * expect dense-favoured weighting for Indic per the cross-lingual BM25
 * limitation documented in bm25.ts.
 */

export interface RankedList {
  ids: string[];
  weight?: number; // defaults to 1
}

const DEFAULT_K = 60;

export function rrf(lists: RankedList[], k: number = DEFAULT_K): Map<string, number> {
  const scores = new Map<string, number>();
  for (const { ids, weight = 1 } of lists) {
    ids.forEach((id, rank) => {
      const contribution = weight / (k + rank + 1);
      scores.set(id, (scores.get(id) ?? 0) + contribution);
    });
  }
  return scores;
}

export function rrfRanked(
  lists: RankedList[],
  topK: number,
  k: number = DEFAULT_K,
): { id: string; score: number }[] {
  const scores = rrf(lists, k);
  return Array.from(scores.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, topK)
    .map(([id, score]) => ({ id, score }));
}
