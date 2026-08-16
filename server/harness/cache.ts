/**
 * Semantic cache (ARCHITECTURE.md §7): if cos(q_emb, cached_q_emb) > 0.97,
 * serve the cached response. CLAUDE.md invariant #4: cached and uncached
 * latency must be reported separately, never conflated -- this module's
 * `get()` return type distinguishes a hit from a miss precisely so the
 * caller can tag which bucket a request's latency belongs to.
 */
import { cosineSim } from "../maun/ood";

export interface CacheEntry<T> {
  queryEmbedding: Float32Array;
  value: T;
}

export interface CacheResult<T> {
  hit: boolean;
  value?: T;
  similarity?: number;
}

const DEFAULT_THRESHOLD = 0.97;

export class SemanticCache<T> {
  private entries: CacheEntry<T>[] = [];

  constructor(
    private readonly threshold: number = DEFAULT_THRESHOLD,
    private readonly maxEntries: number = 1000,
  ) {}

  get(queryEmbedding: Float32Array): CacheResult<T> {
    let best: { entry: CacheEntry<T>; sim: number } | null = null;
    for (const entry of this.entries) {
      const sim = cosineSim(queryEmbedding, entry.queryEmbedding);
      if (!best || sim > best.sim) best = { entry, sim };
    }
    if (best && best.sim >= this.threshold) {
      return { hit: true, value: best.entry.value, similarity: best.sim };
    }
    return { hit: false };
  }

  set(queryEmbedding: Float32Array, value: T): void {
    this.entries.push({ queryEmbedding, value });
    if (this.entries.length > this.maxEntries) {
      this.entries.shift(); // drop oldest -- simple FIFO, not LRU; fine at this scale
    }
  }

  get size(): number {
    return this.entries.length;
  }
}
