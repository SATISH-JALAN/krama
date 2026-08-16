/**
 * Hand-rolled BM25, Indic-aware tokenizer. Built AND queried here, in TS, at
 * server boot -- not pre-built offline in Python (design correction, see
 * MEMORY.md/ingest/05_build_index.py: splitting build/query across two
 * languages risks the tokenizer silently drifting out of sync with no test
 * able to catch it; building both in the same process removes that risk for
 * free, since "does build match query" is trivially true by construction).
 *
 * Tokenization per ARCHITECTURE.md §5.2:
 *   1. Unicode NFC normalize (CLAUDE.md #3) -- nukta double-encoding.
 *   2. Intl.Segmenter word-cluster segmentation (built into Bun) -- correct
 *      grapheme-aware splitting for Devanagari/Bengali/Tamil conjuncts, not
 *      whitespace splitting, which is nearly useless on these scripts.
 *   3. Strip ZWJ/ZWNJ (U+200D/U+200C) -- inconsistently present in translated
 *      text.
 *   4. No stemming.
 * BM25 params k1=0.9, b=0.4 (MS MARCO-tuned defaults, better than classic
 * 1.2/0.75 for short passages, per ARCHITECTURE.md §5.2).
 */

const ZWJ_ZWNJ = /[‌‍]/g;

export function tokenize(text: string, lang: string = "en"): string[] {
  const normalized = text.normalize("NFC").replace(ZWJ_ZWNJ, "");
  const segmenter = new Intl.Segmenter(lang, { granularity: "word" });
  const tokens: string[] = [];
  for (const { segment, isWordLike } of segmenter.segment(normalized)) {
    if (isWordLike) tokens.push(segment.toLowerCase());
  }
  return tokens;
}

export interface Bm25Doc {
  id: string;
  text: string;
  lang: string;
}

interface Posting {
  docIndex: number;
  termFreq: number;
}

const K1 = 0.9;
const B = 0.4;

export class Bm25Index {
  private docIds: string[] = [];
  private docLengths: number[] = [];
  private avgDocLength = 0;
  private postings = new Map<string, Posting[]>();

  build(docs: Bm25Doc[]): void {
    this.docIds = [];
    this.docLengths = [];
    this.postings = new Map();

    let totalLength = 0;
    docs.forEach((doc, docIndex) => {
      const tokens = tokenize(doc.text, doc.lang);
      this.docIds.push(doc.id);
      this.docLengths.push(tokens.length);
      totalLength += tokens.length;

      const termFreqs = new Map<string, number>();
      for (const tok of tokens) termFreqs.set(tok, (termFreqs.get(tok) ?? 0) + 1);

      for (const [term, freq] of termFreqs) {
        const list = this.postings.get(term) ?? [];
        list.push({ docIndex, termFreq: freq });
        this.postings.set(term, list);
      }
    });

    this.avgDocLength = docs.length > 0 ? totalLength / docs.length : 0;
  }

  search(query: string, lang: string, topK: number = 10): { id: string; score: number }[] {
    const N = this.docIds.length;
    if (N === 0) return [];

    const queryTerms = new Set(tokenize(query, lang));
    const scores = new Map<number, number>();

    for (const term of queryTerms) {
      const list = this.postings.get(term);
      if (!list) continue;

      const n = list.length; // number of docs containing this term
      const idf = Math.log((N - n + 0.5) / (n + 0.5) + 1);

      for (const { docIndex, termFreq } of list) {
        const docLen = this.docLengths[docIndex];
        const denom = termFreq + K1 * (1 - B + (B * docLen) / (this.avgDocLength || 1));
        const termScore = idf * ((termFreq * (K1 + 1)) / denom);
        scores.set(docIndex, (scores.get(docIndex) ?? 0) + termScore);
      }
    }

    return Array.from(scores.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, topK)
      .map(([docIndex, score]) => ({ id: this.docIds[docIndex], score }));
  }

  get size(): number {
    return this.docIds.length;
  }
}
