/**
 * krama -- extractive span selection (ARCHITECTURE.md §5.4). Scores each
 * candidate sentence against the query and returns the best one plus its
 * neighbor, with a citation back to the source chunk. This is what makes
 * the fast path a real grounded answer instead of a placeholder -- the
 * returned text is a real quoted span, so it's grounded by construction.
 *
 * score(sent) = 0.5*cos(q_emb, sent_emb) + 0.3*bm25_overlap(query, sent)
 *             + 0.2*qtype_prior(qtype, sent)
 *
 * Sentence embeddings are precomputed offline (same source as strategy
 * D/E's chunks_DE.jsonl + embeddings_DE.npy) -- this module does no
 * embedding inference itself, pure arithmetic at runtime per the doc.
 *
 * qtype_prior caveat, not fixed here: the PERSON heuristic
 * (capitalized-word detection) only means anything for Latin-script text --
 * Devanagari/Bengali/Tamil have no letter case, so it silently contributes
 * nothing for those scripts. A real fix would need a proper NER model;
 * flagged rather than pretending it works cross-script.
 */
import { cosineSim } from "../maun/ood";
import { tokenize } from "../bm25";

export interface ExtractCandidate {
  chunkId: string;
  parentPassageId: string;
  text: string;
  embedding: Float32Array;
  lang: string;
  sentenceIdx?: number;
  numSentences?: number;
}

export interface ExtractResult {
  text: string;
  chunkId: string;
  score: number;
  neighborText?: string;
}

function hasDigits(text: string): boolean {
  return /\d/.test(text);
}

function hasCapitalizedWord(text: string): boolean {
  return /\b[A-Z][a-z]+/.test(text);
}

const KNOWN_QTYPE_PRIORS: Record<string, (text: string) => number> = {
  NUMERIC: (text) => (hasDigits(text) ? 1 : 0),
  PERSON: (text) => (hasCapitalizedWord(text) ? 1 : 0),
};

function qtypePrior(queryType: string, text: string): number {
  return KNOWN_QTYPE_PRIORS[queryType]?.(text) ?? 0;
}

function bm25Overlap(queryTokens: string[], candidateTokens: string[]): number {
  if (queryTokens.length === 0) return 0;
  const candidateSet = new Set(candidateTokens);
  const overlap = queryTokens.filter((t) => candidateSet.has(t)).length;
  return overlap / queryTokens.length;
}

const WEIGHTS = { cosine: 0.5, overlap: 0.3, prior: 0.2 };

export function scoreCandidate(
  queryTokens: string[],
  queryEmbedding: Float32Array,
  queryType: string,
  candidate: ExtractCandidate,
): number {
  const cosScore = cosineSim(queryEmbedding, candidate.embedding);
  const overlapScore = bm25Overlap(queryTokens, tokenize(candidate.text, candidate.lang));
  const priorScore = qtypePrior(queryType, candidate.text);
  return WEIGHTS.cosine * cosScore + WEIGHTS.overlap * overlapScore + WEIGHTS.prior * priorScore;
}

/**
 * Pick the best-scoring candidate. If the candidate pool includes its
 * siblings (same parentPassageId, consecutive sentenceIdx), attach the
 * following sentence as neighborText for extra context in the returned
 * answer -- matches ARCHITECTURE.md §5.4's "top sentence plus its neighbor."
 */
export function extractSpan(
  query: string,
  queryType: string,
  queryEmbedding: Float32Array,
  candidates: ExtractCandidate[],
  lang: string,
): ExtractResult | null {
  if (candidates.length === 0) return null;

  const queryTokens = tokenize(query, lang);

  let best: ExtractCandidate | null = null;
  let bestScore = -Infinity;
  for (const candidate of candidates) {
    const score = scoreCandidate(queryTokens, queryEmbedding, queryType, candidate);
    if (score > bestScore) {
      bestScore = score;
      best = candidate;
    }
  }
  if (!best) return null;

  let neighborText: string | undefined;
  if (best.sentenceIdx !== undefined) {
    const neighbor = candidates.find(
      (c) =>
        c.parentPassageId === best!.parentPassageId &&
        c.sentenceIdx === best!.sentenceIdx! + 1,
    );
    neighborText = neighbor?.text;
  }

  return { text: best.text, chunkId: best.chunkId, score: bestScore, neighborText };
}
