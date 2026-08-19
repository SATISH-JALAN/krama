/**
 * krama -- script-based language detection for the TEXT path.
 *
 * The voice path doesn't need this: Sarvam auto-detects when no
 * language_code is sent, and returns what it heard. But a typed query has
 * no such signal, and /query requires a concrete language because retrieval
 * is language-scoped (see bm25.ts / ghana/bruteforce.ts).
 *
 * Script is a reliable proxy here precisely because the four supported
 * languages use four disjoint scripts -- Devanagari, Bengali, Tamil and
 * Latin. This is NOT general language identification (it cannot tell Hindi
 * from Marathi, which share Devanagari) and deliberately doesn't try: the
 * corpus is hi/bn/ta/en, so the only question that matters is which of
 * those four a query is written in.
 *
 * Counts characters rather than testing the first one, so a query with an
 * English loanword or a digit in it still routes on its dominant script.
 */
const SCRIPT_RANGES: { lang: string; re: RegExp }[] = [
  { lang: "hi", re: /[ऀ-ॿ]/g }, // Devanagari
  { lang: "bn", re: /[ঀ-৿]/g }, // Bengali
  { lang: "ta", re: /[஀-௿]/g }, // Tamil
];

export const SUPPORTED_LANGS = ["hi", "bn", "ta", "en"] as const;

/**
 * Returns a 2-letter code from SUPPORTED_LANGS. Falls back to "en" for text
 * in none of the Indic scripts, which is also the right answer for the
 * empty string -- L0's gibberish guard rejects that on content, and having
 * detection throw for it would just move the same refusal earlier and make
 * this function awkward to call.
 */
export function detectLang(text: string): string {
  let best = "en";
  let bestCount = 0;
  for (const { lang, re } of SCRIPT_RANGES) {
    const count = (text.match(re) ?? []).length;
    if (count > bestCount) {
      best = lang;
      bestCount = count;
    }
  }
  return best;
}

/**
 * Resolves the `lang` a client asked for. "auto" (or anything unsupported)
 * defers to the script; an explicit supported code is honoured as given, so
 * a user who deliberately picks a language in the UI still overrides
 * detection rather than fighting it.
 */
export function resolveLang(requested: string | undefined, text: string): string {
  if (requested && (SUPPORTED_LANGS as readonly string[]).includes(requested)) return requested;
  return detectLang(text);
}
