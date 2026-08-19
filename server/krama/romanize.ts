/**
 * Deterministic script-to-Roman transliteration for the non-English answer
 * languages (hi/bn/ta) -- NOT translation. Same words, same meaning, just
 * written in Latin letters, matching how users already type "Hinglish" /
 * "Banglish" / "Tanglish". Built on @indic-transliteration/sanscript
 * (devanagari/bengali/tamil -> a custom casual roman scheme derived from
 * its stock "itrans_lowercase").
 *
 * Real bug found and worked around before shipping this, not assumed
 * correct: sanscript's stock roman schemes silently DROP one common Tamil
 * letter -- 'ன' (dental/alveolar na, pivot char Devanagari 'ऩ', appears in
 * everyday words like நிறுவனம் "company"/"organization") -- because their
 * `extra_consonants` group only maps that pivot char for the Dravidian-
 * specific schemes ("n2" notation), not the casual lowercase-itrans one.
 * Verified directly (`நிறுவனம்` round-tripped to `niRuவனm` with the raw
 * Tamil character leaking straight through unconverted) before patching it
 * here to map to plain "n" -- the same letter Tamil's other nasal 'ந'
 * already gets, since English doesn't distinguish them either and that's
 * how this is actually typed casually.
 */
import Sanscript from "@indic-transliteration/sanscript";

const ROMAN_SCHEME_NAME = "krama_hinglish";
let schemeRegistered = false;

function ensureScheme(): void {
  if (schemeRegistered) return;
  const base = JSON.parse(JSON.stringify(Sanscript.schemes.itrans_lowercase));
  base.extra_consonants["ऩ"] = "n";
  Sanscript.addRomanScheme(ROMAN_SCHEME_NAME, base);
  schemeRegistered = true;
}

const SOURCE_SCHEME: Record<string, string> = {
  hi: "devanagari",
  bn: "bengali",
  ta: "tamil",
};

// Defense in depth, not just the 'ऩ' patch above: sanscript transliterates
// through Devanagari as an internal pivot, and at least one other real gap
// was found the same way -- Bengali "য়" (YA + a combining nukta, U+09BC,
// two codepoints, not one) isn't recognized as the single cluster its own
// `extra_consonants` table maps, leaving the bare nukta mark in the output
// ("vyavasaaya়ika" instead of "vyavasaayika"). Rather than chase every such
// gap character-by-character, strip any codepoint still in a source script's
// Unicode block after transliteration -- a stray original-script char in
// otherwise-Roman output is always a bug, never intentional.
const LEFTOVER_SOURCE_SCRIPT = /[ऀ-ॿঀ-৿஀-௿]/g;

const ROMANIZATION_LABEL: Record<string, string> = {
  hi: "Hinglish",
  bn: "Banglish",
  ta: "Tanglish",
};

export function romanizationLabel(lang: string): string | null {
  return ROMANIZATION_LABEL[lang] ?? null;
}

/** Returns null for English (nothing to romanize) or empty input. */
export function romanize(text: string, lang: string): string | null {
  const scheme = SOURCE_SCHEME[lang];
  if (!scheme || !text.trim()) return null;
  ensureScheme();
  const out = Sanscript.t(text, scheme, ROMAN_SCHEME_NAME);
  // Sanscript itself (not left as source-script text) renders the
  // Devanagari/Bengali sentence-final danda ("।"/double "॥") as ASCII "|"/
  // "||" -- not meaningfully "roman" to a reader, swap for the punctuation
  // a Latin-alphabet reader actually expects.
  const punctuated = out.replace(/\|\|/g, ".").replace(/\|/g, ".");
  return punctuated.replace(LEFTOVER_SOURCE_SCRIPT, "").replace(/ {2,}/g, " ").trim();
}
