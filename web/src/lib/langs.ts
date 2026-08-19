export interface LangOption {
  code: string; // matches server contract's 2-char lang field
  label: string;
  nativeLabel: string;
  bcp47: string; // for Web Speech API / SpeechSynthesis
}

// "auto" leads and is the default: forcing a language on the STT call made
// English speech come back transcribed into Devanagari, because Sarvam
// honours an explicit language_code instead of detecting. With "auto" the
// code is omitted, Sarvam reports what it actually heard, and typed text
// falls to server-side script detection (krama/detect_lang.ts). The
// explicit entries remain as a deliberate override.
export const LANGUAGES: LangOption[] = [
  { code: "auto", label: "Auto-detect", nativeLabel: "auto", bcp47: "hi-IN" },
  { code: "hi", label: "Hindi", nativeLabel: "हिन्दी", bcp47: "hi-IN" },
  { code: "bn", label: "Bengali", nativeLabel: "বাংলা", bcp47: "bn-IN" },
  { code: "ta", label: "Tamil", nativeLabel: "தமிழ்", bcp47: "ta-IN" },
  { code: "en", label: "English", nativeLabel: "English", bcp47: "en-IN" },
];

export function langByCode(code: string): LangOption {
  return LANGUAGES.find((l) => l.code === code) ?? LANGUAGES[0];
}

/** True when the server should decide the language rather than the client. */
export function isAutoLang(code: string): boolean {
  return code === "auto";
}
