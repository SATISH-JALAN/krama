export interface LangOption {
  code: string; // matches server contract's 2-char lang field
  label: string;
  nativeLabel: string;
  bcp47: string; // for Web Speech API / SpeechSynthesis
}

export const LANGUAGES: LangOption[] = [
  { code: "hi", label: "Hindi", nativeLabel: "हिन्दी", bcp47: "hi-IN" },
  { code: "bn", label: "Bengali", nativeLabel: "বাংলা", bcp47: "bn-IN" },
  { code: "ta", label: "Tamil", nativeLabel: "தமிழ்", bcp47: "ta-IN" },
  { code: "en", label: "English", nativeLabel: "English", bcp47: "en-IN" },
];

export function langByCode(code: string): LangOption {
  return LANGUAGES.find((l) => l.code === code) ?? LANGUAGES[0];
}
