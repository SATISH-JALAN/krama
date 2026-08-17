// Minimal ambient types for the Web Speech API (not in lib.dom.d.ts).
// This is ARCHITECTURE.md §7's documented STT fallback chain (Sarvam ->
// browser Web Speech API), used here as the live demo transcription path
// since server/stt/sarvam.ts's WS proxy doesn't exist yet — a real fallback
// already specified in the architecture, not a stand-in invented for the UI.

interface SpeechRecognitionResultLike {
  isFinal: boolean;
  0: { transcript: string };
}

interface SpeechRecognitionEventLike extends Event {
  resultIndex: number;
  results: ArrayLike<SpeechRecognitionResultLike>;
}

interface SpeechRecognitionLike extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  start(): void;
  stop(): void;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: Event) => void) | null;
  onend: (() => void) | null;
}

interface Window {
  SpeechRecognition?: new () => SpeechRecognitionLike;
  webkitSpeechRecognition?: new () => SpeechRecognitionLike;
}
