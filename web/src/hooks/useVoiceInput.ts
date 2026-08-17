import { useCallback, useRef, useState } from "react";
import { PcmCapture } from "../worklet/pcm";

export type VoiceInputState = "idle" | "listening" | "unsupported" | "denied";

interface UseVoiceInputOpts {
  bcp47: string;
  // Called once per utterance with BOTH the Web Speech transcript (for a
  // no-server/mock fallback and instant interim display) AND the real
  // accumulated Int16 PCM for the same utterance (for the real Sarvam STT
  // path via /query/voice, server/stt/sarvam.ts). The caller decides which
  // one actually drives the answer -- see App.tsx's submitVoice.
  onFinalTranscript: (text: string, pcm: Int16Array) => void;
}

function concatInt16(chunks: Int16Array[]): Int16Array {
  const total = chunks.reduce((sum, c) => sum + c.length, 0);
  const out = new Int16Array(total);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}

// Combines the real AudioWorklet PCM capture (E6.1) with the Web Speech API.
// PCM frames are accumulated for the whole utterance and hand off to real
// Sarvam batch STT server-side (server/stt/sarvam.ts via /query/voice) --
// batch, not streaming, matching the project's approved STT decision.
// Web Speech's own transcript is kept too, purely as interim captions and
// as the fallback source when no live server is configured (App.tsx's
// queryBackend mock path can't take audio).
export function useVoiceInput({ bcp47, onFinalTranscript }: UseVoiceInputOpts) {
  const [state, setState] = useState<VoiceInputState>("idle");
  const [interimText, setInterimText] = useState("");
  // A ref, not state: onLevel fires roughly every audio render quantum
  // (>100/sec) -- routing that through React state would thrash re-renders.
  // Consumers read this directly in their own rAF loop (see components/Mic).
  const levelRef = useRef(0);

  const pcmRef = useRef<PcmCapture | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const framesRef = useRef<Int16Array[]>([]);

  const stop = useCallback(() => {
    pcmRef.current?.stop();
    pcmRef.current = null;
    recognitionRef.current?.stop();
    recognitionRef.current = null;
    setState("idle");
    levelRef.current = 0;
  }, []);

  const start = useCallback(async () => {
    const SpeechRecognitionCtor = window.SpeechRecognition ?? window.webkitSpeechRecognition;

    framesRef.current = [];
    try {
      const pcm = new PcmCapture();
      await pcm.start({
        onFrame: (frame) => {
          framesRef.current.push(frame);
        },
        onLevel: (rms) => {
          levelRef.current = rms;
        },
      });
      pcmRef.current = pcm;
    } catch {
      setState("denied");
      return;
    }

    if (!SpeechRecognitionCtor) {
      setState("unsupported");
      return;
    }

    const recognition = new SpeechRecognitionCtor();
    recognition.lang = bcp47;
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.onresult = (event) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          onFinalTranscript(result[0].transcript, concatInt16(framesRef.current));
          setInterimText("");
        } else {
          interim += result[0].transcript;
        }
      }
      if (interim) setInterimText(interim);
    };
    recognition.onerror = () => setState("denied");
    recognition.onend = () => {
      if (recognitionRef.current === recognition) stop();
    };
    recognitionRef.current = recognition;
    recognition.start();
    setState("listening");
  }, [bcp47, onFinalTranscript, stop]);

  return { state, interimText, levelRef, start, stop };
}
