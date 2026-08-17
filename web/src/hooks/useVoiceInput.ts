import { useCallback, useRef, useState } from "react";
import { PcmCapture } from "../worklet/pcm";

export type VoiceInputState = "idle" | "listening" | "unsupported" | "denied";

interface UseVoiceInputOpts {
  bcp47: string;
  onFinalTranscript: (text: string) => void;
}

// Combines the real AudioWorklet PCM capture (E6.1 — feeds the waveform ring,
// and is the on-ramp for a future Sarvam WS connection) with the Web Speech
// API for live demo transcription today (see lib/speech.d.ts for why this
// stands in for Sarvam right now). The two are independent: PCM capture is
// real audio-pipeline plumbing, SpeechRecognition is just today's transcript
// source. Losing one doesn't break the other.
export function useVoiceInput({ bcp47, onFinalTranscript }: UseVoiceInputOpts) {
  const [state, setState] = useState<VoiceInputState>("idle");
  const [interimText, setInterimText] = useState("");
  // A ref, not state: onLevel fires roughly every audio render quantum
  // (>100/sec) -- routing that through React state would thrash re-renders.
  // Consumers read this directly in their own rAF loop (see components/Mic).
  const levelRef = useRef(0);

  const pcmRef = useRef<PcmCapture | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);

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

    try {
      const pcm = new PcmCapture();
      await pcm.start({
        onFrame: () => {
          // Real Int16 PCM frames land here — nowhere to send them yet
          // (server/stt/sarvam.ts's WS proxy doesn't exist), but the capture
          // path is genuinely wired and ready for it.
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
          onFinalTranscript(result[0].transcript);
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
