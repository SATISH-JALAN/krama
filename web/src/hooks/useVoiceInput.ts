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

// ~4.5s of continuous silence (measured off the real mic RMS, not the Web
// Speech API's own end-of-speech detection) auto-stops and submits the
// utterance -- tap-to-talk, not tap-to-talk-then-tap-again. A hard cap
// backstops it in case something holds the level artificially high (e.g.
// a noisy room) and silence never registers.
const SILENCE_TIMEOUT_MS = 4_500;
const SILENCE_RMS_THRESHOLD = 0.02;
const SILENCE_CHECK_INTERVAL_MS = 200;
const MAX_UTTERANCE_MS = 20_000;

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
// queryBackend mock path can't take audio) -- but it is NOT what decides
// when an utterance ends. Web Speech's continuous+interimResults "isFinal"
// event is inconsistent across browsers (sometimes never fires, sometimes
// fires mid-sentence), which is what made the mic feel unreliable. Ending
// an utterance is driven entirely by real mic silence (or a manual tap)
// instead, so it behaves the same regardless of Web Speech's mood, and
// still works at all in browsers that don't implement Web Speech.
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
  const interimTextRef = useRef("");
  const lastLoudAtRef = useRef(0);
  const startedAtRef = useRef(0);
  const silenceTimerRef = useRef<number | null>(null);
  const finalizedRef = useRef(false);

  const clearSilenceTimer = () => {
    if (silenceTimerRef.current !== null) {
      window.clearInterval(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
  };

  // Releases mic/recognition resources without submitting anything --
  // permission-denied and internal cleanup paths.
  const cleanup = useCallback(() => {
    clearSilenceTimer();
    pcmRef.current?.stop();
    pcmRef.current = null;
    if (recognitionRef.current) {
      // Detach handlers before stop() -- SpeechRecognition.stop() fires
      // onend asynchronously, which would otherwise re-enter this cleanup.
      recognitionRef.current.onresult = null;
      recognitionRef.current.onerror = null;
      recognitionRef.current.onend = null;
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }
    setState("idle");
    setInterimText("");
    interimTextRef.current = "";
    levelRef.current = 0;
  }, []);

  // Ends the utterance and hands off whatever was captured -- shared by
  // manual tap-to-stop and the silence auto-timeout, so both behave
  // identically instead of only the (unreliable) Web Speech path ever
  // actually submitting anything.
  const finalize = useCallback(() => {
    if (finalizedRef.current) return;
    finalizedRef.current = true;
    const pcm = concatInt16(framesRef.current);
    const text = interimTextRef.current;
    cleanup();
    if (pcm.length > 0) onFinalTranscript(text, pcm);
  }, [cleanup, onFinalTranscript]);

  const stop = useCallback(() => {
    if (state === "listening") finalize();
    else cleanup();
  }, [state, finalize, cleanup]);

  const start = useCallback(async () => {
    const SpeechRecognitionCtor = window.SpeechRecognition ?? window.webkitSpeechRecognition;

    framesRef.current = [];
    interimTextRef.current = "";
    finalizedRef.current = false;
    startedAtRef.current = performance.now();
    lastLoudAtRef.current = performance.now();

    try {
      const pcm = new PcmCapture();
      await pcm.start({
        onFrame: (frame) => {
          framesRef.current.push(frame);
        },
        onLevel: (rms) => {
          levelRef.current = rms;
          if (rms > SILENCE_RMS_THRESHOLD) lastLoudAtRef.current = performance.now();
        },
      });
      pcmRef.current = pcm;
    } catch {
      setState("denied");
      return;
    }

    silenceTimerRef.current = window.setInterval(() => {
      const now = performance.now();
      const silentFor = now - lastLoudAtRef.current;
      const totalFor = now - startedAtRef.current;
      if (silentFor >= SILENCE_TIMEOUT_MS || totalFor >= MAX_UTTERANCE_MS) {
        finalize();
      }
    }, SILENCE_CHECK_INTERVAL_MS);

    // Web Speech is optional now -- it only drives live interim captions.
    // Its absence (Firefox, some mobile browsers) no longer blocks voice
    // input entirely: the real PCM capture + silence auto-stop above still
    // drive the real Sarvam STT path on their own.
    if (SpeechRecognitionCtor) {
      const recognition = new SpeechRecognitionCtor();
      recognition.lang = bcp47;
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.onresult = (event) => {
        let interim = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          interim += event.results[i][0].transcript;
        }
        if (interim) {
          interimTextRef.current = interim;
          setInterimText(interim);
        }
      };
      // A recognition error shouldn't kill the real PCM/Sarvam path -- just
      // drop live captions for the rest of this utterance.
      recognition.onerror = () => {
        recognitionRef.current = null;
      };
      recognition.onend = () => {
        recognitionRef.current = null;
      };
      recognitionRef.current = recognition;
      recognition.start();
    }

    setState("listening");
  }, [bcp47, finalize]);

  return { state, interimText, levelRef, start, stop };
}
