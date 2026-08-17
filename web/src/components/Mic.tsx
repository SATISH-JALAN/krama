import { useEffect, useRef, type RefObject } from "react";
import type { VoiceInputState } from "../hooks/useVoiceInput";
import "./Mic.css";

interface Props {
  state: VoiceInputState;
  levelRef: RefObject<number>;
  busy: boolean;
  onToggle: () => void;
}

const STATUS_LABEL: Record<VoiceInputState, string> = {
  idle: "tap to speak",
  listening: "listening…",
  denied: "mic access denied — type below instead",
  unsupported: "speech recognition unsupported here — type below instead",
};

export default function Mic({ state, levelRef, busy, onToggle }: Props) {
  const ringRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let raf: number;
    const tick = () => {
      if (ringRef.current) {
        // RMS of mic input is typically small (~0.01-0.3); scale + clamp so
        // normal speech visibly moves the ring without needing to shout.
        const scaled = Math.min(1, (levelRef.current ?? 0) * 6);
        ringRef.current.style.setProperty("--level", scaled.toFixed(3));
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [levelRef]);

  const disabled = state === "unsupported" || busy;

  return (
    <div className="mic-stage">
      <div className={`mic-ring ${state}`} ref={ringRef}>
        <div className="mic-ring-halo" aria-hidden />
        <button
          className="mic-button"
          onClick={onToggle}
          disabled={disabled}
          aria-pressed={state === "listening"}
          aria-label={state === "listening" ? "Stop listening" : "Start listening"}
        >
          <MicIcon />
        </button>
      </div>
      <p className="mic-status mono">{busy ? "retrieving…" : STATUS_LABEL[state]}</p>
    </div>
  );
}

function MicIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" aria-hidden>
      <rect x="9" y="2" width="6" height="12" rx="3" fill="currentColor" />
      <path
        d="M5 11a7 7 0 0 0 14 0M12 18v3"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  );
}
