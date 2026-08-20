import { useEffect, useRef } from "react";
import "./Transcript.css";

interface Props {
  value: string;
  interimText: string;
  onChange: (text: string) => void;
  onSubmit: () => void;
  disabled: boolean;
}

export default function Transcript({ value, interimText, onChange, onSubmit, disabled }: Props) {
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (interimText) onChange(interimText);
  }, [interimText, onChange]);

  return (
    <form
      className="transcript"
      onSubmit={(e) => {
        e.preventDefault();
        if (value.trim()) onSubmit();
      }}
    >
      <textarea
        ref={inputRef}
        className={`transcript-input ${interimText ? "interim" : ""}`}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="or type a question here…"
        rows={1}
        disabled={disabled}
      />
      <button type="submit" className="transcript-submit btn-hh" disabled={disabled || !value.trim()}>
        ask →
      </button>
    </form>
  );
}
