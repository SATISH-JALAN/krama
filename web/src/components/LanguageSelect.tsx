import { LANGUAGES } from "../lib/langs";
import "./LanguageSelect.css";

interface Props {
  value: string;
  onChange: (code: string) => void;
}

export default function LanguageSelect({ value, onChange }: Props) {
  return (
    <label className="lang-select">
      <span className="eyebrow">lang</span>
      <select value={value} onChange={(e) => onChange(e.target.value)} aria-label="Query language">
        {LANGUAGES.map((l) => (
          <option key={l.code} value={l.code}>
            {l.nativeLabel}
          </option>
        ))}
      </select>
    </label>
  );
}
