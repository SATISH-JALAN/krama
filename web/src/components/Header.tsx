import LanguageSelect from "./LanguageSelect";
import "./Header.css";

interface Props {
  lang: string;
  onLangChange: (code: string) => void;
  source: "live" | "mock" | null;
}

export default function Header({ lang, onLangChange, source }: Props) {
  return (
    <header className="header">
      <div className="header-brand">
        <span className="header-wordmark mono">KRAMA</span>
        <span className="header-tagline">voice → grounded answer, &lt;200ms core</span>
      </div>
      <div className="header-controls">
        <span className={`source-badge mono ${source ?? "unset"}`}>
          <span className="source-dot" aria-hidden />
          {source === "live" ? "live server" : source === "mock" ? "mock demo" : "not yet queried"}
        </span>
        <LanguageSelect value={lang} onChange={onLangChange} />
      </div>
    </header>
  );
}
