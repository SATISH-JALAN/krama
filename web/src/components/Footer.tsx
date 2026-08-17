import { MODULE_GLOSSARY } from "../lib/contracts";
import "./Footer.css";

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-glossary">
        {Object.entries(MODULE_GLOSSARY).map(([name, meaning]) => (
          <span key={name} className="glossary-item">
            <span className="glossary-name mono">{name}</span>
            <span className="glossary-meaning">{meaning}</span>
          </span>
        ))}
      </div>
      <p className="footer-note mono">HH Goa 2026 · shortlisting task 2 · built solo, zero budget</p>
    </footer>
  );
}
