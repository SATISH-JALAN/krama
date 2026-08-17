import { useCallback, useState } from "react";
import Header from "./components/Header";
import Mic from "./components/Mic";
import Transcript from "./components/Transcript";
import Answer from "./components/Answer";
import Waterfall from "./components/Waterfall";
import Footer from "./components/Footer";
import { useVoiceInput } from "./hooks/useVoiceInput";
import { queryBackend } from "./lib/api";
import { langByCode } from "./lib/langs";
import type { GroundedAnswer } from "./lib/contracts";
import "./App.css";

export default function App() {
  const [lang, setLang] = useState("hi");
  const [queryText, setQueryText] = useState("");
  const [result, setResult] = useState<GroundedAnswer | null>(null);
  const [source, setSource] = useState<"live" | "mock" | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = useCallback(
    async (text: string) => {
      if (!text.trim() || busy) return;
      setBusy(true);
      const { data, source: src } = await queryBackend(text, lang);
      setResult(data);
      setSource(src);
      setBusy(false);
    },
    [lang, busy],
  );

  const voice = useVoiceInput({
    bcp47: langByCode(lang).bcp47,
    onFinalTranscript: (text) => {
      setQueryText(text);
      voice.stop();
      void submit(text);
    },
  });

  return (
    <div className="app">
      <Header lang={lang} onLangChange={setLang} source={source} />

      <main className="stage">
        <Mic
          state={voice.state}
          levelRef={voice.levelRef}
          busy={busy}
          onToggle={() => (voice.state === "listening" ? voice.stop() : void voice.start())}
        />

        <Transcript
          value={queryText}
          interimText={voice.interimText}
          onChange={setQueryText}
          onSubmit={() => void submit(queryText)}
          disabled={busy}
        />

        <Answer result={result} busy={busy} />

        <Waterfall trace={result?.trace} />
      </main>

      <Footer />
    </div>
  );
}
