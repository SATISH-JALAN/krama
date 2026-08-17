import { useCallback, useState } from "react";
import Header from "./components/Header";
import Mic from "./components/Mic";
import Transcript from "./components/Transcript";
import Answer from "./components/Answer";
import Waterfall from "./components/Waterfall";
import Footer from "./components/Footer";
import { useVoiceInput } from "./hooks/useVoiceInput";
import { queryBackend, queryVoice } from "./lib/api";
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

  // Real path: send the whole utterance's PCM to /query/voice (real Sarvam
  // batch STT server-side). Only falls back to the Web Speech transcript
  // (via the text-only submit() above, which itself degrades to the mock
  // responder) when no live server is configured or the voice route
  // fails/times out -- there's no audio-capable mock to fall back to.
  const submitVoice = useCallback(
    async (webSpeechText: string, pcm: Int16Array) => {
      if (busy) return;
      setBusy(true);
      const voiceResult = await queryVoice(pcm, lang);
      if (voiceResult) {
        setQueryText(voiceResult.transcript);
        setResult(voiceResult);
        setSource("live");
        setBusy(false);
      } else {
        setQueryText(webSpeechText);
        setBusy(false);
        void submit(webSpeechText);
      }
    },
    [lang, busy, submit],
  );

  const voice = useVoiceInput({
    bcp47: langByCode(lang).bcp47,
    onFinalTranscript: (text, pcm) => {
      voice.stop();
      void submitVoice(text, pcm);
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
