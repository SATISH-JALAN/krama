import { useCallback, useState } from "react";
import Header from "./components/Header";
import Mic from "./components/Mic";
import Transcript from "./components/Transcript";
import Answer from "./components/Answer";
import Waterfall from "./components/Waterfall";
import Footer from "./components/Footer";
import { useVoiceInput } from "./hooks/useVoiceInput";
import { queryBackend, queryVoice, querySynthesize } from "./lib/api";
import { langByCode } from "./lib/langs";
import { totalTraceMs, type GroundedAnswer } from "./lib/contracts";
import "./App.css";

// Mirrors handleSynthesisQuery()'s own eligibility rule. A grounded answer
// gets rewritten more naturally; an off_topic/no_grounding refusal gets a
// labelled general-knowledge answer instead of a dead end; gibberish and
// unsafe input get neither.
function synthesisWorthTrying(data: GroundedAnswer): boolean {
  if (!data.refused) return true;
  return data.refusalReason === "off_topic" || data.refusalReason === "no_grounding";
}

export default function App() {
  const [lang, setLang] = useState("hi");
  const [queryText, setQueryText] = useState("");
  const [result, setResult] = useState<GroundedAnswer | null>(null);
  const [source, setSource] = useState<"live" | "mock" | null>(null);
  const [busy, setBusy] = useState(false);
  const [synthesizing, setSynthesizing] = useState(false);
  // Distinct from "synthesizing" (in flight) and from a successful
  // result landing on `result.synthesized` -- this is only set once an
  // attempt finishes without a usable answer, so the UI can say WHY
  // instead of a single generic "unavailable" for every different reason.
  const [synthesizeStatus, setSynthesizeStatus] = useState<"not_configured" | "failed" | null>(null);

  // Fires after the fast answer already landed, never blocks it (CLAUDE.md
  // #4 -- LLM synthesis stays off the t0->t1 core budget entirely).
  //
  // A refused query is still worth sending: handleSynthesisQuery() answers
  // off_topic/no_grounding refusals from the model's own general knowledge
  // (returned with grounded=false) instead of dead-ending, because the
  // corpus genuinely has no passage for plenty of reasonable questions.
  // Mirrors the server's own eligibility rule -- gibberish and unsafe input
  // stay refused outright and are never handed to an LLM.
  const triggerSynthesis = useCallback(
    async (text: string, queryLang: string) => {
      setSynthesizing(true);
      setSynthesizeStatus(null);
      const outcome = await querySynthesize(text, queryLang);
      setSynthesizing(false);
      if (outcome.status === "ok" && outcome.data.synthesized) {
        setResult((prev) => (prev ? { ...prev, synthesized: outcome.data.synthesized! } : prev));
      } else if (outcome.status === "not_configured") {
        setSynthesizeStatus("not_configured");
      } else {
        // "ok" with synthesized === null (every provider failed, e.g. a
        // real 429 quota exhaustion hit during testing) and "failed"
        // (network error/timeout) both read the same to a user: it didn't
        // work this time, not "it's not set up here".
        setSynthesizeStatus("failed");
      }
    },
    [],
  );

  const submit = useCallback(
    async (text: string) => {
      if (!text.trim() || busy) return;
      setBusy(true);
      setSynthesizeStatus(null);
      const { data, source: src } = await queryBackend(text, lang);
      setResult(data);
      setSource(src);
      setBusy(false);
      // The mock responder has no LLM behind it, so only ask a live server.
      if (src === "live" && synthesisWorthTrying(data)) void triggerSynthesis(text, lang);
    },
    [lang, busy, triggerSynthesis],
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
      setSynthesizeStatus(null);
      const voiceResult = await queryVoice(pcm, lang);
      if (voiceResult) {
        setQueryText(voiceResult.transcript);
        setResult(voiceResult);
        setSource("live");
        setBusy(false);
        if (synthesisWorthTrying(voiceResult))
          void triggerSynthesis(voiceResult.transcript, voiceResult.detectedLang);
      } else {
        setQueryText(webSpeechText);
        setBusy(false);
        void submit(webSpeechText);
      }
    },
    [lang, busy, submit, triggerSynthesis],
  );

  // useVoiceInput already stops/cleans up the mic+recognition before
  // calling this (silence auto-timeout or a manual tap both go through the
  // same finalize path) -- nothing left to tear down here, just submit.
  const voice = useVoiceInput({
    bcp47: langByCode(lang).bcp47,
    onFinalTranscript: (text, pcm) => {
      void submitVoice(text, pcm);
    },
  });

  return (
    <div className="app">
      <Header
        lang={lang}
        onLangChange={setLang}
        source={source}
        latencyMs={totalTraceMs(result?.trace)}
        cached={result?.cached}
      />

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

        <Answer
          result={result}
          busy={busy}
          synthesizing={synthesizing}
          synthesizeStatus={synthesizeStatus}
          lang={lang}
        />

        <Waterfall trace={result?.trace} />
      </main>

      <Footer />
    </div>
  );
}
