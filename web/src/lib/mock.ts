// Demo-mode responder, used only when no live server answers at VITE_API_URL
// (index.ts can't boot yet without a real corpus + HNSW artifacts — see
// server/index.ts's own docstring). Every value here is fabricated for the
// UI to have something honest-looking to render — CLAUDE.md invariant #6
// ("never invent a benchmark number") means this must stay visibly labelled
// as mock in the UI, never presented as a measured result.
import type { GroundedAnswer, Span } from "./contracts";

const MOCK_PASSAGES: Record<string, { answer: string; citation: string }> = {
  hi: {
    answer:
      "कॉर्पोरेशन एक कानूनी इकाई है जो अपने मालिकों से अलग होती है, जिसे राज्य द्वारा एक चार्टर के माध्यम से बनाया जाता है और जिसे स्वतंत्र रूप से अनुबंध करने, संपत्ति रखने और मुकदमा करने या मुकदमा किए जाने का अधिकार होता है।",
    citation: "chunk_7f19a2",
  },
  bn: {
    answer:
      "কর্পোরেশন হলো একটি আইনি সত্তা যা তার মালিকদের থেকে পৃথক, রাষ্ট্রের একটি সনদের মাধ্যমে গঠিত, এবং স্বাধীনভাবে চুক্তি করার, সম্পত্তি রাখার এবং মামলা করার বা মামলার শিকার হওয়ার অধিকার রাখে।",
    citation: "chunk_3c8e91",
  },
  ta: {
    answer:
      "நிறுவனம் என்பது அதன் உரிமையாளர்களிடமிருந்து தனித்துவமான ஒரு சட்ட நிறுவனம் ஆகும், இது அரசால் ஒரு சாசனத்தின் மூலம் உருவாக்கப்படுகிறது, மேலும் இது சுயாதீனமாக ஒப்பந்தம் செய்யவும், சொத்து வைத்திருக்கவும், வழக்கு தொடரவும் அல்லது வழக்கு தொடரப்படவும் உரிமை உடையது.",
    citation: "chunk_9b0d44",
  },
  en: {
    answer:
      "A corporation is a legal entity separate from its owners, created through a charter granted by the state, with the independent right to contract, hold property, and sue or be sued.",
    citation: "chunk_1a4f02",
  },
};

function jitter(base: number, spread: number): number {
  return Math.round((base + (Math.random() - 0.5) * spread) * 10) / 10;
}

function buildTrace(): Span[] {
  const stages: [string, number][] = [
    ["l0_input_guard", jitter(0.4, 0.3)],
    ["embed_query", jitter(9, 4)],
    ["l1_safety_guard", jitter(0.3, 0.2)],
    ["hnsw_search", jitter(3.5, 2)],
    ["bm25_search", jitter(5, 3)],
    ["l2_ood_guard", jitter(0.2, 0.1)],
    ["fuse_rrf", jitter(0.6, 0.4)],
    ["extract_span", jitter(1.2, 0.8)],
  ];
  return stages.map(([name, ms]) => ({ name, ms: Math.max(0.1, ms), ok: true }));
}

export async function mockQuery(text: string, lang: string): Promise<GroundedAnswer> {
  // Simulate the actual t0->t1 wall-clock a real fast path would take, so the
  // waterfall's total roughly matches the artificial delay below.
  const trace = buildTrace();
  const coreMs = trace.reduce((sum, s) => sum + s.ms, 0);
  await new Promise((r) => setTimeout(r, coreMs));

  const trimmed = text.trim();
  if (trimmed.length === 0) {
    return { answer: "", citations: [], confidence: 0, refused: true, refusalReason: "empty_or_gibberish" };
  }

  const passage = MOCK_PASSAGES[lang] ?? MOCK_PASSAGES.en;
  return {
    answer: passage.answer,
    citations: [passage.citation],
    confidence: Math.round((0.72 + Math.random() * 0.2) * 100) / 100,
    refused: false,
    trace,
    synthesized: {
      answer: passage.answer + (lang === "en" ? " (Synthesized, off the critical path.)" : ""),
      streaming: false,
    },
  };
}
