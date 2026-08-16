import { describe, expect, test } from "bun:test";
import { Bm25Index, tokenize } from "./bm25";

describe("tokenize", () => {
  test("segments English text into lowercase word tokens", () => {
    expect(tokenize("What is a Corporation?", "en")).toEqual([
      "what",
      "is",
      "a",
      "corporation",
    ]);
  });

  test("NFC-normalizes before tokenizing so nukta variants match", () => {
    // QA as precomposed (U+0958) vs KA+combining-NUKTA (U+0915 U+093C) --
    // built via chr()-equivalent (String.fromCodePoint) so the two really
    // are different raw input, not silently normalized before the test runs
    // (same class of pitfall documented in ingest/test_prefixing.py).
    const precomposed = String.fromCodePoint(0x0958) + "ी";
    const decomposed = String.fromCodePoint(0x0915) + String.fromCodePoint(0x093c) + "ी";
    expect(precomposed).not.toBe(decomposed);
    expect(tokenize(precomposed, "hi")).toEqual(tokenize(decomposed, "hi"));
  });

  test("strips ZWJ/ZWNJ", () => {
    const withZwnj = "क" + String.fromCodePoint(0x200c) + "ल";
    const without = "कल";
    expect(tokenize(withZwnj, "hi")).toEqual(tokenize(without, "hi"));
  });

  test("segments Devanagari without whitespace-splitting garbage", () => {
    const tokens = tokenize("कॉर्पोरेशन क्या है?", "hi");
    expect(tokens.length).toBeGreaterThan(0);
    expect(tokens).not.toContain("");
  });
});

describe("Bm25Index", () => {
  const docs = [
    { id: "d1", text: "A corporation is a legal entity separate from its owners.", lang: "en" },
    { id: "d2", text: "Bananas are a good source of potassium and fiber.", lang: "en" },
    { id: "d3", text: "A company can be incorporated in a specific state or country.", lang: "en" },
  ];

  test("ranks the lexically relevant document above an unrelated one", () => {
    const idx = new Bm25Index();
    idx.build(docs);
    const results = idx.search("corporation legal entity", "en", 10);
    expect(results.length).toBeGreaterThan(0);
    expect(results[0].id).toBe("d1");
    const bananaResult = results.find((r) => r.id === "d2");
    // banana doc shares no terms with the query -- should score 0 and thus
    // never appear in results at all (only docs with >=1 matching term score)
    expect(bananaResult).toBeUndefined();
  });

  test("returns empty for an empty index", () => {
    const idx = new Bm25Index();
    idx.build([]);
    expect(idx.search("anything", "en")).toEqual([]);
    expect(idx.size).toBe(0);
  });

  test("respects topK", () => {
    const idx = new Bm25Index();
    idx.build(docs);
    const results = idx.search("a", "en", 1);
    expect(results.length).toBeLessThanOrEqual(1);
  });

  test("size reflects the built corpus", () => {
    const idx = new Bm25Index();
    idx.build(docs);
    expect(idx.size).toBe(3);
  });

  test("cross-lingual query barely matches (ARCHITECTURE.md §5.2 expectation)", () => {
    const idx = new Bm25Index();
    idx.build(docs); // English-only corpus
    const results = idx.search("निगम कानूनी इकाई", "hi"); // Hindi query, same meaning as d1
    // No lexical overlap possible -- Hindi tokens never equal English tokens,
    // so this MUST return nothing. This is the documented division of labor:
    // BM25 covers same-language exact matches, dense embeddings cover
    // cross-lingual. If this test ever starts returning results, something
    // is wrong with tokenization (e.g. over-aggressive normalization
    // collapsing distinct scripts together).
    expect(results).toEqual([]);
  });
});
