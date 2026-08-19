import { describe, expect, test } from "bun:test";
import { romanize, romanizationLabel } from "./romanize";

// Roman-only means no leftover source-script Unicode ranges -- catches the
// silent-passthrough failure mode this module exists to avoid.
const DEVANAGARI_RANGE = /[ऀ-ॿ]/;
const BENGALI_RANGE = /[ঀ-৿]/;
const TAMIL_RANGE = /[஀-௿]/;

describe("romanize", () => {
  test("hindi: produces roman-only output", () => {
    const out = romanize("निगम एक कानूनी इकाई है।", "hi");
    expect(out).not.toBeNull();
    expect(out).not.toMatch(DEVANAGARI_RANGE);
    expect(out).toContain("nigama");
  });

  test("bengali: produces roman-only output", () => {
    const out = romanize("একটি কর্পোরেশন একটি আইনি সত্তা।", "bn");
    expect(out).not.toBeNull();
    expect(out).not.toMatch(BENGALI_RANGE);
  });

  test("bengali: handles YYA (য়, base+nukta, two codepoints) without leaking the nukta", () => {
    // sanscript recognizes this cluster in Devanagari (the pivot script) but
    // not reliably when Bengali is the *source* script -- caught by testing
    // against a real retrieved passage, not a hand-picked example (the first
    // version of this test used a string that didn't happen to contain এ২)।
    const out = romanize("ব্যবসায়িক প্রতিষ্ঠান", "bn");
    expect(out).not.toBeNull();
    expect(out).not.toMatch(BENGALI_RANGE);
    expect(out).not.toContain("়");
  });

  test("tamil: produces roman-only output, including the dental na (ன) regression case", () => {
    // நிறுவனம் ("company"/"organization") -- sanscript's stock roman
    // schemes drop this letter entirely (see this module's own docstring);
    // this is the exact case that caught it before shipping.
    const out = romanize("நிறுவனம் என்பது ஒரு சட்ட நிறுவனம்.", "ta");
    expect(out).not.toBeNull();
    expect(out).not.toMatch(TAMIL_RANGE);
    expect(out).toContain("niRuvanam");
  });

  test("english: nothing to romanize, returns null", () => {
    expect(romanize("what is a corporation", "en")).toBeNull();
  });

  test("empty text returns null", () => {
    expect(romanize("", "hi")).toBeNull();
    expect(romanize("   ", "hi")).toBeNull();
  });

  test("unknown lang returns null", () => {
    expect(romanize("some text", "xx")).toBeNull();
  });

  test("swaps the Devanagari/Bengali danda for a period", () => {
    expect(romanize("यह एक वाक्य है।", "hi")).not.toContain("।");
    expect(romanize("यह एक वाक्य है।", "hi")).toContain(".");
  });
});

describe("romanizationLabel", () => {
  test("returns the per-language casual label", () => {
    expect(romanizationLabel("hi")).toBe("Hinglish");
    expect(romanizationLabel("bn")).toBe("Banglish");
    expect(romanizationLabel("ta")).toBe("Tanglish");
  });

  test("returns null for english and unknown langs", () => {
    expect(romanizationLabel("en")).toBeNull();
    expect(romanizationLabel("xx")).toBeNull();
  });
});
