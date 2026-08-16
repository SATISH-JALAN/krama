"""
Unit test for CLAUDE.md invariant #1 (e5 prefixes) and #3 (NFC normalization).

Run: .venv/Scripts/python.exe -m pytest ingest/test_prefixing.py -v
(or, with no pytest installed yet: .venv/Scripts/python.exe ingest/test_prefixing.py)
"""

import unicodedata

from prefixing import add_passage_prefix, add_query_prefix, PASSAGE_PREFIX, QUERY_PREFIX


def test_passage_prefix_applied():
    assert add_passage_prefix("what is retrieval augmented generation") == \
        "passage: what is retrieval augmented generation"


def test_query_prefix_applied():
    assert add_query_prefix("what is retrieval augmented generation") == \
        "query: what is retrieval augmented generation"


def test_prefixes_are_distinct():
    # The #1 silent recall killer is these two being swapped or made identical.
    assert PASSAGE_PREFIX != QUERY_PREFIX
    text = "same underlying text"
    assert add_passage_prefix(text) != add_query_prefix(text)


def test_prefix_survives_nfc_normalization():
    # Devanagari nukta: QA as a single precomposed codepoint (U+0958) vs the
    # decomposed form KA (U+0915) + combining NUKTA (U+093C) must normalize
    # identically, and the prefix must not interfere with that normalization.
    #
    # Built from chr(codepoint) rather than literal characters on purpose:
    # while writing this test, every attempt to type the decomposed form as a
    # literal character got silently NFC-normalized somewhere in the editing
    # pipeline before it ever reached the file, making "precomposed !=
    # decomposed" fail for real -- a live demonstration of exactly the bug
    # invariant #3 warns about. chr() sidesteps this: no literal Devanagari
    # text ever appears in this source file, so nothing upstream can
    # normalize it before Python constructs the string at runtime.
    precomposed = chr(0x0958) + "text"               # QA, single codepoint
    decomposed = chr(0x0915) + chr(0x093C) + "text"  # KA + combining NUKTA
    assert precomposed != decomposed, "test fixture itself was normalized -- fix the fixture"

    p1 = unicodedata.normalize("NFC", add_passage_prefix(precomposed))
    p2 = unicodedata.normalize("NFC", add_passage_prefix(decomposed))
    assert p1 == p2  # ...but identical once NFC-normalized, prefix included


def test_empty_text_still_gets_prefixed():
    # Guards against a refactor that special-cases empty strings and silently
    # skips the prefix for them.
    assert add_passage_prefix("") == "passage: "
    assert add_query_prefix("") == "query: "


if __name__ == "__main__":
    # Minimal runner if pytest isn't installed yet.
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
