"""Tiny stdlib HTTP client for the Bun eval adapter.

urllib rather than `requests` on purpose: this module is imported into the eval
suite's virtualenv, and adding a dependency there is a way to make someone
else's `pip install -r requirements.txt` fail for a reason that has nothing to
do with the evaluation.
"""

import json
import os
import urllib.error
import urllib.request

BASE_URL = os.environ.get("KRAMA_EVAL_ADAPTER_URL", "http://127.0.0.1:3100")

# Generation goes out to a real LLM with a retry chain behind it
# (llm/chain.ts's fallback + harness/retry.ts's 30s deadline), so a short
# timeout here would fail runs that were about to succeed.
EMBED_TIMEOUT_S = float(os.environ.get("KRAMA_EVAL_EMBED_TIMEOUT", 300))
GENERATE_TIMEOUT_S = float(os.environ.get("KRAMA_EVAL_GENERATE_TIMEOUT", 180))


class AdapterUnreachable(RuntimeError):
    """Raised with an actionable message instead of a bare URLError.

    The suite promises its failures name the exact missing piece rather than
    dumping a stack trace; a shim that swallows that promise is worse than no
    shim, so the message says what to start and how.
    """


def _post(path: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise AdapterUnreachable(
            f"Krama's eval adapter is not answering at {BASE_URL}{path} ({e}).\n"
            f"Start it from the repo root with:\n"
            f"    bun run eval:adapter\n"
            f"and override the address with KRAMA_EVAL_ADAPTER_URL if it is not on port 3100."
        ) from e


def health() -> dict:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise AdapterUnreachable(
            f"Krama's eval adapter is not answering at {BASE_URL}/health ({e}).\n"
            f"Start it from the repo root with:\n"
            f"    bun run eval:adapter"
        ) from e


def embed_texts(texts: list, kind: str) -> list:
    return _post("/embed", {"texts": list(texts), "kind": kind}, EMBED_TIMEOUT_S)["vectors"]


def generate(query: str, passages: list, lang: str = "en") -> dict:
    return _post("/generate", {"query": query, "passages": passages, "lang": lang}, GENERATE_TIMEOUT_S)
