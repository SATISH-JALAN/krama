# Runs BeaconBandhu/rag-local-eval-loop against Krama.
#
# Nothing from that suite is copied into this repo -- it has its own eval/
# package that would collide with Krama's, and its own venv holding faiss /
# sentence-transformers / the judge SDKs, which have no business in Krama's
# dependency tree. This script points the suite at this directory with
# --rag-root instead (the alternative its own runbook documents).
#
# Prerequisites, in order:
#   1. The eval suite cloned + its venv built at $EvalLoop below.
#   2. Krama's eval adapter running:  bun run eval:adapter
#   3. A JUDGE credential in this shell -- OPENAI_API_KEY or ANTHROPIC_API_KEY.
#      This is the suite's own judge for faithfulness/correctness and has
#      nothing to do with Krama's GEMINI_API_KEY. Without it those two checks
#      report SKIPPED; retrieval, reliability and latency still run for real.
#
# Every argument is forwarded to eval.runner, e.g.
#   .\run-eval.ps1 --num-answerable 3 --num-unanswerable 3 --workers 1
#   .\run-eval.ps1 --num-answerable 50 --num-unanswerable 50

$ErrorActionPreference = "Stop"

$EvalLoop  = if ($env:RAG_EVAL_LOOP) { $env:RAG_EVAL_LOOP } else { "C:\tools\rag-local-eval-loop" }
$KramaRoot = $PSScriptRoot
$AdapterUrl = if ($env:KRAMA_EVAL_ADAPTER_URL) { $env:KRAMA_EVAL_ADAPTER_URL } else { "http://127.0.0.1:3100" }

if (-not (Test-Path (Join-Path $EvalLoop ".venv\Scripts\python.exe"))) {
    Write-Error "No eval-loop venv at $EvalLoop\.venv. Clone the suite and build its venv, or set `$env:RAG_EVAL_LOOP."
    exit 1
}

# Fail here, with a fixable message, rather than 200 requests deep into a run.
try {
    $null = Invoke-RestMethod -Uri "$AdapterUrl/health" -TimeoutSec 10
} catch {
    Write-Error "Krama's eval adapter is not answering at $AdapterUrl. Start it in another terminal:`n    bun run eval:adapter"
    exit 1
}

# The suite only picks a judge key out of the target's .env when the target has
# an importable app/config.py calling load_dotenv() (eval/judge.py:117) -- Krama
# is TypeScript and has no such module, so the key would otherwise have to be
# re-exported into every new shell. Read it here instead, so it can live once in
# .env (already gitignored) next to GEMINI_API_KEY. An already-exported env var
# wins, so a one-off override still works.
$dotenv = Join-Path $KramaRoot ".env"
if (Test-Path $dotenv) {
    foreach ($line in Get-Content $dotenv) {
        if ($line -match '^\s*(OPENAI_API_KEY|ANTHROPIC_API_KEY|ANTHROPIC_AUTH_TOKEN)\s*=\s*(.+?)\s*$') {
            $name, $value = $Matches[1], $Matches[2].Trim('"', "'")
            if (-not [Environment]::GetEnvironmentVariable($name)) {
                Set-Item -Path "env:$name" -Value $value
                Write-Host "Judge credential $name loaded from .env"
            }
        }
    }
}

# Opt-in free judge: set $env:KRAMA_FREE_JUDGE = "1" to drive the judge from the
# Gemini free tier instead of a paid OpenAI/Anthropic key. eval/judge.py builds a
# bare openai.OpenAI(), which honours OPENAI_BASE_URL, and Google publishes an
# OpenAI-compatible endpoint -- so the suite needs no patching. Verified against
# judge.py's exact call shape (max_completion_tokens + response_format
# json_object): both models below return contract-valid JSON.
#
# READ THIS BEFORE USING IT FOR ANYTHING YOU REPORT. Krama generates with Gemini
# (llm/gemini.ts). A Gemini judge is therefore grading its own family's output,
# which eval/judge.py calls out by name as a bias risk: "judging a model with
# itself ... a model is more likely to rate its own output favorably." Fine for
# iterating for free; not fine as the number you publish or submit. For that,
# use a genuinely independent judge (an Anthropic or OpenAI key) so faithfulness
# and correctness are not self-graded.
if ($env:KRAMA_FREE_JUDGE -eq "1") {
    if (-not $env:GEMINI_API_KEY -and (Test-Path $dotenv)) {
        foreach ($line in Get-Content $dotenv) {
            if ($line -match '^\s*GEMINI_API_KEY\s*=\s*(.+?)\s*$') {
                $env:GEMINI_API_KEY = $Matches[1].Trim('"', "'")
            }
        }
    }
    if (-not $env:GEMINI_API_KEY) {
        Write-Error "KRAMA_FREE_JUDGE=1 but no GEMINI_API_KEY in the environment or .env."
        exit 1
    }
    $env:OPENAI_API_KEY = $env:GEMINI_API_KEY
    $env:OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    if (-not $env:EVAL_JUDGE_MODEL_OPENAI) { $env:EVAL_JUDGE_MODEL_OPENAI = "gemini-3.6-flash" }
    Write-Host "FREE JUDGE: Gemini ($env:EVAL_JUDGE_MODEL_OPENAI) via Google's OpenAI-compatible endpoint." -ForegroundColor Cyan
    Write-Host "  Krama also generates with Gemini, so faithfulness/correctness are SELF-GRADED here." -ForegroundColor Yellow
    Write-Host "  Treat those two numbers as directional only -- do not report them." -ForegroundColor Yellow
    Write-Host ""
}

if (-not $env:OPENAI_API_KEY -and -not $env:ANTHROPIC_API_KEY -and -not $env:ANTHROPIC_AUTH_TOKEN) {
    Write-Host "WARNING: no judge credential (OPENAI_API_KEY / ANTHROPIC_API_KEY)." -ForegroundColor Yellow
    Write-Host "         Faithfulness and correctness will report SKIPPED; everything else is real." -ForegroundColor Yellow
    Write-Host ""
}

$env:EVAL_EMBEDDER_MODULE  = "evalshim.embedder"
$env:EVAL_GENERATOR_MODULE = "evalshim.generator"

# --workers 1: Krama's embedder is a single onnxruntime session with
# intraOpNumThreads=2 (ghana/embed.ts) behind one Bun process. Parallel workers
# would queue on it anyway and would make the latency percentiles measure
# contention rather than Krama. Override by passing --workers N yourself; the
# last occurrence wins in argparse.
Push-Location $EvalLoop
try {
    & (Join-Path $EvalLoop ".venv\Scripts\python.exe") -m eval.runner `
        --rag-root $KramaRoot --workers 1 @args
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
