# CLAUDE.md

## Purpose

This is a personal fork of **OpenPronounce** (Wav2Vec2 phone recognition, edit-distance alignment, DTW acoustic comparison, prosody). The goal is a **pronunciation lab for one learner** (native Spanish speaker, working toward professional B2 English). It complements conversational practice, where transcription hides pronunciation errors.

Success means answering "what specific pronunciation problem should I work on next?", not "what score did I get?". Intelligibility matters, accent elimination does not.

Full context — motivation, learner profile, the `ship`/`chip` failure case, known weaknesses, dataset plan, feedback model, metrics, backlog, non-goals — lives in [`docs/vision.md`](docs/vision.md). Read the relevant section before working on evaluation, scoring, or feedback.

## Current focus

Build the evaluation harness **before** changing pronunciation logic (`docs/vision.md` §9–12, §25). Existing starting points: `benchmarks/speechocean762.py`, `benchmarks/word_detection.py`.

First regression case (`docs/vision.md` §7, §22): expected `ship`, produced ≈ `chip`. Score ≈ 26 and PER ≈ 0.67, yet `errors: []` and "Your pronunciation is excellent!". The fix must test semantic consistency, never an exact score.

## Commands

```bash
# Dev dependencies (.venv is uv-managed and has no pip)
uv pip install --python .venv/Scripts/python.exe -e ".[app,dev]"

# Tests (same as CI: .github/workflows/tests.yml)
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m pytest --cov=openpronounce --cov-report=term-missing

# Web app + API on http://127.0.0.1:8000 (sets ffmpeg/espeak-ng paths for Windows)
./run-local.sh          # Git Bash
.\run-local.ps1         # PowerShell

# CLI
.venv/Scripts/python.exe -m openpronounce.cli <audio> "<expected text>"

# Evaluation harness (lab/README.md): labeled recordings -> lab/runs/<timestamp>/
./run-lab.sh --validate-only                     # Git Bash; .\run-lab.ps1 in PowerShell
./run-lab.sh --name baseline --skip-missing      # all flags: python -m lab.evaluate --help
.\record-lab.ps1                                 # PowerShell: record missing takes (raw + trimmed + sidecar); python -m lab.record --help
```

Windows notes: `python` on PATH is the Microsoft Store alias; always use `.venv/Scripts/python.exe`. espeak-ng is expected at `C:\Program Files\eSpeak NG` and ffmpeg in the WinGet package dir (see `run-local.*`). Models download from Hugging Face on first run.

## Code map

| Path | Role |
|---|---|
| `openpronounce/speech.py` | Pipeline: embeddings, ASR, phonemization, `compare_transcriptions`, `compute_pronunciation_score`, `_feedback` |
| `openpronounce/phones.py` | Direct phone recognition (`wav2vec2-lv-60-espeak-cv-ft`), per-phone confidences, word error reporting thresholds |
| `openpronounce/audio.py` | Audio loading/conversion, reference speech generation |
| `openpronounce/tts.py` | TTS backends (`OPENPRONOUNCE_TTS`: gtts default, piper, kokoro) |
| `openpronounce/languages.py` | Language registry, acoustic score calibration |
| `openpronounce/cli.py` | CLI entry point |
| `server.py` | FastAPI web UI + JSON API |
| `benchmarks/` | speechocean762 and word-detection benchmarks |
| `lab/` | Personal evaluation harness: corpus CSV, `lab/record.py` (recording helper), `python -m lab.evaluate`, raw outputs and report per run |
| `tests/` | pytest suite |

Known fact: `_feedback` (`speech.py`) depends only on word-level errors, not on the score. Word errors are gated by `PHONE_ERROR_THRESHOLD`, `PHONE_ERROR_MIN_EDITS`, and `NEAR_PHONE_COST` (`phones.py`).

## Development rules

1. Inspect the existing implementation and run the tests before modifying behavior.
2. **Measure before modifying.** Do not change thresholds, weights, or models without a hypothesis and before/after results on the controlled dataset.
3. **Do not tune against a single recording.** No one-off exceptions. Personalization, if any, is a separate explicit layer.
4. **Keep raw outputs.** Never discard lower-level signals (phones, confidences, PER, WER, acoustic distance) because a higher-level score exists.
5. **Separate recognition from interpretation.** A model prediction is not a teaching conclusion.
6. **Confidence matters.** Low-confidence detections are surfaced as `uncertain`, never asserted as learner errors. Prefer "uncertain" over wrong advice.
7. ASR output is not pronunciation truth. The phone recognizer is a separate signal.
8. Every confirmed bug becomes a regression test when practical.
9. Keep changes small and reviewable. Modify incrementally. Do not rewrite the Python ML stack or add models without a clear experimental hypothesis.
10. Avoid infrastructure, abstractions, or architecture churn that does not directly serve the pronunciation-learning objective (see non-goals in `docs/vision.md` §24).
11. Near-term work belongs in the **evaluation / interpretation layer**, not in replacing the recognition stack.

> Guiding principle: **Do not teach from a model output until we have enough evidence to trust its interpretation.**
