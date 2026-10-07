# Feature: eval-harness

Locator: `odd/tasks/eval-harness.md` · Engram mirror: `odd/eval-harness/tasks` · Branch: `feat/eval-harness` (from `docs/fork-claude-md` @ 80ecec9)

## Objective

Process a directory of labeled personal recordings through the current OpenPronounce pipeline and produce machine-readable raw results plus a compact human-readable report (`docs/vision.md` §9–12, §25), without changing pronunciation logic.

## Problem / why

Pronunciation logic must not be changed before there is a controlled, reproducible baseline on the learner's own voice. The existing `benchmarks/` measure speechocean762 (Mandarin-speaking learners), which says nothing about this learner. The known `ship`→`chip` inconsistency (`docs/vision.md` §7) cannot be fixed responsibly without a corpus.

## Scope

- Phase 0: dev dependencies installed, full test suite run, package versions recorded.
- Phase 1: corpus metadata schema, evaluation CLI, raw JSON per sample, run metadata, summary report with detection outcome per label, unit tests without models, recording guide.
- Out of scope: any change to scoring, thresholds, feedback, or models (`openpronounce/`).

## Constraints

- Audio recordings are personal voice data: not committed by default (gitignored), kept on disk.
- Raw model outputs are preserved in full; the report is a view, never the only copy.
- No hardcoded thresholds that pretend to be a diagnosis; the report shows signals and label-vs-output agreement only.
- Windows local setup: `.venv/Scripts/python.exe`, espeak-ng and ffmpeg paths as in `run-local.*`.

## Delivery strategy

`exception-ok`: personal fork with a single reviewer (the user), who asked for minimal interruptions on 2026-10-07. Forecast ≈ 550 authored lines (T1 ≈ 60, T2 ≈ 450, T3 ≈ 40). Branch is pushed to the fork `origin` only; upstream is never touched.

## Tasks

- [x] **T1 Baseline (Phase 0).** Install `.[app,dev]`, run the full suite, record versions and results in `docs/baseline.md`; commit the `run-local.*` launchers referenced by `CLAUDE.md`. Route: delegated (full suite → verification worker).
- [ ] **T2 Harness core (Phase 1).** `lab/` package: corpus CSV schema + validation, `python -m lab.evaluate`, raw JSON per sample, `results.jsonl`, `run.json`, `report.md`; unit tests with an injected fake pipeline; `lab/README.md` recording guide; `run-lab.*` launchers; gitignore audio and runs. Route: delegated writer (2+ non-trivial files).
- [ ] **T3 End-to-end smoke run.** Run the harness with the real models on an example corpus built from `assets/`; fix defects found; record evidence. Route: delegated verification.
- [ ] **T4 First personal corpus (user).** Record the 5 `ship` samples (2× good, sheep, chip, sip) per `lab/README.md`; then run the first baseline. Owner: user.

## Acceptance criteria

- `python -m lab.evaluate --corpus <csv>` processes every sample and writes, per run: raw JSON per sample (full pipeline output + metadata), `results.jsonl`, `run.json` (git commit, package versions, TTS backend, language, device), `report.md`.
- Invalid metadata (missing column, unknown label, missing audio file, duplicate id) fails fast with a clear message before any model loads.
- A pipeline failure on one sample is recorded and does not abort the run.
- Report shows per-sample signals (score, ASR, expected vs heard phones with confidences, PER, WER, acoustic distance, flagged words, feedback) and a summary of label-vs-detection agreement (TP/FP/TN/FN, precision, recall; `uncertain` excluded) plus positive-feedback-on-intentional-error cases.
- Existing test suite still passes; new harness tests pass without downloading models.

## Checks

- `.venv/Scripts/python.exe -m pytest` (full suite)
- `.venv/Scripts/python.exe -m pytest tests/test_lab_evaluate.py`
- T3 smoke run output readback

## Progress / evidence

| Task | Route | Commit | Checks | Review tier / outcome |
|---|---|---|---|---|
| T1 | delegated (full suite → worker) | see git log | 99 passed (with and without espeak exports); parent spot check 99 passed | pending assess |
| T2 | delegated | — | — | — |
| T3 | delegated | — | — | — |

## Next step

T2.
