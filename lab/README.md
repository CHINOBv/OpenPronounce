# Pronunciation lab

Run your own labeled recordings through the current OpenPronounce pipeline and keep every output, so that any later change to scoring or feedback can be measured before and after on the same corpus. The harness reports signals and how they agree with your label. It does not diagnose and adds no thresholds of its own. Background: [`docs/vision.md`](../docs/vision.md) §9–12.

## Quick path

1. Record the five `ship` samples into `lab/corpus/audio/` (protocol and checklist below).
2. Check the corpus: `./run-lab.sh --validate-only`.
3. Run it: `./run-lab.sh --name baseline`.
4. Open `lab/runs/<timestamp>-baseline/report.md`.

To try the harness without recording anything, use the bundled samples: `./run-lab.sh --corpus lab/corpus/example.csv --name example`.

## Recording protocol

Keep the conditions identical across takes, so that differences come from your pronunciation and not from the setup.

| Rule | Why |
|---|---|
| Same microphone, same distance, same room | Microphone and room change the acoustic distance |
| Quiet room, no music or fan | Noise becomes phones |
| Say only the target word, with about 0.5 s of silence before and after | Clipped onsets look like deleted phones |
| One take per file | One file = one row = one label |
| File name = `<sample_id>.wav` | Matches `audio_file` in the CSV |
| WAV, mono if the tool offers it | Any format ffmpeg reads works (m4a, mp3, flac); stereo is downmixed and audio resampled to 16 kHz on load |

Suggested tools: Audacity (set the project to mono, export as WAV) or Windows Sound Recorder. If you save another format, change the extension in `audio_file`.

## First set: `ship`

Already listed in `lab/corpus/samples.csv`; only the audio is missing.

- [ ] `ship_good_01.wav`: say "ship" as well as you can
- [ ] `ship_good_02.wav`: same again (stability)
- [ ] `ship_as_sheep_01.wav`: say "sheep" on purpose (`/ɪ/` → `/iː/`)
- [ ] `ship_as_chip_01.wav`: say "chip" on purpose (`/ʃ/` → `/tʃ/`, the case of `docs/vision.md` §7)
- [ ] `ship_as_sip_01.wav`: say "sip" on purpose (`/ʃ/` → `/s/`)

Fill in `microphone`, `recording_context` and `recorded_on` as you record. While some files are still missing, run with `--skip-missing`.

## Corpus CSV

UTF-8 (a BOM from Excel is fine), one row per recording; blank lines are ignored, also before the header. Unknown extra columns are kept and copied to the outputs. A row with more cells than the header is an error; missing trailing cells are empty.

| Column | Required | Content |
|---|---|---|
| `sample_id` | yes | Unique even ignoring case (`Ship_01` and `ship_01` would share a file on Windows), letters, digits, `_` and `-` only; names the output files |
| `expected_text` | yes | What the pipeline is told you said |
| `intended_pronunciation` | yes (may be empty) | What you actually tried to produce, e.g. `chip` |
| `label` | yes | `good`, `intentional_error` or `uncertain` |
| `audio_file` | yes | Path relative to the CSV's directory |
| `target_phoneme`, `intended_substitution` | no | e.g. `ʃ` and `ʃ->tʃ` |
| `microphone`, `recording_context`, `recorded_on`, `notes` | no | Free text |

| Label | Meaning | Detection when the pipeline flags a word / flags nothing |
|---|---|---|
| `good` | You said the target as well as you can | FP / TN |
| `intentional_error` | You produced `intended_pronunciation` on purpose | TP / FN |
| `uncertain` | You are not sure what came out | NA (excluded from precision and recall) |

Validation lists every problem at once (line number and reason) and stops before any model loads.

## Running

`./run-lab.sh` (Git Bash) and `.\run-lab.ps1` (PowerShell) set the ffmpeg and espeak-ng paths, then run `python -m lab.evaluate` with your arguments.

| Flag | Default | Effect |
|---|---|---|
| `--corpus PATH` | `lab/corpus/samples.csv` | Corpus to run |
| `--out DIR` | `lab/runs` | Where run directories go |
| `--name NAME` | none | Suffix of the run directory, e.g. `baseline` |
| `--lang CODE` | `en` | Pipeline language |
| `--only ID [ID ...]` | all | Run only these samples |
| `--skip-missing` | off | Skip rows whose audio file does not exist yet (listed in the report) |
| `--validate-only` | off | Check the corpus and stop; no model, no output |

Before the first sample, a preflight checks that espeak-ng returns phones (without it the pipeline would silently report no errors) and makes the TTS reference of every distinct `expected_text` with the pipeline's own call: cached references cost nothing, missing ones are synthesized then (gTTS, the default, needs network access; offline backends are in [`docs/reference-voice.md`](../docs/reference-voice.md)). The first real run downloads the models from Hugging Face, so the first analyzed sample's elapsed time includes model loading (`cold_start` in its raw record, noted in the report).

| Exit code | Meaning |
|---|---|
| `0` | Done, even if some samples failed (they are in the report) |
| `1` | Unexpected error (traceback); `run.json` says `aborted` if the run had started |
| `2` | Usage error (bad flag or `--name`) |
| `3` | Invalid corpus |
| `4` | Preflight failed |
| `5` | Nothing to evaluate |
| `130` | Interrupted (Ctrl+C); the partial run is kept |

## Outputs

Each run writes `lab/runs/<YYYYMMDD-HHMMSS>[-<name>]/`:

| File | Content |
|---|---|
| `raw/<sample_id>.json` | The CSV row, audio path, sha256 and duration, the TTS reference (`reference`: path in the run, sha256, cache source), the full pipeline output (prosody included), per-word phone reports of every word (`diagnostics.word_reports`, also below the reporting threshold; `diagnostics.error` if computing them failed, the result stands), the error if the sample failed (`stage`: `analysis`, or `row` when its results line could not be derived; the pipeline output is kept), elapsed time, `cold_start` |
| `references/<sample_id>.wav` | Copy of the TTS reference the acoustic distance was measured against |
| `results.jsonl` | One line per sample: score, acoustic distance, reference sha256, ASR, PER, WER, expected and heard phones with confidences, reported errors, feedback, weighted edits per word (with word position), `detection`, `flags`. No prosody |
| `run.json` | `status` (`running` while the run goes, then `completed`, `interrupted` or `aborted`), timestamps, arguments, corpus sha256, git commit and dirty flag, package and espeak-ng versions, TTS backend, device, models and their Hugging Face revisions (`model_revisions`, the commit in the local cache; `null` when not cached or a local path), phone thresholds, counts (`not_run` after an interruption), summary |
| `report.md` | Summary (TP/FP/TN/FN, precision, recall, score range per label, positive feedback on intentional errors), one row per sample, then word-level edits (`word #position`, 3 decimals) next to the reporting threshold. Written for the finished samples also when the run is interrupted |

`flags` are facts, not judgments: `positive_feedback` (the feedback says "excellent"), `asr_mismatch` (ASR words differ from the expected words), `no_expected_phones` (espeak-ng gave nothing), `no_heard_phones`, `missing_words_with_errors` (the result has no `words_with_errors`, so the detection is unknown: NA).

## Privacy

Recordings (`lab/corpus/audio/*`) and runs (`lab/runs/`) are gitignored: your voice stays on this machine unless you commit it on purpose. The CSVs carry no audio and are committed.
