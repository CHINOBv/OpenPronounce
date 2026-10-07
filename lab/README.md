# Pronunciation lab

Run your own labeled recordings through the current OpenPronounce pipeline and keep every output, so that any later change to scoring or feedback can be measured before and after on the same corpus. The harness reports signals and how they agree with your label. It does not diagnose and adds no thresholds of its own. Background: [`docs/vision.md`](../docs/vision.md) §9–12.

## Quick path

PowerShell, from the repository root:

1. Turn off the microphone enhancements once ([step 1 below](#1-turn-off-microphone-enhancements-once)).
2. Record the 13 `ship` takes: `.\record-lab.ps1` (it asks for every sample whose audio is missing).
3. Check the corpus: `.\run-lab.ps1 --validate-only`.
4. Run it: `.\run-lab.ps1 --name baseline`.
5. Open `lab\runs\<timestamp>-baseline\report.md`.

To try the harness without recording anything, use the bundled samples: `.\run-lab.ps1 --corpus lab/corpus/example.csv --name example`. In Git Bash, `./run-lab.sh` takes the same flags.

## Recording

### 1. Turn off microphone enhancements (once)

Windows noise suppression replaces the silence around your word with digital silence and can clip the edges of the word. The first attempt shows that pattern (98 % exact zeros in the silence) and every take scored 10–29, mostly because of the setup.

| Where | Setting |
|---|---|
| Settings → System → Sound → Input → your microphone (Configuración → Sistema → Sonido → Entrada) | **Audio enhancements** (Mejoras de audio): **Off** |
| Realtek Audio Console, Microphone tab, if installed | **Noise Suppression** and **Acoustic Echo Cancellation**: off |
| Classic Sound panel (`mmsys.cpl`) → Recording → microphone → Properties → Advanced | Untick **Enable audio enhancements** |

Then make a test take and check it: `.\record-lab.ps1 --only ship_good_01`. After you stop, the stats line must **not** end in `(digital silence)`: the noise floor has to be above one 16-bit step (0.000031); the gated first attempt showed about 0.000004. Keep the take if it looks fine, or press Ctrl+C to discard it. If the gating stays, try the other input from `.\record-lab.ps1 --list-devices`.

### 2. Record the takes

```powershell
.\record-lab.ps1                              # every sample whose audio is missing, in corpus order
.\record-lab.ps1 --only ship_as_chip_02       # just these samples
.\record-lab.ps1 --only ship_good_01 --redo   # record an existing take again
```

For each sample:

1. Read what to say, e.g. `[8/13] ship_as_chip_01: say "chip" on purpose (target: ship)`.
2. Press Enter, wait for `Recording`, say the word once, wait about a second, press Enter.
3. Read the stats line and any warning.
4. `Enter` keeps the take, `r` records it again, `p` plays the trimmed take.

Ctrl+C stops the session at any point: kept takes stay, the current take is discarded, and the next session starts with the samples still missing.

| Rule | Why |
|---|---|
| Same microphone, same distance, same room | Microphone and room change the acoustic distance |
| Quiet room, no music or fan | Noise becomes phones |
| One word per take, then wait a second before stopping | Stopping on the word can cut its end; the silence is trimmed anyway |

| Flag | Default | Effect |
|---|---|---|
| `--device NAME` | `$env:OPENPRONOUNCE_MIC`, else the first device | DirectShow microphone, e.g. `"Micrófono (Realtek(R) Audio)"` |
| `--list-devices` | | List the microphones ffmpeg sees, then stop |
| `--only ID [ID ...]` | all missing | Record only these samples |
| `--redo` | off | Also record samples that already have audio; the old take is replaced only when you keep the new one |
| `--max-seconds S` | `6` | Hard stop of a take |
| `--pad S` | `0.2` | Seconds of the original audio kept before and after the speech |
| `--top-db DB` | `40` | Trim threshold below the loudest part (`librosa.effects.trim`) |
| `--corpus PATH` | `lab/corpus/samples.csv` | Corpus to record |

To keep a device for the whole PowerShell session: `$env:OPENPRONOUNCE_MIC = "Micrófono (Realtek(R) Audio)"`.

| Warning (code in the sidecar) | Meaning | What to do |
|---|---|---|
| `digital_gating` | Noise floor below one 16-bit step: the input is gated | Step 1 above |
| `clipping` | Peak at 0.99 or more | Move back or lower the input level |
| `too_quiet` | Peak below 0.05 | Move closer or raise the input level |
| `speech_too_short` | Less than 0.15 s of speech (20 ms frames within `--top-db` of the loudest) | Retry: the word was probably cut |
| `max_seconds` | The take hit `--max-seconds` | Retry and stop sooner |

Exit codes: `0` done, `2` usage error, `3` invalid corpus, `4` ffmpeg or the microphone failed, `130` interrupted.

### Files of a take

The pipeline does not trim silence, and long silence hurts it: trimming the same first-attempt audio raised `ship_good_02` from 29 to 72 and fixed its ASR (SHIPLEFF → SHIP). So each kept take leaves three files in `lab/corpus/audio/`:

| File | Content |
|---|---|
| `<sample_id>.wav` | The trimmed take that `lab.evaluate` reads (`audio_file`): the speech `librosa.effects.trim` finds plus `--pad` seconds each side, the same samples as the raw take (no resampling, no gain) |
| `raw/<sample_id>.wav` | The raw take as ffmpeg captured it (mono, 16 kHz, 16-bit), kept as evidence |
| `<sample_id>.recording.json` | Sidecar: time, device, ffmpeg command, path and sha256 of both files, trim parameters and bounds, stats (durations, peak, RMS of the trimmed take, noise floor), warnings |

A take is first written as `*.take.wav` and renamed when you keep it, so the ffmpeg command in the sidecar names that pending file. `lab.evaluate` copies the sidecar into each raw result (`recording`) and its warnings into `results.jsonl` (`recording_warnings`).

Recordings made with another tool still work (any format ffmpeg reads; stereo is downmixed and audio resampled to 16 kHz on load), they just have no sidecar (`recording: null`). Keep about 0.5 s of silence around the word.

### Checklist: `ship` (13 takes)

Already listed in `lab/corpus/samples.csv`; only the audio is missing. Several takes per contrast show how stable the signals are.

- [ ] `ship_good_01`: say "ship" as well as you can
- [ ] `ship_good_02`: same again
- [ ] `ship_good_03`: same again
- [ ] `ship_good_04`: same again
- [ ] `ship_as_sheep_01`: say "sheep" on purpose (`/ɪ/` → `/iː/`)
- [ ] `ship_as_sheep_02`: same again
- [ ] `ship_as_sheep_03`: same again
- [ ] `ship_as_chip_01`: say "chip" on purpose (`/ʃ/` → `/tʃ/`, the case of `docs/vision.md` §7)
- [ ] `ship_as_chip_02`: same again
- [ ] `ship_as_chip_03`: same again
- [ ] `ship_as_sip_01`: say "sip" on purpose (`/ʃ/` → `/s/`)
- [ ] `ship_as_sip_02`: same again
- [ ] `ship_as_sip_03`: same again

The sidecar records the device and the time, so `microphone` and `recorded_on` in the CSV are optional. While some files are still missing, run with `--skip-missing`.

The first attempt (five 3 s `ffmpeg` takes with enhancements on, in `lab/corpus/audio/archive/20261007-ffmpeg-3s/`) was set aside because gated silence and untrimmed audio, not pronunciation, explain most of its low scores (runs `20261007-104447-baseline` and `20261007-104609-trimmed-experiment`).

## Corpus CSV

UTF-8 (a BOM from Excel is fine), one row per recording; blank lines are ignored, also before the header. Unknown extra columns are kept and copied to the outputs. A row with more cells than the header is an error; missing trailing cells are empty.

| Column | Required | Content |
|---|---|---|
| `sample_id` | yes | Unique even ignoring case (`Ship_01` and `ship_01` would share a file on Windows), letters, digits, `_` and `-` only; names the output files |
| `expected_text` | yes | What the pipeline is told you said |
| `intended_pronunciation` | yes (may be empty) | What you actually tried to produce, e.g. `chip` |
| `label` | yes | `good`, `intentional_error` or `uncertain` |
| `audio_file` | yes | Path relative to the CSV's directory (`.wav` for `lab.record`) |
| `target_phoneme`, `intended_substitution` | no | e.g. `ʃ` and `ʃ->tʃ` |
| `microphone`, `recording_context`, `recorded_on`, `notes` | no | Free text |

| Label | Meaning | Detection when the pipeline flags a word / flags nothing |
|---|---|---|
| `good` | You said the target as well as you can | FP / TN |
| `intentional_error` | You produced `intended_pronunciation` on purpose | TP / FN |
| `uncertain` | You are not sure what came out | NA (excluded from precision and recall) |

Validation lists every problem at once (line number and reason) and stops before any model loads.

## Running

`.\run-lab.ps1` (PowerShell) and `./run-lab.sh` (Git Bash) set the ffmpeg and espeak-ng paths, then run `python -m lab.evaluate` with your arguments.

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
| `raw/<sample_id>.json` | The CSV row, audio path, sha256 and duration, the recording sidecar (`recording`, `null` without one), the TTS reference (`reference`: path in the run, sha256, cache source), the full pipeline output (prosody included), per-word phone reports of every word (`diagnostics.word_reports`, also below the reporting threshold; `diagnostics.error` if computing them failed, the result stands), the error if the sample failed (`stage`: `analysis`, or `row` when its results line could not be derived; the pipeline output is kept), elapsed time, `cold_start` |
| `references/<sample_id>.wav` | Copy of the TTS reference the acoustic distance was measured against |
| `results.jsonl` | One line per sample: score, acoustic distance, reference sha256, ASR, PER, WER, expected and heard phones with confidences, reported errors, feedback, weighted edits per word (with word position), `detection`, `flags`, `recording_warnings` (`null` without a sidecar). No prosody |
| `run.json` | `status` (`running` while the run goes, then `completed`, `interrupted` or `aborted`), timestamps, arguments, corpus sha256, git commit and dirty flag, package and espeak-ng versions, TTS backend, device, models and their Hugging Face revisions (`model_revisions`, the commit in the local cache; `null` when not cached or a local path), phone thresholds, counts (`not_run` after an interruption), summary |
| `report.md` | Summary (TP/FP/TN/FN, precision, recall, score range per label, positive feedback on intentional errors), one row per sample, then word-level edits (`word #position`, 3 decimals) next to the reporting threshold. Written for the finished samples also when the run is interrupted |

`flags` are facts, not judgments: `positive_feedback` (the feedback says "excellent"), `asr_mismatch` (ASR words differ from the expected words), `no_expected_phones` (espeak-ng gave nothing), `no_heard_phones`, `missing_words_with_errors` (the result has no `words_with_errors`, so the detection is unknown: NA).

## Privacy

Recordings (`lab/corpus/audio/*`: trimmed and raw takes, sidecars, the archive) and runs (`lab/runs/`) are gitignored: your voice stays on this machine unless you commit it on purpose. The CSVs carry no audio and are committed.
