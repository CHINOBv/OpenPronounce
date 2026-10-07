# Baseline (Phase 0)

Recorded 2026-10-07 on commit `80ecec9` (branch `feat/eval-harness`), before any change to pronunciation logic.

## Environment

- OS: Windows 11 (`Windows-11-10.0.26300-SP0`), CPU only (`torch.cuda.is_available()` is `False`).
- Python 3.12.15 in a uv-managed `.venv` (uv 0.12.23). The venv has no `pip`; install with `uv pip`.
- Dev dependencies installed with `uv pip install --python .venv/Scripts/python.exe -e ".[app,dev]"`. Added: pytest, pytest-cov, coverage, pluggy, iniconfig. torch was not changed.
- Optional TTS backends `piper-tts` and `kokoro` are not installed (gTTS only).

## Versions

| Component | Version |
|---|---|
| torch | 2.14.1+cpu |
| transformers | 5.17.0 |
| huggingface_hub | 1.16.1 |
| librosa | 1.0.0 |
| soundfile | 0.14.0 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| phonemizer | 3.4.0 |
| fastdtw | 0.3.4 |
| Levenshtein | 0.27.5 |
| gTTS | 2.5.4 |
| fastapi | 0.142.2 |
| uvicorn | 0.54.0 |
| python-multipart | 0.0.32 |
| jinja2 | 3.1.6 |
| httpx | 0.28.1 |
| pytest / pytest-cov | 9.1.1 / 7.1.0 |
| eSpeak NG | 1.52.0 (`C:\Program Files\eSpeak NG`) |
| ffmpeg | 9.0.1-essentials_build (gyan.dev, via winget) |

## How the tests were run

Environment exports (same as `run-local.sh`):

```bash
export PATH="$LOCALAPPDATA/Microsoft/WinGet/Packages/Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.1-essentials_build/bin:/c/Program Files/eSpeak NG:$PATH"
export PHONEMIZER_ESPEAK_LIBRARY="C:\\Program Files\\eSpeak NG\\libespeak-ng.dll"
export PHONEMIZER_ESPEAK_PATH="C:\\Program Files\\eSpeak NG\\espeak-ng.exe"
export PYTHONUTF8=1 HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
.venv/Scripts/python.exe -m pytest -q -rs
```

`pyproject.toml` adds `-ra --tb=short --strict-markers --disable-warnings --durations=10` and ignores User/Deprecation/Future warnings, so warnings are not shown.

## Results

99 tests collected: test_audio 16, test_device 3, test_languages 15, test_phones 24, test_server 5, test_speech 36.

| Run | Environment | Result |
|---|---|---|
| 1 | Default Git Bash shell, no exports | 99 passed in 6.82 s |
| 2 | With the exports above | 99 passed in 6.07 s |
| 3 | espeak-ng and ffmpeg removed from `PATH`, no `PHONEMIZER_*` vars | 18 failed, 80 passed, 1 skipped in 6.55 s |
| 4 | Same as 3, plus the two `PHONEMIZER_*` vars | 98 passed, 1 skipped in 5.90 s |

Run 1 passes because both tools are already on the Windows `Path` (ffmpeg in the user Path, eSpeak NG in the machine Path), and phonemizer finds `libespeak-ng.dll` by itself.

- Skip (runs 3 and 4): `tests/test_audio.py:43` `test_load_browser_webm_opus_through_ffmpeg`, "ffmpeg not installed".
- Failures (run 3): 18 tests, all phonemizer-dependent: test_phones 9 (`TestExpectedPhones`, `TestComparePhones`, `TestPhonemizerFallback`, `TestConfidenceRule`), test_speech 6 (`TestPhonemeFunctions`, `TestTranscriptionComparison`), test_languages 2 (`TestFrenchPhones`), test_server 1 (`test_phonemes`). All are assertion errors on empty results (for example `[] != ['world']`, `0 not greater than 0`), not import or library errors.

## Notes

- Without espeak-ng, phonemization does not raise. `openpronounce/phones.py` (lines 266-289) and `openpronounce/speech.py` `_phonemize_word` catch the exception, log it at DEBUG level and return empty phone lists. The result is empty expected phones and no reported word errors. This is a silent-failure mode to keep in mind when reading "no errors" results.
- Printing IPA to the console without `PYTHONUTF8=1` raises `UnicodeEncodeError` (cp1252 charmap).
- `python` on PATH is the Microsoft Store alias; always use `.venv/Scripts/python.exe`.
- The suite mocks model inference and the gTTS network call; no test exceeded 1.7 s and no model was downloaded. It does not validate real Wav2Vec2 output.
