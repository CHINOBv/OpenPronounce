"""Run a labeled corpus through the current OpenPronounce pipeline and keep every output.

    python -m lab.evaluate --corpus lab/corpus/samples.csv [--name baseline] [--only ID ...]

Each run writes ``<out>/<YYYYMMDD-HHMMSS>[-<name>]/`` with ``raw/<sample_id>.json`` (full
pipeline output), ``results.jsonl`` (one compact line per sample), ``run.json``
(environment, versions, thresholds) and ``report.md``. See ``lab/README.md``.

The harness reports signals and label-vs-output agreement only; it never changes or
reinterprets the pipeline. openpronounce (and torch) is imported only once the corpus
is valid, so a bad CSV fails immediately.
"""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback

from . import corpus, report

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLING_RATE = 16000
EXIT_INVALID_CORPUS = 2
EXIT_PREFLIGHT_FAILED = 3
EXIT_NOTHING_TO_EVALUATE = 4
PACKAGES = ("torch", "transformers", "phonemizer", "librosa", "numpy")
PHONE_THRESHOLDS = ("PHONE_ERROR_THRESHOLD", "PHONE_ERROR_MIN_EDITS", "NEAR_PHONE_COST", "FINAL_EXTRA_COST",
                    "FINAL_DELETION_COST", "PHONE_PLAUSIBLE_POSTERIOR")


class PreflightError(RuntimeError):
    """The environment cannot produce meaningful results (e.g. espeak-ng is unreachable)."""


# ---------------------------------------------------------------------------
# Pipeline (the real analyzer and preflight; tests inject fakes)
# ---------------------------------------------------------------------------

def analyze_with_pipeline(waveform, expected_text, lang):
    """Run the current pipeline on a 16 kHz waveform and return ``(result, diagnostics)``.

    ``result`` is the unmodified output of ``speech.compare_audio_with_text``.
    ``diagnostics`` is ``{"word_reports": [...]}``, the per-word phone reports of every
    word, including those below the reporting thresholds that ``result`` leaves out;
    None when the phone model is disabled.
    """
    from openpronounce import phones, speech

    # compare_audio_with_text calls phones.recognize_phones through the module attribute,
    # so a temporary wrapper captures the recognition without running the phone model
    # twice (TestPipelineCapture guards that call shape).
    captured = []
    recognize_phones = phones.recognize_phones

    def capture(*args, **kwargs):
        recognition = recognize_phones(*args, **kwargs)
        captured.append(recognition)
        return recognition

    phones.recognize_phones = capture
    try:
        result = speech.compare_audio_with_text(waveform, expected_text, lang=lang)
    finally:
        phones.recognize_phones = recognize_phones
    if not captured:
        return result, None
    # Coupled on purpose to the private phones._word_reports: it is the exact per-word
    # computation that compare_phones thresholds, and the public result keeps only the
    # words above the thresholds. Re-running it on the captured recognition needs no model
    # call. TestPipelineCapture checks the keys this harness reads.
    word_reports = phones._word_reports(captured[-1], expected_text, result["language"])
    return result, {"word_reports": _plain(word_reports)}


def check_phonemizer(lang):
    """Fail when espeak-ng yields no phones: the pipeline would then silently report no errors."""
    from openpronounce import phones, speech

    _, groups = phones.get_expected_phones("ship", lang)
    if not any(groups) or not speech.get_phonemes("ship", lang):
        raise PreflightError(
            f"espeak-ng returned no phones for 'ship' (lang {lang!r}); every sample would get no errors. "
            "Set PHONEMIZER_ESPEAK_LIBRARY (path to libespeak-ng.dll) and PHONEMIZER_ESPEAK_PATH "
            "(path to espeak-ng.exe), or run through ./run-lab.sh or .\\run-lab.ps1, which set them."
        )


def load_audio(path):
    from openpronounce import audio

    return audio.load(path, sr=SAMPLING_RATE)


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _json_default(value):
    if hasattr(value, "tolist"):  # numpy scalars and arrays
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _plain(value):
    """``value`` as plain JSON types (numpy values converted), or None."""
    return None if value is None else json.loads(json.dumps(value, default=_json_default))


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
        f.write("\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path):
    """``path`` relative to the working directory when possible, with forward slashes."""
    try:
        path = os.path.relpath(path)
    except ValueError:  # another drive on Windows
        pass
    return path.replace("\\", "/")


def git_info():
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout

    try:
        return {"commit": git("rev-parse", "HEAD").strip(), "dirty": bool(git("status", "--porcelain").strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _espeak_version():
    try:
        from phonemizer.backend import EspeakBackend

        return ".".join(str(part) for part in EspeakBackend.version())
    except Exception:  # noqa: BLE001 - informative only, never fails the run
        return None


def pipeline_info(lang):
    """Versions, models, device, TTS backend and thresholds the pipeline runs with."""
    import openpronounce
    from openpronounce import phones, speech, tts
    from openpronounce.device import get_device
    from openpronounce.languages import get_language

    try:
        backend, voice = tts.resolve(lang)
        tts_info = {"backend": backend, "voice": voice}
    except Exception as e:  # noqa: BLE001 - every sample will fail and say why
        tts_info = {"error": str(e)}
    tts_info["OPENPRONOUNCE_TTS"] = os.environ.get("OPENPRONOUNCE_TTS")
    return {
        "versions": {"python": platform.python_version(), "openpronounce": openpronounce.__version__,
                     **{name: _package_version(name) for name in PACKAGES}},
        "espeak_ng": _espeak_version(),
        "tts": tts_info,
        "device": str(get_device()),
        "asr_model": get_language(lang).asr_model,
        "embedding_model": speech.MODEL_NAME,
        "phone_model": {"name": phones.PHONE_MODEL_NAME, "enabled": phones.is_enabled()},
        "phone_thresholds": {name: getattr(phones, name) for name in PHONE_THRESHOLDS},
    }


def make_run_dir(out, name):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = os.path.join(out, f"{stamp}-{name}" if name else stamp)
    path, n = base, 1
    while os.path.exists(path):
        n += 1
        path = f"{base}-{n}"
    os.makedirs(os.path.join(path, "raw"))
    return path


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def evaluate_sample(sample, analyze, lang):
    """Raw record of one sample. A failure is recorded in ``error``, never raised."""
    record = {
        "sample": sample.fields,
        "audio": {"path": _display_path(sample.audio_path), "sha256": None, "duration_s": None},
        "result": None,
        "diagnostics": None,
        "error": None,
        "elapsed_s": None,
    }
    start = time.perf_counter()
    try:
        record["audio"]["sha256"] = file_sha256(sample.audio_path)
        waveform = load_audio(sample.audio_path)
        record["audio"]["duration_s"] = round(len(waveform) / SAMPLING_RATE, 3)
        result, diagnostics = analyze(waveform, sample.expected_text, lang)
        record["result"], record["diagnostics"] = _plain(result), _plain(diagnostics)
    except Exception as e:  # noqa: BLE001 - one bad sample must not abort the run
        record["error"] = {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}
    record["elapsed_s"] = round(time.perf_counter() - start, 3)
    return record


def _run_name(value):
    if not corpus.SAMPLE_ID_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(f"must match {corpus.SAMPLE_ID_RE.pattern}")
    return value


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m lab.evaluate",
                                     description="Run a labeled corpus through the current pipeline.")
    parser.add_argument("--corpus", default="lab/corpus/samples.csv", help="corpus CSV (default: %(default)s)")
    parser.add_argument("--out", default="lab/runs", help="directory of the runs (default: %(default)s)")
    parser.add_argument("--name", type=_run_name, help="suffix of the run directory, e.g. baseline")
    parser.add_argument("--lang", default="en", help="language code (default: %(default)s)")
    parser.add_argument("--only", nargs="+", metavar="ID", help="evaluate only these sample ids")
    parser.add_argument("--skip-missing", action="store_true", help="skip samples whose audio file does not exist")
    parser.add_argument("--validate-only", action="store_true", help="validate the corpus and stop")
    return parser


def main(argv=None, analyze=None, preflight=None):
    args = build_parser().parse_args(argv)
    analyze = analyze or analyze_with_pipeline
    preflight = preflight or check_phonemizer

    try:
        selected = corpus.load_corpus(args.corpus, skip_missing=args.skip_missing, only=args.only)
    except corpus.CorpusError as e:
        print(e, file=sys.stderr)
        return EXIT_INVALID_CORPUS
    print(f"{args.corpus}: {len(selected.samples)} sample(s) ready, {len(selected.missing)} skipped")
    for sample in selected.missing:
        print(f"  skipped (audio not found): {sample.sample_id} -> {sample.fields['audio_file']}")
    if args.validate_only:
        return 0
    if not selected.samples:
        print("nothing to evaluate", file=sys.stderr)
        return EXIT_NOTHING_TO_EVALUATE

    try:
        preflight(args.lang)
    except Exception as e:  # noqa: BLE001 - any failure here means no meaningful run
        print(f"preflight failed: {e}", file=sys.stderr)
        return EXIT_PREFLIGHT_FAILED

    run_dir = make_run_dir(args.out, args.name)
    run = {
        "run_id": os.path.basename(run_dir),
        "started_at": _now(),
        "finished_at": None,
        "args": vars(args),
        "corpus": {"path": _display_path(args.corpus), "sha256": file_sha256(args.corpus)},
        "git": git_info(),
        "lang": args.lang,
        **pipeline_info(args.lang),
    }

    rows, word_reports = [], {}
    total = len(selected.samples)
    with open(os.path.join(run_dir, "results.jsonl"), "w", encoding="utf-8") as results:
        for index, sample in enumerate(selected.samples, 1):
            record = evaluate_sample(sample, analyze, args.lang)
            _write_json(os.path.join(run_dir, "raw", f"{sample.sample_id}.json"), record)
            row = report.result_row(record)
            results.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
            results.flush()
            rows.append(row)
            word_reports[sample.sample_id] = (record["diagnostics"] or {}).get("word_reports")
            if row["error"]:
                outcome = f"FAILED {row['error']['type']}: {row['error']['message']}"
            else:
                outcome = f"score {row['score']}, flagged {row['words_with_errors'] or '-'}, {row['detection']}"
            print(f"[{index}/{total}] {sample.sample_id}: {outcome} ({record['elapsed_s']:.1f} s)")

    failed = sum(1 for r in rows if r["error"])
    run.update({
        "finished_at": _now(),
        "counts": {"selected": total, "processed": total - failed, "failed": failed,
                   "missing": len(selected.missing)},
        "missing_samples": [s.sample_id for s in selected.missing],
        "summary": report.summarize(rows),
    })
    _write_json(os.path.join(run_dir, "run.json"), run)
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report.render_markdown(run, rows, word_reports))
    counts = run["summary"]["counts"]
    print(f"{run_dir}: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + f", {failed} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
