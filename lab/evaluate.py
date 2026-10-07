"""Run a labeled corpus through the current OpenPronounce pipeline and keep every output.

    python -m lab.evaluate --corpus lab/corpus/samples.csv [--name baseline] [--only ID ...]

Each run writes ``<out>/<YYYYMMDD-HHMMSS>[-<name>]/`` with ``raw/<sample_id>.json`` (full
pipeline output), ``references/<sample_id>.<ext>`` (the TTS reference behind the acoustic
distance), ``results.jsonl`` (one compact line per sample), ``run.json`` (environment,
versions, thresholds; written at the start, completed at the end) and ``report.md``. See
``lab/README.md``.

The harness reports signals and label-vs-output agreement only; it never changes or
reinterprets the pipeline. openpronounce (and torch) is imported only once the corpus
is valid, so a bad CSV fails immediately.
"""

import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback

from . import corpus, report

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLING_RATE = 16000
# 1 is Python's own exit code for an unexpected error and 2 argparse's for a usage error.
EXIT_INVALID_CORPUS = 3
EXIT_PREFLIGHT_FAILED = 4
EXIT_NOTHING_TO_EVALUATE = 5
EXIT_INTERRUPTED = 130  # 128 + SIGINT, the shell convention for Ctrl+C
EXIT_CODES_HELP = ("exit codes: 0 done (failed samples are in the report), 1 unexpected error, 2 usage error, "
                   f"{EXIT_INVALID_CORPUS} invalid corpus, {EXIT_PREFLIGHT_FAILED} preflight failed, "
                   f"{EXIT_NOTHING_TO_EVALUATE} nothing to evaluate, {EXIT_INTERRUPTED} interrupted "
                   "(the partial run is kept)")
PACKAGES = ("torch", "transformers", "phonemizer", "librosa", "numpy")
PHONE_THRESHOLDS = ("PHONE_ERROR_THRESHOLD", "PHONE_ERROR_MIN_EDITS", "NEAR_PHONE_COST", "FINAL_EXTRA_COST",
                    "FINAL_DELETION_COST", "PHONE_PLAUSIBLE_POSTERIOR")


class PreflightError(RuntimeError):
    """The environment cannot produce meaningful results (e.g. espeak-ng is unreachable)."""


def _error_info(exc, stage):
    """Record of the exception being handled: where it happened, what it was, its traceback."""
    return {"stage": stage, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Pipeline (the real analyzer and preflight; tests inject fakes)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _capturing(module, name):
    """Wrap ``module.<name>`` for the duration of the block; yield the list of the values it returns."""
    original = getattr(module, name)
    returned = []

    def wrapper(*args, **kwargs):
        value = original(*args, **kwargs)
        returned.append(value)
        return value

    setattr(module, name, wrapper)
    try:
        yield returned
    finally:
        setattr(module, name, original)


def analyze_with_pipeline(waveform, expected_text, lang):
    """Run the current pipeline on a 16 kHz waveform and return ``(result, diagnostics)``.

    ``result`` is the unmodified output of ``speech.compare_audio_with_text``.
    ``diagnostics`` holds ``word_reports``, the per-word phone reports of every word,
    including those below the reporting thresholds that ``result`` leaves out (None when
    the phone model is disabled; None plus ``error`` when computing them failed, ``result``
    stands either way), and ``reference_audio``, the TTS file behind ``acoustic_distance``.
    """
    from openpronounce import audio, phones, speech

    # compare_audio_with_text calls phones.recognize_phones and audio.text2speech through
    # their modules, so temporary wrappers capture the recognition and the reference file
    # without running anything twice (TestPipelineCapture guards that call shape).
    with _capturing(phones, "recognize_phones") as recognitions, _capturing(audio, "text2speech") as references:
        result = speech.compare_audio_with_text(waveform, expected_text, lang=lang)
    diagnostics = {"word_reports": None, "reference_audio": references[-1] if references else None}
    if recognitions:
        # Coupled on purpose to the private phones._word_reports: it is the exact per-word
        # computation that compare_phones thresholds, and the public result keeps only the
        # words above the thresholds. Re-running it on the captured recognition needs no model
        # call. TestPipelineCapture checks the keys this harness reads.
        try:
            word_reports = phones._word_reports(recognitions[-1], expected_text, result["language"])
            diagnostics["word_reports"] = _plain(word_reports)
        except Exception as e:  # noqa: BLE001 - the per-word view is extra, the result stands
            diagnostics["error"] = _error_info(e, "diagnostics")
    return result, diagnostics


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


def check_tts(lang, texts):
    """Fail when a TTS reference cannot be made: every sample needs one for its acoustic distance.

    Makes the reference of every text with the pipeline's own call (``audio.text2speech``,
    same arguments): a cached reference costs nothing, a missing one is synthesized now
    (gTTS needs network access) and the run then finds it in the cache.
    """
    from openpronounce import audio, tts
    from openpronounce.languages import get_language

    code = get_language(lang).code
    for text in texts:
        try:
            audio.text2speech(text, lang=code)
        except Exception as e:  # noqa: BLE001 - any failure here fails every sample with this text
            backend = os.environ.get("OPENPRONOUNCE_TTS") or tts.DEFAULT_BACKEND
            raise PreflightError(
                f"the TTS reference for {text!r} could not be made (backend {backend}): {type(e).__name__}: {e}. "
                "gTTS, the default backend, needs network access; docs/reference-voice.md lists offline backends."
            ) from e


def check_pipeline(lang, texts):
    """The real preflight: espeak-ng yields phones and the TTS references of ``texts`` can be made."""
    check_phonemizer(lang)
    check_tts(lang, texts)


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
    """``value`` as plain JSON types (numpy values converted, always a copy), or None."""
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


def model_revision(name):
    """Commit of model ``name`` in the local Hugging Face cache (what ``from_pretrained`` resolves), or None.

    Reads the cache only, never the network: None for a model not downloaded yet, a local
    directory instead of a repo id, or any lookup failure.
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(name, "config.json")
    except Exception:  # noqa: BLE001 - informative only, never fails the run
        return None
    if not isinstance(path, str):  # None, or the marker of a file known to be missing
        return None
    # <cache>/models--<org>--<name>/snapshots/<commit>/config.json
    return os.path.basename(os.path.dirname(path))


def pipeline_info(lang):
    """Versions, models (and their revisions), device, TTS backend and thresholds the pipeline runs with."""
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
    asr_model = get_language(lang).asr_model
    models = dict.fromkeys((asr_model, speech.MODEL_NAME, phones.PHONE_MODEL_NAME))
    return {
        "versions": {"python": platform.python_version(), "openpronounce": openpronounce.__version__,
                     **{name: _package_version(name) for name in PACKAGES}},
        "espeak_ng": _espeak_version(),
        "tts": tts_info,
        "device": str(get_device()),
        "asr_model": asr_model,
        "embedding_model": speech.MODEL_NAME,
        "phone_model": {"name": phones.PHONE_MODEL_NAME, "enabled": phones.is_enabled()},
        # Read again when the run ends: a first run downloads the models during the run.
        "model_revisions": {name: model_revision(name) for name in models},
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

def keep_reference(source, references_dir, sample_id):
    """Copy the TTS reference ``source`` to ``references/<sample_id><ext>``; return its record."""
    os.makedirs(references_dir, exist_ok=True)
    target = os.path.join(references_dir, sample_id + os.path.splitext(source)[1])
    shutil.copyfile(source, target)
    return {"path": f"references/{os.path.basename(target)}", "sha256": file_sha256(target),
            "source": _display_path(source)}


def recording_path(audio_path):
    """Sidecar that ``python -m lab.record`` writes next to a take: ``<stem>.recording.json``."""
    return os.path.splitext(audio_path)[0] + ".recording.json"


def read_recording(audio_path):
    """The recording sidecar of ``audio_path``; None without one, ``{"error": ...}`` when unreadable."""
    path = recording_path(audio_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:  # ValueError covers JSONDecodeError and bad UTF-8
        return {"error": f"{type(e).__name__}: {e}"}
    return data if isinstance(data, dict) else {"error": "the sidecar is not a JSON object"}


def evaluate_sample(sample, analyze, lang, references_dir, cold_start=False):
    """Raw record of one sample. A failure is recorded in ``error``, never raised.

    ``cold_start`` marks the first analysis of the run: its ``elapsed_s`` includes the lazy
    loading of the models. It is recorded only when the sample reaches the analyzer.
    ``recording`` is the take's lab.record sidecar, None when it has none.
    """
    record = {
        "sample": sample.fields,
        "audio": {"path": _display_path(sample.audio_path), "sha256": None, "duration_s": None},
        "recording": read_recording(sample.audio_path),
        "reference": None,
        "result": None,
        "diagnostics": None,
        "error": None,
        "elapsed_s": None,
        "cold_start": False,
    }
    start = time.perf_counter()
    reference = None
    try:
        record["audio"]["sha256"] = file_sha256(sample.audio_path)
        waveform = load_audio(sample.audio_path)
        record["audio"]["duration_s"] = round(len(waveform) / SAMPLING_RATE, 3)
        record["cold_start"] = cold_start
        result, diagnostics = analyze(waveform, sample.expected_text, lang)
        record["result"] = _plain(result)
        record["diagnostics"] = _plain(diagnostics)
        # The reference file is kept in the run (below); the copy of diagnostics drops its cache path.
        reference = (record["diagnostics"] or {}).pop("reference_audio", None)
    except Exception as e:  # noqa: BLE001 - one bad sample must not abort the run
        record["error"] = _error_info(e, "analysis")
    record["elapsed_s"] = round(time.perf_counter() - start, 3)
    if reference:
        try:
            record["reference"] = keep_reference(reference, references_dir, sample.sample_id)
        except Exception as e:  # noqa: BLE001 - the result stands without its reference copy
            record["reference"] = {"source": _display_path(reference), "error": _error_info(e, "reference")}
    return record


def derive_row(record):
    """``report.result_row(record)`` plus ``recording_warnings`` (None without a sidecar).

    A failure is recorded on the record (stage "row"), never raised.
    """
    try:
        row = report.result_row(record)
    except Exception as e:  # noqa: BLE001 - one malformed result must not abort the run
        record["error"] = _error_info(e, "row")
        row = report.result_row(dict(record, result=None, diagnostics=None))
    row["recording_warnings"] = (record.get("recording") or {}).get("warnings")
    return row


def _evaluate_samples(samples, analyze, lang, run_dir, rows, word_reports):
    """Evaluate ``samples`` in order, appending to ``rows`` and ``word_reports`` as each one finishes."""
    cold_start = True  # the first analysis of the run also loads the models
    with open(os.path.join(run_dir, "results.jsonl"), "w", encoding="utf-8") as results:
        for index, sample in enumerate(samples, 1):
            record = evaluate_sample(sample, analyze, lang, os.path.join(run_dir, "references"), cold_start)
            cold_start = cold_start and not record["cold_start"]
            row = derive_row(record)
            _write_json(os.path.join(run_dir, "raw", f"{sample.sample_id}.json"), record)
            results.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
            results.flush()
            rows.append(row)
            word_reports[sample.sample_id] = (record["diagnostics"] or {}).get("word_reports")
            if row["error"]:
                outcome = f"FAILED {row['error']['type']}: {row['error']['message']}"
            else:
                outcome = f"score {row['score']}, flagged {row['words_with_errors'] or '-'}, {row['detection']}"
            loading = ", includes model loading" if record["cold_start"] else ""
            print(f"[{index}/{len(samples)}] {sample.sample_id}: {outcome} ({record['elapsed_s']:.1f} s{loading})")


def _finish_run(run, run_dir, status, selected, rows, word_reports):
    """Final ``run.json`` and ``report.md``, with partial counts when the run did not complete."""
    failed = sum(1 for r in rows if r["error"])
    total = len(selected.samples)
    run.update({
        "status": status,
        "finished_at": _now(),
        "model_revisions": {name: model_revision(name) for name in run.get("model_revisions") or {}},
        "counts": {"selected": total, "processed": len(rows) - failed, "failed": failed,
                   "not_run": total - len(rows), "missing": len(selected.missing)},
        "missing_samples": [s.sample_id for s in selected.missing],
        "summary": report.summarize(rows),
    })
    _write_json(os.path.join(run_dir, "run.json"), run)
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report.render_markdown(run, rows, word_reports))


def _run_name(value):
    if not corpus.SAMPLE_ID_RE.fullmatch(value):
        raise argparse.ArgumentTypeError(f"must match {corpus.SAMPLE_ID_RE.pattern}")
    return value


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m lab.evaluate",
                                     description="Run a labeled corpus through the current pipeline.",
                                     epilog=EXIT_CODES_HELP)
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
    preflight = preflight or check_pipeline

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
        preflight(args.lang, list(dict.fromkeys(s.expected_text for s in selected.samples)))
    except Exception as e:  # noqa: BLE001 - any failure here means no meaningful run
        print(f"preflight failed: {e}", file=sys.stderr)
        return EXIT_PREFLIGHT_FAILED

    run = {
        "run_id": None,
        "status": "running",
        "started_at": _now(),
        "finished_at": None,
        "args": vars(args),
        "corpus": {"path": _display_path(args.corpus), "sha256": file_sha256(args.corpus)},
        "git": git_info(),
        "lang": args.lang,
        **pipeline_info(args.lang),
    }
    # The run directory is created only once everything above worked, so a failed start
    # leaves no empty run behind. run.json is written first: an interrupted or crashed run
    # still says what it was.
    run_dir = make_run_dir(args.out, args.name)
    run["run_id"] = os.path.basename(run_dir)
    _write_json(os.path.join(run_dir, "run.json"), run)

    rows, word_reports = [], {}
    try:
        _evaluate_samples(selected.samples, analyze, args.lang, run_dir, rows, word_reports)
    except KeyboardInterrupt:
        _finish_run(run, run_dir, "interrupted", selected, rows, word_reports)
        print(f"interrupted after {len(rows)} of {len(selected.samples)} sample(s); "
              f"partial run in {_display_path(run_dir)}", file=sys.stderr)
        return EXIT_INTERRUPTED
    except Exception as e:  # noqa: BLE001 - recorded in run.json, then raised as is
        run["abort_error"] = f"{type(e).__name__}: {e}"
        _finish_run(run, run_dir, "aborted", selected, rows, word_reports)
        raise
    _finish_run(run, run_dir, "completed", selected, rows, word_reports)
    counts = run["summary"]["counts"]
    failed = run["counts"]["failed"]
    print(f"{_display_path(run_dir)}: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + f", {failed} failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
