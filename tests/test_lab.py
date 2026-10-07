"""Tests of the evaluation harness (``lab/``). No model is loaded: the pipeline is faked."""

import csv
import glob
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from lab import corpus, evaluate, report

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER = ["sample_id", "expected_text", "intended_pronunciation", "label", "audio_file"]
FEEDBACK_OK = "🔊 Feedback on your pronunciation:\n✅ Your pronunciation is excellent! 🎉\n"
FEEDBACK_BAD = "🔊 Feedback on your pronunciation:\n❌ You need to better pronounce these words: ship\n"


def write_csv(path, header, rows, bom=False):
    with open(path, "w", encoding="utf-8-sig" if bom else "utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_wav(path, seconds=0.5, sr=44100):
    t = np.arange(int(sr * seconds)) / sr
    sf.write(path, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)


def fake_result(score, transcribe, heard, confidences, words_with_errors, feedback):
    errors = [{"position": 0, "word": w, "expected": "ʃɪp", "actual": "".join(heard)} for w in words_with_errors]
    return {
        "score": score,
        "distance": 120,
        "acoustic_distance": 7.5,
        "differences": {
            "word_error_rate": 0.0 if transcribe == "SHIP" else 1.0,
            "phoneme_error_rate": 0.0 if heard == ["ʃ", "ɪ", "p"] else 0.6667,
            "errors": errors,
            "words_with_errors": words_with_errors,
            "expected_phones": [["ʃ", "ɪ", "p"]],
            "heard_phones": heard,
            "heard_phones_confidence": confidences,
            "feedback": feedback,
            "transcribe": transcribe,
        },
        "feedback": feedback,
        "transcribe": transcribe,
        "language": "en",
        "prosody": {"f0": [110.0, 112.5, 0.0], "energy": [10.0, 200.0, 5.0]},
    }


def row(sample_id, label, words_with_errors=(), feedback=FEEDBACK_OK, score=90.0, failed=False):
    """A minimal results row, as built by report.result_row."""
    return {
        "sample_id": sample_id,
        "label": label,
        "score": None if failed else score,
        "words_with_errors": None if failed else list(words_with_errors),
        "detection": report.detection(label, list(words_with_errors), failed=failed),
        "flags": None if failed else {"positive_feedback": "excellent" in feedback.lower()},
        "error": {"type": "RuntimeError", "message": "boom"} if failed else None,
    }


# ---------------------------------------------------------------------------
# Corpus validation
# ---------------------------------------------------------------------------

class TestCorpus(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        os.makedirs(os.path.join(self.dir, "audio"))
        self.csv = os.path.join(self.dir, "samples.csv")

    def tearDown(self):
        self.tmp.cleanup()

    def audio(self, name):
        write_wav(os.path.join(self.dir, "audio", name))
        return f"audio/{name}"

    def test_valid_corpus(self):
        write_csv(self.csv, HEADER + ["microphone", "take"], [
            ["ship_good_01", "ship", "ship", "good", self.audio("a.wav"), "headset", "1"],
            ["ship_as_chip_01", "ship", "chip", "intentional_error", self.audio("b.wav"), "headset", "2"],
        ])
        result = corpus.load_corpus(self.csv)
        self.assertEqual([s.sample_id for s in result.samples], ["ship_good_01", "ship_as_chip_01"])
        self.assertEqual(result.missing, [])
        sample = result.samples[1]
        self.assertEqual(sample.label, "intentional_error")
        self.assertEqual(sample.expected_text, "ship")
        # Extra columns are passed through, missing optional columns are empty.
        self.assertEqual(sample.fields["take"], "2")
        self.assertEqual(sample.fields["notes"], "")
        # Audio paths are relative to the CSV, not to the working directory.
        self.assertEqual(os.path.normcase(sample.audio_path),
                         os.path.normcase(os.path.join(self.dir, "audio", "b.wav")))
        self.assertEqual(sample.row, 3)

    def test_all_problems_are_reported_together(self):
        header = [c for c in HEADER if c != "intended_pronunciation"]
        write_csv(self.csv, header, [
            ["ok_01", "ship", "good", self.audio("ok.wav")],
            ["bad_label", "ship", "excellent", self.audio("x.wav")],
            ["ok_01", "ship", "good", self.audio("y.wav")],
            ["ship 01", "ship", "good", self.audio("z.wav")],
            ["empty_text", "  ", "good", self.audio("w.wav")],
            ["no_audio", "ship", "good", "audio/missing.wav"],
        ])
        with self.assertRaises(corpus.CorpusError) as ctx:
            corpus.load_corpus(self.csv)
        problems = "\n".join(ctx.exception.problems)
        self.assertIn("intended_pronunciation", problems)
        self.assertIn("line 3", problems)
        self.assertIn("excellent", problems)
        self.assertIn("duplicate sample_id 'ok_01'", problems)
        self.assertIn("'ship 01'", problems)
        self.assertIn("line 6", problems)
        self.assertIn("expected_text is empty", problems)
        self.assertIn("missing.wav", problems)
        self.assertEqual(len(ctx.exception.problems), 6)
        # The message lists everything too.
        self.assertIn("missing.wav", str(ctx.exception))

    def test_skip_missing(self):
        write_csv(self.csv, HEADER, [
            ["recorded", "ship", "ship", "good", self.audio("r.wav")],
            ["not_yet", "ship", "sheep", "intentional_error", "audio/not_yet.wav"],
        ])
        with self.assertRaises(corpus.CorpusError):
            corpus.load_corpus(self.csv)
        result = corpus.load_corpus(self.csv, skip_missing=True)
        self.assertEqual([s.sample_id for s in result.samples], ["recorded"])
        self.assertEqual([s.sample_id for s in result.missing], ["not_yet"])

    def test_skip_missing_does_not_hide_other_problems(self):
        write_csv(self.csv, HEADER, [["not_yet", "ship", "sheep", "wrong", "audio/not_yet.wav"]])
        with self.assertRaises(corpus.CorpusError) as ctx:
            corpus.load_corpus(self.csv, skip_missing=True)
        self.assertEqual(len(ctx.exception.problems), 1)
        self.assertIn("wrong", ctx.exception.problems[0])

    def test_bom_is_accepted(self):
        write_csv(self.csv, HEADER, [["ship_good_01", "ship", "ship", "good", self.audio("a.wav")]], bom=True)
        result = corpus.load_corpus(self.csv)
        self.assertEqual(result.samples[0].sample_id, "ship_good_01")

    def test_only(self):
        write_csv(self.csv, HEADER, [
            ["a", "ship", "ship", "good", self.audio("a.wav")],
            ["b", "ship", "ship", "good", "audio/b.wav"],  # missing, but not selected
        ])
        result = corpus.load_corpus(self.csv, only=["a"])
        self.assertEqual([s.sample_id for s in result.samples], ["a"])
        with self.assertRaises(corpus.CorpusError) as ctx:
            corpus.load_corpus(self.csv, only=["a", "nope"])
        self.assertIn("'nope'", ctx.exception.problems[0])

    def test_missing_corpus_file(self):
        with self.assertRaises(corpus.CorpusError) as ctx:
            corpus.load_corpus(os.path.join(self.dir, "nope.csv"))
        self.assertIn("not found", ctx.exception.problems[0])

    def test_validation_does_not_import_torch(self):
        write_csv(self.csv, HEADER, [["a", "ship", "ship", "good", "audio/a.wav"]])
        code = (
            "import sys; from lab import evaluate; "
            f"rc = evaluate.main(['--corpus', {self.csv!r}, '--validate-only']); "
            "print(rc, 'torch' in sys.modules, 'openpronounce' in sys.modules)"
        )
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True,
                             check=True).stdout.splitlines()[-1].split()
        self.assertNotEqual(out[0], "0")  # a.wav is missing
        self.assertEqual(out[1:], ["False", "False"])


# ---------------------------------------------------------------------------
# Derived signals
# ---------------------------------------------------------------------------

class TestDetection(unittest.TestCase):

    def test_mapping(self):
        cases = [
            ("good", [], False, "TN"),
            ("good", ["ship"], False, "FP"),
            ("intentional_error", ["ship"], False, "TP"),
            ("intentional_error", [], False, "FN"),
            ("uncertain", [], False, "NA"),
            ("uncertain", ["ship"], False, "NA"),
            ("good", [], True, "NA"),
            ("intentional_error", ["ship"], True, "NA"),
        ]
        for label, words, failed, expected in cases:
            with self.subTest(label=label, words=words, failed=failed):
                self.assertEqual(report.detection(label, words, failed=failed), expected)


class TestFlags(unittest.TestCase):

    def flags(self, **changes):
        result = fake_result(26.3, "TRIP", ["tʃ", "y", "p"], [0.9, 0.1, 0.9], [], FEEDBACK_OK)
        result["differences"].update(changes.pop("differences", {}))
        result.update(changes)
        return report.flags(result, "ship")

    def test_positive_feedback(self):
        self.assertTrue(self.flags()["positive_feedback"])
        self.assertTrue(self.flags(feedback="EXCELLENT")["positive_feedback"])
        self.assertFalse(self.flags(feedback=FEEDBACK_BAD)["positive_feedback"])

    def test_asr_mismatch(self):
        self.assertTrue(self.flags()["asr_mismatch"])
        self.assertFalse(self.flags(transcribe="SHIP")["asr_mismatch"])
        result = fake_result(98.9, "HELLO I AM A DEVELOPER", [], [], [], FEEDBACK_OK)
        self.assertFalse(report.flags(result, "Hello, I am a developer.")["asr_mismatch"])

    def test_no_expected_phones(self):
        self.assertFalse(self.flags()["no_expected_phones"])
        self.assertTrue(self.flags(differences={"expected_phones": [[]]})["no_expected_phones"])
        # Without the phone model, the text-based phonemes are used.
        result = fake_result(90.0, "SHIP", [], [], [], FEEDBACK_OK)
        del result["differences"]["expected_phones"]
        result["differences"]["expected_phonemes"] = ["ʃɪp"]
        self.assertFalse(report.flags(result, "ship")["no_expected_phones"])
        result["differences"]["expected_phonemes"] = []
        self.assertTrue(report.flags(result, "ship")["no_expected_phones"])

    def test_no_heard_phones(self):
        self.assertFalse(self.flags()["no_heard_phones"])
        self.assertTrue(self.flags(differences={"heard_phones": []})["no_heard_phones"])
        result = fake_result(90.0, "SHIP", [], [], [], FEEDBACK_OK)
        del result["differences"]["heard_phones"]
        self.assertTrue(report.flags(result, "ship")["no_heard_phones"])


class TestSummary(unittest.TestCase):

    def test_counts_precision_recall(self):
        rows = [
            row("g1", "good", score=95.0),
            row("g2", "good", ["ship"], FEEDBACK_BAD, score=70.0),
            row("e1", "intentional_error", ["ship"], FEEDBACK_BAD, score=30.0),
            row("e2", "intentional_error", score=26.3),
            row("e3", "intentional_error", ["ship"], FEEDBACK_BAD, score=40.0),
            row("u1", "uncertain", score=60.0),
            row("f1", "good", failed=True),
        ]
        summary = report.summarize(rows)
        self.assertEqual(summary["counts"], {"TP": 2, "FP": 1, "TN": 1, "FN": 1, "NA": 2})
        self.assertAlmostEqual(summary["precision"], 2 / 3)
        self.assertAlmostEqual(summary["recall"], 2 / 3)
        self.assertEqual(summary["scores"]["good"], {"n": 2, "min": 70.0, "mean": 82.5, "max": 95.0})
        self.assertEqual(summary["scores"]["uncertain"]["n"], 1)
        self.assertEqual(summary["positive_feedback_on_error"], ["e2"])
        self.assertEqual(summary["failed"], ["f1"])

    def test_zero_denominators(self):
        summary = report.summarize([row("u1", "uncertain"), row("g1", "good")])
        self.assertIsNone(summary["precision"])
        self.assertIsNone(summary["recall"])
        self.assertIsNone(summary["scores"]["intentional_error"])
        self.assertEqual(report.format_ratio(None), "n/a")
        self.assertEqual(report.format_ratio(0.5), "0.50")

    def test_empty(self):
        summary = report.summarize([])
        self.assertEqual(summary["counts"], {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "NA": 0})
        self.assertIsNone(summary["precision"])


# ---------------------------------------------------------------------------
# Real analyzer: capture of the phone recognition (pipeline faked)
# ---------------------------------------------------------------------------

class TestPipelineCapture(unittest.TestCase):

    def test_speech_calls_recognize_phones_through_the_module(self):
        # The capture in evaluate.analyze_with_pipeline relies on this call shape.
        from openpronounce import speech

        self.assertIn("phones.recognize_phones(", inspect.getsource(speech.compare_audio_with_text))

    def test_word_reports_contract(self):
        # Keys that report.result_row reads from the private phones._word_reports (espeak, no model).
        from openpronounce import phones

        (word,) = phones._word_reports(["tʃ", "ɪ", "p"], "ship", "en")
        self.assertLessEqual({"position", "word", "expected", "distance", "weighted_edits", "phones"}, set(word))
        self.assertEqual(word["expected"], ["ʃ", "ɪ", "p"])
        self.assertGreater(word["distance"], 0)
        self.assertLessEqual({"expected", "heard", "confidence"}, set(word["phones"][0]))

    def test_word_reports_of_the_captured_recognition(self):
        from openpronounce import phones

        recognition = phones.PhoneRecognition(["tʃ", "y", "p"], [0.9, 0.1, 0.9], [(0, 2), (2, 4), (4, 6)],
                                              np.zeros((6, 3)), ("<pad>", "a", "b"))
        reports = [{"position": 0, "word": "ship", "expected": ["ʃ", "ɪ", "p"], "actual": ["tʃ", "y", "p"],
                    "distance": 2, "phones": [], "weighted_edits": np.float64(1.5)}]

        def fake_compare(waveform, text, lang="en"):
            phones.recognize_phones(waveform, 16000, lang=lang)
            return {"language": "en", "score": 26.3}

        with patch.object(phones, "recognize_phones", return_value=recognition) as recognize, \
                patch.object(phones, "_word_reports", return_value=reports) as word_reports, \
                patch("openpronounce.speech.compare_audio_with_text", side_effect=fake_compare):
            result, diagnostics = evaluate.analyze_with_pipeline(np.zeros(16000), "ship", "en")
            self.assertIs(phones.recognize_phones, recognize)  # restored
        word_reports.assert_called_once_with(recognition, "ship", "en")
        self.assertEqual(result["score"], 26.3)
        self.assertEqual(diagnostics["word_reports"][0]["weighted_edits"], 1.5)
        json.dumps(diagnostics)  # numpy values were converted

    def test_no_diagnostics_without_phone_recognition(self):
        from openpronounce import phones

        original = phones.recognize_phones
        with patch("openpronounce.speech.compare_audio_with_text", return_value={"language": "en"}):
            _, diagnostics = evaluate.analyze_with_pipeline(np.zeros(16000), "ship", "en")
        self.assertIsNone(diagnostics)
        self.assertIs(phones.recognize_phones, original)

    def test_recognize_phones_is_restored_on_failure(self):
        from openpronounce import phones

        original = phones.recognize_phones
        with patch("openpronounce.speech.compare_audio_with_text", side_effect=RuntimeError("tts down")):
            with self.assertRaises(RuntimeError):
                evaluate.analyze_with_pipeline(np.zeros(16000), "ship", "en")
        self.assertIs(phones.recognize_phones, original)


# ---------------------------------------------------------------------------
# End to end with a fake analyzer
# ---------------------------------------------------------------------------

class TestEvaluate(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.out = os.path.join(self.dir, "runs")
        os.makedirs(os.path.join(self.dir, "audio"))
        for name in ("ship_good_01", "ship_as_chip_01", "ship_broken_01"):
            write_wav(os.path.join(self.dir, "audio", f"{name}.wav"))
        self.csv = os.path.join(self.dir, "samples.csv")
        write_csv(self.csv, HEADER + ["target_phoneme", "intended_substitution"], [
            ["ship_good_01", "ship", "ship", "good", "audio/ship_good_01.wav", "", ""],
            ["ship_as_chip_01", "ship", "chip", "intentional_error", "audio/ship_as_chip_01.wav", "ʃ", "ʃ->tʃ"],
            ["ship_broken_01", "ship", "ship", "uncertain", "audio/ship_broken_01.wav", "", ""],
        ])
        self.waveform_lengths = []

    def tearDown(self):
        self.tmp.cleanup()

    def fake_analyze(self, waveform, expected_text, lang):
        self.waveform_lengths.append(len(waveform))
        n = len(self.waveform_lengths)
        if n == 1:
            result = fake_result(94.61, "SHIP", ["ʃ", "ɪ", "p"], [0.99, 0.95, 0.98], [], FEEDBACK_OK)
            reports = [{"position": 0, "word": "ship", "expected": ["ʃ", "ɪ", "p"], "actual": ["ʃ", "ɪ", "p"],
                        "distance": 0, "phones": [], "weighted_edits": 0}]
            return result, {"word_reports": reports}
        if n == 2:
            # The ship -> chip case of docs/vision.md §7: low score, no flagged word, positive feedback.
            result = fake_result(26.3, "TRIP", ["tʃ", "y", "p"], [0.91, 0.12, 0.98], [], FEEDBACK_OK)
            reports = [{"position": 0, "word": "ship", "expected": ["ʃ", "ɪ", "p"], "actual": ["tʃ", "y", "p"],
                        "distance": 2, "weighted_edits": 1.1,
                        "phones": [{"expected": "ʃ", "heard": "tʃ", "confidence": 1.0},
                                   {"expected": "ɪ", "heard": "y", "confidence": 0.1},
                                   {"expected": "p", "heard": "p", "confidence": 0.0}]}]
            return result, {"word_reports": reports}
        raise RuntimeError("synthetic pipeline failure")

    def run_main(self, *extra, preflight=None):
        argv = ["--corpus", self.csv, "--out", self.out, *extra]
        return evaluate.main(argv, analyze=self.fake_analyze, preflight=preflight or (lambda lang: None))

    def test_end_to_end(self):
        self.assertEqual(self.run_main("--name", "smoke"), 0)
        run_dirs = glob.glob(os.path.join(self.out, "*-smoke"))
        self.assertEqual(len(run_dirs), 1)
        run_dir = run_dirs[0]
        for name in ("run.json", "results.jsonl", "report.md", "raw/ship_good_01.json",
                     "raw/ship_as_chip_01.json", "raw/ship_broken_01.json"):
            self.assertTrue(os.path.isfile(os.path.join(run_dir, name)), name)

        # The waveform handed to the pipeline is 16 kHz (the files are 44.1 kHz, 0.5 s).
        self.assertEqual(self.waveform_lengths[0], 8000)

        with open(os.path.join(run_dir, "raw", "ship_as_chip_01.json"), encoding="utf-8") as f:
            raw = json.load(f)
        self.assertEqual(raw["sample"]["intended_substitution"], "ʃ->tʃ")
        self.assertEqual(raw["result"]["prosody"]["f0"], [110.0, 112.5, 0.0])
        self.assertEqual(raw["diagnostics"]["word_reports"][0]["weighted_edits"], 1.1)
        self.assertIsNone(raw["error"])
        with open(os.path.join(self.dir, "audio", "ship_as_chip_01.wav"), "rb") as f:
            self.assertEqual(raw["audio"]["sha256"], hashlib.sha256(f.read()).hexdigest())
        self.assertAlmostEqual(raw["audio"]["duration_s"], 0.5, places=2)

        with open(os.path.join(run_dir, "raw", "ship_broken_01.json"), encoding="utf-8") as f:
            broken = json.load(f)
        self.assertIsNone(broken["result"])
        self.assertEqual(broken["error"]["type"], "RuntimeError")
        self.assertIn("synthetic pipeline failure", broken["error"]["traceback"])

        with open(os.path.join(run_dir, "results.jsonl"), encoding="utf-8") as f:
            lines = f.read().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertNotIn("prosody", "\n".join(lines))
        rows = {r["sample_id"]: r for r in map(json.loads, lines)}
        chip = rows["ship_as_chip_01"]
        self.assertEqual(chip["detection"], "FN")
        self.assertTrue(chip["flags"]["positive_feedback"])
        self.assertTrue(chip["flags"]["asr_mismatch"])
        self.assertEqual(chip["heard_phones"], ["tʃ", "y", "p"])
        self.assertEqual(chip["acoustic_distance"], 7.5)
        self.assertEqual(chip["word_edits"], [{"word": "ship", "n_phones": 3, "distance": 2,
                                               "weighted_edits": 1.1, "reported": False}])
        self.assertEqual(rows["ship_good_01"]["detection"], "TN")
        self.assertEqual(rows["ship_broken_01"]["detection"], "NA")
        self.assertEqual(rows["ship_broken_01"]["error"]["type"], "RuntimeError")

        with open(os.path.join(run_dir, "run.json"), encoding="utf-8") as f:
            run = json.load(f)
        self.assertEqual(run["lang"], "en")
        self.assertEqual(run["counts"], {"selected": 3, "processed": 2, "failed": 1, "missing": 0})
        self.assertIn("PHONE_ERROR_THRESHOLD", run["phone_thresholds"])
        self.assertIn("transformers", run["versions"])
        self.assertIn("commit", run["git"])
        self.assertEqual(len(run["corpus"]["sha256"]), 64)
        self.assertIsNotNone(run["finished_at"])

        with open(os.path.join(run_dir, "report.md"), encoding="utf-8") as f:
            markdown = f.read()
        for sample_id in ("ship_good_01", "ship_as_chip_01", "ship_broken_01"):
            self.assertIn(sample_id, markdown)
        self.assertIn("tʃ(0.91)", markdown)
        self.assertIn("ʃ → tʃ", markdown)
        self.assertIn("n/a", markdown)  # precision: no flagged word at all
        self.assertIn("synthetic pipeline failure", markdown)

    def test_validate_only(self):
        self.assertEqual(self.run_main("--validate-only"), 0)
        self.assertFalse(os.path.exists(self.out))
        self.assertEqual(self.waveform_lengths, [])
        os.remove(os.path.join(self.dir, "audio", "ship_good_01.wav"))
        self.assertNotEqual(self.run_main("--validate-only"), 0)
        self.assertEqual(self.run_main("--validate-only", "--skip-missing"), 0)

    def test_skip_missing_run(self):
        os.remove(os.path.join(self.dir, "audio", "ship_good_01.wav"))
        self.assertEqual(self.run_main("--skip-missing", "--only", "ship_good_01", "ship_as_chip_01"), 0)
        (run_dir,) = glob.glob(os.path.join(self.out, "*"))
        with open(os.path.join(run_dir, "run.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["counts"], {"selected": 1, "processed": 1, "failed": 0, "missing": 1})

    def test_invalid_corpus_stops_before_preflight(self):
        write_csv(self.csv, HEADER, [["x", "ship", "ship", "nope", "audio/ship_good_01.wav"]])
        preflight_calls = []
        self.assertNotEqual(self.run_main(preflight=preflight_calls.append), 0)
        self.assertEqual(preflight_calls, [])
        self.assertFalse(os.path.exists(self.out))

    def test_preflight_failure(self):
        def broken_preflight(lang):
            raise evaluate.PreflightError("espeak-ng returned no phones")

        self.assertNotEqual(self.run_main(preflight=broken_preflight), 0)
        self.assertEqual(self.waveform_lengths, [])
        self.assertFalse(os.path.exists(self.out))


class TestCorpusFiles(unittest.TestCase):
    """The committed corpus files are valid (audio presence aside)."""

    def test_samples_csv(self):
        result = corpus.load_corpus(os.path.join(REPO_ROOT, "lab", "corpus", "samples.csv"), skip_missing=True)
        ids = [s.sample_id for s in result.samples + result.missing]
        self.assertEqual(ids, ["ship_good_01", "ship_good_02", "ship_as_sheep_01", "ship_as_chip_01",
                               "ship_as_sip_01"])

    def test_example_csv(self):
        result = corpus.load_corpus(os.path.join(REPO_ROOT, "lab", "corpus", "example.csv"))
        self.assertEqual(result.missing, [])
        self.assertEqual({s.label for s in result.samples}, set(corpus.LABELS))
        self.assertTrue(all(s.fields["notes"] for s in result.samples))


if __name__ == "__main__":
    unittest.main()
