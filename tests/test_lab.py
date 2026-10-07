"""Tests of the evaluation harness (``lab/``). No model is loaded: the pipeline is faked."""

import contextlib
import csv
import datetime
import glob
import hashlib
import inspect
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import call, patch

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


def record(result, diagnostics=None, label="intentional_error", sample_id="ship_as_chip_01"):
    """A raw record, as built by evaluate.evaluate_sample."""
    return {
        "sample": {"sample_id": sample_id, "label": label, "expected_text": "ship", "intended_pronunciation": "chip"},
        "audio": {"path": "audio/x.wav", "sha256": "0" * 64, "duration_s": 0.5},
        "reference": None,
        "result": result,
        "diagnostics": diagnostics,
        "error": None,
        "elapsed_s": 1.0,
        "cold_start": False,
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

    def test_ids_that_differ_only_in_case_are_duplicates(self):
        # Ship_01 and ship_01 would write the same raw/<id>.json on Windows.
        write_csv(self.csv, HEADER, [
            ["Ship_01", "ship", "ship", "good", self.audio("a.wav")],
            ["ship_01", "ship", "ship", "good", self.audio("b.wav")],
        ])
        with self.assertRaises(corpus.CorpusError) as ctx:
            corpus.load_corpus(self.csv)
        (problem,) = ctx.exception.problems
        self.assertIn("line 3", problem)
        self.assertIn("'ship_01'", problem)
        self.assertIn("'Ship_01'", problem)

    def test_blank_leading_lines_are_skipped(self):
        with open(self.csv, "w", encoding="utf-8", newline="") as f:
            f.write("\n  \n,,,\n")
            writer = csv.writer(f)
            writer.writerow(HEADER)
            writer.writerow(["ship_good_01", "ship", "ship", "good", self.audio("a.wav")])
        result = corpus.load_corpus(self.csv)
        self.assertEqual([s.sample_id for s in result.samples], ["ship_good_01"])
        self.assertEqual(result.samples[0].row, 5)  # line numbers stay those of the file

    def test_only_blank_lines_is_empty(self):
        with open(self.csv, "w", encoding="utf-8") as f:
            f.write("\n\n")
        with self.assertRaises(corpus.CorpusError) as ctx:
            corpus.load_corpus(self.csv)
        self.assertIn("empty", ctx.exception.problems[0])

    def test_row_width(self):
        write_csv(self.csv, HEADER, [
            ["wide", "ship", "ship", "good", self.audio("a.wav"), "stray"],
            ["short", "ship", "ship"],  # missing cells are empty, so the required ones are reported
        ])
        with self.assertRaises(corpus.CorpusError) as ctx:
            corpus.load_corpus(self.csv)
        self.assertEqual(ctx.exception.problems, [
            "line 2: 6 fields, the header has 5",
            "line 3: label '' must be one of: good, intentional_error, uncertain",
            "line 3: audio_file is empty",
        ])

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
            # No words_with_errors in the result: unknown, not "nothing flagged".
            ("good", None, False, "NA"),
            ("intentional_error", None, False, "NA"),
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

    def test_missing_words_with_errors(self):
        self.assertFalse(self.flags()["missing_words_with_errors"])
        result = fake_result(26.3, "TRIP", ["tʃ", "y", "p"], [0.9, 0.1, 0.9], [], FEEDBACK_OK)
        del result["differences"]["words_with_errors"]
        self.assertTrue(report.flags(result, "ship")["missing_words_with_errors"])
        row_ = report.result_row(record(result))
        self.assertEqual(row_["detection"], "NA")
        summary = report.summarize([row_])
        self.assertEqual(summary["counts"]["FN"], 0)
        self.assertEqual(summary["unknown_detection"], ["ship_as_chip_01"])
        markdown = report.render_markdown({"summary": summary}, [row_], {})
        self.assertIn("no `words_with_errors`", markdown)
        self.assertIn("ship_as_chip_01", markdown)


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


class TestMarkdown(unittest.TestCase):

    def test_tts_label(self):
        def header(tts):
            return "\n".join(report._header({"tts": tts}, []))

        # The gTTS "voice" is the Google Translate domain.
        self.assertIn("gtts (tld com)", header({"backend": "gtts", "voice": "com"}))
        self.assertIn("piper (voice en_US-lessac-medium)", header({"backend": "piper", "voice": "en_US-lessac-medium"}))
        self.assertIn("unavailable: unknown backend 'x'", header({"error": "unknown backend 'x'"}))

    def test_every_heard_phone_is_shown(self):
        self.assertEqual(report._heard_phones_text(["tʃ", "y", "p"], [0.91, 0.12, 0.98]), "tʃ(0.91) y(0.12) p(0.98)")
        self.assertEqual(report._heard_phones_text(["tʃ", "y", "p"], [0.91]), "tʃ(0.91) y(?) p(?)")
        self.assertEqual(report._heard_phones_text(["tʃ", "y"], None), "tʃ(?) y(?)")

    def test_word_level_positions_and_decimals(self):
        result = fake_result(40.0, "SHIP SHIP", ["ʃ", "ɪ", "p", "tʃ", "ɪ", "p"], [0.9] * 6, [], FEEDBACK_OK)
        reports = [
            {"position": 0, "word": "ship", "expected": ["ʃ", "ɪ", "p"], "actual": ["ʃ", "ɪ", "p"],
             "distance": 0, "phones": [], "weighted_edits": 0.0},
            {"position": 1, "word": "ship", "expected": ["ʃ", "ɪ", "p"], "actual": ["tʃ", "ɪ", "p"],
             "distance": 1, "weighted_edits": 1.199,
             "phones": [{"expected": "ʃ", "heard": "tʃ", "confidence": 1.199}]},
        ]
        row_ = report.result_row(record(result, {"word_reports": reports}))
        self.assertEqual([w["position"] for w in row_["word_edits"]], [0, 1])
        run = {"phone_thresholds": {"PHONE_ERROR_THRESHOLD": 0.4, "PHONE_ERROR_MIN_EDITS": 2}}
        markdown = "\n".join(report._word_section(run, [row_], {"ship_as_chip_01": reports}))
        self.assertIn("| ship #0 |", markdown)
        self.assertIn("| ship #1 |", markdown)
        self.assertIn("- ship #1: ʃ → tʃ (1.20)", markdown)
        # Three decimals: 1.199 is below the 1.200 at which a 3-phone word is reported.
        self.assertIn("| 1.199 | 1.200 | no |", markdown)

    def test_diagnostics_error_is_surfaced(self):
        result = fake_result(26.3, "TRIP", ["tʃ", "y", "p"], [0.91, 0.12, 0.98], [], FEEDBACK_OK)
        diagnostics = {"word_reports": None, "error": {"stage": "diagnostics", "type": "KeyError",
                                                       "message": "'position'", "traceback": "..."}}
        row_ = report.result_row(record(result, diagnostics))
        # The pipeline result still drives detection and flags.
        self.assertEqual(row_["detection"], "FN")
        self.assertTrue(row_["flags"]["positive_feedback"])
        self.assertEqual(row_["diagnostics_error"], {"stage": "diagnostics", "type": "KeyError", "message": "'position'"})
        markdown = "\n".join(report._word_section({}, [row_], {}))
        self.assertIn("Phone-level diagnostics failed: KeyError: 'position'", markdown)

    def test_cold_start_note(self):
        rows = [dict(row("g1", "good"), elapsed_s=12.5, cold_start=True),
                dict(row("g2", "good"), elapsed_s=1.5, cold_start=False)]
        header = "\n".join(report._header({}, rows))
        self.assertIn("g1 (12.5 s) includes lazy model loading", header)


class TestReportingRule(unittest.TestCase):
    """report._reporting_value restates the compare_phones rule for display; it must not drift."""

    def test_matches_compare_phones(self):
        from openpronounce import phones

        thresholds = {name: getattr(phones, name) for name in evaluate.PHONE_THRESHOLDS}
        for n_phones in range(1, 9):
            value = report._reporting_value(n_phones, thresholds)
            for edits in (value - 1e-6, value + 1e-6):
                word = {"position": 0, "word": "w", "expected": ["p"] * n_phones, "actual": [], "distance": 1,
                        "phones": [], "weighted_edits": edits}
                # Synthetic word report and expected phones: no espeak-ng, no model.
                with patch.object(phones, "get_expected_phones", return_value=(["w"], [["p"] * n_phones])), \
                        patch.object(phones, "_word_reports", return_value=[word]):
                    reported = bool(phones.compare_phones([], "w", "en")["words_with_errors"])
                with self.subTest(n_phones=n_phones, edits=edits):
                    self.assertEqual(reported, edits > value)


# ---------------------------------------------------------------------------
# Real analyzer: capture of the phone recognition (pipeline faked)
# ---------------------------------------------------------------------------

class TestPipelineCapture(unittest.TestCase):

    def test_speech_calls_the_captured_functions_through_their_modules(self):
        # The captures in evaluate.analyze_with_pipeline rely on this call shape.
        from openpronounce import speech

        source = inspect.getsource(speech.compare_audio_with_text)
        self.assertIn("phones.recognize_phones(", source)
        self.assertIn("audio.text2speech(", source)

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

    def test_no_word_reports_without_phone_recognition(self):
        from openpronounce import phones

        original = phones.recognize_phones
        with patch("openpronounce.speech.compare_audio_with_text", return_value={"language": "en"}):
            _, diagnostics = evaluate.analyze_with_pipeline(np.zeros(16000), "ship", "en")
        self.assertIsNone(diagnostics["word_reports"])
        self.assertIsNone(diagnostics["reference_audio"])
        self.assertNotIn("error", diagnostics)
        self.assertIs(phones.recognize_phones, original)

    def test_captured_functions_are_restored_on_failure(self):
        from openpronounce import audio, phones

        originals = phones.recognize_phones, audio.text2speech
        with patch("openpronounce.speech.compare_audio_with_text", side_effect=RuntimeError("tts down")):
            with self.assertRaises(RuntimeError):
                evaluate.analyze_with_pipeline(np.zeros(16000), "ship", "en")
        self.assertEqual((phones.recognize_phones, audio.text2speech), originals)

    def test_reference_audio_is_captured(self):
        from openpronounce import audio

        def fake_compare(waveform, text, lang="en"):
            audio.text2speech(text, lang=lang)
            return {"language": "en", "score": 26.3}

        with patch.object(audio, "text2speech", return_value="cache/tts-1.wav") as text2speech, \
                patch("openpronounce.speech.compare_audio_with_text", side_effect=fake_compare):
            _, diagnostics = evaluate.analyze_with_pipeline(np.zeros(16000), "ship", "en")
            self.assertIs(audio.text2speech, text2speech)  # restored
        text2speech.assert_called_once_with("ship", lang="en")
        self.assertEqual(diagnostics["reference_audio"], "cache/tts-1.wav")

    def test_diagnostics_failure_keeps_the_result(self):
        from openpronounce import phones

        def fake_compare(waveform, text, lang="en"):
            phones.recognize_phones(waveform, 16000, lang=lang)
            return {"language": "en", "score": 26.3}

        with patch.object(phones, "recognize_phones", return_value=["tʃ", "y", "p"]), \
                patch.object(phones, "_word_reports", side_effect=KeyError("position")), \
                patch("openpronounce.speech.compare_audio_with_text", side_effect=fake_compare):
            result, diagnostics = evaluate.analyze_with_pipeline(np.zeros(16000), "ship", "en")
        self.assertEqual(result["score"], 26.3)
        self.assertIsNone(diagnostics["word_reports"])
        self.assertEqual(diagnostics["error"]["type"], "KeyError")
        self.assertIn("position", diagnostics["error"]["traceback"])

    def test_tts_preflight(self):
        from openpronounce import audio

        with patch.object(audio, "text2speech", side_effect=OSError("no network")) as text2speech:
            with self.assertRaises(evaluate.PreflightError) as ctx:
                evaluate.check_tts("en", ["ship", "sheep"])
        text2speech.assert_called_once_with("ship", lang="en")  # the pipeline's own call
        self.assertIn("'ship'", str(ctx.exception))
        self.assertIn("no network", str(ctx.exception))
        with patch.object(audio, "text2speech", return_value="cache/tts.wav") as text2speech:
            evaluate.check_tts("en", ["ship", "sheep"])
        self.assertEqual(text2speech.call_args_list, [call("ship", lang="en"), call("sheep", lang="en")])

    def test_model_revision(self):
        snapshot = os.path.join("hub", "models--facebook--x", "snapshots", "abc123", "config.json")
        with patch("huggingface_hub.try_to_load_from_cache", return_value=snapshot) as lookup:
            self.assertEqual(evaluate.model_revision("facebook/x"), "abc123")
        lookup.assert_called_once_with("facebook/x", "config.json")
        # Not cached, cached as missing, or not a repo id (a local directory): unknown.
        for outcome in ({"return_value": None}, {"return_value": object()},
                        {"side_effect": ValueError("Repo id must be in the form 'namespace/repo_name'")}):
            with self.subTest(outcome=outcome), patch("huggingface_hub.try_to_load_from_cache", **outcome):
                self.assertIsNone(evaluate.model_revision("facebook/x"))


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
        # The TTS reference the fake pipeline "compared against" (a cache file outside the run).
        self.reference = os.path.join(self.dir, "tts-ship.wav")
        write_wav(self.reference, sr=16000)
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
            return result, {"word_reports": reports, "reference_audio": self.reference}
        if n == 2:
            # The ship -> chip case of docs/vision.md §7: low score, no flagged word, positive feedback.
            result = fake_result(26.3, "TRIP", ["tʃ", "y", "p"], [0.91, 0.12, 0.98], [], FEEDBACK_OK)
            reports = [{"position": 0, "word": "ship", "expected": ["ʃ", "ɪ", "p"], "actual": ["tʃ", "y", "p"],
                        "distance": 2, "weighted_edits": 1.1,
                        "phones": [{"expected": "ʃ", "heard": "tʃ", "confidence": 1.0},
                                   {"expected": "ɪ", "heard": "y", "confidence": 0.1},
                                   {"expected": "p", "heard": "p", "confidence": 0.0}]}]
            return result, {"word_reports": reports, "reference_audio": self.reference}
        raise RuntimeError("synthetic pipeline failure")

    def run_main(self, *extra, preflight=None, analyze=None):
        argv = ["--corpus", self.csv, "--out", self.out, *extra]
        return evaluate.main(argv, analyze=analyze or self.fake_analyze,
                             preflight=preflight or (lambda lang, texts: None))

    def run_dir(self):
        (run_dir,) = glob.glob(os.path.join(self.out, "*"))
        return run_dir

    def read_json(self, *path):
        with open(os.path.join(*path), encoding="utf-8") as f:
            return json.load(f)

    def test_end_to_end(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(self.run_main("--name", "smoke"), 0)
        run_dirs = glob.glob(os.path.join(self.out, "*-smoke"))
        self.assertEqual(len(run_dirs), 1)
        run_dir = run_dirs[0]
        for name in ("run.json", "results.jsonl", "report.md", "raw/ship_good_01.json",
                     "raw/ship_as_chip_01.json", "raw/ship_broken_01.json"):
            self.assertTrue(os.path.isfile(os.path.join(run_dir, name)), name)
        # The final console line shows the run directory with forward slashes only.
        last_line = stdout.getvalue().splitlines()[-1]
        self.assertTrue(last_line.startswith(evaluate._display_path(run_dir) + ":"), last_line)
        self.assertNotIn("\\", last_line)

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

        # The TTS reference behind acoustic_distance is copied into the run and hashed.
        with open(self.reference, "rb") as f:
            reference_sha = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(raw["reference"]["path"], "references/ship_as_chip_01.wav")
        self.assertEqual(raw["reference"]["sha256"], reference_sha)
        self.assertEqual(raw["reference"]["source"], evaluate._display_path(self.reference))
        self.assertNotIn("reference_audio", raw["diagnostics"])
        with open(os.path.join(run_dir, "references", "ship_as_chip_01.wav"), "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), reference_sha)
        # Only the first analyzed sample pays the lazy model loading.
        self.assertTrue(self.read_json(run_dir, "raw", "ship_good_01.json")["cold_start"])
        self.assertFalse(raw["cold_start"])

        with open(os.path.join(run_dir, "raw", "ship_broken_01.json"), encoding="utf-8") as f:
            broken = json.load(f)
        self.assertIsNone(broken["result"])
        self.assertIsNone(broken["reference"])
        self.assertEqual(broken["error"]["type"], "RuntimeError")
        self.assertEqual(broken["error"]["stage"], "analysis")
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
        self.assertEqual(chip["word_edits"], [{"word": "ship", "position": 0, "n_phones": 3, "distance": 2,
                                               "weighted_edits": 1.1, "reported": False}])
        self.assertEqual(chip["reference_sha256"], reference_sha)
        self.assertEqual(rows["ship_good_01"]["detection"], "TN")
        self.assertEqual(rows["ship_broken_01"]["detection"], "NA")
        self.assertEqual(rows["ship_broken_01"]["error"]["type"], "RuntimeError")

        with open(os.path.join(run_dir, "run.json"), encoding="utf-8") as f:
            run = json.load(f)
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["lang"], "en")
        self.assertEqual(run["counts"], {"selected": 3, "processed": 2, "failed": 1, "not_run": 0, "missing": 0})
        self.assertIn("PHONE_ERROR_THRESHOLD", run["phone_thresholds"])
        self.assertIn("transformers", run["versions"])
        self.assertIn("commit", run["git"])
        self.assertEqual(set(run["model_revisions"]),
                         {run["asr_model"], run["embedding_model"], run["phone_model"]["name"]})
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
        self.assertIn("ship_good_01 (", markdown)  # the cold-start note
        self.assertIn("includes lazy model loading", markdown)

    def test_preflight_gets_the_distinct_texts(self):
        calls = []
        self.assertEqual(self.run_main(preflight=lambda lang, texts: calls.append((lang, texts))), 0)
        self.assertEqual(calls, [("en", ["ship"])])

    def test_cold_start_is_the_first_analyzed_sample(self):
        # The first sample fails before reaching the pipeline, so the second one loads the models.
        load_audio = evaluate.load_audio

        def flaky_load(path):
            if "ship_good_01" in path:
                raise RuntimeError("cannot decode")
            return load_audio(path)

        with patch.object(evaluate, "load_audio", side_effect=flaky_load):
            self.assertEqual(self.run_main(), 0)
        raw = [self.read_json(self.run_dir(), "raw", f"{i}.json") for i in
               ("ship_good_01", "ship_as_chip_01", "ship_broken_01")]
        self.assertEqual([r["cold_start"] for r in raw], [False, True, False])

    def test_run_json_is_written_at_start(self):
        statuses = []

        def analyze(waveform, expected_text, lang):
            statuses.append(self.read_json(self.run_dir(), "run.json")["status"])
            return self.fake_analyze(waveform, expected_text, lang)

        self.assertEqual(self.run_main(analyze=analyze), 0)
        self.assertEqual(statuses[0], "running")
        self.assertEqual(self.read_json(self.run_dir(), "run.json")["status"], "completed")

    def test_interrupted_run_keeps_partial_results(self):
        def analyze(waveform, expected_text, lang):
            if self.waveform_lengths:
                raise KeyboardInterrupt
            return self.fake_analyze(waveform, expected_text, lang)

        self.assertEqual(self.run_main(analyze=analyze), evaluate.EXIT_INTERRUPTED)
        run_dir = self.run_dir()
        run = self.read_json(run_dir, "run.json")
        self.assertEqual(run["status"], "interrupted")
        self.assertIsNotNone(run["finished_at"])
        self.assertEqual(run["counts"], {"selected": 3, "processed": 1, "failed": 0, "not_run": 2, "missing": 0})
        with open(os.path.join(run_dir, "results.jsonl"), encoding="utf-8") as f:
            self.assertEqual(len(f.read().splitlines()), 1)
        with open(os.path.join(run_dir, "report.md"), encoding="utf-8") as f:
            markdown = f.read()
        self.assertIn("interrupted", markdown)
        self.assertIn("ship_good_01", markdown)
        self.assertNotIn("ship_as_chip_01", markdown)

    def test_aborted_run_is_recorded_and_raised(self):
        with patch.object(evaluate, "derive_row", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.run_main()
        run = self.read_json(self.run_dir(), "run.json")
        self.assertEqual(run["status"], "aborted")
        self.assertIn("disk full", run["abort_error"])

    def test_row_derivation_failure_is_isolated(self):
        def analyze(waveform, expected_text, lang):
            result, diagnostics = self.fake_analyze(waveform, expected_text, lang)
            if len(self.waveform_lengths) == 1:
                result["differences"]["errors"] = ["not a dict"]  # breaks report.result_row
            return result, diagnostics

        self.assertEqual(self.run_main(analyze=analyze), 0)
        run_dir = self.run_dir()
        raw = self.read_json(run_dir, "raw", "ship_good_01.json")
        self.assertEqual(raw["result"]["score"], 94.61)  # the pipeline output is kept
        self.assertEqual(raw["error"]["stage"], "row")
        self.assertEqual(raw["error"]["type"], "AttributeError")
        with open(os.path.join(run_dir, "results.jsonl"), encoding="utf-8") as f:
            rows = {r["sample_id"]: r for r in map(json.loads, f.read().splitlines())}
        self.assertEqual(rows["ship_good_01"]["detection"], "NA")
        self.assertEqual(rows["ship_good_01"]["error"]["stage"], "row")
        self.assertEqual(rows["ship_as_chip_01"]["detection"], "FN")  # the run went on
        self.assertEqual(self.read_json(run_dir, "run.json")["counts"]["failed"], 2)

    def test_no_run_dir_when_run_metadata_fails(self):
        with patch.object(evaluate, "pipeline_info", side_effect=RuntimeError("torch broken")):
            with self.assertRaises(RuntimeError):
                self.run_main()
        self.assertFalse(os.path.exists(self.out))

    def test_exit_codes(self):
        codes = [evaluate.EXIT_INVALID_CORPUS, evaluate.EXIT_PREFLIGHT_FAILED, evaluate.EXIT_NOTHING_TO_EVALUATE,
                 evaluate.EXIT_INTERRUPTED]
        self.assertEqual(len(set(codes)), len(codes))
        # 0 done, 1 Python's own exit code for a traceback, 2 argparse's for a usage error.
        self.assertTrue(set(codes).isdisjoint({0, 1, 2}))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            self.run_main("--name", "not valid")
        self.assertEqual(ctx.exception.code, 2)
        for name in ("ship_good_01", "ship_as_chip_01", "ship_broken_01"):
            os.remove(os.path.join(self.dir, "audio", f"{name}.wav"))
        self.assertEqual(self.run_main("--skip-missing"), evaluate.EXIT_NOTHING_TO_EVALUATE)

    def test_run_dir_naming(self):
        with patch.object(evaluate, "datetime") as clock:
            clock.datetime.now.return_value = datetime.datetime(2026, 10, 7, 9, 48, 35)
            first = evaluate.make_run_dir(self.out, "baseline")
            second = evaluate.make_run_dir(self.out, "baseline")
            unnamed = evaluate.make_run_dir(self.out, None)
        self.assertEqual([os.path.basename(p) for p in (first, second, unnamed)],
                         ["20261007-094835-baseline", "20261007-094835-baseline-2", "20261007-094835"])
        self.assertTrue(os.path.isdir(os.path.join(first, "raw")))

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
            self.assertEqual(json.load(f)["counts"],
                             {"selected": 1, "processed": 1, "failed": 0, "not_run": 0, "missing": 1})

    def test_invalid_corpus_stops_before_preflight(self):
        write_csv(self.csv, HEADER, [["x", "ship", "ship", "nope", "audio/ship_good_01.wav"]])
        preflight_calls = []
        self.assertEqual(self.run_main(preflight=lambda *a: preflight_calls.append(a)), evaluate.EXIT_INVALID_CORPUS)
        self.assertEqual(preflight_calls, [])
        self.assertFalse(os.path.exists(self.out))

    def test_preflight_failure(self):
        def broken_preflight(lang, texts):
            raise evaluate.PreflightError("espeak-ng returned no phones")

        self.assertEqual(self.run_main(preflight=broken_preflight), evaluate.EXIT_PREFLIGHT_FAILED)
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
