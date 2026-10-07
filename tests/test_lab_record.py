"""Tests of the recording helper (``lab/record.py``). No microphone and no ffmpeg: capture is faked."""

import contextlib
import csv
import datetime
import glob
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from lab import corpus, evaluate, record

SR = 16000
HEADER = ["sample_id", "expected_text", "intended_pronunciation", "label", "audio_file"]
MIC = "Micrófono (Realtek(R) Audio)"
SETTINGS = record.Settings(max_seconds=6.0, pad=0.2, top_db=40.0)

# Real stderr of `ffmpeg -list_devices true -f dshow -i dummy` (ffmpeg 9), plus a device of both kinds.
LISTING = """\
[in#0 @ 00000179a10e0840] "ASUS FHD webcam" (video)
[in#0 @ 00000179a10e0840]   Alternative name "@device_pnp_\\\\?\\usb#vid_3277&pid_0010&mi_00#7&df5034d\\global"
[in#0 @ 00000179a10e0840] "Micrófono (Realtek(R) Audio)" (audio)
[in#0 @ 00000179a10e0840]   Alternative name "@device_cm_{33D9A762-90C8-11D0-BD43}\\wave_{41BAEBA6-11AC}"
[in#0 @ 00000179a10e0840] "Varios micrófonos (Realtek(R) Audio)" (audio)
[in#0 @ 00000179a10e0840]   Alternative name "@device_cm_{33D9A762-90C8-11D0-BD43}\\wave_{A98AE997-CAF2}"
[in#0 @ 00000179a10e0840] "Capture card" (audio, video)
Error opening input file dummy.
"""
# Older ffmpeg builds list devices under section headers instead of tagging each line.
LISTING_OLD = """\
[dshow @ 000001] DirectShow video devices (some may be both video and audio devices)
[dshow @ 000001]  "Integrated Camera"
[dshow @ 000001]     Alternative name "@device_pnp_\\\\?\\usb#vid_04f2"
[dshow @ 000001] DirectShow audio devices
[dshow @ 000001]  "Microphone (USB Audio Device)"
[dshow @ 000001]     Alternative name "@device_cm_{33D9A762}\\wave_{1234}"
dummy: Immediate exit requested
"""


def tone(seconds, amp=0.3, freq=220.0):
    t = np.arange(int(round(SR * seconds))) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def noise(seconds, level=1e-3, seed=0):
    return np.random.default_rng(seed).normal(0.0, level, int(round(SR * seconds)))


def silence(seconds):
    return np.zeros(int(round(SR * seconds)))


def clean_take():
    """A word in a quiet but live room: low noise around it, never exact zero."""
    return np.concatenate([noise(0.5, seed=1), tone(0.4), noise(0.5, seed=2)])


def gated_take(before=0.5, word=0.4, after=0.5):
    """A word surrounded by exact digital zero, as with Windows noise suppression on."""
    return np.concatenate([silence(before), tone(word), silence(after)])


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sample(sample_id="ship_as_sheep_01", label="intentional_error", intended="sheep", expected="ship"):
    fields = {"sample_id": sample_id, "expected_text": expected, "intended_pronunciation": intended,
              "label": label, "audio_file": f"audio/{sample_id}.wav"}
    return corpus.Sample(2, fields, os.path.join("corpus", "audio", f"{sample_id}.wav"))


class Script:
    """Answers to ``input()`` prompts, in order; an exception class in the list is raised instead."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt=""):
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError(f"unexpected prompt {prompt!r}")
        answer = self.answers.pop(0)
        if isinstance(answer, type) and issubclass(answer, BaseException):
            raise answer
        return answer


class FakeCapture:
    """Stands in for ffmpeg: ``stop`` writes ``signal`` (16 kHz, 16-bit) where ffmpeg would."""

    def __init__(self, command, output, signal):
        self.command, self.output, self.signal = command, output, signal
        self.stopped = self.killed = False

    def wait_until_recording(self):
        if isinstance(self.signal, Exception):
            raise self.signal

    def stop(self):
        sf.write(self.output, self.signal, SR, subtype="PCM_16")
        self.stopped = True

    def kill(self):
        self.killed = True


class FakeRunner:
    def __init__(self, *takes):
        self.takes = list(takes)
        self.captures = []

    def __call__(self, command, output):
        capture = FakeCapture(command, output, self.takes.pop(0))
        self.captures.append(capture)
        return capture


# ---------------------------------------------------------------------------
# Pure parts
# ---------------------------------------------------------------------------

class TestDevices(unittest.TestCase):

    def test_audio_devices_with_accents_in_order(self):
        self.assertEqual(record.parse_dshow_devices(LISTING),
                         [MIC, "Varios micrófonos (Realtek(R) Audio)", "Capture card"])

    def test_older_section_format(self):
        self.assertEqual(record.parse_dshow_devices(LISTING_OLD), ["Microphone (USB Audio Device)"])

    def test_no_devices(self):
        self.assertEqual(record.parse_dshow_devices("Error opening input file dummy.\n"), [])
        self.assertEqual(record.parse_dshow_devices('[in#0 @ 01] "ASUS FHD webcam" (video)\n'), [])


class TestCommand(unittest.TestCase):

    def value(self, command, flag):
        return command[command.index(flag) + 1]

    def test_capture_command(self):
        command = record.ffmpeg_command(MIC, "lab/corpus/audio/raw/x.take.wav", 6.0, ffmpeg="C:/bin/ffmpeg.exe")
        self.assertEqual(command[0], "C:/bin/ffmpeg.exe")
        self.assertEqual(command[-1], "lab/corpus/audio/raw/x.take.wav")
        self.assertEqual(self.value(command, "-f"), "dshow")
        # One argument, no shell quoting: the device name goes to ffmpeg as is.
        self.assertEqual(self.value(command, "-i"), f"audio={MIC}")
        self.assertLess(command.index("-f"), command.index("-i"))
        for flag, expected in (("-ac", "1"), ("-ar", "16000"), ("-c:a", "pcm_s16le"), ("-t", "6")):
            self.assertEqual(self.value(command, flag), expected)
            self.assertGreater(command.index(flag), command.index("-i"), f"{flag} is an output option")
        self.assertIn("-y", command)
        # stdin stays open: `q` on it is how a take is stopped.
        self.assertNotIn("-nostdin", command)

    def test_fractional_max_seconds(self):
        self.assertEqual(self.value(record.ffmpeg_command(MIC, "x.wav", 2.5), "-t"), "2.5")


class TestTrim(unittest.TestCase):

    def test_pads_around_the_speech(self):
        signal = gated_take(before=1.0, word=0.4, after=1.0)
        trim = record.trim_take(signal, SR, top_db=40, pad=0.2)
        # librosa works on 2048-sample frames: the found speech is within ~0.15 s of the tone.
        self.assertAlmostEqual(trim.speech_start / SR, 1.0, delta=0.15)
        self.assertAlmostEqual(trim.speech_end / SR, 1.4, delta=0.15)
        self.assertEqual(trim.start, trim.speech_start - int(0.2 * SR))
        self.assertEqual(trim.end, trim.speech_end + int(0.2 * SR))

    def test_padding_is_clamped_at_the_edges(self):
        at_start = np.concatenate([tone(0.4), silence(0.05)])
        trim = record.trim_take(at_start, SR, top_db=40, pad=0.2)
        self.assertEqual(trim.start, 0)
        at_end = np.concatenate([silence(0.05), tone(0.4)])
        trim = record.trim_take(at_end, SR, top_db=40, pad=0.2)
        self.assertEqual(trim.end, len(at_end))

    def test_zero_pad(self):
        trim = record.trim_take(gated_take(), SR, top_db=40, pad=0)
        self.assertEqual((trim.start, trim.end), (trim.speech_start, trim.speech_end))

    def test_silence_has_no_speech(self):
        # librosa alone would keep all of an all-zero signal (every frame equals the maximum).
        self.assertIsNone(record.trim_take(silence(1.0), SR, top_db=40, pad=0.2))
        self.assertIsNone(record.trim_take(np.zeros(0), SR, top_db=40, pad=0.2))


class TestStats(unittest.TestCase):

    def analyze(self, signal, max_seconds=6.0):
        trim = record.trim_take(signal, SR, top_db=40, pad=0.2)
        stats = record.take_stats(signal, SR, trim, top_db=40)
        return trim, stats, record.take_warnings(stats, max_seconds)

    def test_clean_take(self):
        trim, stats, warnings = self.analyze(clean_take())
        self.assertEqual(stats["raw_duration_s"], 1.4)
        self.assertEqual(stats["trimmed_duration_s"], round((trim.end - trim.start) / SR, 3))
        self.assertAlmostEqual(stats["speech_duration_s"], 0.4, delta=0.05)
        self.assertAlmostEqual(stats["peak"], 0.3, delta=0.01)
        # RMS of the trimmed take, mostly the tone (0.3 / sqrt(2) ~ 0.21).
        self.assertGreater(stats["rms"], 0.1)
        self.assertLess(stats["rms"], 0.22)
        self.assertGreater(stats["noise_floor"], 5e-4)
        self.assertLess(stats["noise_floor"], 2e-3)
        self.assertEqual(warnings, [])

    def test_digital_gating(self):
        _, stats, warnings = self.analyze(gated_take())
        self.assertEqual(stats["noise_floor"], 0)
        self.assertEqual(warnings, ["digital_gating"])
        self.assertIn("Audio enhancements", record.WARNINGS["digital_gating"])

    def test_digital_gating_with_one_bit_values(self):
        # As in the archived T4a takes: 98.4 % exact zeros, the rest +-1 LSB (noise floor ~4e-6, not 0).
        lsb = np.random.default_rng(3).choice([-1, 0, 1], size=int(0.5 * SR), p=[0.008, 0.984, 0.008]) / 32768
        _, stats, warnings = self.analyze(np.concatenate([lsb, tone(0.4), lsb]))
        self.assertGreater(stats["noise_floor"], 0)
        self.assertEqual(warnings, ["digital_gating"])
        # A live microphone in a quiet room sits far above one 16-bit step.
        self.assertNotIn("digital_gating", self.analyze(clean_take())[2])

    def test_clipping(self):
        clipped = np.concatenate([noise(0.5), np.clip(tone(0.4, amp=1.5), -1, 1), noise(0.5)])
        self.assertEqual(self.analyze(clipped)[2], ["clipping"])

    def test_too_quiet(self):
        quiet = np.concatenate([noise(0.5, level=1e-4), tone(0.4, amp=0.01), noise(0.5, level=1e-4)])
        self.assertEqual(self.analyze(quiet)[2], ["too_quiet"])

    def test_speech_too_short(self):
        short = np.concatenate([noise(0.5), tone(0.1), noise(0.5)])
        _, stats, warnings = self.analyze(short)
        self.assertAlmostEqual(stats["speech_duration_s"], 0.1, delta=0.03)
        self.assertEqual(warnings, ["speech_too_short"])

    def test_hit_max_seconds(self):
        long_take = np.concatenate([noise(1.0), tone(0.4), noise(0.6)])
        self.assertEqual(self.analyze(long_take, max_seconds=2.0)[2], ["max_seconds"])
        self.assertEqual(self.analyze(long_take, max_seconds=2.5)[2], [])

    def test_no_speech(self):
        trim, stats, warnings = self.analyze(silence(1.0))
        self.assertIsNone(trim)
        self.assertIsNone(stats["trimmed_duration_s"])
        self.assertIsNone(stats["speech_duration_s"])
        self.assertEqual(stats["peak"], 0)
        self.assertEqual(warnings, ["too_quiet", "digital_gating"])


class TestPrompt(unittest.TestCase):

    def test_prompts(self):
        self.assertEqual(record.prompt_text(sample(label="good", intended="ship")), 'say "ship" as well as you can')
        self.assertEqual(record.prompt_text(sample()), 'say "sheep" on purpose (target: ship)')
        self.assertEqual(record.prompt_text(sample(intended="")), "make the intended error on purpose (target: ship)")
        self.assertEqual(record.prompt_text(sample(label="uncertain", intended="")),
                         'say "ship" (label uncertain, target: ship)')


class TestPaths(unittest.TestCase):

    def test_take_paths(self):
        paths = record.take_paths(sample("ship_good_01"))
        audio = os.path.join("corpus", "audio")
        self.assertEqual(paths.trimmed, os.path.join(audio, "ship_good_01.wav"))
        self.assertEqual(paths.raw, os.path.join(audio, "raw", "ship_good_01.wav"))
        self.assertEqual(paths.sidecar, os.path.join(audio, "ship_good_01.recording.json"))
        # lab.evaluate finds the sidecar by the same rule.
        self.assertEqual(paths.sidecar, evaluate.recording_path(paths.trimmed))
        self.assertNotIn(paths.pending_raw, (paths.raw, paths.trimmed))
        self.assertNotIn(paths.pending_trimmed, (paths.raw, paths.trimmed))


# ---------------------------------------------------------------------------
# Work list and interactive session (through main)
# ---------------------------------------------------------------------------

class SessionCase(unittest.TestCase):
    IDS = ("ship_good_01", "ship_good_02", "ship_as_chip_01")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.audio = os.path.join(self.dir, "audio")
        os.makedirs(self.audio)
        self.csv = os.path.join(self.dir, "samples.csv")
        with open(self.csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            for sample_id in self.IDS:
                good = "good" in sample_id
                writer.writerow([sample_id, "ship", "ship" if good else "chip",
                                 "good" if good else "intentional_error", f"audio/{sample_id}.wav"])
        self.lines = []
        self.played = []
        self.devices = [MIC, "Varios micrófonos (Realtek(R) Audio)"]
        self.list_calls = 0
        patches = (patch.object(record.shutil, "which", return_value="C:/bin/ffmpeg.exe"),
                   patch.object(record, "list_dshow_devices", side_effect=self.list_devices),
                   patch.dict(os.environ, {"OPENPRONOUNCE_MIC": ""}))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def list_devices(self, ffmpeg):
        self.list_calls += 1
        return list(self.devices)

    def play(self, path):
        self.played.append((path, os.path.isfile(path)))

    def main(self, *argv, runner=None, answers=()):
        self.read = Script(answers)
        self.runner = runner or FakeRunner()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = record.main(["--corpus", self.csv, *argv], runner=self.runner, read=self.read,
                               write=self.lines.append, play=self.play)
        self.stderr = stderr.getvalue()
        return code

    def output(self):
        return "\n".join(self.lines)

    def path(self, *parts):
        return os.path.join(self.audio, *parts)

    def sidecar(self, sample_id):
        with open(self.path(f"{sample_id}.recording.json"), encoding="utf-8") as f:
            return json.load(f)

    def leftovers(self):
        return glob.glob(os.path.join(self.audio, "**", "*.take.wav"), recursive=True)


class TestWorkList(SessionCase):

    def load(self, only=None):
        return corpus.load_corpus(self.csv, skip_missing=True, only=only)

    def test_missing_only_in_corpus_order(self):
        sf.write(self.path("ship_good_02.wav"), clean_take(), SR)
        work, already = record.select_work(self.load(), redo=False)
        self.assertEqual([s.sample_id for s in work], ["ship_good_01", "ship_as_chip_01"])
        self.assertEqual([s.sample_id for s in already], ["ship_good_02"])

    def test_only(self):
        work, _ = record.select_work(self.load(only=["ship_as_chip_01"]), redo=False)
        self.assertEqual([s.sample_id for s in work], ["ship_as_chip_01"])

    def test_redo_takes_every_selected_sample(self):
        sf.write(self.path("ship_good_02.wav"), clean_take(), SR)
        work, already = record.select_work(self.load(), redo=True)
        self.assertEqual([s.sample_id for s in work], list(self.IDS))
        self.assertEqual(already, [])

    def test_nothing_to_record(self):
        sf.write(self.path("ship_good_01.wav"), clean_take(), SR)
        self.assertEqual(self.main("--only", "ship_good_01"), 0)
        self.assertIn("nothing to record", self.output())
        self.assertIn("--redo", self.output())
        self.assertEqual(self.runner.captures, [])

    def test_unknown_only_id_is_an_invalid_corpus(self):
        self.assertEqual(self.main("--only", "nope"), record.EXIT_INVALID_CORPUS)
        self.assertIn("nope", self.stderr)

    def test_non_wav_audio_file_is_rejected(self):
        with open(self.csv, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["ship_m4a_01", "ship", "ship", "good", "audio/ship_m4a_01.m4a"])
        self.assertEqual(self.main("--only", "ship_m4a_01"), record.EXIT_INVALID_CORPUS)
        self.assertIn("ship_m4a_01", self.stderr)
        self.assertIn(".wav", self.stderr)


class TestSession(SessionCase):

    def test_keep_writes_raw_trimmed_and_sidecar(self):
        code = self.main("--only", "ship_good_01", runner=FakeRunner(gated_take()), answers=["", "", ""])
        self.assertEqual(code, 0)
        self.assertIn('ship_good_01: say "ship" as well as you can', self.output())
        self.assertIn(f'"{MIC}"', self.output())  # the first listed device, named
        raw, trimmed = self.path("raw", "ship_good_01.wav"), self.path("ship_good_01.wav")
        self.assertEqual(self.leftovers(), [])

        raw_samples, raw_sr = sf.read(raw, dtype="int16")
        trimmed_samples, trimmed_sr = sf.read(trimmed, dtype="int16")
        self.assertEqual((raw_sr, trimmed_sr), (SR, SR))
        self.assertEqual(len(raw_samples), int(1.4 * SR))
        self.assertLess(len(trimmed_samples), len(raw_samples))

        side = self.sidecar("ship_good_01")
        self.assertEqual(side["sample_id"], "ship_good_01")
        self.assertIsNotNone(datetime.datetime.fromisoformat(side["recorded_at"]).tzinfo)
        self.assertEqual(side["device"], MIC)
        self.assertEqual(side["ffmpeg_command"], self.runner.captures[0].command)
        self.assertIn(f"audio={MIC}", side["ffmpeg_command"])
        self.assertEqual(side["raw"], {"path": "raw/ship_good_01.wav", "sha256": sha256(raw)})
        self.assertEqual(side["trimmed"], {"path": "ship_good_01.wav", "sha256": sha256(trimmed)})
        self.assertEqual((side["trim"]["top_db"], side["trim"]["pad_s"], side["max_seconds"]), (40, 0.2, 6))
        # The trimmed take is the raw samples between the recorded bounds, untouched.
        start = round(side["trim"]["start_s"] * SR)
        self.assertEqual(round(side["trim"]["end_s"] * SR) - start, len(trimmed_samples))
        np.testing.assert_array_equal(raw_samples[start:start + len(trimmed_samples)], trimmed_samples)
        self.assertEqual(set(side["stats"]), {"raw_duration_s", "trimmed_duration_s", "speech_duration_s",
                                              "peak", "rms", "noise_floor"})
        self.assertEqual(side["stats"]["raw_duration_s"], 1.4)
        self.assertEqual(side["warnings"], ["digital_gating"])
        self.assertIn("Audio enhancements", self.output())

    def test_retry_replaces_the_take(self):
        runner = FakeRunner(clean_take(), gated_take(before=0.3, word=0.5, after=0.3))
        code = self.main("--only", "ship_good_01", runner=runner, answers=["", "", "r", "", "", ""])
        self.assertEqual(code, 0)
        self.assertEqual(len(runner.captures), 2)
        side = self.sidecar("ship_good_01")
        self.assertEqual(side["stats"]["raw_duration_s"], 1.1)
        self.assertEqual(side["raw"]["sha256"], sha256(self.path("raw", "ship_good_01.wav")))
        self.assertEqual(self.leftovers(), [])

    def test_play_then_keep(self):
        code = self.main("--only", "ship_good_01", runner=FakeRunner(clean_take()), answers=["", "", "p", "x", ""])
        self.assertEqual(code, 0)
        (played, existed), = self.played
        self.assertTrue(existed, "the pending trimmed take is played before it is kept")
        self.assertTrue(played.endswith(".take.wav"), played)
        self.assertIn("press Enter to keep", self.output())  # "x" is not an answer
        self.assertTrue(os.path.isfile(self.path("ship_good_01.wav")))

    def test_no_speech_offers_a_retry(self):
        code = self.main("--only", "ship_good_01", runner=FakeRunner(silence(1.0), clean_take()),
                         answers=["", "", "", "", "", ""])
        self.assertEqual(code, 0)
        self.assertTrue(any("No speech found" in p for p in self.read.prompts))
        self.assertEqual(self.sidecar("ship_good_01")["warnings"], [])

    def test_no_speech_then_skip(self):
        code = self.main("--only", "ship_good_01", runner=FakeRunner(silence(1.0)), answers=["", "", "s"])
        self.assertEqual(code, 0)
        self.assertFalse(os.path.exists(self.path("ship_good_01.wav")))
        self.assertFalse(os.path.exists(self.path("ship_good_01.recording.json")))
        self.assertEqual(self.leftovers(), [])
        self.assertIn("0 kept, 1 skipped", self.output())

    def test_ctrl_c_while_recording_keeps_accepted_takes(self):
        runner = FakeRunner(gated_take(), gated_take())
        code = self.main(runner=runner, answers=["", "", "", "", KeyboardInterrupt])
        self.assertEqual(code, record.EXIT_INTERRUPTED)
        self.assertTrue(runner.captures[1].killed)
        self.assertFalse(runner.captures[1].stopped)
        self.assertTrue(os.path.isfile(self.path("ship_good_01.wav")))
        self.assertTrue(os.path.isfile(self.path("ship_good_01.recording.json")))
        self.assertFalse(os.path.exists(self.path("ship_good_02.wav")))
        self.assertEqual(self.leftovers(), [])
        self.assertIn("1 take(s) kept", self.output())

    def test_interrupt_at_the_keep_prompt_discards_the_take(self):
        # Closed input (EOF) stops like Ctrl+C.
        code = self.main("--only", "ship_good_01", runner=FakeRunner(gated_take()), answers=["", "", EOFError])
        self.assertEqual(code, record.EXIT_INTERRUPTED)
        self.assertEqual(os.listdir(self.audio), ["raw"])
        self.assertEqual(os.listdir(self.path("raw")), [])

    def test_redo_keeps_the_previous_take_until_the_new_one_is_kept(self):
        self.main("--only", "ship_good_01", runner=FakeRunner(clean_take()), answers=["", "", ""])
        before = self.sidecar("ship_good_01")
        code = self.main("--only", "ship_good_01", "--redo", runner=FakeRunner(gated_take()),
                         answers=["", "", KeyboardInterrupt])
        self.assertEqual(code, record.EXIT_INTERRUPTED)
        self.assertEqual(self.sidecar("ship_good_01"), before)
        self.assertEqual(sha256(self.path("ship_good_01.wav")), before["trimmed"]["sha256"])
        self.assertEqual(self.main("--only", "ship_good_01", "--redo", runner=FakeRunner(gated_take()),
                                   answers=["", "", ""]), 0)
        self.assertNotEqual(self.sidecar("ship_good_01")["trimmed"]["sha256"], before["trimmed"]["sha256"])

    def test_device_that_cannot_be_opened(self):
        failure = record.CaptureError("Could not find audio only device with name [nope]")
        code = self.main("--only", "ship_good_01", "--device", "nope", runner=FakeRunner(failure), answers=[""])
        self.assertEqual(code, record.EXIT_RECORDER_FAILED)
        self.assertIn("Could not find audio only device", self.output())
        self.assertIn("--list-devices", self.output())
        self.assertTrue(self.runner.captures[0].killed)
        self.assertEqual(self.list_calls, 0)  # --device given: no need to list

    def test_device_choice(self):
        self.main("--only", "ship_good_01", "--device", "Other mic", runner=FakeRunner(clean_take()),
                  answers=["", "", ""])
        self.assertIn("audio=Other mic", self.runner.captures[0].command)
        with patch.dict(os.environ, {"OPENPRONOUNCE_MIC": "Env mic"}):
            self.main("--only", "ship_good_02", runner=FakeRunner(clean_take()), answers=["", "", ""])
        self.assertIn("audio=Env mic", self.runner.captures[0].command)
        self.assertEqual(self.list_calls, 0)

    def test_no_audio_device(self):
        self.devices = []
        self.assertEqual(self.main("--only", "ship_good_01"), record.EXIT_RECORDER_FAILED)
        self.assertIn("no DirectShow audio device", self.stderr)

    def test_ffmpeg_missing(self):
        with patch.object(record.shutil, "which", return_value=None):
            self.assertEqual(self.main("--only", "ship_good_01"), record.EXIT_RECORDER_FAILED)
        self.assertIn("ffmpeg not found", self.stderr)
        self.assertIn("record-lab.ps1", self.stderr)


class TestListDevices(SessionCase):

    def test_list_devices(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(self.main("--list-devices"), 0)
        self.assertIn(f'"{MIC}"', stdout.getvalue())
        self.assertIn('"Varios micrófonos (Realtek(R) Audio)"', stdout.getvalue())

    def test_list_devices_without_any(self):
        self.devices = []
        self.assertEqual(self.main("--list-devices"), record.EXIT_RECORDER_FAILED)


# ---------------------------------------------------------------------------
# The real capture class, with a stand-in for ffmpeg (no device involved)
# ---------------------------------------------------------------------------

FAKE_FFMPEG = """\
import sys
output, mode = sys.argv[1], sys.argv[2]
if mode == "fail":
    sys.stderr.write("Could not find audio only device with name [nope] among source devices\\n")
    sys.exit(1)
with open(output, "wb") as f:
    f.write(b"\\0" * 4096)
    f.flush()
    if mode == "stop-on-q":
        sys.stdin.read(1)
    else:
        import time
        time.sleep(60)
"""


class TestFfmpegCapture(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.script = os.path.join(self.tmp.name, "fake_ffmpeg.py")
        with open(self.script, "w", encoding="utf-8") as f:
            f.write(FAKE_FFMPEG)
        self.output = os.path.join(self.tmp.name, "take.wav")

    def tearDown(self):
        self.tmp.cleanup()

    def start(self, mode):
        return record.FfmpegCapture([sys.executable, "-I", self.script, self.output, mode], self.output)

    def test_q_stops_the_capture(self):
        capture = self.start("stop-on-q")
        capture.wait_until_recording(timeout=20)
        capture.stop(timeout=20)
        self.assertEqual(capture.process.returncode, 0)
        self.assertEqual(os.path.getsize(self.output), 4096)

    def test_device_failure_is_reported(self):
        with open(self.output, "wb") as f:
            f.write(b"\0" * 9999)  # an older take: removed first, its size must not look like a live capture
        capture = self.start("fail")
        with self.assertRaises(record.CaptureError) as ctx:
            capture.wait_until_recording(timeout=20)
        self.assertIn("Could not find audio only device", str(ctx.exception))
        self.assertFalse(os.path.exists(self.output))

    def test_kill(self):
        capture = self.start("hang")
        capture.wait_until_recording(timeout=20)
        capture.kill()
        self.assertIsNotNone(capture.process.poll())


if __name__ == "__main__":
    unittest.main()
