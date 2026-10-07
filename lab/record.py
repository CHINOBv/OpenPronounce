"""Record the corpus takes with ffmpeg: keep the raw take, write a silence-trimmed take for evaluation.

    python -m lab.record [--device NAME] [--only ID ...] [--redo]

For each sample whose audio file is missing (every selected sample with ``--redo``), in
corpus order: Enter starts an ffmpeg DirectShow capture (mono, 16 kHz, 16-bit PCM), Enter
stops it. The take is trimmed with ``librosa.effects.trim(top_db)`` plus ``--pad`` seconds of
the original audio on each side (the same samples, no resampling or gain) and written at the
CSV's ``audio_file``. The raw take goes to ``<audio dir>/raw/<sample_id>.wav`` and a
``<stem>.recording.json`` sidecar (device, ffmpeg command, hashes, trim, levels, warnings)
goes next to the trimmed take; ``lab.evaluate`` copies it into its raw output.

A take is written to pending ``*.take.wav`` files first and replaces the sample's files only
once it is kept, so a retry, a skip or Ctrl+C never destroys an earlier kept take.
See ``lab/README.md``.
"""

import argparse
import contextlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import NamedTuple

import numpy as np
import soundfile as sf

from . import corpus, evaluate

SAMPLING_RATE = 16000
FRAME_S = 0.02  # frames of the noise floor and of the speech span
CLIPPING_PEAK = 0.99
QUIET_PEAK = 0.05
MIN_SPEECH_S = 0.15
# One 16-bit step (~ -90 dBFS): a live microphone and preamp always hiss above it. The T4a takes
# sat at ~4e-6 (98 % exact zeros, the rest +-1 LSB), the mark of noise suppression gating the input.
GATING_FLOOR = 1 / 32768
MAX_SECONDS_SLACK = 0.05  # a take this close to --max-seconds was cut by the hard stop

EXIT_INVALID_CORPUS = evaluate.EXIT_INVALID_CORPUS
EXIT_RECORDER_FAILED = 4
EXIT_INTERRUPTED = evaluate.EXIT_INTERRUPTED
EXIT_CODES_HELP = (f"exit codes: 0 done, 2 usage error, {EXIT_INVALID_CORPUS} invalid corpus, "
                   f"{EXIT_RECORDER_FAILED} ffmpeg or the microphone failed, {EXIT_INTERRUPTED} interrupted "
                   "(kept takes stay)")

WARNINGS = {
    "clipping": "the peak is at full scale, the take is clipped: move back from the microphone or lower its level",
    "too_quiet": "the peak is below 0.05: speak closer to the microphone or raise its level",
    "digital_gating": ("the silence is digital (below one 16-bit step), so Windows or the driver gates the "
                       "microphone: turn off Settings > System > Sound > (your microphone) > Audio enhancements, "
                       "and noise suppression in Realtek Audio Console if present (lab/README.md)"),
    "speech_too_short": "less than 0.15 s of speech: the word may be cut, retry",
    "max_seconds": "the take reached --max-seconds and was cut by the hard stop: retry and stop sooner",
}


class CaptureError(RuntimeError):
    """ffmpeg could not record: device not found or busy, or no readable audio written."""


class Settings(NamedTuple):
    max_seconds: float
    pad: float
    top_db: float


class Trim(NamedTuple):
    """Sample indices: ``start:end`` is the kept take, ``speech_start:speech_end`` what librosa found."""

    start: int
    end: int
    speech_start: int
    speech_end: int


class Take(NamedTuple):
    trim: Trim | None  # None: no speech found
    stats: dict
    warnings: list
    sr: int
    recorded_at: str


class TakePaths(NamedTuple):
    raw: str
    trimmed: str  # the CSV's audio_file, what lab.evaluate reads
    sidecar: str
    pending_raw: str
    pending_trimmed: str


# ---------------------------------------------------------------------------
# Pure parts
# ---------------------------------------------------------------------------

_DEVICE_LINE = re.compile(r'^(?:\[[^\]]*\]\s*)?"(?P<name>[^"]+)"\s*(?:\((?P<types>[^)]*)\))?$')


def parse_dshow_devices(text):
    """Audio device names, in order, from the stderr of ``ffmpeg -list_devices true -f dshow -i dummy``.

    Recent ffmpeg tags each device line with its media types (``"name" (audio)``); older
    builds list the devices under "DirectShow audio/video devices" headers instead.
    "Alternative name" lines are ignored.
    """
    names, section = [], None
    for line in text.splitlines():
        if "DirectShow audio devices" in line:
            section = "audio"
            continue
        if "DirectShow video devices" in line:
            section = "video"
            continue
        match = _DEVICE_LINE.match(line.strip())
        if not match:
            continue
        types = match["types"]
        kinds = {kind.strip() for kind in types.split(",")} if types is not None else {section}
        if "audio" in kinds and match["name"] not in names:
            names.append(match["name"])
    return names


def ffmpeg_command(device, output, max_seconds, ffmpeg="ffmpeg"):
    """ffmpeg arguments capturing DirectShow ``device`` to a mono 16 kHz 16-bit WAV at ``output``.

    ``-t`` is the hard stop. stdin is left open: ``q`` on it ends the capture cleanly.
    ``-flush_packets 1`` writes audio as it arrives, so a growing file shows the capture started.
    """
    return [ffmpeg, "-hide_banner", "-nostats", "-loglevel", "error", "-y",
            "-f", "dshow", "-i", f"audio={device}",
            "-ac", "1", "-ar", str(SAMPLING_RATE), "-c:a", "pcm_s16le", "-t", f"{max_seconds:g}",
            "-flush_packets", "1", output]


def trim_take(signal, sr, top_db, pad):
    """Bounds of the speech in ``signal`` (floats) plus ``pad`` seconds each side, clamped; None if silent.

    The speech is what ``librosa.effects.trim(top_db=top_db)`` keeps, as in the trim
    experiment of T4a. An all-zero signal has no speech (librosa alone would keep all of it).
    """
    if signal.size == 0 or not np.any(signal):
        return None
    import librosa  # slow import, needed only once a take exists

    _, (speech_start, speech_end) = librosa.effects.trim(np.asarray(signal, dtype=np.float32), top_db=top_db)
    speech_start, speech_end = int(speech_start), int(speech_end)
    if speech_end <= speech_start:
        return None
    margin = int(round(pad * sr))
    return Trim(max(0, speech_start - margin), min(signal.size, speech_end + margin), speech_start, speech_end)


def _frame_rms(signal, sr):
    """RMS of consecutive 20 ms frames (one frame when shorter) and the frame length in samples."""
    size = max(1, int(round(FRAME_S * sr)))
    count = signal.size // size
    if count == 0:
        return (np.sqrt(np.mean(np.square(signal), keepdims=True)) if signal.size else np.zeros(0)), signal.size
    return np.sqrt(np.mean(np.square(signal[:count * size].reshape(count, size)), axis=1)), size


def _level(value):
    """A level with 6 significant digits: small levels stay distinguishable from zero."""
    return float(f"{float(value):.6g}")


def take_stats(signal, sr, trim, top_db):
    """Durations and levels of a raw take (``signal`` floats in [-1, 1]).

    ``rms`` is measured on the trimmed take (the raw one without speech), ``noise_floor`` is the
    10th percentile of the 20 ms frame RMS of the raw take, and ``speech_duration_s`` spans the
    20 ms frames within ``top_db`` of the loudest one (librosa's 128 ms frames would add ~0.14 s).
    """
    frames, frame_size = _frame_rms(signal, sr)
    kept = signal[trim.start:trim.end] if trim else signal
    speech = None
    if trim and frames.size:
        loud = np.flatnonzero(frames >= frames.max() * 10 ** (-top_db / 20))
        speech = (loud[-1] - loud[0] + 1) * frame_size / sr
    return {
        "raw_duration_s": round(signal.size / sr, 3),
        "trimmed_duration_s": round((trim.end - trim.start) / sr, 3) if trim else None,
        "speech_duration_s": round(float(speech), 3) if speech is not None else None,
        "peak": _level(np.max(np.abs(signal))) if signal.size else 0.0,
        "rms": _level(np.sqrt(np.mean(np.square(kept)))) if kept.size else 0.0,
        "noise_floor": _level(np.percentile(frames, 10)) if frames.size else 0.0,
    }


def take_warnings(stats, max_seconds):
    """Codes of :data:`WARNINGS` that apply to a take's ``stats``."""
    speech = stats["speech_duration_s"]
    checks = (
        ("clipping", stats["peak"] >= CLIPPING_PEAK),
        ("too_quiet", stats["peak"] < QUIET_PEAK),
        ("digital_gating", stats["noise_floor"] < GATING_FLOOR),
        ("speech_too_short", speech is not None and speech < MIN_SPEECH_S),
        ("max_seconds", stats["raw_duration_s"] >= max_seconds - MAX_SECONDS_SLACK),
    )
    return [code for code, applies in checks if applies]


def prompt_text(sample):
    """What to say for ``sample``, e.g. ``say "sheep" on purpose (target: ship)``."""
    expected = sample.expected_text
    intended = sample.fields.get("intended_pronunciation", "")
    if sample.label == "good":
        return f'say "{expected}" as well as you can'
    if sample.label == "intentional_error":
        if not intended:
            return f"make the intended error on purpose (target: {expected})"
        return f'say "{intended}" on purpose (target: {expected})'
    return f'say "{intended or expected}" (label uncertain, target: {expected})'


def select_work(selected, redo):
    """``(to_record, already_recorded)`` in CSV order: samples missing audio, every one with ``redo``."""
    if redo:
        return sorted(selected.samples + selected.missing, key=lambda s: s.row), []
    return list(selected.missing), list(selected.samples)


def take_paths(sample):
    folder = os.path.dirname(sample.audio_path)
    raw_dir = os.path.join(folder, "raw")
    return TakePaths(
        raw=os.path.join(raw_dir, f"{sample.sample_id}.wav"),
        trimmed=sample.audio_path,
        sidecar=evaluate.recording_path(sample.audio_path),
        pending_raw=os.path.join(raw_dir, f"{sample.sample_id}.take.wav"),
        pending_trimmed=os.path.splitext(sample.audio_path)[0] + ".take.wav",
    )


def describe_take(take):
    """Console lines for a take: durations, levels, then one line per warning."""
    s = take.stats
    found = (f"trimmed {s['trimmed_duration_s']:.2f} s (speech {s['speech_duration_s']:.2f} s)" if take.trim
             else "no speech found")
    floor = f"{s['noise_floor']:.6f}" + (" (digital silence)" if s["noise_floor"] < GATING_FLOOR else "")
    lines = [f"  raw {s['raw_duration_s']:.2f} s, {found}, peak {s['peak']:.3f}, rms {s['rms']:.4f}, "
             f"noise floor {floor}"]
    return lines + [f"  warning: {WARNINGS[code]}" for code in take.warnings]


def _seconds(samples, sr):
    return round(samples / sr, 6)  # 6 decimals: the sample index is recoverable at 16 kHz


def sidecar_record(sample, device, command, paths, take, settings):
    """Contents of ``<stem>.recording.json`` for a kept take (paths relative to the sidecar)."""
    folder = os.path.dirname(paths.sidecar)

    def file_record(path):
        return {"path": os.path.relpath(path, folder).replace("\\", "/"), "sha256": evaluate.file_sha256(path)}

    trim = take.trim
    return {
        "sample_id": sample.sample_id,
        "recorded_at": take.recorded_at,
        "device": device,
        "ffmpeg_command": command,
        "raw": file_record(paths.raw),
        "trimmed": file_record(paths.trimmed),
        "trim": {"top_db": settings.top_db, "pad_s": settings.pad,
                 "start_s": _seconds(trim.start, take.sr), "end_s": _seconds(trim.end, take.sr),
                 "speech_start_s": _seconds(trim.speech_start, take.sr),
                 "speech_end_s": _seconds(trim.speech_end, take.sr)},
        "max_seconds": settings.max_seconds,
        "stats": take.stats,
        "warnings": take.warnings,
    }


# ---------------------------------------------------------------------------
# ffmpeg and audio I/O
# ---------------------------------------------------------------------------

def list_dshow_devices(ffmpeg):
    result = subprocess.run([ffmpeg, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                            stdin=subprocess.DEVNULL, capture_output=True, timeout=30)
    return parse_dshow_devices(result.stderr.decode("utf-8", "replace"))


class FfmpegCapture:
    """A running ffmpeg capture to ``output``: ``wait_until_recording``, then ``stop`` (sends ``q``) or ``kill``."""

    STARTED_BYTES = 1024  # beyond the WAV header: audio is reaching the file

    def __init__(self, command, output):
        self.output = output
        with contextlib.suppress(FileNotFoundError):
            os.remove(output)  # a stale take's size must not look like a live capture
        self.stderr = ""
        self._stderr_file = tempfile.TemporaryFile()  # a file, not a pipe: ffmpeg can never block on it
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                        stderr=self._stderr_file)

    def wait_until_recording(self, timeout=5.0):
        """Return once audio reaches the file; raise :class:`CaptureError` if ffmpeg exits first.

        Without audio after ``timeout`` it returns anyway: the take's stats will show what happened.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self._finish()
                raise CaptureError(f"ffmpeg exited with code {self.process.returncode}: "
                                   f"{self.stderr or 'no message'}")
            with contextlib.suppress(OSError):
                if os.path.getsize(self.output) >= self.STARTED_BYTES:
                    return
            time.sleep(0.05)

    def stop(self, timeout=5.0):
        """Ask ffmpeg to finish the file (``q`` on stdin); kill it if it has not exited after ``timeout``."""
        if self.process.poll() is None:
            with contextlib.suppress(OSError):
                self.process.stdin.write(b"q")
                self.process.stdin.flush()
            try:
                self.process.wait(timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self._finish()
        if self.process.returncode != 0 and not os.path.isfile(self.output):
            raise CaptureError(f"ffmpeg exited with code {self.process.returncode}: {self.stderr or 'no message'}")

    def kill(self):
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        self._finish()

    def _finish(self):
        """Close stdin and keep ffmpeg's messages in ``stderr``."""
        if self._stderr_file.closed:
            return
        with contextlib.suppress(OSError):
            self.process.stdin.close()
        self._stderr_file.seek(0)
        self.stderr = self._stderr_file.read().decode("utf-8", "replace").strip()
        self._stderr_file.close()


def play_file(path):
    """Play a WAV synchronously with winsound (Windows): no player window, no lock left on the file."""
    try:
        import winsound
    except ImportError:
        print(f"  no player here; the take is {evaluate._display_path(path)}")
        return
    try:
        winsound.PlaySound(path, winsound.SND_FILENAME)
    except RuntimeError as e:
        print(f"  could not play the take: {e}")


def analyze_take(paths, settings):
    """Read the pending raw take, trim it, and write the pending trimmed take when speech was found."""
    recorded_at = evaluate._now()
    try:
        samples, sr = sf.read(paths.pending_raw, dtype="int16")
    except (OSError, RuntimeError) as e:  # soundfile reports a missing or broken file as RuntimeError
        raise CaptureError(f"no readable take at {evaluate._display_path(paths.pending_raw)}: {e}") from e
    if samples.ndim > 1:
        samples = samples[:, 0]
    signal = samples.astype(np.float64) / 32768.0
    trim = trim_take(signal, sr, settings.top_db, settings.pad)
    stats = take_stats(signal, sr, trim, settings.top_db)
    if trim:
        # The int16 samples themselves: the trimmed take is bit-identical to that stretch of the raw one.
        sf.write(paths.pending_trimmed, samples[trim.start:trim.end], sr, subtype="PCM_16")
    return Take(trim, stats, take_warnings(stats, settings.max_seconds), sr, recorded_at)


def _keep(sample, device, command, paths, take, settings):
    """Move the pending take into place, then write its sidecar (the old one goes first: no stale hashes)."""
    with contextlib.suppress(FileNotFoundError):
        os.remove(paths.sidecar)
    os.replace(paths.pending_trimmed, paths.trimmed)
    os.replace(paths.pending_raw, paths.raw)
    evaluate._write_json(paths.sidecar, sidecar_record(sample, device, command, paths, take, settings))


# ---------------------------------------------------------------------------
# Interactive session
# ---------------------------------------------------------------------------

def _ask_keep(read, write, play, trimmed):
    """True to keep the take, False to retry; ``p`` plays it and asks again."""
    while True:
        answer = read("  [Enter] keep  r retry  p play ").strip().lower()
        if answer in ("", "r"):
            return answer == ""
        if answer == "p":
            play(trimmed)
        else:
            write("  press Enter to keep, r to retry or p to play")


def record_sample(sample, device, ffmpeg, settings, runner, read, write, play):
    """Record ``sample`` until a take is kept (True) or the sample is skipped (False)."""
    paths = take_paths(sample)
    os.makedirs(os.path.dirname(paths.raw), exist_ok=True)
    output = evaluate._display_path(paths.pending_raw)
    command = ffmpeg_command(device, output, settings.max_seconds, ffmpeg)
    try:
        while True:
            read("  Enter to start recording ")
            capture = runner(command, output)
            try:
                capture.wait_until_recording()
                read(f"  Recording: say it, wait a second, then press Enter "
                     f"(hard stop after {settings.max_seconds:g} s) ")
                capture.stop()
            except BaseException:  # Ctrl+C included: never leave ffmpeg running
                capture.kill()
                raise
            take = analyze_take(paths, settings)
            for line in describe_take(take):
                write(line)
            if take.trim is None:
                if read("  No speech found. [Enter] retry  s skip ").strip().lower() == "s":
                    return False
            elif _ask_keep(read, write, play, paths.pending_trimmed):
                _keep(sample, device, command, paths, take, settings)
                return True
    finally:
        for path in (paths.pending_raw, paths.pending_trimmed):
            with contextlib.suppress(FileNotFoundError):
                os.remove(path)


def record_session(work, device, ffmpeg, settings, runner, read, write, play):
    """Record every sample of ``work`` in order; return the exit code."""
    kept = skipped = 0
    try:
        for index, sample in enumerate(work, 1):
            write(f"[{index}/{len(work)}] {sample.sample_id}: {prompt_text(sample)}")
            if record_sample(sample, device, ffmpeg, settings, runner, read, write, play):
                kept += 1
            else:
                skipped += 1
                write(f"  skipped {sample.sample_id}")
    except (KeyboardInterrupt, EOFError):
        write(f"\ninterrupted: {kept} take(s) kept, the current take discarded")
        return EXIT_INTERRUPTED
    except CaptureError as e:
        write(f"recording failed: {e}")
        write(f'device "{device}": check the name with --list-devices, choose one with --device or OPENPRONOUNCE_MIC')
        return EXIT_RECORDER_FAILED
    write(f"done: {kept} kept, {skipped} skipped")
    return 0


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _positive(value):
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return number


def _non_negative(value):
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return number


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m lab.record", epilog=EXIT_CODES_HELP,
                                     description="Record the corpus takes: raw take kept, trimmed take evaluated.")
    parser.add_argument("--corpus", default="lab/corpus/samples.csv", help="corpus CSV (default: %(default)s)")
    parser.add_argument("--device", help="DirectShow audio device (default: $OPENPRONOUNCE_MIC, else the first one)")
    parser.add_argument("--list-devices", action="store_true", help="list the DirectShow audio devices and stop")
    parser.add_argument("--only", nargs="+", metavar="ID", help="record only these sample ids")
    parser.add_argument("--redo", action="store_true", help="also record samples that already have audio")
    parser.add_argument("--max-seconds", type=_positive, default=6.0,
                        help="hard stop of a take in seconds (default: %(default)s)")
    parser.add_argument("--pad", type=_non_negative, default=0.2,
                        help="seconds kept around the speech (default: %(default)s)")
    parser.add_argument("--top-db", type=_positive, default=40.0,
                        help="librosa trim threshold below the peak (default: %(default)s)")
    return parser


def _find_ffmpeg():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg not found on PATH: run through .\\record-lab.ps1, which adds the WinGet ffmpeg to PATH",
              file=sys.stderr)
    return ffmpeg


def _audio_devices(ffmpeg):
    names = list_dshow_devices(ffmpeg)
    if not names:
        print("no DirectShow audio device found (is a microphone connected and allowed in Windows privacy "
              "settings?)", file=sys.stderr)
    return names


def main(argv=None, runner=None, read=input, write=print, play=None):
    args = build_parser().parse_args(argv)
    if args.list_devices:
        ffmpeg = _find_ffmpeg()
        names = _audio_devices(ffmpeg) if ffmpeg else []
        if not names:
            return EXIT_RECORDER_FAILED
        print("DirectShow audio devices (pass one to --device or set OPENPRONOUNCE_MIC):")
        for name in names:
            print(f'  "{name}"')
        return 0

    try:
        selected = corpus.load_corpus(args.corpus, skip_missing=True, only=args.only)
    except corpus.CorpusError as e:
        print(e, file=sys.stderr)
        return EXIT_INVALID_CORPUS
    work, already = select_work(selected, args.redo)
    not_wav = [s.sample_id for s in work if os.path.splitext(s.audio_path)[1].lower() != ".wav"]
    if not_wav:
        print(f"lab.record writes WAV files; make audio_file end in .wav for: {', '.join(not_wav)}", file=sys.stderr)
        return EXIT_INVALID_CORPUS
    if already:
        write(f"already recorded ({len(already)}, --redo records them again): "
              + ", ".join(s.sample_id for s in already))
    if not work:
        write("nothing to record (--redo records existing takes again)")
        return 0

    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return EXIT_RECORDER_FAILED
    device = args.device or os.environ.get("OPENPRONOUNCE_MIC")
    if device:
        write(f'device "{device}"')
    else:
        names = _audio_devices(ffmpeg)
        if not names:
            return EXIT_RECORDER_FAILED
        device = names[0]
        write(f'using the first DirectShow audio device "{device}" '
              "(choose with --device or OPENPRONOUNCE_MIC, see --list-devices)")
    write(f"{len(work)} sample(s) to record; Ctrl+C stops, kept takes stay")
    settings = Settings(args.max_seconds, args.pad, args.top_db)
    return record_session(work, device, ffmpeg, settings, runner or FfmpegCapture, read, write, play or play_file)


if __name__ == "__main__":
    sys.exit(main())
