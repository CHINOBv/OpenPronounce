"""Corpus of labeled recordings: CSV schema and validation.

This module deliberately imports neither torch nor openpronounce, so that a broken
CSV fails in a fraction of a second, before any model is loaded.
"""

import csv
import os
import re
from dataclasses import dataclass, field

REQUIRED_COLUMNS = ("sample_id", "expected_text", "intended_pronunciation", "label", "audio_file")
OPTIONAL_COLUMNS = ("target_phoneme", "intended_substitution", "microphone", "recording_context",
                    "recorded_on", "notes")
# good: the target was said as well as the learner can; intentional_error: the learner
# deliberately produced something else (``intended_pronunciation``); uncertain: neither is sure.
LABELS = ("good", "intentional_error", "uncertain")
SAMPLE_ID_RE = re.compile(r"[A-Za-z0-9_-]+")


class CorpusError(ValueError):
    """The corpus cannot be used. ``problems`` lists every problem found, one string each."""

    def __init__(self, path, problems):
        self.path = path
        self.problems = list(problems)
        lines = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"invalid corpus {path} ({len(self.problems)} problem(s)):\n{lines}")


@dataclass(frozen=True)
class Sample:
    """One CSV row. ``fields`` holds every column (optional ones default to ""), ``row`` its line number."""

    row: int
    fields: dict
    audio_path: str

    @property
    def sample_id(self):
        return self.fields["sample_id"]

    @property
    def label(self):
        return self.fields["label"]

    @property
    def expected_text(self):
        return self.fields["expected_text"]


@dataclass
class Corpus:
    """Validated samples, in CSV order. ``missing`` lists the samples skipped because their audio is absent."""

    path: str
    columns: list
    samples: list = field(default_factory=list)
    missing: list = field(default_factory=list)


def load_corpus(path, skip_missing=False, only=None):
    """Read and validate the corpus CSV at ``path``; raise :class:`CorpusError` listing every problem.

    ``audio_file`` is resolved relative to the CSV's directory. A missing audio file is a
    problem unless ``skip_missing`` is true, in which case the sample goes to
    ``Corpus.missing``. ``only`` restricts the corpus to these sample ids; the audio of
    unselected samples is not checked.
    """
    if not os.path.isfile(path):
        raise CorpusError(path, [f"corpus file not found: {path}"])
    base_dir = os.path.dirname(os.path.abspath(path))
    problems = []

    # utf-8-sig drops the BOM that Excel and Notepad add to "UTF-8" CSV files.
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        # Blank lines (or rows of empty cells) are skipped everywhere, before the header too.
        rows = [(reader.line_num, values) for values in reader if any(v.strip() for v in values)]

    if not rows:
        raise CorpusError(path, ["the file is empty (no header row)"])
    header = [name.strip() for name in rows.pop(0)[1]]
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing_columns:
        problems.append(f"missing required column(s): {', '.join(missing_columns)}")
    duplicates = sorted({c for c in header if header.count(c) > 1})
    if duplicates:
        problems.append(f"duplicate column(s): {', '.join(duplicates)}")
    columns = header + [c for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS if c not in header]

    samples = []
    first_seen = {}  # lower-cased sample_id -> (line, sample_id) of its first row
    for line, values in rows:
        if len(values) > len(header):
            problems.append(f"line {line}: {len(values)} fields, the header has {len(header)}")
            continue
        fields = dict.fromkeys(columns, "")
        fields.update(zip(header, (v.strip() for v in values)))
        row_problems = list(_row_problems(fields, header))
        sample_id = fields["sample_id"]
        if SAMPLE_ID_RE.fullmatch(sample_id):
            # Ids name output files, and Windows file names ignore case: Ship_01 and ship_01 collide.
            key = sample_id.lower()
            if key not in first_seen:
                first_seen[key] = (line, sample_id)
            else:
                first_line, first_id = first_seen[key]
                same = "" if first_id == sample_id else f" as {first_id!r}; ids differing only in case collide"
                row_problems.insert(0, f"duplicate sample_id {sample_id!r} (first on line {first_line}{same})")
        problems.extend(f"line {line}: {p}" for p in row_problems)
        audio_file = fields["audio_file"]
        samples.append(Sample(line, fields, os.path.normpath(os.path.join(base_dir, audio_file))))

    if only:
        selected = set(only)
        known = {s.sample_id for s in samples}
        problems.extend(f"--only: unknown sample_id {sample_id!r}" for sample_id in only if sample_id not in known)
        samples = [s for s in samples if s.sample_id in selected]

    present, missing = [], []
    for sample in samples:
        if not sample.fields["audio_file"] or os.path.isfile(sample.audio_path):
            present.append(sample)
        elif skip_missing:
            missing.append(sample)
        else:
            problems.append(f"line {sample.row}: audio file not found: {sample.fields['audio_file']} "
                            f"(resolved to {sample.audio_path})")

    if problems:
        raise CorpusError(path, problems)
    return Corpus(path, columns, present, missing)


def _row_problems(fields, header):
    """Problems of one row on its own (duplicates are checked across rows by the caller).

    Columns absent from ``header`` are already reported once, so they are not checked per row.
    """
    sample_id = fields["sample_id"]
    if "sample_id" in header:
        if not sample_id:
            yield "sample_id is empty"
        elif not SAMPLE_ID_RE.fullmatch(sample_id):
            yield f"sample_id {sample_id!r} must match {SAMPLE_ID_RE.pattern} (it names the output files)"
    if "expected_text" in header and not fields["expected_text"]:
        yield "expected_text is empty"
    if "label" in header and fields["label"] not in LABELS:
        yield f"label {fields['label']!r} must be one of: {', '.join(LABELS)}"
    if "audio_file" in header and not fields["audio_file"]:
        yield "audio_file is empty"
