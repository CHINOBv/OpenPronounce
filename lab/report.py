"""Pure derivations over pipeline outputs: per-sample rows, detection outcome, summary, markdown.

Nothing here judges pronunciation. These functions restate what the pipeline returned
and compare it with the human label; they add no threshold of their own.
"""

import itertools
import re

from .corpus import LABELS

DETECTIONS = ("TP", "FP", "TN", "FN", "NA")
# Word that speech._feedback uses when no word is flagged ("Your pronunciation is excellent!").
POSITIVE_FEEDBACK_MARKER = "excellent"


# ---------------------------------------------------------------------------
# Per sample
# ---------------------------------------------------------------------------

def detection(label, words_with_errors, failed=False):
    """Label vs. pipeline output. Flagged = the pipeline reported at least one word.

    good: TN when nothing is flagged, FP otherwise; intentional_error: TP when something
    is flagged, FN otherwise; uncertain samples, failed analyses and results without
    ``words_with_errors`` (None: unknown, not "nothing flagged"): NA.
    """
    if failed or words_with_errors is None or label not in ("good", "intentional_error"):
        return "NA"
    flagged = bool(words_with_errors)
    if label == "good":
        return "FP" if flagged else "TN"
    return "TP" if flagged else "FN"


def _normalized_words(text):
    """Lower-case words without punctuation, to compare ASR output with the expected text."""
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def _flat_expected_phones(differences):
    """Expected phones of the phone model (per word), or the text-based phonemes without it."""
    if "expected_phones" in differences:
        return [phone for word in differences["expected_phones"] for phone in word]
    return list(differences.get("expected_phonemes") or [])


def flags(result, expected_text):
    """Factual flags on one pipeline result (no thresholds)."""
    differences = result.get("differences") or {}
    return {
        "positive_feedback": POSITIVE_FEEDBACK_MARKER in (result.get("feedback") or "").lower(),
        "asr_mismatch": _normalized_words(result.get("transcribe") or "") != _normalized_words(expected_text),
        # espeak-ng unreachable: the pipeline then reports no error at all.
        "no_expected_phones": not _flat_expected_phones(differences),
        "no_heard_phones": not differences.get("heard_phones"),
        # Without it the detection is unknown (NA), not "nothing flagged".
        "missing_words_with_errors": differences.get("words_with_errors") is None,
    }


def _word_edits(diagnostics, reported_positions):
    """Compact per-word view of ``diagnostics.word_reports``: edits of every word, reported or not."""
    if not diagnostics or diagnostics.get("word_reports") is None:
        return None
    return [{
        "word": r["word"],
        "position": r["position"],
        "n_phones": len(r["expected"]),
        "distance": r["distance"],
        "weighted_edits": round(r["weighted_edits"], 3),
        "reported": r["position"] in reported_positions,
    } for r in diagnostics["word_reports"]]


def _short_error(error):
    """``error`` without its traceback, or None."""
    return {key: error[key] for key in ("stage", "type", "message") if key in error} if error else None


def result_row(record):
    """One ``results.jsonl`` line from a raw record: metadata, signals, detection, flags. No prosody."""
    sample = record["sample"]
    failed = record["error"] is not None or record["result"] is None
    result = record["result"] or {}
    differences = result.get("differences") or {}
    errors = differences.get("errors")
    diagnostics = record["diagnostics"] or {}
    return {
        "sample_id": sample["sample_id"],
        "label": sample["label"],
        "sample": sample,
        "audio": record["audio"],
        "reference_sha256": (record.get("reference") or {}).get("sha256"),
        "score": result.get("score"),
        "acoustic_distance": result.get("acoustic_distance"),
        "transcribe": result.get("transcribe"),
        "word_error_rate": differences.get("word_error_rate"),
        "phoneme_error_rate": differences.get("phoneme_error_rate"),
        "expected_phones": differences.get("expected_phones"),
        "heard_phones": differences.get("heard_phones"),
        "heard_phones_confidence": differences.get("heard_phones_confidence"),
        "errors": errors,
        "words_with_errors": differences.get("words_with_errors"),
        "feedback": result.get("feedback"),
        "word_edits": _word_edits(diagnostics, {e.get("position") for e in errors or []}),
        "diagnostics_error": _short_error(diagnostics.get("error")),
        "detection": detection(sample["label"], differences.get("words_with_errors"), failed=failed),
        "flags": None if failed else flags(result, sample["expected_text"]),
        "error": _short_error(record["error"]),
        "elapsed_s": record["elapsed_s"],
        "cold_start": record.get("cold_start", False),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def format_ratio(value):
    return "n/a" if value is None else f"{value:.2f}"


def summarize(rows):
    """Detection counts, precision/recall (None without a denominator), score range per label."""
    counts = dict.fromkeys(DETECTIONS, 0)
    for row in rows:
        counts[row["detection"]] += 1
    scores = {}
    for label in LABELS:
        values = [r["score"] for r in rows if r["label"] == label and r["error"] is None and r["score"] is not None]
        scores[label] = {"n": len(values), "min": min(values), "mean": sum(values) / len(values),
                         "max": max(values)} if values else None
    return {
        "counts": counts,
        "precision": _ratio(counts["TP"], counts["TP"] + counts["FP"]),
        "recall": _ratio(counts["TP"], counts["TP"] + counts["FN"]),
        "scores": scores,
        "positive_feedback_on_error": [r["sample_id"] for r in rows if r["label"] == "intentional_error"
                                       and (r["flags"] or {}).get("positive_feedback")],
        "unknown_detection": [r["sample_id"] for r in rows if (r["flags"] or {}).get("missing_words_with_errors")],
        "failed": [r["sample_id"] for r in rows if r["error"] is not None],
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _cell(value):
    """Markdown table cell: '-' for nothing, 2 decimals for numbers, escaped pipes, one line."""
    if value is None or value == "" or value == []:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return " ".join(str(value).split()).replace("|", "\\|")


def _table(header, rows, align=None):
    align = align or ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _expected_phones_text(expected_phones):
    if not expected_phones:
        return None
    return " / ".join(" ".join(word) for word in expected_phones)


def _heard_phones_text(phones, confidences):
    """Every heard phone with its confidence; ``p(?)`` when the confidence is missing."""
    if not phones:
        return None
    confidences = list(confidences or [])[:len(phones)]
    return " ".join(f"{p}(?)" if c is None else f"{p}({c:.2f})" for p, c in itertools.zip_longest(phones, confidences))


def _active_flags(row_flags):
    return ", ".join(name for name, on in (row_flags or {}).items() if on) or None


def _error_text(error):
    """``Type: message`` of a row error, prefixed with its stage unless it is the analysis itself."""
    text = f"{error['type']}: {_cell(error['message'])}"
    stage = error.get("stage")
    return text if stage in (None, "analysis") else f"{stage}: {text}"


def _decimals3(value):
    return None if value is None else f"{value:.3f}"


def _reporting_value(n_phones, thresholds):
    """Weighted edits at which ``phones.compare_phones`` reports a word of ``n_phones`` phones.

    Restates its rule (edits / phones >= PHONE_ERROR_THRESHOLD or edits >= PHONE_ERROR_MIN_EDITS)
    for display only; whether a word was actually reported comes from the pipeline output.
    """
    threshold, min_edits = thresholds.get("PHONE_ERROR_THRESHOLD"), thresholds.get("PHONE_ERROR_MIN_EDITS")
    if threshold is None or min_edits is None:
        return None
    return float(min(threshold * n_phones, min_edits))


def _tts_text(tts):
    if "error" in tts:
        return f"unavailable: {tts['error']}"
    # The gTTS "voice" is the Google Translate domain (tld), e.g. com or co.uk.
    kind = "tld" if tts.get("backend") == "gtts" else "voice"
    return f"{tts.get('backend')} ({kind} {tts.get('voice', '-')})"


def _header(run, rows):
    git = run.get("git") or {}
    commit = (git.get("commit") or "unknown")[:12]
    if git.get("dirty"):
        commit += " (dirty)"
    corpus = run.get("corpus") or {}
    phone_model = run.get("phone_model") or {}
    revisions = ", ".join(f"{name} @ {(revision or 'unknown')[:12]}"
                          for name, revision in (run.get("model_revisions") or {}).items())
    counts = run.get("counts") or {}
    samples = (f"Samples: {counts.get('selected', 0)} selected, {counts.get('processed', 0)} processed, "
               f"{counts.get('failed', 0)} failed, {counts.get('missing', 0)} skipped (audio not found)")
    if counts.get("not_run"):
        samples += f", {counts['not_run']} not run"
    lines = [
        f"# Lab run {run.get('run_id', '')}",
        "",
        _table(["", ""], [
            ["Status", run.get("status")],
            ["Started", run.get("started_at")],
            ["Commit", commit],
            ["Corpus", f"{corpus.get('path')} (sha256 {(corpus.get('sha256') or '')[:12]})"],
            ["Language", run.get("lang")],
            ["TTS", _tts_text(run.get("tts") or {})],
            ["Device", run.get("device")],
            ["Phone model", f"{phone_model.get('name')} (enabled: {phone_model.get('enabled')})"],
            ["Model revisions", revisions],
        ]),
        "",
        samples + ".",
    ]
    cold = [r for r in rows if r.get("cold_start")]
    if cold:
        total = sum(r.get("elapsed_s") or 0 for r in rows)
        lines += ["", f"Elapsed: {total:.1f} s in total; {cold[0]['sample_id']} ({cold[0]['elapsed_s']:.1f} s) "
                      "includes lazy model loading."]
    return lines


def _summary_section(summary, rows, missing):
    counts = summary["counts"]
    errors_by_id = {r["sample_id"]: r["error"] for r in rows if r["error"]}
    score_rows = []
    for label in LABELS:
        s = summary["scores"][label]
        score_rows.append([label, s["n"], s["min"], s["mean"], s["max"]] if s else [label, 0, None, None, None])
    lines = [
        "## Summary",
        "",
        "Label vs. pipeline output. Flagged = at least one word reported; `uncertain`, failed samples and "
        "results without `words_with_errors` are NA.",
        "",
        _table(["TP", "FP", "TN", "FN", "NA", "precision", "recall"],
               [[*(str(counts[k]) for k in DETECTIONS),
                 format_ratio(summary["precision"]), format_ratio(summary["recall"])]],
               ["--:"] * 7),
        "",
        _table(["label", "n", "score min", "mean", "max"], score_rows, ["---", "--:", "--:", "--:", "--:"]),
        "",
        "Positive feedback on an intentional error: " + (", ".join(summary["positive_feedback_on_error"]) or "none"),
        "",
        "Failed: " + ("; ".join(f"{i} ({_error_text(e)})" for i, e in errors_by_id.items()) or "none"),
    ]
    if summary.get("unknown_detection"):
        lines += ["", "Detection unknown (NA), the result has no `words_with_errors`: "
                  + ", ".join(summary["unknown_detection"])]
    if missing:
        lines += ["", "Skipped (audio not found): " + ", ".join(missing)]
    return lines


def _samples_section(rows):
    table_rows = []
    for r in rows:
        table_rows.append([
            r["sample_id"], r["label"], r["sample"].get("intended_pronunciation"), r["score"], r["transcribe"],
            _expected_phones_text(r["expected_phones"]),
            _heard_phones_text(r["heard_phones"], r["heard_phones_confidence"]),
            r["phoneme_error_rate"], r["word_error_rate"], r["acoustic_distance"],
            ", ".join(r["words_with_errors"] or []), r["detection"], _active_flags(r["flags"]),
        ])
    header = ["id", "label", "intended", "score", "ASR", "expected phones", "heard phones (confidence)",
              "PER", "WER", "acoustic distance", "flagged words", "detection", "flags"]
    align = ["---"] * 3 + ["--:"] + ["---"] * 3 + ["--:"] * 3 + ["---"] * 3
    return ["## Samples", "", _table(header, table_rows, align)]


def _word_section(run, rows, word_reports):
    thresholds = run.get("phone_thresholds") or {}
    lines = [
        "## Word level",
        "",
        "Weighted edits = sum of the per-phone error confidences of `phones.compare_phones`. A word is "
        "reported at min(PHONE_ERROR_THRESHOLD x phones, PHONE_ERROR_MIN_EDITS) with this run's values "
        f"(PHONE_ERROR_THRESHOLD = {thresholds.get('PHONE_ERROR_THRESHOLD')}, "
        f"PHONE_ERROR_MIN_EDITS = {thresholds.get('PHONE_ERROR_MIN_EDITS')}). "
        "Phone lines read expected → heard (error confidence), for words with at least one edit.",
    ]
    for r in rows:
        lines += ["", f"### {r['sample_id']}", ""]
        if r["error"]:
            lines.append(f"Failed: {_error_text(r['error'])}")
            continue
        lines.append(f"Feedback: {_cell(r['feedback'])}")
        if r["word_edits"] is None:
            error = r.get("diagnostics_error")
            lines += ["", f"Phone-level diagnostics failed: {error['type']}: {_cell(error['message'])}." if error
                      else "No phone-level diagnostics (phone model disabled)."]
            continue
        # Words carry their position (#n), so that a repeated word is unambiguous. Three decimals
        # keep a value just below the reporting threshold from reading as equal to it.
        lines += ["", _table(
            ["word", "phones", "edit distance", "weighted edits", "reported at", "reported"],
            [[f"{w['word']} #{w['position']}", w["n_phones"], w["distance"], _decimals3(w["weighted_edits"]),
              _decimals3(_reporting_value(w["n_phones"], thresholds)), "yes" if w["reported"] else "no"]
             for w in r["word_edits"]],
            ["---", "--:", "--:", "--:", "--:", "---"],
        )]
        phone_lines = [
            f"- {w['word']} #{w['position']}: "
            + ", ".join(f"{p['expected']} → {p['heard'] or '∅'} ({p['confidence']:.2f})" for p in w["phones"])
            for w in word_reports.get(r["sample_id"]) or [] if w["distance"] and w["phones"]
        ]
        if phone_lines:
            lines += [""] + phone_lines
    return lines


def render_markdown(run, rows, word_reports):
    """The human-readable ``report.md``. ``word_reports`` maps sample ids to ``diagnostics.word_reports``."""
    summary = run.get("summary") or summarize(rows)
    sections = [
        _header(run, rows),
        _summary_section(summary, rows, run.get("missing_samples") or []),
        _samples_section(rows),
        _word_section(run, rows, word_reports),
    ]
    return "\n\n".join("\n".join(section) for section in sections) + "\n"
