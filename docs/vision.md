# OpenPronounce Fork — Vision and Evaluation Plan

Long-form context for this fork: why it exists, who it serves, what is known to be broken, and how improvements must be measured. The operational rules for working in the repository live in `CLAUDE.md`.

## 1. Project Overview

This repository is a fork of **OpenPronounce**, an open-source pronunciation assessment system based primarily on Wav2Vec2, phoneme recognition, edit-distance alignment, DTW-based acoustic comparison, and prosody extraction.

The purpose of this fork is **not** to build a generic commercial pronunciation platform.

The purpose is to evolve OpenPronounce into a **personal pronunciation laboratory and learning-assistance system** for David, focused on complementing an existing English study plan and improving real spoken English performance.

This project should remain pragmatic, measurable, and evidence-driven.

The main goal is not to maximize a synthetic pronunciation score. The goal is to make pronunciation feedback accurate enough to help identify, explain, practice, and track real pronunciation problems over time.

---

## 2. Why This Project Exists

The project originates from a limitation encountered during conversational English practice with ChatGPT.

During real-time voice conversations, the speech system generally provides ChatGPT with a transcription of what the user said. That transcription is useful for evaluating:

- grammar;
- vocabulary;
- sentence structure;
- fluency at the linguistic level;
- ability to formulate ideas;
- recurring language mistakes.

However, transcription alone is not enough to reliably evaluate pronunciation.

A speech recognizer can:

- correctly infer a word even when pronunciation is imperfect;
- silently normalize or correct a badly pronounced word;
- misrecognize a correctly pronounced word;
- hide the specific sound that caused the mistake;
- provide no direct information about which phoneme was actually produced.

This creates a blind spot.

Example:

The learner may try to say:

`ship`

The transcription system may still return:

`ship`

even if the vowel or consonant was not produced correctly.

Therefore, conversational speaking practice and pronunciation assessment should be treated as **two complementary systems**.

### Conversational practice

Used to improve:

- fluency;
- spontaneous speech;
- vocabulary;
- grammar;
- sentence construction;
- communication under pressure;
- technical and professional English.

### Pronunciation laboratory

Used to improve:

- phoneme production;
- vowel contrasts;
- consonant contrasts;
- word-level pronunciation;
- stress;
- rhythm;
- prosody;
- recurring pronunciation patterns;
- intelligibility.

The intention is for both systems to complement the same English study plan.

---

## 3. Primary User

The primary and initial user is:

**David**

This project should be optimized first for a single learner rather than for a broad public audience.

The learner:

- is a native Spanish speaker;
- is currently working toward a functional B2 level of English;
- needs English for professional software-engineering environments;
- wants to communicate effectively in interviews, meetings, technical discussions, architecture conversations, and daily work;
- already practices conversational English separately;
- wants pronunciation analysis to complement that conversational practice.

This means personalization is acceptable and desirable.

However, personalization must not become blind overfitting.

The system should distinguish between:

1. genuine pronunciation errors;
2. acceptable accent variation;
3. ASR/model uncertainty;
4. microphone/environment artifacts;
5. false positives produced by the pronunciation model.

---

## 4. Learning Goal

The broader learning goal is not accent elimination.

The target is:

> Clear, intelligible, consistent spoken English suitable for professional daily use.

Native-like pronunciation may be desirable in specific cases, but it is not the primary success criterion.

The system should prioritize mistakes that:

- reduce intelligibility;
- change the perceived word;
- cause common lexical confusion;
- repeatedly appear in the learner's speech;
- negatively affect professional communication.

Examples of useful contrasts include:

- `ship` vs `sheep`;
- `ship` vs `chip`;
- `ship` vs `sip`;
- `/ɪ/` vs `/iː/`;
- `/ʃ/` vs `/tʃ/`;
- `/ʃ/` vs `/s/`;
- `/θ/` vs `/t/` or `/s/`;
- `/ð/` vs `/d/`;
- `/v/` vs `/b/`;
- English rhotic sounds;
- word endings;
- consonant clusters;
- stress patterns.

---

## 5. Current Technical Base

OpenPronounce already provides a significant portion of the required technical foundation.

The existing project includes:

- Python;
- FastAPI;
- PyTorch;
- Hugging Face Transformers;
- Wav2Vec2;
- phonemizer / espeak-ng;
- Levenshtein alignment;
- fastdtw;
- librosa;
- soundfile;
- TTS reference generation;
- phoneme-level recognition;
- ASR transcription;
- acoustic distance;
- phoneme error rate;
- word error rate;
- prosody extraction;
- browser microphone recording;
- REST API endpoints.

The current pronunciation pipeline roughly performs:

1. audio normalization to 16 kHz mono;
2. Wav2Vec2 embedding extraction;
3. ASR transcription;
4. expected-text phonemization;
5. direct phone recognition from audio;
6. expected vs detected phone comparison;
7. edit-distance alignment;
8. reference audio generation via TTS;
9. DTW comparison between learner and reference embeddings;
10. pitch and energy extraction;
11. score calculation;
12. feedback generation.

The current aggregate pronunciation score combines:

- acoustic similarity;
- phoneme error rate;
- word error rate.

This is useful as a starting point, but must not be treated as ground truth.

---

## 6. The Core Problem We Are Trying to Solve

The problem is not simply:

> "Make OpenPronounce produce a better score."

The actual problem is:

> Build a pronunciation evaluation system whose feedback is reliable enough to guide a real learner.

A score is useful only when its underlying diagnosis is trustworthy.

The system must answer questions such as:

- What did the learner probably say?
- Which phoneme was expected?
- Which phoneme was probably produced?
- How confident is the model?
- Is the mismatch meaningful?
- Is it likely to affect intelligibility?
- Is the problem reproducible?
- Is it a known recurring learner mistake?
- Should the learner actually spend time correcting it?
- Is the model uncertain enough that the learner should simply retry?

---

## 7. First Confirmed Failure Case

A controlled test was performed with the target word:

`ship`

### Correct pronunciation attempt

Observed result:

- score: approximately `94.61`;
- transcription: `SHIP`;
- expected phones: `/ʃ ɪ p/`;
- detected phones: `/ʃ ɪ p/`;
- phoneme error rate: `0.0`.

This is consistent with a correct result.

### Intentional incorrect pronunciation

The learner intentionally pronounced the target closer to:

`chip`

while the expected word remained:

`ship`.

Observed result:

- score: approximately `26.3`;
- ASR transcription: `TRIP`;
- expected phones: `/ʃ ɪ p/`;
- detected phones: approximately `/tʃ y p/`;
- phoneme error rate: approximately `0.6667`;
- word error rate: `1.0`;
- acoustic distance: significantly worse than the correct attempt.

This is useful because the phone recognizer did detect an initial affricate closer to `/tʃ/`, which matches the intentional error.

However, the returned feedback still said:

`Your pronunciation is excellent!`

and returned no word-level errors.

This is a serious pedagogical inconsistency.

The scoring layer recognized that the pronunciation was poor.

The feedback layer did not.

This case must eventually become a regression test.

Unverified hypothesis for the inconsistency: `_feedback` in `openpronounce/speech.py` only looks at word-level errors and ignores the score, and the `ship` substitutions may fall below the word-reporting threshold described in section 8.2. Note that `tests/test_speech.py` currently asserts the "excellent" message for the no-error path.

---

## 8. Current Known Weaknesses

The current system has several known or suspected weaknesses.

### 8.1 Score and feedback can disagree

A pronunciation can receive:

- a very low score;
- a high phoneme error rate;
- a wrong ASR transcription;

while still producing:

`errors: []`

and:

`Your pronunciation is excellent!`

This should not happen.

### 8.2 Threshold-based word error reporting can hide important mistakes

The existing phone-level logic uses weighted edit thresholds.

A word may contain a meaningful sound substitution without crossing the configured threshold required to mark the whole word as incorrect.

Short words are especially sensitive to this problem.

Current implementation (`openpronounce/phones.py`): each wrong phone gets an error weight (1.0 for a substitution, `NEAR_PHONE_COST = 0.5` for a close phone, reduced further when the expected phone is still plausible in the posteriors). A word is reported only when the weights reach `PHONE_ERROR_THRESHOLD = 0.4` of its phones or `PHONE_ERROR_MIN_EDITS = 2`. For a 3-phone word such as `ship` that means 1.2, so two near-phone substitutions (1.0) are not enough.

### 8.3 ASR is not pronunciation truth

The word recognizer may:

- infer the intended word;
- misrecognize another word;
- hide pronunciation detail.

The direct phone recognizer should be treated as a separate signal.

### 8.4 Phone recognition is uncertain

The phone model may itself make mistakes.

Low-confidence phones should not automatically be treated as learner errors.

Example:

A detected phone with very low confidence should be interpreted differently from a high-confidence substitution.

### 8.5 Synthetic-reference acoustic distance is imperfect

The current implementation compares learner audio against synthesized reference speech.

This may introduce variation caused by:

- TTS voice;
- speaker identity;
- pitch;
- timing;
- microphone;
- room acoustics;
- speaking speed.

Acoustic distance is useful, but should not dominate interpretation without validation.

### 8.6 The system was not calibrated specifically for this learner

The current thresholds and score weights were not trained specifically for David's voice, accent, microphones, environment, or learning goals.

That does not mean they should simply be customized immediately.

Calibration must be performed with a controlled evaluation process.

---

## 9. Engineering Principle: Measure Before Modifying

Do not begin by changing thresholds, replacing models, or adding new AI components.

First build an evaluation harness.

The project must follow this sequence:

```text
Current OpenPronounce
        |
        v
Controlled Evaluation Dataset
        |
        v
Baseline Results
        |
        v
Failure Classification
        |
        v
Hypothesis
        |
        v
Experiment
        |
        v
Measured Comparison
        |
        v
Regression Test
```

Do not optimize based on anecdotal examples alone.

---

## 10. Initial Evaluation Dataset

The first dataset should be small, controlled, and easy to reproduce.

Initial target:

- approximately 20-30 controlled recordings;
- same microphone;
- similar microphone distance;
- same environment;
- low background noise;
- consistent recording setup.

Each sample should contain:

```text
sample_id
expected_text
intended_pronunciation
label
audio_file
microphone
recording_context
raw_model_output
notes
```

Suggested labels:

- `good`
- `intentional_error`
- `uncertain`

Additional metadata may later include:

- target phoneme;
- intended substitution;
- perceived human result;
- model confidence;
- learner confidence.

---

## 11. Initial Minimal-Pair Test Set

Start with controlled pronunciation contrasts.

Example:

### `ship`

Record:

- `ship -> ship`
- `ship -> ship`
- `ship -> sheep`
- `ship -> chip`
- `ship -> sip`

These intentionally isolate different error dimensions.

| Target | Intentional production | Primary contrast |
|---|---|---|
| ship | ship | baseline |
| ship | sheep | `/ɪ/` vs `/iː/` |
| ship | chip | `/ʃ/` vs `/tʃ/` |
| ship | sip | `/ʃ/` vs `/s/` |

Expand later to other high-value Spanish-speaker contrasts.

---

## 12. Required Raw Data

Do not rely only on the final score.

For each analysis, preserve at minimum:

```json
{
  "score": 0,
  "acoustic_distance": 0,
  "transcribe": "",
  "differences": {
    "word_error_rate": 0,
    "phoneme_error_rate": 0,
    "errors": [],
    "expected_phones": [],
    "heard_phones": [],
    "heard_phones_confidence": []
  }
}
```

The original audio should also be preserved.

Prosody arrays do not need to be included in every human-readable experiment report unless the experiment specifically concerns:

- pitch;
- stress;
- intonation;
- rhythm;
- energy.

---

## 13. Desired Feedback Model

The long-term goal is to move from generic feedback such as:

`You need to better pronounce this word.`

toward structured diagnostic feedback.

Example:

```text
Pronunciation score: 26/100

Target:
ship

Detected pronunciation:
likely /tʃ ? p/

Primary issue:
Expected /ʃ/
Detected /tʃ/

Interpretation:
The initial consonant sounds closer to "chip" than "ship".

Confidence:
Medium

Vowel:
The vowel detector had low confidence.
Do not treat this as a confirmed vowel error yet.

ASR:
Expected: ship
Recognized: trip

Recommended exercise:
Practice the /ʃ/ vs /tʃ/ contrast.
```

The system should separate:

- detected fact;
- confidence;
- interpretation;
- teaching recommendation.

Do not present uncertain model output as fact.

---

## 14. Feedback Severity

Future feedback should distinguish at least:

### Correct

Strong evidence that pronunciation matches the target.

### Minor deviation

Difference exists but is unlikely to significantly affect intelligibility.

### Meaningful pronunciation error

Difference is likely to alter or degrade the perceived word.

### Uncertain

The model does not have enough confidence to provide reliable feedback.

The system should prefer:

`uncertain`

over confidently giving incorrect pedagogical advice.

---

## 15. Evaluation Metrics

Do not optimize only for correlation with the aggregate pronunciation score.

Relevant metrics include:

### Detection quality

- true positive rate;
- false positive rate;
- false negative rate;
- precision;
- recall.

### Error classification

Can the system correctly identify the actual contrast?

Example:

```text
expected /ʃ/
actual /tʃ/
```

not merely:

```text
pronunciation bad
```

### Confidence calibration

A low-confidence model prediction should fail gracefully.

### Stability

The same correctly pronounced word recorded several times should produce reasonably stable results.

### Microphone robustness

Eventually test whether microphone changes materially alter diagnosis.

### Learner usefulness

The feedback must result in a clear action the learner can perform.

---

## 16. Anti-Overfitting Rule

Do not tune a threshold because a single sample appears wrong.

Any scoring or threshold modification should be supported by multiple controlled samples.

Prefer:

- reproducible patterns;
- small experiments;
- before/after comparisons;
- regression tests.

Avoid:

- one-off hardcoded exceptions for David;
- arbitrary score changes;
- tuning until a desired score appears.

Personalization may later be introduced explicitly as a separate layer.

---

## 17. Architecture Direction

For now, preserve the existing Python-based pronunciation engine.

Do not rewrite the system in another language.

Python is appropriate because the project depends heavily on:

- PyTorch;
- Hugging Face;
- audio processing;
- ML inference;
- phonetics tooling.

Potential future architecture:

```text
Web / PWA Client
       |
       v
Pronunciation API
       |
       +--> Audio normalization
       |
       +--> ASR signal
       |
       +--> Phone recognition signal
       |
       +--> Acoustic signal
       |
       +--> Prosody signal
       |
       v
Evaluation / Interpretation Layer
       |
       +--> Confidence handling
       +--> Error classification
       +--> Severity
       +--> Learner history
       |
       v
Pedagogical Feedback Layer
       |
       +--> Explanation
       +--> Exercise recommendation
       +--> Progress tracking
```

The main near-term work should happen in the:

**Evaluation / Interpretation Layer**

rather than replacing the recognition stack.

---

## 18. Possible Future Components

Note: a partial Goodness-of-Pronunciation-style plausibility check already exists (`PHONE_PLAUSIBLE_POSTERIOR` in `openpronounce/phones.py`).

These are possibilities, not current requirements.

Do not add them without evidence.

Potential future experiments:

- Whisper / faster-whisper;
- WhisperX;
- forced alignment;
- alternative phone recognizers;
- Goodness of Pronunciation scoring;
- speaker-normalized acoustic embeddings;
- Praat / Parselmouth;
- better prosody scoring;
- multiple TTS reference voices;
- phoneme confusion matrices;
- learner-specific error history;
- spaced repetition;
- minimal-pair exercise generation;
- LLM-generated pedagogical explanations;
- MCP integration;
- conversational-learning integration.

Any external model must have a clear experimental hypothesis.

---

## 19. Relationship With Conversational English Practice

This project is not intended to replace speaking practice.

The study system should eventually contain two distinct feedback loops.

### Loop A: Real-time conversation

Focus:

- fluency;
- grammar;
- vocabulary;
- spontaneous expression;
- professional communication;
- interviews;
- technical discussion.

### Loop B: Pronunciation lab

Focus:

- pronunciation;
- phonetics;
- intelligibility;
- minimal pairs;
- repeated drills;
- sound-level correction.

Findings from the pronunciation lab should inform future conversation practice.

Example:

If the pronunciation system repeatedly detects problems with:

`/ɪ/ vs /iː/`

then conversational sessions may intentionally include words containing those sounds.

Likewise, recurring words that are difficult during conversation may be added to the pronunciation lab.

---

## 20. Long-Term Study Integration

The pronunciation system should complement the learner's broader English study plan.

The target is functional B2 English for daily professional use.

A future workflow may look like:

```text
Conversation practice
       |
       v
Recurring pronunciation issue detected
       |
       v
Pronunciation Lab
       |
       v
Targeted drills
       |
       v
Measured improvement
       |
       v
Return to natural conversation
```

The system should avoid turning the study plan into endless isolated pronunciation drills.

Pronunciation practice should remain connected to real communication.

---

## 21. Initial Technical Backlog

### Phase 0 - Preserve baseline

- [ ] Confirm project runs locally.
- [ ] Confirm all existing tests pass.
- [ ] Record current package versions.
- [ ] Preserve upstream behavior before modifications.
- [ ] Add a clear development branch strategy.

### Phase 1 - Evaluation harness

- [ ] Define dataset directory structure.
- [ ] Define sample metadata schema.
- [ ] Store raw API outputs.
- [ ] Store audio files.
- [ ] Support labels: `good`, `intentional_error`, `uncertain`.
- [ ] Create CLI/script to evaluate all samples.
- [ ] Produce machine-readable results.
- [ ] Produce a compact human-readable report.

### Phase 2 - Baseline corpus

- [ ] Add `ship` correct samples.
- [ ] Add `ship -> sheep`.
- [ ] Add `ship -> chip`.
- [ ] Add `ship -> sip`.
- [ ] Add additional controlled words.
- [ ] Record approximately 20-30 initial samples.

### Phase 3 - Failure analysis

- [ ] Identify false positives.
- [ ] Identify false negatives.
- [ ] Identify low-confidence phone errors.
- [ ] Compare word ASR vs phone recognition.
- [ ] Analyze short-word threshold behavior.
- [ ] Analyze score vs feedback inconsistencies.

### Phase 4 - Feedback consistency

- [ ] Ensure low score cannot incorrectly produce "excellent pronunciation".
- [ ] Separate score from diagnosis.
- [ ] Add uncertainty-aware feedback.
- [ ] Add severity levels.
- [ ] Add regression tests for confirmed failures.

### Phase 5 - Pronunciation diagnostics

- [ ] Report expected vs detected phoneme.
- [ ] Report confidence.
- [ ] Explain likely articulation mistake.
- [ ] Suggest focused minimal-pair practice.
- [ ] Avoid explanations when model confidence is too low.

### Phase 6 - Personal progress

- [ ] Track recurring pronunciation errors.
- [ ] Track improvement over time.
- [ ] Add learner-specific exercises.
- [ ] Rank high-impact pronunciation weaknesses.
- [ ] Integrate results into the broader English study plan.

---

## 22. First Regression Case

Create a regression test representing the observed case:

### Input

Expected:

`ship`

Intentional production:

approximately `chip`

### Observed raw characteristics

- aggregate score around `26`;
- wrong ASR transcription;
- phoneme error rate around `0.66`;
- detected initial phone closer to `/tʃ/`;
- current word-level `errors` empty;
- current feedback says pronunciation is excellent.

### Required future behavior

The system must **not** return positive pronunciation feedback.

Expected interpretation should be equivalent to:

- pronunciation incorrect;
- likely initial consonant substitution;
- detected sound closer to `/tʃ/` than `/ʃ/`;
- low-confidence phones must be identified as uncertain rather than asserted as errors.

Do not hardcode the exact score.

Test semantic consistency.

---
## 23. Definition of Success

The project will be successful when it can reliably help answer:

> "What specific pronunciation problem should I work on next?"

rather than simply:

> "What score did I get?"

A successful system should:

- detect meaningful pronunciation errors;
- avoid frequent false alarms;
- distinguish uncertainty from error;
- identify specific sound contrasts;
- explain the issue clearly;
- suggest an appropriate drill;
- track recurring patterns;
- show measurable improvement;
- complement conversational English practice;
- support progress toward functional professional B2 English.

---

## 24. Non-Goals

At the current stage, do not prioritize:

- multi-tenant SaaS architecture;
- payments;
- public user registration;
- enterprise scalability;
- mobile-native applications;
- social features;
- gamification;
- accent elimination;
- support for many languages;
- production cloud infrastructure;
- replacing every OpenPronounce dependency;
- training a speech model from scratch.

These may be considered later only if the project's purpose changes.

---

## 25. Immediate Next Step

The immediate next task is:

> Build the evaluation harness before changing pronunciation logic.

Existing starting points: `benchmarks/speechocean762.py` and `benchmarks/word_detection.py`. Decide whether to extend them before writing a new harness.

The first deliverable should allow a directory of labeled recordings to be processed through the current OpenPronounce pipeline and produce a structured result containing:

- sample metadata;
- expected text;
- intended pronunciation;
- human label;
- score;
- ASR output;
- expected phones;
- detected phones;
- phone confidences;
- phoneme error rate;
- word error rate;
- acoustic distance;
- reported errors;
- generated feedback.

Once the initial corpus exists, analyze failure patterns before modifying the model or thresholds.

---

## 26. Guiding Principle

The guiding principle of this fork is:

> **Do not teach from a model output until we have enough evidence to trust its interpretation.**

The ML components provide signals.

The application must turn those signals into reliable learning feedback.

That interpretation layer is the core of this project.
