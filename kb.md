# Experiment Knowledge Base — Amazon ML Challenge Entity Resolution

> This file is a running log for the coding agent to record every experiment run during this challenge. Append new entries — never delete or overwrite past ones. The goal is to avoid re-testing failed approaches and to always know the current best configuration at a glance.

---

## Current Best (update this section every time a new best is found)

| Field | Value |
|---|---|
| Experiment ID | — |
| Val macro F_0.5 | — |
| Public leaderboard score | — |
| Blocking method | — |
| Feature set | — |
| Model | — |
| Threshold | — |
| Last updated | — |

---

## How to Log an Experiment

Every run — successful or failed — gets one entry in the table below AND, if it's non-trivial, a short narrative block underneath. Do not skip failed/negative-result runs; they're as valuable as wins for avoiding repeated dead ends.

### Experiment Table

| ID | Date | Phase | Blocking method | Recall ceiling | Features used | Model | Threshold | Val Precision | Val Recall | Val F_0.5 | Public LB score | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP-001 | | | | | | | | | | | | |

**Column definitions:**
- **Phase**: which phase of the build guide this corresponds to (Blocking / Feature Eng / GBM / Fine-tune / Threshold sweep)
- **Blocking method**: token / phonetic / embedding / union of which — be specific (e.g. "token+soundex" vs "token+soundex+e5-embedding-top20")
- **Recall ceiling**: measured recall of the candidate set against ground truth on val split — the hard gate number
- **Features used**: comma list or reference to a named feature-set version (see Feature Set Registry below) if the list gets long
- **Model**: exact model/config (e.g. "LightGBM, 500 trees, depth 6" or "Qwen2.5-3B QLoRA r16")
- **Threshold**: the decision threshold used for this run's reported scores
- **Val Precision/Recall/F_0.5**: macro per-entity, computed the same way the competition scores it — not raw pair-level metrics
- **Public LB score**: only filled if this exact config was submitted
- **Notes**: one-line takeaway — what worked, what didn't, what to try next

### Narrative Block Template (for any experiment worth explaining beyond the table)

```
## EXP-XXX — <short title>
**Date:** 
**Hypothesis:** what you expected this change to do
**Change from previous best:** exactly what was different
**Result:** val F_0.5 before → after, precision/recall breakdown
**Why (if known):** root cause of the improvement or regression
**Next step suggested by this result:**
```

---

## Feature Set Registry

Track named, versioned feature sets here instead of re-listing every feature in every table row.

| Feature Set ID | Features included | Introduced in |
|---|---|---|
| FS-v1 | Jaccard, Levenshtein, TF-IDF cosine (base tips-to-success set) | EXP-001 |
| FS-v2 | FS-v1 + Jaro-Winkler, char n-gram cosine, token sort ratio | — |
| FS-v3 | FS-v2 + component-level address parsing (street/city/postal), numeric token match | — |
| FS-v4 | FS-v3 + acronym match, legal-suffix-stripped exact match, length ratios, source-pair categorical | — |

Add new versions as feature sets evolve. Never mutate a past version in place — increment instead, so old experiment rows stay reproducible.

---

## Blocking Recall Ceiling Log

Since this is the hardest gate in the pipeline, track it separately from the main experiment table so degradation is easy to spot.

| Date | Blocking config | Recall ceiling (val) | Candidate set size (avg per S1 entity) | Total candidate pairs generated |
|---|---|---|---|---|
| 2026-09-25 | Name Token (len>=3) + Soundex/Metaphone + 4-char prefix | 53.08% | 49.97 | ~50,000 (on 1,000 S1 val sample across 10.3M pool) |

---

## EDA / Problem Analysis Notes

Feeds directly into Documentation_template.md section 2.1. Log concrete observations as they're found during data exploration — not general problem-statement restatement, actual patterns seen in the provided train data.

| Observation | Example (anonymized/illustrative) | Impact on design decision |
|---|---|---|
| Indic Script Transliteration in S2/S3 | `Raj Investments LLP` (S1) vs `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி` (S2 Tamil) | Name-only blocking fails completely on transliterations; address-based blocking channels are strictly necessary to catch them. |
| Singletons constitute 5.58% of entities | 123,247 out of 2,206,821 S1 entities have 0 matches | Over-predicting on singletons turns score from 1.0 to 0.0; decision threshold must be high and conservative. |
| High Address Fidelity on Garbled/DBA Names | `Maure Williams Colombier Inc` vs `Dréxkor` (both at `85 Wayne Avenue, Ticonderoga, NY`) | Address component matching (street number + street/locality) acts as a high-precision alternative channel. |
| Web Domain Style Names | `maurewilliamscolombier.com` vs `Maure Williams Colombier Inc` | Need web suffix stripping (`.com`, `www.`) and character n-gram / concatenated token comparison. |

---

## Qualitative Error Analysis

Feeds directly into Documentation_template.md section 5. Beyond the aggregate metrics, capture concrete patterns in what's going wrong — with real (or representative) examples pulled from val-split misclassifications.

### Common False Positives (wrong merges)
- Common corporate tokens (`Enterprises`, `Solutions`, `Services`) without distinctive root word overlap.

### Common False Negatives (missed matches)
- Transliterated business names in Indic scripts when address lacks distinct postal codes or street numbers.

---

## Failed Approaches (do not repeat without a new reason to believe it'll work)

| Approach | Why it failed | Date tried |
|---|---|---|
| Indexing short (<5 digit) numeric address tokens | Caused massive candidate explosion and slowed inverted index scanning by 4x. | 2026-09-25 |
| Pure Name-based blocking without address channel | Missed ~47% of matches due to Indic script transliteration and DBA names. | 2026-09-25 |

---

## Threshold Sweep Log

Record sweep results separately since threshold is retuned every time the model or features change.

| Date | Model/Feature version | Threshold tested | Val Precision | Val Recall | Val F_0.5 | Selected? |
|---|---|---|---|---|---|---|
| | | | | | | |

---

## Submission Log

Every leaderboard upload, regardless of whether it improved score — for tracking submission budget and score trajectory over time.

| Submission # | Date/time | Config used (Experiment ID) | Public LB score | Rank at time of submission | Notes |
|---|---|---|---|---|---|
| | | | | | |

---

## Open Questions / Hypotheses to Test

Running list of ideas not yet tried — the agent should pull from here when deciding what to try next, and move items to the experiment table once tested.

- [ ] Does embedding-based blocking (e5-small) meaningfully raise recall ceiling over token+phonetic alone?
- [ ] Does per-country address abbreviation dictionary outperform one global dictionary?
- [ ] Does a per-entity top-k cap improve precision without hurting recall much?
- [ ] Does isotonic calibration change the optimal threshold materially vs. raw GBM probabilities?
- [ ] Fine-tuned cross-encoder vs GBM — does it actually beat the baseline on this specific dataset, or does the labeled pair count limit it?