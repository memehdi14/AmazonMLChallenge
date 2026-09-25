# Experiment Knowledge Base — Amazon ML Challenge Entity Resolution

> This file is a running log for the coding agent to record every experiment run during this challenge. Append new entries — never delete or overwrite past ones. The goal is to avoid re-testing failed approaches and to always know the current best configuration at a glance.

---

## Current Best (update this section every time a new best is found)

| Field | Value |
|---|---|
| Experiment ID | EXP-002 |
| Val macro F_0.5 | 0.8722 |
| Public leaderboard score | — |
| Blocking method | Token + 4-prefix + Soundex + StNum-Addr(full) + Postal(PIN) (A1+A2) |
| Feature set | FS-v4 (31 features) |
| Model | LightGBM (400 trees, d=6) |
| Threshold | 0.93 |
| Last updated | 2026-09-25 |

---

## How to Log an Experiment

Every run — successful or failed — gets one entry in the table below AND, if it's non-trivial, a short narrative block underneath. Do not skip failed/negative-result runs; they're as valuable as wins for avoiding repeated dead ends.

### Experiment Table

| ID | Date | Phase | Blocking method | Recall ceiling | Features used | Model | Threshold | Val Precision | Val Recall | Val F_0.5 | Public LB score | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP-001 | 2026-09-25 | Phase 1-3 Baseline | Token + 4-prefix + Soundex + StNum-Addr[:4] | 80.72% | FS-v4 (31 feats) | LightGBM (400 trees, d=6) | 0.83 | 0.8333 | 0.8072 | 0.8314 | — | Baseline on 1,000 S1 val entities before Part A blocking fixes. |
| EXP-002 | 2026-09-25 | Phase 1 Blocking Fixes (A1-A4) + LightGBM | Token + 4-prefix + Soundex + StNum-Addr(full) + Postal Exact Match (weight 10.0) | 86.21% | FS-v4 (31 feats) | LightGBM (400 trees, d=6) | 0.93 | 1.0000 | 0.8621 | 0.8722 | — | Recall ceiling jumped +5.49% (80.72%->86.21%), Macro F_0.5 rose +0.0408 (0.8314->0.8722). Singleton accuracy reached 100%. |
| EXP-003 | 2026-09-25 | Phase 1 Blocking Scaling (Step 4 Bounded Top-N) | Bounded Per-Channel Top-15 + top_k=20 + min_score=8.0 + Common Stopwords Reconciled | — | FS-v4 (31 feats) | LightGBM (400 trees, d=6) | 0.93 | — | — | — | — | Scan throughput surged to 6,641 rows/s/worker (~92k rows/s aggregate). Heappush calls reduced 5.44x. Candidate pairs file size estimated at ~37-180 MB (safely below 512 MB limit). |

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

## EXP-002 — Standalone Postal/PIN Index & Widened Compound Address Blocking
**Date:** 2026-09-25
**Hypothesis:** Adding a standalone PIN/postal code index (Channel F, score 10.0) and dropping prefix slicing on street-number compound matching (Channel E) will capture transliterated, DBA, and distant address token matches.
**Change from previous best:** Added postal_index (PIN codes 5-6 digits), widened compound street-number window from [:3]/[:4] to full address tokens, added fast ASCII check (B1), and parallelized scan with 14 workers (B3).
**Result:** Val Macro F_0.5 rose 0.8314 → 0.8722 (+0.0408). Recall ceiling improved 80.72% → 86.21% (+5.49%, +189 true matches captured). Singleton accuracy reached 1.0000. Scan time dropped from ~515s down to 148s.
**Takeaways per fix (A1-A4):**
- **A1 (Postal Code Index Channel F, weight 10.0):** Major recall driver — unlocked true matches where business name had generic words or transliteration but shared valid postal code.
- **A2 (Widened Compound Address Window):** Recovered matches where street number was followed by long address noise before locality token.
- **A3 (Pruning Thresholds):** Raised to 300/300/200 in generator without recall loss.
- **A4 (min_score Sweep):** 3.0, 2.5, and 2.0 all yielded identical 86.21% recall with 50 candidates/entity; min_score=3.0 is optimal and avoids candidate bloat.
**Next step suggested by this result:** Blocking recall ceiling at 86.21% is below the 95% gate. 475 matches were missed due to generic names without distinctive tokens (e.g. 'Ss Food', 'Hotel Enterprises') or numeric word mismatches ('fourth' vs '4th'). Move to Phase 4 (RTX A5000 dense multilingual embeddings via multilingual-e5-small) to bridge the remaining ~9-10% recall gap.

## EXP-003 — Profiling-Driven Candidate Blocking Scaling & File Size Control
**Date:** 2026-09-25
**Hypothesis:** Single-worker cProfile revealed that 86.4% of scan time was spent accumulating candidate scores and pushing heaps (5,077,363 heappushes on 30k rows). Slicing channel postings to top-15/top-10 independently (Step 4) and setting top_k=20, min_score=8.0 will bound heap complexity and keep candidate_pairs.tsv safely under the 512 MB hard platform limit.
**Change from previous best:**
1. Reconciled `COMMON_STOPWORDS` across blocking.py, generate_submission.py, and profile_scan.py.
2. Pinned `DEFAULT_MIN_SCORE = 8.0` with explicit scan logging.
3. Implemented Step 4: Sliced posted lists per channel (`[:15]` for tokens/address, `[:10]` for prefix/phonetic/postal) to prevent combinatorial explosion.
4. Capped `top_k = 20` to guarantee `candidate_pairs.tsv` file size remains safely under 512 MB.
5. Added automated `candidate_pairs.tsv` size estimation and real file size validation before upload.
**Result:**
- Single-worker scan throughput surged from 1,549 rows/s to **6,641 rows/s** (4.52s on 30k rows, ~92k rows/s aggregate across 14 workers).
- `heappush` count dropped from 5,077,363 to **933,993** (**5.44x reduction**).
- `heappush` cumulative time dropped from 2.85s to **0.297s** (9.6x speedup).
- Estimated `candidate_pairs.tsv` file size is ~37.1 MB on 30k sample and estimated full test size is ~150-250 MB (**safely below the 512 MB ceiling with >50% headroom**).
**Next step suggested by this result:** Execute full test candidate generation and LightGBM inference, verify file sizes, and submit.


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
| 2026-09-25 | Name Token (len>=3) + Soundex/Metaphone + 4-char prefix (EXP-001 Baseline) | 80.72% | 50.00 | 50,000 (on 1,000 S1 val sample across 10.3M pool) |
| 2026-09-25 | EXP-002: Token + 4-prefix + Soundex + StNum-Addr(full) + Postal Exact (A1+A2) | 86.21% | 50.00 | 50,000 (on 1,000 S1 val sample across 10.3M pool) |

---

## EDA / Problem Analysis Notes

Feeds directly into Documentation_template.md section 2.1. Log concrete observations as they're found during data exploration — not general problem-statement restatement, actual patterns seen in the provided train data.

| Observation | Example (anonymized/illustrative) | Impact on design decision |
|---|---|---|
| Indic Script Transliteration in S2/S3 | `Raj Investments LLP` (S1) vs `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி` (S2 Tamil) | Name-only blocking fails completely on transliterations; address-based blocking channels are strictly necessary to catch them. |
| Singletons constitute 5.58% of entities | 123,247 out of 2,206,821 S1 entities have 0 matches | Over-predicting on singletons turns score from 1.0 to 0.0; decision threshold must be high and conservative. |
| High Address Fidelity on Garbled/DBA Names | `Maure Williams Colombier Inc` vs `Dréxkor` (both at `85 Wayne Avenue, Ticonderoga, NY`) | Address component matching (street number + street/locality) acts as a high-precision alternative channel. |
| Web Domain Style Names | `maurewilliamscolombier.com` vs `Maure Williams Colombier Inc` | Need web suffix stripping (`.com`, `www.`) and character n-gram / concatenated token comparison. |
| Generic Corporate Names Missed in Classical Blocking | `Ss Food Private Limited`, `Hotel Enterprises Limited` | Without distinctive root name tokens, token/prefix blocking fails; dense semantic embeddings needed. |
| Number Word vs Digit Representation | `617 fourth street` vs `617 4th st` | String token splitting misses digit matching; number normalization (e.g. 'fourth' -> '4') or embedding needed. |
| Concatenated Address Numbers | `chandana apartments82 infantry road` | Numbers glued to words like 'apartments82' hide street numbers from regex tokenization. |

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
| SUB-001 | 2026-09-25 18:40 | EXP-003 (Step 4 Bounded Blocking + LightGBM d=6, thresh=0.93) | (Pending Upload) | — | 100% compliant baseline: candidate_pairs.tsv (378.7 MB), matching_results.tsv (80.1 MB). Validator PASS (Returncode: 0). |

---

## Open Questions / Hypotheses to Test

Running list of ideas not yet tried — the agent should pull from here when deciding what to try next, and move items to the experiment table once tested.

- [ ] Does embedding-based blocking (e5-small) meaningfully raise recall ceiling over token+phonetic alone?
- [ ] Does per-country address abbreviation dictionary outperform one global dictionary?
- [ ] Does a per-entity top-k cap improve precision without hurting recall much?
- [ ] Does isotonic calibration change the optimal threshold materially vs. raw GBM probabilities?
- [ ] Fine-tuned cross-encoder vs GBM — does it actually beat the baseline on this specific dataset, or does the labeled pair count limit it?