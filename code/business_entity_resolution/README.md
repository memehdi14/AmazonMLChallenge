# Business Entity Resolution Pipeline — ML Challenge 2026

An end-to-end, high-recall, precision-calibrated entity resolution pipeline built for the Amazon ML Challenge 2026.

## Architecture Overview

1. **Multilingual Normalization (`normalize.py`)**:
   - Accents folded via Unicode NFKD (ensures zero-shot generalization to French entities).
   - Legal suffix expansion and isolation across US, India, and France (`pvt ltd`, `inc`, `corp`, `sarl`, `sas`, etc.).
   - Country-aware street abbreviation dictionary (`rd` → `road`, `bd` → `boulevard`, etc.).
   - Landmark extraction into dedicated fields (`near SBI ATM`, etc.).

2. **Multi-Channel Candidate Generation / Blocking (`blocking.py`)**:
   - Inverted token indexing on distinctive name tokens.
   - 4-character prefix hashing on stripped entity names.
   - Phonetic hashing (Double Metaphone & Soundex) on leading brand tokens.
   - 5-to-6 digit numeric postal code / PIN code channel.
   - Memory-efficient streaming candidate heap pruning (retaining top-$k$ candidates).

3. **Pairwise Feature Engineering (`features.py`)**:
   - Normalized Levenshtein distance, Jaro-Winkler prefix similarity, Token Sort / Token Set ratios via C++ RapidFuzz.
   - Character 3-gram Jaccard, Token Jaccard, acronym expansion matching.
   - Address numeric token overlap, street name similarity, landmark match indicators.
   - Soft country match indicator and source partition flags (`S2` vs `S3`).

4. **Calibrated Matching Classifier (`train_gbm.py`)**:
   - LightGBM gradient boosted decision trees trained with class reweighting.
   - Decision threshold sweep targeting **macro-averaged $F_{0.5}$** (weighting precision 2×).
   - Singletons explicitly scored: predicting empty matches rewards 1.0, avoiding costly false merges.

## Environment & Requirements

```bash
pip install -r requirements.txt
```

Core dependencies:
- Python 3.10+
- `lightgbm>=4.0.0`
- `rapidfuzz>=3.0.0`
- `jellyfish>=1.0.0`
- `pandas>=2.0.0`
- `numpy>=1.26.0`
- `scikit-learn>=1.3.0`

## Reproduction Steps

### 1. Validate on Held-Out Split
```bash
python src/pipeline.py --mode validate --top-k 50
```

### 2. Generate Final Test Submissions
```bash
python src/pipeline.py --mode test --output-dir ../../output --top-k 50
```

### 3. Verify Formatting with Submission Validator
```bash
python ../../Dataset/student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../Dataset/student_resource/dataset/test
```
