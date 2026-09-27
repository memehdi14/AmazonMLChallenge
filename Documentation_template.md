
# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** orions  
**Team Members:** Namdar MohammadMehdi, Lakshya Jain, Jiya Patel (Nirma University)  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We resolve each Source 1 business against ~10M Source 2/3 records with a three-stage pipeline:
(1) **blocking** — the union of a rare-token IDF pass and an exact GPU kNN over a *fine-tuned*
multilingual-e5-small encoder (MIT, 118M params), which keeps **99.2 %** of true matches in only
~25 candidates per entity; (2) a **two-stage LightGBM** matcher over 61 hand-engineered pair features
(fuzzy name/address, phonetic skeletons for transliterated names, number/postal agreement, blocking
scores, candidate-competition context); (3) a **metric-aware decision layer** — one-to-one
assignment plus a per-entity set selection that maximises expected F0.5 (empty set allowed).
Held-out macro F0.5 on 200k training entities: **0.9699** (US 0.9734, India 0.9646). Everything
is country-agnostic, so the unseen country (France) runs through exactly the same code.

---

## 2. Methodology

### 2.1 Problem Analysis
EDA on the training split (`src/eda.py`):

| Fact | Value | Consequence for the design |
|---|---|---|
| Records | S1 2.21M · S2 5.03M · S3 5.29M (test: 1.73M · 4.89M · 5.08M) | all-pairs is impossible → blocking; memory-lean code for a 16 GB laptop |
| Singletons | 5.6 % of S1 | "predict empty" matters but is rare |
| Matches per S1 | mostly 2–6, mean ≈ 3.5 (max 11) | recall matters as much as the singleton decision |
| One-to-one | **0** S2/S3 ids appear under two S1s | each S2/S3 record may be assigned to at most one S1 |
| Country | 100 % of true pairs share the country label | block within each country value (open set) |
| Distractors | ~26 % of S2/S3 records match no S1 | the matcher must reject look-alikes |
| Scripts | ~9 % of S2 names in Devanagari/Bengali | phonetic transliterations of *English* names |

Noise observed in true pairs (examples from the data):
- **Names:** legal words anywhere (`Private Anand Foundation Ltd`), junk (`***`, `<<`, `(ID: 54473)`,
  `[Ltd]`), domain-style names (`UROLOGYSTRATEGICHEALTH.COM`, `@GIBSONGALLARDO`), aliases
  (`Yumabrixveo fka Allsun Allen, DO`, `… a/k/a Caum, LLC`, `name | www.x.com`), OCR digit swaps
  (`N0SE`, `BAILEY6ULF`), transliterations (`राम मार्केटिंग प्राइवेट लिमिटेड` → "raam maarketting praaivett limittedd").
- **Addresses:** `ST` rendered as `SAINT`, component reordering, state codes vs names vs native script
  (`MH` / `Maharashtra` / `महाराष्ट्र`), dropped components, mutated house numbers (`165` vs `16`).

### 2.2 Solution Strategy
**Approach Type:** Blocking + learned pair classifier + metric-aware decision layer (hybrid lexical + dense retrieval).  
**Core Innovation:** a contrastively fine-tuned multilingual bi-encoder for blocking (recall@20 jumps
from 83 % for lexical blocking to 99.2 %), phonetic "consonant skeletons" that align transliterated
and typo'd names, candidate-competition features, and a decision layer that optimises the exact
per-entity F0.5 with the one-to-one constraint discovered in EDA.

```
raw TSVs → normalise (hand-written rules) → blocking: token-IDF ∪ fine-tuned-e5 kNN (per country, ≤25/S1)
        → 61 pair features → LightGBM stage 1 → + per-S1 probability context → LightGBM stage 2
        → one-to-one + expected-F0.5 set selection → matching_results.tsv (candidates → candidate_pairs.tsv)
```

---

## 3. Candidate Generation (Blocking)

**Normalisation (`src/normalize.py`, hand-written, country-agnostic).** Unidecode transliteration,
lowercase, `&`→`and`, merged initials (`M.G.`→`mg`), junk/ID-tag removal, alias split
(`fka / aka / dba / |`), URL stems, OCR digit repair inside words. Name views: *clean*, *core*
(legal words removed — pvt/ltd/llc/inc/corp/sarl/sas/sa/eurl/…, also by phonetic skeleton so
"praaivett limittedd" is removed), *skeleton* (phonetic consonant key: bh/w→v, ph→f, sh→s, …,
drop non-initial vowels, squeeze repeats — "sebhen"/"seven" → `svn`), *no-space* (for domain-style
names) and *alias*. Addresses are mapped to canonical **short** forms in both directions
(road→rd, street/str/saint→st, rue→r, avenue/av→ave, boulevard/bd→blvd, nagar→ngr, chemin→ch, …),
and postal codes, house numbers and landmark tokens ("near/opp …") are extracted.

**Blocking keys used** (all computed within each country value; an unseen label on the pool side
falls back to the whole pool):
1. **Rare-token pass** — IDF-weighted cosine over `n:` name-skeleton tokens, `s:` whole no-space
   name and `a:` address tokens; tokens with pool document frequency > 3000 are dropped (keeps the
   sparse product sparse). Top-12 per S1.
2. **Dense pass** — `intfloat/multilingual-e5-small` (MIT) **fine-tuned** on 400k training pairs
   (only non-validation S1s) with symmetric InfoNCE; batches are built from S1s in the same
   city/state so in-batch negatives are hard. Records are encoded as `name | address`; exact
   inner-product kNN on GPU, chunked over queries and pool (pool embeddings are a disk memmap).
   Top-20 per S1.
3. **Union**, then keep the 25 best per S1 by best rank across passes.

**Candidate pairs generated:** 42,789,509 on test (24.7 per S1); 54,599,219 on train.
Reduction ratio ≈ 1 − 24.7 / 3.9M (same-country pool) ≈ **99.9994 %**.

**How we ensured true matches were not lost:** recall was measured on held-out S1s against the
*full* pool (distractor density matters) after every change (`src/exp_blocking.py`):

| Pass (20k validation S1 vs full 10.3M pool) | @5 | @10 | @20 | @30 |
|---|---|---|---|---|
| rare-token IDF | 70.6 % | 78.6 % | 83.2 % | 85.3 % |
| fine-tuned e5 kNN | 90.7 % | 98.5 % | 99.2 % | 99.4 % |
| union (best rank) | 93.4 % | 98.8 % | 99.4 % | 99.5 % |

Final configuration on the 200k training sample: **recall 99.22 %**, oracle macro F0.5 (perfect
matcher on these candidates) **0.9976**. Every final match is a subset of its candidates (enforced in code).

---

## 4. Matching Model

**Features used (61, `src/features.py`):**
- **Name features:** Levenshtein ratio, partial ratio, token-sort and token-set ratio, Jaro-Winkler on
  the core name; ratio on the clean name; ratio / token-set / Jaccard / overlap on phonetic
  skeletons; ratio and partial ratio on the no-space name; exact-core and no-space equality;
  alias token-set similarity.
- **Address features:** ratio, partial, token-set, token-sort, token Jaccard/overlap; house-number
  equality; number-set Jaccard / overlap / conflict; postal-code Jaccard; landmark overlap; missing flags.
- **Blocking features:** token-pass score & rank, embedding cosine & rank, best rank.
- **Context features:** rank and gap-to-best of this candidate among its S1's candidates (for
  combined, name, address, skeleton, embedding and token scores); and — computed over **all** S1s,
  exactly as at test time — how many S1s compete for this candidate and this S1's rank/gap among them.
- **Meta:** source (S2/S3), string lengths. Country is never one-hot encoded.

**Model type:** LightGBM binary classifier (MIT), 127 leaves, lr 0.05, early stopping; 5 folds
grouped by S1 (crc32 hash of the S1 id), fold models averaged at inference. **Stage 2** adds
per-S1 probability context computed from the out-of-fold stage-1 probabilities (rank, gap to best,
second best, probability mass, count > 0.5) and is kept because it improves OOF F0.5 (0.9685 → 0.9699).
Training data = all blocking candidates of a 200k-S1 random sample (4.9M pairs), i.e. exactly the
inference distribution, with hard negatives for free.

Most important features (gain): embedding-candidate rank among competing S1s, embedding cosine,
embedding gap, house-number conflict, number Jaccard, clean-name ratio, address token-set, skeleton Jaccard.

**Threshold selection method:** per-entity decision tuned on out-of-fold probabilities to maximise
the exact macro F0.5 (singletons included). (1) **One-to-one:** each S2/S3 record is kept only for
its highest-probability S1. (2) **Set selection:** for each S1, candidates sorted by probability,
choose the prefix (possibly empty) maximising approximate expected F0.5,
`1.25·Σp_top-k / (0.25·Σp + k)` vs. `α·Π(1−p)` for the empty set; α = 1.5 was chosen by grid search
against absolute/relative thresholds. No tuning on the public leaderboard.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9699** on 200,000 held-out training S1s (each scored by a fold model
  that never saw it), computed with the exact formula of the problem statement (verified on its worked
  example = 0.714). US 0.9734 · India 0.9646 · 79.1 % of entities perfect · 0.84 % scored 0.
  By number of true matches: 0 → 0.954, 1 → 0.893, 2 → 0.967, 3+ → 0.975–0.980.
  Pair level: precision 0.989, recall 0.938.
- **Generalisation to an unseen country:** leave-one-country-out (train one country, score the other,
  stage 1): US→India 0.904, India→US 0.918. France is scored by models trained on *both* countries, and
  all rules are generic; expected leaderboard ≈ 0.96.
- **Common false positives (wrong merges):** 7,105 false pairs on 200k S1 (632 on true singletons).
  Dominant pattern — **near-duplicate decoys**: the same (or near-same) name at the same street with a
  slightly different house number, e.g. `Selia Alvarez Horizon Twin LLC, 18730 Little Lane` →
  `… LTD, 1873 Little Ln`; `Delhi Sports, Flat No 243` → `Flat No 247`; `Johnson and Roe, 3099
  Breckenridge Ln` → `3104 Breckenridge Ln`. Also: generic public names in a different city
  (`Department of Aging`, Conroe vs Temple, TX), a different name at the exact same address
  (`Slate Inc` → `QU0XYLO`, same street address), and name-only records with an empty address.
- **Common false negatives (missed matches):** 42,850 missed pairs — 5,387 never reached the model
  (blocking), 37,463 rejected by the model/decision. The mirror image of the decoys: **true** matches
  whose house number was also mutated (`14298` vs `14306`, `206` vs `207`, `6138` vs `138`, `91A` vs `51A`),
  so number noise on true pairs and decoys are hard to separate. Also: a completely different trade name
  at the same address (`Starshine Medical` → `Ectodova`), records with an empty address and a noisy name,
  and native-script names (`আল বিজনেস প্রাইভেট লিমিটেড` for `Al Business Private Limited`). F0.5
  deliberately trades some of this recall for precision.

---

## 6. Conclusion
A fine-tuned dense retriever solved candidate generation (99.2 % recall at ~25 candidates), after which
a feature-rich LightGBM with competition context and an F0.5-aware one-to-one decision layer reaches
0.97 macro F0.5 on held-out data. The biggest lessons: measure blocking against the full pool, exploit
structural facts from EDA (one-to-one, same-country), and engineer for memory at 10M-record scale.
Next steps would be a cross-encoder re-scorer for the ~37k rejected true pairs and training on all 2.2M S1s.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` — all source in `src/`, `README.md` with exact steps, pinned
`requirements.txt`, `configs/default.yaml`. One command reproduces both outputs:
`python -m src.reproduce` (normalise → fine-tune encoder → embed → block → features → LightGBM → decide
→ validator). Entry point for single stages: `python -m src.run --stage <stage> --split <train|test>`.
Models: `intfloat/multilingual-e5-small` (MIT, 118M params, fine-tuned) and LightGBM (MIT). No external
data, APIs, geocoding or lookups are used; all normalisation rules are hand-written.

### B. Additional Results

| Experiment | Blocking recall | Oracle F0.5 | OOF macro F0.5 |
|---|---|---|---|
| Token pass only, 5k S1 (smoke) | 0.833 | 0.927 | 0.880 |
| Token ∪ fine-tuned e5 (cap 25), stage 1, 200k S1 | 0.9922 | 0.9976 | 0.9685 |
| + stage 2 probability context (submitted v1) | 0.9922 | 0.9976 | 0.9699 |
| same, LightGBM trained on 500k S1, lr 0.1 (**v2**) | 0.9922 | 0.9976 | **0.9719** |

Learning curve (stage 1, fixed 40k-S1 validation): 20k → 0.9617, 40k → 0.9640, 80k → 0.9661,
160k → 0.9683 training S1 — about +0.002 per doubling with no plateau, which motivated v2.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
