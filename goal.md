# Amazon ML Challenge — Full Execution Plan

**Task:** Business Entity Resolution ($F_{0.5}$ macro-averaged, precision-weighted 2x)

> [!NOTE]
> **Fine-tuning is optional.** Nothing in the problem statement requires an LLM. A classical pipeline — blocking + feature engineering + gradient-boosted classifier — is fully compliant and sufficient to get scored. Fine-tuning is a ceiling-raising upgrade to attempt only after a working baseline is submitted, not a prerequisite.

> [!NOTE]
> **On the expected-score ranges below:** These are rough heuristics based on how blocking → GBM → fine-tuned-matcher pipelines typically progress on entity resolution tasks with this noise profile. They are not measured on your data and could be off either direction depending on real dataset noise. Treat them as sanity-check ranges, not targets to force.

---

## 0. Hard Constraints (Violating any of these gets you rejected, not just scored low)

| Constraint | Detail |
| :--- | :--- |
| **Output format** | `matching_results.tsv` (scored) + `candidate_pairs.tsv` (audited, not scored), both tab-separated |
| **Coverage** | Every Source 1 test entity appears exactly once in both files |
| **ID validity** | `matched_entity_ids` / `candidate_entity_ids`: comma-separated, S2-/S3- prefixed only, must exist in test set, no duplicates, empty string for no match |
| **Subset rule** | Every ID in `matching_results.tsv` must also appear in that entity's `candidate_pairs.tsv` row |
| **External data** | **Strictly prohibited** — no APIs, no geocoding, no business registries, no internet augmentation. Pure computation on provided fields only. Local embedding models are fine — they compute similarity from provided text, they don't query external services. |
| **Model license** | *(Only if using a fine-tuned model)* $\le 8\text{B}$ parameters AND **MIT or Apache 2.0 license** |
| **Country handling** | **Open set** — train has US/India only, test adds France. Do not hard-code, filter, or one-hot to {US, India} |

> [!IMPORTANT]
> **Do not use Gemma or Llama** — Gemma is under Google's custom Gemma Terms of Use, Llama models are under Meta's custom license. Neither is MIT/Apache 2.0. Using either as your final fine-tuned model violates the license constraint.

---

## Compute Timeline

| Resource | Available | Primary Use |
| :--- | :--- | :--- |
| **RTX 3060** | Now | Blocking (token/phonetic), feature engineering, GBM baseline |
| **RTX A5000 (16GB)** | This evening | Embedding-based blocking, GBM tuning, first submission |
| **Trainium 2** | Tomorrow 5pm | *(Optional)* QLoRA fine-tune of embedder + cross-encoder matcher |

---

## Model Choice (Only relevant if attempting the optional fine-tune)

| Model | Params | License | Fit |
| :--- | :--- | :--- | :--- |
| **Qwen2.5-3B** ✅ *(Recommended)* | 3B | Apache 2.0 | Strong multilingual grounding — helps with transliteration and Indian address noise |
| **Qwen2.5-7B** | 7B | Apache 2.0 | Stretch option if Trainium 2 throughput allows |
| **Phi-3-mini** | 3.8B | MIT | Strong per-param reasoning, fast fallback |
| **Mistral-7B-v0.3** | 7B | Apache 2.0 | Solid general baseline |

**Final pick if fine-tuning:** `Qwen2.5-3B`, QLoRA, sequence-classification cross-encoder.

---

## Execution Phases

### Phase 0 — Setup
- **Goal:** Clean, split data ready for pipeline work. No leakage risk.
- **Tasks:**
  - Load all TSVs with `sep="\t"`, verify row counts and `entity_id` prefixes.
  - Normalize name: lowercase, strip punctuation, collapse whitespace, expand legal-suffix abbreviations (`Corp`/`Corporation`, `Pvt`/`Private`, `Ltd`/`Limited`, `Inc`/`Incorporated`), `&` → `and`.
  - Normalize address: lowercase, expand street abbreviations (`Rd` → `Road`, `St` → `Street`) using per-country dictionaries — US/India conventions differ, and France has zero training examples, so don't build a France-specific rule; let features generalize instead.
  - Strip landmark phrases (`"Near X"`, `"Nr. X"`) into a separate flag rather than deleting silently — presence of a landmark reference is itself a feature.
  - Split train into train/val by `Source1` `entity_id`, never by pair.
- **Done when:** Normalized fields spot-checked on 20 random rows, val split has zero entity overlap with train.
- **Expected outcome:** No score yet — this is infra, not modeling.

---

### Phase 1 — Blocking / Candidate Generation (3060, now)
- **Goal:** Recall ceiling $\ge 95\%$ on val split. This is the hardest gate in the whole pipeline — you cannot match a record you never consider, and nothing downstream fixes a blocking miss.
- **Tasks:**
  - Token-blocking: shared word/n-gram overlap on normalized name.
  - Phonetic blocking: Soundex or Metaphone key on normalized name.
  - Union all strategies → write to `candidate_pairs.tsv` format.
  - Measure recall = $(\text{true matches found in candidates}) / (\text{total true matches})$ on val split.
- **Done when:** Recall ceiling $\ge 95\%$. If below, widen blocking before proceeding — do not move to Phase 2 with an unverified ceiling.
- **Expected recall ceiling:** 85–95% with token + phonetic alone (typically the weak point before embeddings are added).

---

### Phase 2 — Feature Engineering (3060, now)
- **Goal:** A labeled pairwise dataset with discriminative features. More features than the bare minimum is fine and often necessary — nothing in the rules caps feature count, only what data sources can feed them (provided fields only).
- **Name similarity:**
  - Jaccard (token-set overlap)
  - Levenshtein distance (normalized 0–1)
  - Jaro-Winkler similarity — often beats Levenshtein on short business names, rewards common prefixes
  - Character n-gram ($n=2,3$) cosine/Jaccard — catches typos and transliteration noise word-level Jaccard misses
  - Token sort ratio / token set ratio — handles word-order transpositions
  - TF-IDF cosine similarity
  - Acronym match flag (do one name's initials match the other's expansion?)
  - Legal-suffix-stripped exact match flag
- **Address similarity:**
  - Component-level parsing (street number, street name, city, state/region, postal code) with per-component similarity — usually the single biggest $F_{0.5}$ lift, since it turns one noisy blob comparison into several precise ones
  - Numeric token match (street number / PIN exact match) — high-precision signal
  - Landmark-stripped comparison
  - TF-IDF cosine on full normalized address
  - *(Optional)* sentence-embedding cosine — a local pretrained embedder (e.g. `multilingual-e5-small`) computing semantic similarity is compliant; it's not an external lookup
- **Structural / meta features:**
  - Name length ratio, address length ratio
  - Country match/mismatch flag (soft signal, never a filter)
  - Source-pair identity (`S1-S2` vs `S1-S3`) as categorical — sources may carry different noise profiles
- **Done when:** Feature dataframe built for train + val, class balance checked (expect heavy negative skew).
- **Checkpoint:** After Phase 3's first model trains, check feature importance and drop anything that doesn't move val $F_{0.5}$.

---

### Phase 3 — GBM Baseline Model (3060, now)
- **Goal:** A trained, calibrated, threshold-tuned classifier — first real $F_{0.5}$ number, and a fully compliant standalone submission path.
- **Tasks:**
  - Train LightGBM or CatBoost on pairwise features (logistic regression as a lighter fallback).
  - Handle class imbalance via class weighting.
  - Calibrate output probabilities (isotonic regression) on val split.
  - Sweep decision threshold to maximize macro per-entity $F_{0.5}$ — compute $F_{0.5}$ per Source 1 entity, then average; not global pair-level F1.
  - Bias toward a higher threshold than a pure-F1-optimal cutoff: a false merge costs $\sim 2\times$ a missed match, so when unsure, do not merge.
  - Add per-Source1-entity top-k cap if precision suffers from over-prediction.
  - Verify singleton entities (no true match) score near 1.0 after thresholding — don't let the model default to "always predict something".
- **Done when:** Val $F_{0.5}$ computed and recorded.
- **Expected $F_{0.5}$ (val):** Roughly 0.55–0.70 — wide range because it depends heavily on real noise level vs. the problem statement's illustrative examples.

---

### Phase 4 — Embedding-Based Blocking Upgrade (A5000, this evening)
- **Goal:** Push recall ceiling higher, catch transliteration/reordering cases token/phonetic blocking misses.
- **Tasks:**
  - Encode name+address with `multilingual-e5-small/base`.
  - FAISS top-k nearest neighbors per Source1 entity, merged into existing candidates.
  - Re-measure recall ceiling.
- **Done when:** Recall ceiling improves without candidate set exploding (check reduction ratio doesn't collapse toward brute-force).
- **Expected recall ceiling:** 93–98%, roughly +5–10 points over Phase 1.

---

### Phase 5 — First Leaderboard Submission (A5000, this evening)
- **Goal:** A validated, submitted entry — real feedback beats theoretical improvement.
- **Tasks:**
  - Regenerate both output files with updated blocking + tuned GBM.
  - Run `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`.
  - Fix any flagged issues, re-run until PASS.
  - Upload `matching_results.tsv` to the leaderboard.
- **Done when:** `SCORED` status received.
- **Expected public $F_{0.5}$:** Similar to or slightly above Phase 3's val number, $\pm 0.03\text{–}0.05$ variance.

---

### Phase 6 — Error Analysis (A5000, this evening)
- **Goal:** Concrete failure modes to target next — not vague "improve accuracy."
- **Tasks:**
  - Break down val errors: false merges vs. missed matches, by country, by source-pair.
  - Note concrete EDA observations (noise patterns actually seen, not just problem-statement restatement) — feeds the methodology write-up.
  - Identify hardest cases (near-duplicate wrong-entity pairs) — these become fine-tune hard negatives if Phase 7 is attempted.
  - Start drafting `Documentation_template.md` now, not at the end.
- **Done when:** Ranked list of 3–5 concrete weaknesses (e.g. "France addresses underperform," "abbreviated legal suffixes cause false merges").
- **Expected outcome:** No score — diagnostic input to Phase 7.

---

### Phase 7 — (Optional) Trainium 2 Fine-Tuning (Attempt only after Phase 5's baseline is submitted)

#### Task Framing
Sequence classification (`AutoModelForSequenceClassification`, single logit + sigmoid), not generation. Single forward pass per pair — fast, easy to calibrate, and inference-cheap at scale.

#### Input Template
*(Identical between train and inference, max length 128–256 tokens):*
```text
[S1] name: Acme Corp Pvt Ltd | address: 12 MG Road, Near SBI ATM, Pune, MH, India
[S2] name: ACME CORPORATION | address: 12, M.G. Rd, Pune, Maharashtra
```

#### Data Construction
- **Positives:** Every (S1, matched S2/S3) pair from ground truth.
- **Hard negatives:** Non-matching pairs from blocking candidates that scored high on embedding/GBM similarity — the confusable near-misses that actually move the needle.
- **Easy negatives:** Small proportion of random non-candidate pairs.
- **Target ratio:** ~1 positive : 3–5 hard negatives : 1–2 easy negatives.

#### Fine-Tuning Method
QLoRA, not full fine-tune — 4-bit base + LoRA adapters (rank 16–32, alpha 32) on `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. Lower overfitting risk on a likely modest labeled-pair count, and allows iterate → re-mine → retrain within the window.

#### Loss & Imbalance
Binary cross-entropy with class weighting, or focal loss ($\gamma=2$).

#### Calibration & Threshold
Isotonic-calibrate the sigmoid output on val split; sweep for macro per-entity $F_{0.5}$ (expect a higher optimal threshold than plain-F1, given 2x precision weighting); apply per-entity top-k cap if still over-predicting.

#### Inference
Batch the classification head forward pass only over `candidate_pairs.tsv` — never the full cross product.

#### Checklist
- [ ] Confirm Neuron SDK compatibility with PyTorch/HF Transformers before committing time.
- [ ] Contrastive fine-tune blocking embedder using Phase 6 hard negatives.
- [ ] QLoRA fine-tune Qwen2.5-3B cross-encoder on train pairs + hard negatives.
- [ ] Evaluate against the same val harness used for the GBM baseline — apples to apples.
- **Done when:** Clear statement of whether fine-tuned model beats, matches, or loses to GBM baseline, with numbers.
- **Expected $F_{0.5}$ (val):** Roughly 0.65–0.80 if hard-negative mining and calibration go well; usually +5–15 points over the GBM baseline, but also the phase most likely to underdeliver if the labeled pair count is small or Neuron compatibility eats into training time.

---

### Phase 8 — Final Assembly
- **Goal:** Submission-ready package, not a scramble at the deadline.
- **Tasks:**
  - Pick winning model (GBM or fine-tuned, whichever scores higher on val).
  - Finalize both output files, re-validate.
  - Clean `code/business_entity_resolution/src/`, write `README.md` with exact reproduction steps, pin `requirements.txt`.
  - Complete `Documentation_template.md` using the mapping below.
  - Zip as `<team_name>_submission.zip` matching required structure:

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/business_entity_resolution/
│   ├── src/
│   ├── README.md
│   └── requirements.txt
└── Documentation_template.md
```

- **Done when:** A fresh clone of `code/` reproduces both output files from raw data using only the `README.md`.

#### `Documentation_template.md` Section Mapping

| Template Section | Source |
| :--- | :--- |
| **2.1 Problem Analysis** | Phase 6's EDA / noise-pattern observations |
| **2.2 Solution Strategy** | Phases 1–4 summary |
| **3. Candidate Generation** | Phase 1/4 recall ceiling + total candidate pair count |
| **4. Matching Model** | Final feature set + model config + threshold |
| **5. Results & Error Analysis** | Final val/public $F_{0.5}$ + Phase 6's qualitative false-positive/false-negative patterns |
| **6. Conclusion** | Written last, pulled from experiment log narrative entries |

---

## North Star Checkpoints
1. **Recall ceiling $\ge 95\%$** before trusting any matching model's score.
2. **Every threshold decision optimized against macro per-entity $F_{0.5}$**, never raw pair accuracy.
3. **Singletons treated as first-class predictions**, not an afterthought.
4. **A working baseline is submitted before any fine-tuning is attempted** — fine-tuning is a bonus, not a dependency.
5. **Every feature/technique used is derived from provided fields only** — no external lookups, no incompatible-license models.
