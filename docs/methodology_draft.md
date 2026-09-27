# Methodology draft (fill numbers as experiments land)

## EDA facts
- Train: S1 2,206,821 / S2 5,034,616 / S3 5,285,603. Test: S1 1,732,544 (India 810k, US 663k, France 259k).
- Singletons 5.6%. Matches per S1: mostly 2–6 (mean ≈ 3.5), max 11. 7.64M true pairs, 48% from S2.
- One-to-one holds exactly (no S2/S3 id under two S1s). Every true pair shares the country label.
- ~26% of S2/S3 records match no S1 (distractors). ~9% of S2 names are in Devanagari/Bengali
  script — phonetic transliterations of English names ("redd veNcrs praaivett limittedd").
- Name noise: legal words anywhere, junk marks, "(ID: n)", domain-style names, fka/aka aliases,
  OCR digit swaps. Address noise: St→"SAINT", reordering, state codes vs names vs native script,
  dropped components, mutated house numbers.

## Blocking
- Pass 1 — rare-token IDF cosine per country over name phonetic-skeleton tokens, whole no-space
  name, and address tokens (tokens with pool df > 3000 dropped). Token-only recall@20 = 83.2%,
  @50 = 87.3% (20k validation S1 vs full 10.3M pool).
- Pass 2 — multilingual-e5-small (MIT) fine-tuned with in-batch InfoNCE on 400k training pairs,
  batches grouped by city/state for hard negatives; exact GPU kNN per country. Recall@5/10/20/30 = 90.7 / 98.5 / 99.2 / 99.4%
  (20k validation S1 vs full pool) — the fine-tuned encoder carries blocking.
- Union (token top-12 + embedding top-20) capped at 25 per S1 by best rank across passes:
  ~99.1% recall at ~25 candidates per S1 (union@10 per pass: 98.8% at 16/S1; @15: 99.2% at 25/S1).

## Matching
- ~55 features (see src/features.py) + LightGBM, folds grouped by S1 hash.
- Stage 2 (per-S1 probability context) kept only if OOF improves.
- Decision: one-to-one + threshold/expected-F0.5 set selection tuned on OOF.

## Results log
| experiment | blocking recall | oracle F0.5 | OOF F0.5 |
|---|---|---|---|
| token pass only, 5k S1 smoke | 0.833 | 0.927 | 0.880 |
| token + fine-tuned embedding union (cap 25), 200k S1, stage 1 LightGBM | 0.9922 | 0.9976 | 0.9685 |
| + stage 2 (per-S1 probability context) | 0.9922 | 0.9976 | **0.9699** |

Decision tuned on OOF: one-to-one + expected-F0.5 prefix selection (alpha = 1.5).
Top stage-1 features: cand_rank_emb_score, emb_score, cand_gap_emb_score, num_conflict, num_jac,
n_clean_ratio, a_tset, s1_gap_emb_score — i.e. embedding similarity and candidate competition dominate,
with house-number conflict as the strongest string signal.

## Validation analysis (src/analysis.py, 200k-S1 OOF)
- By country: India 0.9646, US 0.9734. By #true matches: 0 (singletons) 0.954, 1 → 0.893, 2 → 0.967, 3+ → 0.975–0.980.
- Pairs: 691,838 gold | TP 648,988 | FP 7,105 (632 on singletons) | FN 42,850 (blocking 5,387; model/decision 37,463).
  Pair precision 0.989, recall 0.938 — the decision layer trades recall for precision, as F0.5 rewards.
- Leave-one-country-out (stage 1 only, 600 rounds, one country as training): US→India 0.9035, India→US 0.9180,
  i.e. a 0.055–0.061 drop vs in-country. France at test time trains on BOTH countries, so its drop should be smaller.

## Leaderboard estimate (v1)
Test S1 mix: India 46.8%, US 38.3%, France 15.0%.
Expected = 0.468·0.9646 + 0.383·0.9734 + 0.150·F_France, with F_France ≈ 0.90–0.94 (LOCO proxy)
→ **≈ 0.962 (range 0.959–0.965)**.
