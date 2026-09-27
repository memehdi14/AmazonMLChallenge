# Business Entity Resolution — reproduction guide

Regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the raw challenge
data using only this folder. No external data, APIs or lookups are used; the only downloaded
artefact is the pretrained open model `intfloat/multilingual-e5-small` (MIT, 118M parameters),
which is fine-tuned here on the training pairs. The matcher is LightGBM (MIT).

## 1. Environment

- Python 3.13 (tested 3.13.12), Windows 11 / Linux, NVIDIA GPU with ≥ 8 GB (tested RTX 4060 Laptop, CUDA 12.4), 16 GB RAM, ~40 GB free disk.

```
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 2. Data layout

Put the challenge files here (paths are set in `configs/default.yaml`):

```
dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

## 3. Run end-to-end (one command)

From this folder:

```
python -m src.reproduce
```

This runs, in order (every step caches to `artefacts/`; `--resume` skips finished steps):

| # | command | what it does | time (RTX 4060) |
|---|---|---|---|
| 1 | `python -m src.run --stage normalize --split train` / `test` | clean names & addresses → parquet | 3 + 3 min |
| 2 | `python -m src.finetune_embed --pairs 400000 --max_steps 2500 --out artefacts/embed_ft` | contrastive fine-tune of the blocking encoder | 15 min |
| 3 | `python -m src.run --stage embed --split train` / `test` | encode all records (disk memmap) | 45 + 75 min |
| 4 | `python -m src.run --stage block --split train` | token + embedding blocking for all train S1; prints recall | 37 min |
| 5 | `python -m src.run --stage features --split train` | pair features for the 200k-S1 training sample | 3 min |
| 6 | `python -m src.run --stage rank --split train` | grouped 5-fold LightGBM (stage 1 + 2), decision tuning; prints OOF F0.5 | 60 min |
| 7 | `python -m src.run --stage block --split test` | blocking for all test S1 → `candidate_pairs.tsv` content | 28 min |
| 8 | `python -m src.run --stage features --split test` | pair features for all test candidates | 12 min |
| 9 | `python -m src.run --stage rank --split test` | score all test candidates | 70 min |
| 10 | `python -m src.run --stage decide --split test` | one-to-one + F0.5 set selection → both TSVs → validator | 3 min |

Outputs land in `output/`; step 10 runs `utils/validate_submission.py` automatically.

Quick check of the train path on 5k S1: `python -m src.run --stage <block|features|rank> --split train --config configs/smoke.yaml`.

## 4. Source map (`src/`)

| file | role |
|---|---|
| `run.py` | CLI and all pipeline stages (normalize, embed, block, features, rank, decide) |
| `normalize.py` | hand-written, country-agnostic cleaning: name views (clean / core / phonetic skeleton / no-space / alias), canonical address forms, postal / numbers / landmarks |
| `finetune_embed.py` | InfoNCE fine-tuning of multilingual-e5-small with same-area in-batch negatives |
| `embed.py` | batched encoding to disk memmap; exact chunked GPU kNN |
| `blocking.py` | per-country rare-token IDF pass, union with embedding pass, per-S1 cap |
| `features.py` | 61 pair features: fuzzy name/address, numbers/postal/landmarks, blocking scores, context |
| `ranker.py` | LightGBM with S1-grouped folds; stage-2 probability-context model |
| `decide.py` | one-to-one assignment + threshold / expected-F0.5 set selection, tuned on OOF |
| `evaluate.py` | official per-entity F0.5, macro average, blocking recall, deterministic folds |
| `io_utils.py` | TSV reading (explicit tab separator) and writing |
| `eda.py`, `exp_blocking.py`, `analysis.py` | EDA, blocking-recall experiments, validation error analysis / leave-one-country-out |
| `reproduce.py` | runs every step above in order |
| `package_submission.py` | builds the submission zip |

## 5. Determinism

Seeds are fixed (`seed: 42`); folds are a crc32 hash of the S1 id; the training sample is a seeded
random draw. GPU kernels (fp16 encoding, kNN) can differ in the last bits across hardware, which may
flip a handful of near-tie candidates.
