# Business Entity Resolution — Amazon ML Challenge 2026

For each Source 1 business, find all matching Source 2 / Source 3 records.
Metric: macro F0.5 per S1 entity (singletons included).

## Key EDA facts (train)

| | |
|---|---|
| S1 / S2 / S3 rows | 2.21M / 5.03M / 5.29M (test: 1.73M / 4.89M / 5.08M, incl. France) |
| Singletons | 5.6% — most S1 have 2–6 matches (mean ≈ 3.5), so recall matters |
| One-to-one | holds exactly: no S2/S3 id appears under two S1s |
| Country | every true pair has the same country label → block within country |
| Scripts | ~9% of S2 names are Devanagari/Bengali transliterations of English names |

## Pipeline

```
normalize  -> artefacts/<split>/s{1,2,3}.parquet   name views (clean/core/phonetic skeleton/no-space/alias),
                                                   canonical short address forms, postal/numbers/landmarks
embed      -> artefacts/<split>/emb_{s1,pool}.npy  fine-tuned multilingual-e5-small (MIT) on 'name | address'
block      -> artefacts/<split>/cands.parquet      union of rare-token IDF pass + GPU embedding kNN, per country,
                                                   capped at 40 per S1 (this IS candidate_pairs.tsv)
features   -> artefacts/<split>/feats/part_*.parquet  fuzzy name/address, numbers/postal, blocking scores, context
rank       -> artefacts/model/                     LightGBM, folds grouped by S1 (crc32 hash), OOF-tuned decision
decide     -> output/matching_results.tsv + output/candidate_pairs.tsv, then utils/validate_submission.py
```

Encoder fine-tuning (optional, GPU): `python -m src.finetune_embed --pairs 400000 --out artefacts/embed_ft`
(contrastive InfoNCE, same-area in-batch negatives, only non-validation S1s).

## Run

```
pip install -r requirements.txt
# data in dataset/train and dataset/test
python -m src.run --stage normalize --split train
python -m src.run --stage normalize --split test
python -m src.finetune_embed --out artefacts/embed_ft        # GPU, ~15 min on RTX 4060
python -m src.run --stage embed    --split train             # GPU, ~40 min on RTX 4060
python -m src.run --stage block    --split train
python -m src.run --stage features --split train
python -m src.run --stage rank     --split train             # prints OOF macro F0.5
python -m src.run --stage embed    --split test
python -m src.run --stage block    --split test
python -m src.run --stage features --split test
python -m src.run --stage rank     --split test
python -m src.run --stage decide   --split test              # writes output/ and validates
```

`configs/smoke.yaml` runs the train path on 5k S1 for a quick end-to-end check.
Useful extras: `python -m src.eda`, `python -m src.exp_blocking --n 20000 [--emb]` (blocking recall vs k).

## Memory notes (16 GB machines)
Embeddings are streamed to disk memmaps; kNN pulls 1M-row pool chunks onto the GPU; features are
written in S1 chunks. Avoid running two heavy stages at once.
