# Submission build notes (branch `subs-final`)

## FINAL (best LB): `v15_frself_s-1.25`, LB **0.986509** (uploaded 27 Sep 2026 21:03 IST, laptop build, md5 `8b68aed84c39980832ca283d1f050748`)
The v11 recipe with the Qwen3 cross-encoder's FRANCE scores replaced by a France self-trained adapter (India/US
byte-identical to v11), France logit shift −1.25. Implied France F0.5 ≈ 0.9650 vs v11's 0.9629.
HPC build of the same recipe: `output/v15_frself_s-1.25` on this branch (md5 `8c3d2b13c13e80e3f6f22d1c23e17ccc`,
not uploaded; differs only by LightGBM thread/platform nondeterminism).

France self-training (Qwen side; code + exact commands in branch `ce-qwen`, `qwen_ce/README.md`):
- start from the v11 Qwen3-4B LoRA adapter; 1 epoch, lr 1e-5 (`qwen_dann.py --max_lambda 0`, i.e. no domain loss)
- data (`prep_selftrain.py` + `prep_selftrain_sib.py`, 300k pairs): 108k France pairs where lgbm_full, ce_full and
  Qwen3 all agree (positives one-to-one), 42k SIBLING hard negatives (a pool record confidently owned by S1-A, all 3
  models >= 0.98, paired with a same-core-name sibling S1-B at another address), 150k labelled India/US replay pairs
- France-all rescore -> `handoff/ce_qwen_frself_out` (v11 Qwen3 files with only France scores replaced)
Build: same as below with `--extra qwen=handoff/ce_qwen_frself_out --shift france:-1.25`.

Previous best (kept for reference): `output/v16_v11_s-1.25`, LB 0.986232, md5 `e8ba1a0fee1a186c2193a24570397aaa`.

It is the v11 recipe (LB 0.986201) with only the France decision moved: logit shift −1.25 instead of −1 on France
(unseen-country) pairs. India/US decisions are identical to the v11 recipe.

## Models (all MIT / Apache-2.0; total ≈ 6.22 B parameters for the final)
| Component | Params | Source |
|---|---|---|
| multilingual-e5-small (fine-tuned blocker) | 0.12 B | `src/embed.py`, `src/finetune_embed.py` |
| xlm-roberta-base CE | 0.28 B | `handoff/ce_out` (branch `ce-results`) |
| xlm-roberta-large CE (`ce_large`) | 0.56 B | branch `ce-large` |
| xlm-roberta-large CE continued on full data (`ce_full`) | 0.56 B | branch `ce-full` |
| xlm-roberta-large France self-trained r1 (`ce_france`, swapped in for France) | 0.56 B | branch `ce-france` |
| Qwen3-4B-Base cross-encoder, LoRA r=32 (`qwen`, India/US scores) | 4.02 B + 0.07 B | branch `ce-qwen`, `qwen_ce/` (README there); weights `final_models/qwen3_v11_adapter` |
| France self-trained LoRA adapter on the SAME Qwen3-4B base (France scores only) | + 0.07 B | `final_models/qwen3_frself_adapter` (adapter_model.safetensors md5 ff0c2b44ede8c3f5db70c4570ba30cd3) |
| LightGBM ranker + LightGBM stacker | negligible | `src/run.py`, `src/stack_submit.py` |

## Exact build (run from the repo root, CPU only)
Inputs, checked out from their branches into `handoff/` (not tracked on `main`):
```
git fetch origin full-results ce-large ce-full ce-france ce-results ce-qwen
for bd in full-results:full_out ce-large:ce_large_out ce-full:ce_full_out ce-france:ce_france_out ce-results:ce_out ce-qwen:ce_qwen_out; do
  git checkout origin/${bd%%:*} -- handoff/${bd##*:} && git reset -q handoff/${bd##*:}; done
rm -f handoff/ce_qwen_out/*_rest*    # v11 coverage: Qwen3 fold-0 without the remainder file
```
Held-out check (India/US fold-0, cross-fitted halves) - expect 0.9903:
```
python -m src.stack_eval --pairs "handoff/full_out/oof_train_full_part*.parquet" \
  --extra ce_large=handoff/ce_large_out --extra ce_full=handoff/ce_full_out --extra qwen=handoff/ce_qwen_out \
  --fill qwen=ce_full --compare_extras --only_all --tag v11
```
Submission (writes one folder per France shift; `-1.25` is the best):
```
python -m src.stack_submit --features extra \
  --extra ce_large=handoff/ce_large_out --extra ce_full=handoff/ce_full_out --extra qwen=handoff/ce_qwen_out \
  --fill qwen=ce_full --pairs "handoff/full_out/oof_train_full_part*.parquet" \
  --lgbm_test "handoff/full_out/probs_model_full_part*.parquet" --swap ce_large=handoff/ce_france_out \
  --shift "france:-1.5,-1.25,-0.75,-0.5" --alpha 1.5 --out output/v16_v11
```
`scripts/hpc_build.sh` wraps the same steps (8 threads); `scripts/fr_diff.py` compares a file to a reference by country.
Threads: `N_CPUS=OMP_NUM_THREADS=RAYON_NUM_THREADS=MKL_NUM_THREADS=6..8`. LightGBM results vary slightly with thread
count / platform (~650 France pairs between the laptop and HPC builds of v11); India/US decisions were identical.

## Leaderboard log (27 Sep)
| File | Change vs v11 | LB |
|---|---|---|
| v11 (26 Sep) | - | 0.986201 |
| v13_q35_s-0.25 | Qwen3.5-4B replaces Qwen3-4B | 0.985613 |
| v14_ens2_s-0.5 | Qwen3 v11 + adapter #2 logit-averaged | 0.985777 |
| **v16_v11_s-1.25** (HPC build, md5 e8ba1a0f…) | France shift −1 → −1.25 | **0.986232** |
| v11s_fr15 (laptop build, md5 22f5d7d1…) | France shift −1 → −1.5 | 0.98615 |
| **v15_frself_s-1.25** (laptop build, md5 8b68aed8…) | France Qwen scores from the France self-trained adapter, shift −1.25 | **0.986509 (FINAL)** |
| v15_frself_s-1.4 (laptop build, md5 9cdc2395…) | same, shift −1.4 | 0.986508 (tie) |

Selected final submission: the 21:03 IST upload, v15_frself_s-1.25, LB 0.986509.

France shift curve on the v11 recipe: −1 → 0.986201, −1.25 → 0.986232, −1.5 → 0.98615, so the optimum is ≈ −1.25.
The −1.25 gain (+0.000031) is within the ~±0.0001 laptop/HPC platform noise; the selected final is the exact file that scored (HPC build).
