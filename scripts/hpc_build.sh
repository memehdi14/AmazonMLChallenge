#!/bin/bash
# HPC (login node, CPU) version of scripts/auto_ens.sh: build a v11-recipe submission with a given Qwen column.
# usage: scripts/hpc_build.sh <tag> <qwen dir> [more adapter dirs to logit-average with it]
#   scripts/hpc_build.sh v11chk handoff/ce_qwen_out                        (reproduce v11: must match its numbers)
#   scripts/hpc_build.sh ens2   handoff/ce_qwen_out handoff/ce_qwen_s2_out (v11 Qwen3 + adapter #2)
# Inputs must already be checked out under handoff/ (see HPC_HANDOFF.md, 27 Sep).
set -e
T=$1; shift
cd "$(dirname "$0")/.."
[ -f /home/soft/anaconda3/etc/profile.d/conda.sh ] && source /home/soft/anaconda3/etc/profile.d/conda.sh && conda activate ber 2>/dev/null || true
export N_CPUS=8 OMP_NUM_THREADS=8 RAYON_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=. PYTHONIOENCODING=utf-8
PY="nice -n 10 python"
Q=$1
if [ $# -gt 1 ]; then Q=handoff/ce_qwen_$T; $PY -m src.ce_avg --out $Q "$@"; fi
X=(--extra ce_large=handoff/ce_large_out --extra ce_full=handoff/ce_full_out --extra qwen=$Q --fill qwen=ce_full)
date; echo "== held-out ($T)"
$PY -m src.stack_eval --pairs "handoff/full_out/oof_train_full_part*.parquet" "${X[@]}" --compare_extras --only_all --tag $T \
    > logs/stack_eval_$T.log 2>&1
grep -E "fill|own|held-out|Traceback" logs/stack_eval_$T.log
echo "v11 reference: held-out 0.9903 (India 0.9906 / US 0.9901), train own-score 770,645, test own-score 2,661,807, France 823,814"
date; echo "== build output/v14_$T"
$PY -m src.stack_submit --features extra "${X[@]}" --pairs "handoff/full_out/oof_train_full_part*.parquet" \
    --lgbm_test "handoff/full_out/probs_model_full_part*.parquet" --swap ce_large=handoff/ce_france_out \
    --shift france:-1 --alpha 1.5 --save_probs --out output/v14_$T > logs/v14_$T.log 2>&1
grep -E "stacker|own|swap|shift|matches|PASS|FAIL|Traceback" logs/v14_$T.log
date
