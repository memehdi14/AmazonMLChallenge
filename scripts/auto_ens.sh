#!/bin/bash
# usage: auto_ens.sh <tag> <new adapter dir> [more member dirs...]  - when <new dir> lands complete on ce-qwen:
# average Qwen3 v11 adapter + members (src.ce_avg) -> held-out -> build output/v14_<tag> (v11 recipe) -> diff vs v11.
set -f
T=$1; shift; NEW=$1
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
D=scripts
has() { git ls-tree -r --name-only origin/ce-qwen | grep -q "$NEW/$1"; }
until git fetch -q origin ce-qwen 2>/dev/null && has ce_oof_fold0 && has ce_test_france_part && git ls-tree -r --name-only origin/ce-qwen | grep -qE "$NEW/ce_test_(contested_)?part"; do sleep 60; done
for d in "$@"; do git checkout origin/ce-qwen -- handoff/$d 2>/dev/null; git reset -q handoff/$d; done
date; PYTHONPATH=. $PY -m src.ce_avg --out handoff/ce_qwen_$T handoff/ce_qwen_out $(printf "handoff/%s " "$@")
X=(--extra ce_large=handoff/ce_large_out --extra ce_full=handoff/ce_full_out --extra qwen=handoff/ce_qwen_$T --fill qwen=ce_full)
PYTHONIOENCODING=utf-8 $PY -m src.stack_eval --pairs "handoff/full_out/oof_train_full_part*.parquet" "${X[@]}" --compare_extras --only_all --tag $T > logs/stack_eval_$T.log 2>&1
grep -E "fill|held-out|Traceback" logs/stack_eval_$T.log; date
PYTHONIOENCODING=utf-8 $PY -m src.stack_submit --features extra "${X[@]}" --pairs "handoff/full_out/oof_train_full_part*.parquet" \
    --lgbm_test "handoff/full_out/probs_model_full_part*.parquet" --swap ce_large=handoff/ce_france_out \
    --shift france:-1 --alpha 1.5 --out output/v14_$T > logs/v14_$T.log 2>&1
grep -E "stacker|own score|swap|shifted|PASS|FAIL|Traceback" logs/v14_$T.log
$PY $D/diff_sub.py output/v11_qwen/matching_results.tsv output/v14_$T/matching_results.tsv
date
