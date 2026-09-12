#!/usr/bin/env bash
# Profile one bench_decode config at two token budgets. Usage: nsys_bd.sh TAG CFG MODEL [presets...]
set -uo pipefail
S=/tmp/claude-1000/-home-inureyes-Development-mlxcel/a7ac83cc-0ca7-4ae0-8f19-a24a07471f2a/scratchpad/1798
WT=/home/inureyes/Development/wt-1798-graph-budget
tag=$1; cfg=$2; model=$3; shift 3
for n in 400 200; do
  python3 $S/bench_bd.py --bin $WT/target/release/mlxcel-bench-decode --model $WT/models/mlx/$model \
    --prompt-file $S/prompt_code0.txt --no-chat-template --out $S/results_nsys.jsonl --configs "$cfg" --rounds 1 --warmup 0 \
    --max-tokens $n --tag "nsys-$tag-$n" \
    --wrap "nsys profile -t cuda --cuda-graph-trace=node --force-overwrite true -o $S/nsys_${tag}_${n}" "$@"
done
echo NSYS_DONE $tag
