#!/usr/bin/env bash
# Fifth #1798 chain: memory footprint against budget size on Laguna, intermediate budgets.
set -uo pipefail
S=/tmp/claude-1000/-home-inureyes-Development-mlxcel/a7ac83cc-0ca7-4ae0-8f19-a24a07471f2a/scratchpad
R=$S/1798
WT=/home/inureyes/Development/wt-1798-graph-budget
BD=$WT/target/release/mlxcel-bench-decode
M=$WT/models/mlx/laguna-xs-2.1-nvfp4
PRESETS=(--preset nograph=MLX_USE_CUDA_GRAPHS=0 --preset ops100=MLX_MAX_OPS_PER_BUFFER=100 \
         --preset mb1000=MLX_MAX_MB_PER_BUFFER=1000 --preset both=MLX_MAX_OPS_PER_BUFFER=100,MLX_MAX_MB_PER_BUFFER=1000 \
         --preset both400=MLX_MAX_OPS_PER_BUFFER=100,MLX_MAX_MB_PER_BUFFER=400 \
         --preset ops50mb1000=MLX_MAX_OPS_PER_BUFFER=50,MLX_MAX_MB_PER_BUFFER=1000 \
         --preset ops100mb100=MLX_MAX_OPS_PER_BUFFER=100,MLX_MAX_MB_PER_BUFFER=100 \
         --preset ops20mb100=MLX_MAX_MB_PER_BUFFER=100)
until grep -q SWEEP4_DONE $R/sweep4.out; do sleep 30; done
$S/gpu_lock.sh acquire 1798 || exit 1
trap '$S/gpu_lock.sh release 1798' EXIT
$R/idle_gate.sh
echo "PHASE_START tune_decode $(date -Is) load=$(cat /proc/loadavg)"
python3 $R/bench_bd.py --bin $BD --model $M --prompt-file $R/prompt_code0.txt --no-chat-template \
  --out $R/results_tune.jsonl --tag tune_decode --configs default,both,ops50mb1000,ops100mb100,ops20mb100 --rounds 3 --warmup 1 --max-tokens 200 "${PRESETS[@]}"
echo "PHASE_START tune_prefill2048 $(date -Is) load=$(cat /proc/loadavg)"
python3 $R/bench_bd.py --bin $BD --model $M --prompt-file $R/prompt_code0.txt --no-chat-template \
  --out $R/results_tune.jsonl --tag tune_prefill2048 --configs default,ops100,mb1000,both400,both,ops50mb1000,ops100mb100,ops20mb100,nograph --rounds 2 --warmup 1 --max-tokens 16 --prompt-tokens 2048 "${PRESETS[@]}"
echo "PHASE_START tune_prefill8192 $(date -Is) load=$(cat /proc/loadavg)"
python3 $R/bench_bd.py --bin $BD --model $M --prompt-file $R/prompt_code0.txt --no-chat-template \
  --out $R/results_tune.jsonl --tag tune_prefill8192 --configs default,both,ops50mb1000 --rounds 1 --warmup 0 --max-tokens 16 --prompt-tokens 8192 "${PRESETS[@]}"
echo "PHASE_END tune $(date -Is) load=$(cat /proc/loadavg)"
echo SWEEP5_DONE
