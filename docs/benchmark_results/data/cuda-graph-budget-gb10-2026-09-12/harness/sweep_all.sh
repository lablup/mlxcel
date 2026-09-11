#!/usr/bin/env bash
# The whole #1798 measurement chain under the GPU lock. Usage: sweep_all.sh PHASE...
# Phases: single | batched | prefill | nsys | dflash
set -uo pipefail
S=/tmp/claude-1000/-home-inureyes-Development-mlxcel/a7ac83cc-0ca7-4ae0-8f19-a24a07471f2a/scratchpad
R=$S/1798
WT=/home/inureyes/Development/wt-1798-graph-budget
BD=$WT/target/release/mlxcel-bench-decode
CLI=$WT/target/release/mlxcel
SRV=$WT/target/release/mlxcel-server
PRESETS=(--preset nograph=MLX_USE_CUDA_GRAPHS=0 --preset ops100=MLX_MAX_OPS_PER_BUFFER=100 \
         --preset mb1000=MLX_MAX_MB_PER_BUFFER=1000 --preset both=MLX_MAX_OPS_PER_BUFFER=100,MLX_MAX_MB_PER_BUFFER=1000 \
         --preset both400=MLX_MAX_OPS_PER_BUFFER=100,MLX_MAX_MB_PER_BUFFER=400)
ARMS6=default,nograph,ops100,mb1000,both,both400
ARMS3=default,both,nograph
$S/gpu_lock.sh acquire 1798 || exit 1
trap '$S/gpu_lock.sh release 1798' EXIT
for phase in "$@"; do
  $R/idle_gate.sh
  echo "PHASE_START $phase $(date -Is) load=$(cat /proc/loadavg)"
  case $phase in
    single)
      for m in laguna-xs-2.1-nvfp4 qwen3.5-4b-4bit llama-3.1-8b-4bit; do
        python3 $R/bench_bd.py --bin $BD --model $WT/models/mlx/$m --prompt-file $R/prompt_code0.txt --no-chat-template \
          --out $R/results_single.jsonl --tag single --configs $ARMS6 --rounds 3 --warmup 1 --max-tokens 200 "${PRESETS[@]}"
      done ;;
    batched)
      for m in laguna-xs-2.1-nvfp4 qwen3.5-4b-4bit; do
        python3 $R/bench_srv.py --server $SRV --client $WT/scripts/bench_serving_concurrency.py --model $WT/models/mlx/$m \
          --out $R/results_batched.jsonl --tag batched --configs $ARMS3 --rounds 3 --concurrency 1,4,8 --max-tokens 200 --prompt-tokens 128 --max-batch 8 "${PRESETS[@]}"
      done ;;
    prefill)
      for m in laguna-xs-2.1-nvfp4 qwen3.5-4b-4bit; do
        python3 $R/bench_bd.py --bin $BD --model $WT/models/mlx/$m --prompt-file $R/prompt_code0.txt --no-chat-template \
          --out $R/results_prefill.jsonl --tag prefill2048 --configs $ARMS3 --rounds 3 --warmup 1 --max-tokens 16 --prompt-tokens 2048 "${PRESETS[@]}"
      done ;;
    nsys)
      for m in laguna-xs-2.1-nvfp4 qwen3.5-4b-4bit; do
        for cfg in default both; do
          $R/nsys_bd.sh "${m%%-*}-$cfg" $cfg $m "${PRESETS[@]}"
        done
      done ;;
    dflash)
      python3 $S/1799/bench_cli.py --bin $CLI --target $WT/models/mlx/laguna-xs-2.1-nvfp4 --draft $WT/models/mlx/laguna-xs-2.1-dflash \
        --prompt-file $R/prompt_code0.txt --out $R/results_dflash.jsonl --tag dflash --configs off,b4,both-off,both-b4,nograph-off,nograph-b4 \
        --rounds 3 --warmup 1 --preset both=MLX_MAX_OPS_PER_BUFFER=100,MLX_MAX_MB_PER_BUFFER=1000 --preset nograph=MLX_USE_CUDA_GRAPHS=0 ;;
  esac
  echo "PHASE_END $phase $(date -Is) load=$(cat /proc/loadavg)"
done
echo SWEEP_ALL_DONE
