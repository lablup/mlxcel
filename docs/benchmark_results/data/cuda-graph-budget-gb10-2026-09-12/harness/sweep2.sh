#!/usr/bin/env bash
# Second #1798 chain: more MoE shapes (single-stream, 3 arms) and an nsys pass on the Llama regression.
set -uo pipefail
S=/tmp/claude-1000/-home-inureyes-Development-mlxcel/a7ac83cc-0ca7-4ae0-8f19-a24a07471f2a/scratchpad
R=$S/1798
WT=/home/inureyes/Development/wt-1798-graph-budget
BD=$WT/target/release/mlxcel-bench-decode
PRESETS=(--preset nograph=MLX_USE_CUDA_GRAPHS=0 --preset ops100=MLX_MAX_OPS_PER_BUFFER=100 \
         --preset mb1000=MLX_MAX_MB_PER_BUFFER=1000 --preset both=MLX_MAX_OPS_PER_BUFFER=100,MLX_MAX_MB_PER_BUFFER=1000)
until grep -q SWEEP_ALL_DONE $R/sweep_all.out; do sleep 30; done
$S/gpu_lock.sh acquire 1798 || exit 1
trap '$S/gpu_lock.sh release 1798' EXIT
for phase in "$@"; do
  $R/idle_gate.sh
  echo "PHASE_START $phase $(date -Is) load=$(cat /proc/loadavg)"
  case $phase in
    moe2)
      for m in gpt-oss-20b-mxfp4 qwen3-30b-a3b-4bit qwen3.5-35b-a3b-4bit gemma-4-26b-a4b-it-4bit; do
        python3 $R/bench_bd.py --bin $BD --model $WT/models/mlx/$m --prompt-file $R/prompt_code0.txt --no-chat-template \
          --out $R/results_moe2.jsonl --tag single --configs default,both,nograph --rounds 3 --warmup 1 --max-tokens 200 "${PRESETS[@]}"
      done ;;
    nsys_llama)
      for cfg in default both; do
        $R/nsys_bd.sh "llama-$cfg" $cfg llama-3.1-8b-4bit "${PRESETS[@]}"
      done ;;
  esac
  echo "PHASE_END $phase $(date -Is) load=$(cat /proc/loadavg)"
done
echo SWEEP2_DONE
