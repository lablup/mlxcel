#!/usr/bin/env bash
# Sixth #1798 chain: the allowlisted families' remaining costs (qwen3.6 same model_type, long-prefill memory on the Qwen MoEs).
set -uo pipefail
S=/tmp/claude-1000/-home-inureyes-Development-mlxcel/a7ac83cc-0ca7-4ae0-8f19-a24a07471f2a/scratchpad
R=$S/1798
WT=/home/inureyes/Development/wt-1798-graph-budget
BD=$WT/target/release/mlxcel-bench-decode
PRESETS=(--preset both=MLX_MAX_OPS_PER_BUFFER=100,MLX_MAX_MB_PER_BUFFER=1000 --preset nograph=MLX_USE_CUDA_GRAPHS=0)
until grep -q SWEEP5_DONE $R/sweep5.out; do sleep 30; done
$S/gpu_lock.sh acquire 1798 || exit 1
trap '$S/gpu_lock.sh release 1798' EXIT
$R/idle_gate.sh
echo "PHASE_START moe3 $(date -Is) load=$(cat /proc/loadavg)"
python3 $R/bench_bd.py --bin $BD --model $WT/models/mlx/qwen3.6-35b-a3b-4bit --prompt-file $R/prompt_code0.txt --no-chat-template \
  --out $R/results_moe2.jsonl --tag single --configs default,both --rounds 3 --warmup 1 --max-tokens 200 "${PRESETS[@]}"
echo "PHASE_START moe_prefill2048 $(date -Is) load=$(cat /proc/loadavg)"
for m in qwen3-30b-a3b-4bit qwen3.5-35b-a3b-4bit; do
  python3 $R/bench_bd.py --bin $BD --model $WT/models/mlx/$m --prompt-file $R/prompt_code0.txt --no-chat-template \
    --out $R/results_prefill.jsonl --tag prefill2048 --configs default,both --rounds 2 --warmup 1 --max-tokens 16 --prompt-tokens 2048 "${PRESETS[@]}"
done
echo "PHASE_END moe3 $(date -Is) load=$(cat /proc/loadavg)"
echo SWEEP6_DONE
