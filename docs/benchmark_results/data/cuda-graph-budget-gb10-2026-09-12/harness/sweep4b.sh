#!/usr/bin/env bash
# Fourth #1798 chain (b): batched serving on Laguna and the qwen3_moe family.
set -uo pipefail
S=/tmp/claude-1000/-home-inureyes-Development-mlxcel/a7ac83cc-0ca7-4ae0-8f19-a24a07471f2a/scratchpad
R=$S/1798
WT=/home/inureyes/Development/wt-1798-graph-budget
SRV=$WT/target/release/mlxcel-server
PRESETS=(--preset nograph=MLX_USE_CUDA_GRAPHS=0 --preset both=MLX_MAX_OPS_PER_BUFFER=100,MLX_MAX_MB_PER_BUFFER=1000)
until grep -q SWEEP3_DONE $R/sweep3.out; do sleep 30; done
$S/gpu_lock.sh acquire 1798 || exit 1
trap '$S/gpu_lock.sh release 1798' EXIT
$R/idle_gate.sh
echo "PHASE_START batched $(date -Is) load=$(cat /proc/loadavg)"
for m in laguna-xs-2.1-nvfp4 qwen3-30b-a3b-4bit; do
  python3 $R/bench_srv.py --server $SRV --client $WT/scripts/bench_serving_concurrency.py --model $WT/models/mlx/$m \
    --out $R/results_batched.jsonl --tag batched --configs default,both,nograph --rounds 3 --concurrency 1,4,8 --max-tokens 200 --prompt-tokens 128 --max-batch 8 "${PRESETS[@]}"
done
echo "PHASE_END batched $(date -Is) load=$(cat /proc/loadavg)"
echo SWEEP4_DONE
