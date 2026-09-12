#!/usr/bin/env bash
# Third #1798 chain: the DFlash phase-split arm re-run with the per-run busy gate.
set -uo pipefail
S=/tmp/claude-1000/-home-inureyes-Development-mlxcel/a7ac83cc-0ca7-4ae0-8f19-a24a07471f2a/scratchpad
R=$S/1798
WT=/home/inureyes/Development/wt-1798-graph-budget
CLI=$WT/target/release/mlxcel
until grep -q SWEEP2_DONE $R/sweep2.out; do sleep 30; done
$S/gpu_lock.sh acquire 1798 || exit 1
trap '$S/gpu_lock.sh release 1798' EXIT
$R/idle_gate.sh
echo "PHASE_START dflash2 $(date -Is) load=$(cat /proc/loadavg)"
python3 $R/bench_cli.py --bin $CLI --target $WT/models/mlx/laguna-xs-2.1-nvfp4 --draft $WT/models/mlx/laguna-xs-2.1-dflash \
  --prompt-file $R/prompt_code0.txt --out $R/results_dflash2.jsonl --tag dflash2 --configs off,b4,both-off,both-b4,nograph-off,nograph-b4 \
  --rounds 3 --warmup 1 --preset both=MLX_MAX_OPS_PER_BUFFER=100,MLX_MAX_MB_PER_BUFFER=1000 --preset nograph=MLX_USE_CUDA_GRAPHS=0
echo "PHASE_END dflash2 $(date -Is) load=$(cat /proc/loadavg)"
echo SWEEP3_DONE
