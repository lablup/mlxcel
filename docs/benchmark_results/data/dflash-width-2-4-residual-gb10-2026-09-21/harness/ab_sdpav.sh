#!/usr/bin/env bash
# A/B on MLXCEL_SDPA_VECTOR_LARGE_D. Qwen 3.5's head_dim is 256, so this gate
# decides whether a single-query attention call takes the fused sdpa_vector
# kernel or the materializing fallback. Classic decode and the burst's
# attend_per_position are both single-query calls, so if the residual is the
# two of them landing on different kernels, turning the gate off puts both on
# the fallback and they converge. If they still differ, the kernel choice is
# not the mechanism.
set -uo pipefail
SP=/tmp/claude-1000/-home-inureyes-Development-mlxcel/a7ac83cc-0ca7-4ae0-8f19-a24a07471f2a/scratchpad
WT=/home/inureyes/Development/mlxcel-wt-1935
BIN=/home/inureyes/Development/mlxcel/target/release/mlxcel-server
H=$WT/docs/benchmark_results/data/draft-block-width-gb10-2026-09-20/harness
for arm in "classic:cls-novec" "4:w4-novec"; do
  W=${arm%%:*}; T=${arm##*:}
  python3 "$SP/transcript.py" --server "$BIN" \
    --target "$WT/models/mlx/qwen3.5-4b-4bit" \
    --drafter "$WT/models/mlx/qwen3.5-4b-dflash" \
    --width "$W" --n 1 --max-tokens 200 --tag "$T" \
    --extra-env "MLXCEL_SDPA_VECTOR_LARGE_D=0" \
    --prompt-file "$H/prompt_retry.txt" --outdir "$SP/arms" --port 18936
  echo "=== $T done ==="
done
echo "AB DONE"
