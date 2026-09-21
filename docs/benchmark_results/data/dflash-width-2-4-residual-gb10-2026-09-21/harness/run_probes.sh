#!/usr/bin/env bash
# The in-process arms, run against an already-built test binary by path so a
# concurrent rebuild of the server cannot reshape them mid-run.
set -uo pipefail
SP=/tmp/claude-1000/-home-inureyes-Development-mlxcel/a7ac83cc-0ca7-4ae0-8f19-a24a07471f2a/scratchpad
WT=/home/inureyes/Development/mlxcel-wt-1935
TB=${TEST_BIN:-/home/inureyes/Development/mlxcel/target/release/deps/mlxcel-69c784c732fe3546}
cd "$WT" || exit 1
export MLX_CUDA_ARCHITECTURES=121 MLX_ENABLE_TF32=1
export MLXCEL_Q35_PROBE_PROMPT="$(python3 -c 'import json;print(",".join(map(str,json.load(open("'"$SP"'/prompt_ids.json")))))')"
export MLXCEL_Q35_PROBE_REFERENCE="$(python3 -c 'import json;print(",".join(map(str,json.load(open("'"$SP"'/classic_ids.json")))))')"

echo "=== ARM 3: speculative against classic at ONE ROW, logit BYTES ==="
"$TB" --ignored --test-threads=1 --nocapture speculative_t1_forward_versus_classic_on_the_real_transcript 2>&1 | grep -E "^\[1935\]|test result|panicked"

echo "=== ARM 1: replay the served width-4 round structure, no drafter ==="
MLXCEL_Q35_PROBE_BLOCK=4 MLXCEL_Q35_PROBE_ACCEPTS="$(cat "$SP/accepts_w4.txt")" MLXCEL_Q35_PROBE_WRONG=9999 \
  "$TB" --ignored --test-threads=1 --nocapture round_loop_cache_dynamics_with_rollback_match_the_chain 2>&1 | grep -E "^\[1935\]|test result|panicked"

echo "=== ARM 2: the served burst wrapper with the real drafter ==="
MLXCEL_Q35_PROBE_WIDTHS=4 \
  "$TB" --ignored --test-threads=1 --nocapture served_burst_wrapper_with_real_drafter_matches_the_chain 2>&1 | grep -E "^\[1935\]|test result|panicked"
echo "ALL PROBES DONE"
