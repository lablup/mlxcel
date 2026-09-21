#!/usr/bin/env bash
set -uo pipefail
SP=/tmp/claude-1000/-home-inureyes-Development-mlxcel/a7ac83cc-0ca7-4ae0-8f19-a24a07471f2a/scratchpad
WT=/home/inureyes/Development/mlxcel-wt-1935
TB=${TEST_BIN:-/home/inureyes/Development/mlxcel/target/release/deps/mlxcel-e4caea9f4277a3b5}
cd "$WT" || exit 1
export MLX_CUDA_ARCHITECTURES=121 MLX_ENABLE_TF32=1
export MLXCEL_Q35_PROBE_PROMPT="$(python3 -c 'import json;print(",".join(map(str,json.load(open("'"$SP"'/prompt_ids.json")))))')"
export MLXCEL_Q35_PROBE_REFERENCE="$(python3 -c 'import json;print(",".join(map(str,json.load(open("'"$SP"'/classic_ids.json")))))')"
echo "test binary: $TB"

echo "=== BISECT: where do logit bytes first differ, and in which layer ==="
MLXCEL_Q35_PROBE_BLOCK=4 MLXCEL_Q35_PROBE_ACCEPTS="$(cat "$SP/accepts_w4.txt")" MLXCEL_Q35_PROBE_WRONG=9999 \
  "$TB" --ignored --test-threads=1 --nocapture block_versus_chain_byte_bisect_on_the_real_transcript 2>&1 | grep -E "^\[1935\]|panicked"

echo "=== PROBE VERDICTS against prompt length ==="
for L in 8 32 64 128 256 512; do
  echo "--- MLXCEL_MTP_PROBE_PROMPT_LEN=$L ---"
  S=$(date +%s.%N)
  MLXCEL_MTP_PROBE_PROMPT_LEN=$L "$TB" --ignored --test-threads=1 --nocapture \
    block_chain_exactness_verdicts_on_the_real_checkpoint 2>&1 | grep -E "^\[1935\] width"
  E=$(date +%s.%N)
  echo "    (arm wall clock including model load: $(python3 -c "print(f'{$E-$S:.1f}s')"))"
done
echo "PROBES2 DONE"
