#!/usr/bin/env bash
set -uo pipefail
# Paths are derived, not hardcoded: SP is this harness directory, WT the repo
# it lives in, and BIN/TEST_BIN are overridable so a run can point at a binary
# built anywhere.
SP=${SP:-$(cd "$(dirname "$0")" && pwd)}
WT=${WT:-$(git -C "$SP" rev-parse --show-toplevel)}
OUT=${OUT:-$SP/../arms}
BIN=${BIN:-$WT/target/release/mlxcel-server}
TB=${TEST_BIN:?set TEST_BIN to the cargo lib-test binary}
cd "$WT" || exit 1
export MLX_CUDA_ARCHITECTURES=121 MLX_ENABLE_TF32=1
export MLXCEL_Q35_PROBE_PROMPT="$(python3 -c 'import json;print(",".join(map(str,json.load(open("'"$SP"'/../prompt_ids.json")))))')"
export MLXCEL_Q35_PROBE_REFERENCE="$(python3 -c 'import json;print(",".join(map(str,json.load(open("'"$SP"'/../classic_ids.json")))))')"
echo "test binary: $TB"

echo "=== BISECT: where do logit bytes first differ, and in which layer ==="
MLXCEL_Q35_PROBE_BLOCK=4 MLXCEL_Q35_PROBE_ACCEPTS="$(cat "$OUT/accepts_w4.txt")" MLXCEL_Q35_PROBE_WRONG=9999 \
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
