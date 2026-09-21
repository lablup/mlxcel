#!/usr/bin/env bash
# A/B the per-row query-layout fix on the served path, against the classic null
# arm from the same binary. `0` is the kill switch that restores the pre-fix
# slicing, so the same binary produces both sides.
set -uo pipefail
SP=/tmp/claude-1000/-home-inureyes-Development-mlxcel/a7ac83cc-0ca7-4ae0-8f19-a24a07471f2a/scratchpad
WT=/home/inureyes/Development/mlxcel-wt-1935
BIN=/home/inureyes/Development/mlxcel/target/release/mlxcel-server
H=$WT/docs/benchmark_results/data/draft-block-width-gb10-2026-09-20/harness
run() {  # width tag extra-env
  python3 "$SP/transcript.py" --server "$BIN" \
    --target "$WT/models/mlx/qwen3.5-4b-4bit" \
    --drafter "$WT/models/mlx/qwen3.5-4b-dflash" \
    --width "$1" --n 1 --max-tokens 200 --tag "$2" --extra-env "$3" \
    --prompt-file "$H/prompt_retry.txt" --outdir "$SP/arms" --port 18937
  echo "=== $2 done ==="
}
run classic fix-classic      "MLXCEL_QWEN35_ATTEND_CONTIGUOUS=q"
run 4       fix-w4-q         "MLXCEL_QWEN35_ATTEND_CONTIGUOUS=q"
run 2       fix-w2-q         "MLXCEL_QWEN35_ATTEND_CONTIGUOUS=q"
run 4       fix-w4-off       "MLXCEL_QWEN35_ATTEND_CONTIGUOUS=0"
# If the query copy alone does not close it, the key and value slices are not
# the matching pair this assumed; `all` and `kv` separate those in the same
# session rather than costing another build.
run 4       fix-w4-all       "MLXCEL_QWEN35_ATTEND_CONTIGUOUS=all"
run classic fix-classic-all  "MLXCEL_QWEN35_ATTEND_CONTIGUOUS=all"
echo "FIX AB DONE"
