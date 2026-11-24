#!/bin/bash
set -euo pipefail
# Run from the repository root. BIN_DIR must contain the three retained binaries.
: "${BIN_DIR:?Set BIN_DIR to the directory containing server-main, server-branch and server-budget}"
BENCH_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
RESULT_DIR=${RESULT_DIR:-/tmp/issue-3385-large-model-rerun}
mkdir -p "$RESULT_DIR"
for variant in branch budget; do
  BENCH_CELLS=8192x4 gpu-lock run --tag issue-3385-large-model -- python3 "$BENCH_DIR/bench_decode.py" --main "$BIN_DIR/server-main" --branch "$BIN_DIR/server-$variant" --models models/qwen3-8b-4bit --rounds 3 --reps 2 --out "$RESULT_DIR/qwen3-8b-$variant.json"
done
