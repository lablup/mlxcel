#!/bin/bash
# Single-model benchmark runner
# Usage: ./run_bench.sh <model_path> <csv_file>
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <model_path> <csv_file>" >&2
    exit 1
fi

MODEL_PATH="$1"
CSV_FILE="$2"
MODEL_NAME=$(basename "$MODEL_PATH")
PROMPT="Explain the concept of machine learning in simple terms."
MAX_TOKENS=100
DATE=$(date +%F)
HARDWARE="${HARDWARE:-NVIDIA_GB10_CUDA13.0}"
MLX_VERSION="${MLX_VERSION:-0.31.1}"
BUILD_TYPE="release"
BINARY="./target/release/mlxcel"

echo ">>> Benchmarking: $MODEL_NAME"

# A failed run is a row too -- let it reach the FAILED branch instead of
# aborting under `set -e`.
OUTPUT=$($BINARY generate -m "$MODEL_PATH" -p "$PROMPT" -n $MAX_TOKENS --profile 2>&1) || true

# Parse results. `sed -n ...p` rather than grep: a run that produced no stats
# must still reach the FAILED row below, and grep's no-match exit 1 would
# abort it under `set -e`.
field() { printf '%s\n' "$OUTPUT" | sed -n -E "s/^[[:space:]]*$1:[[:space:]]*([0-9.]+).*/\1/p"; }
tok_per_sec() { printf '%s\n' "$OUTPUT" | sed -n -E "s/^[[:space:]]*$1:.*\(([0-9.]+) tok\/s\).*/\1/p"; }

PROMPT_TOKENS=$(field "Prompt tokens")
GEN_TOKENS=$(field "Generated tokens")
PREFILL_MS=$(field "Prefill")
PREFILL_TOKS=$(tok_per_sec "Prefill")
DECODE_MS=$(field "Decode")
DECODE_TOKS=$(tok_per_sec "Decode")

if [ -z "$DECODE_TOKS" ]; then
    echo "    FAILED or no output"
    echo "$MODEL_NAME,$MODEL_PATH,$PROMPT_TOKENS,$GEN_TOKENS,,,,,${DATE},${HARDWARE},${MLX_VERSION},${BUILD_TYPE},${MAX_TOKENS},FAILED" >> "$CSV_FILE"
else
    echo "    Prefill: ${PREFILL_TOKS} tok/s | Decode: ${DECODE_TOKS} tok/s"
    echo "$MODEL_NAME,$MODEL_PATH,$PROMPT_TOKENS,$GEN_TOKENS,$PREFILL_MS,$PREFILL_TOKS,$DECODE_MS,$DECODE_TOKS,${DATE},${HARDWARE},${MLX_VERSION},${BUILD_TYPE},${MAX_TOKENS},\"$PROMPT\"" >> "$CSV_FILE"
fi
