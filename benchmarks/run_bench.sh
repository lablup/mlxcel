#!/bin/bash
# Single-model benchmark runner
# Usage: ./run_bench.sh <model_path> <csv_file>

MODEL_PATH="$1"
CSV_FILE="$2"
MODEL_NAME=$(basename "$MODEL_PATH")
PROMPT="Explain the concept of machine learning in simple terms."
MAX_TOKENS=100
DATE="2026-03-15"
HARDWARE="NVIDIA_GB10_CUDA13.0"
MLX_VERSION="0.31.1"
BUILD_TYPE="release"
BINARY="./target/release/mlxcel"

echo ">>> Benchmarking: $MODEL_NAME"

OUTPUT=$($BINARY generate -m "$MODEL_PATH" -p "$PROMPT" -n $MAX_TOKENS --profile 2>&1)

# Parse results
PROMPT_TOKENS=$(echo "$OUTPUT" | grep "Prompt tokens:" | sed 's/.*: *//')
GEN_TOKENS=$(echo "$OUTPUT" | grep "Generated tokens:" | sed 's/.*: *//')
PREFILL_MS=$(echo "$OUTPUT" | grep "Prefill:" | sed 's/.*: *//' | sed 's/ ms.*//')
PREFILL_TOKS=$(echo "$OUTPUT" | grep "Prefill:" | grep -oP '[\d.]+(?= tok/s)')
DECODE_MS=$(echo "$OUTPUT" | grep "Decode:" | sed 's/.*: *//' | sed 's/ ms.*//')
DECODE_TOKS=$(echo "$OUTPUT" | grep "Decode:" | grep -oP '[\d.]+(?= tok/s)')

if [ -z "$DECODE_TOKS" ]; then
    echo "    FAILED or no output"
    echo "$MODEL_NAME,$MODEL_PATH,$PROMPT_TOKENS,$GEN_TOKENS,,,,,${DATE},${HARDWARE},${MLX_VERSION},${BUILD_TYPE},${MAX_TOKENS},FAILED" >> "$CSV_FILE"
else
    echo "    Prefill: ${PREFILL_TOKS} tok/s | Decode: ${DECODE_TOKS} tok/s"
    echo "$MODEL_NAME,$MODEL_PATH,$PROMPT_TOKENS,$GEN_TOKENS,$PREFILL_MS,$PREFILL_TOKS,$DECODE_MS,$DECODE_TOKS,${DATE},${HARDWARE},${MLX_VERSION},${BUILD_TYPE},${MAX_TOKENS},\"$PROMPT\"" >> "$CSV_FILE"
fi
