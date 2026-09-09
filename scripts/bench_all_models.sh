#!/bin/bash
# Continuous batching benchmark for all models in models/ directory
# Usage: ./scripts/bench_all_models.sh [output_file]

set -euo pipefail

MLXCEL="./target/release/mlxcel"
BENCH="./target/release/examples/batch_benchmark"
MODELS_DIR="${MODELS_DIR:-./models}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/store_guard.sh"

PORT=18080
OUTPUT="${1:-benchmark_batching_results.log}"

# Before the first `tee -a "$OUTPUT"` below. A guard placed at the enumeration
# runs after the log has already been created, and $1 here is the log name
# rather than a mode, so a stray argument creates a file named after it.
require_checkpoints "$MODELS_DIR" "continuous-batching sweep"
MAX_TOKENS=50
RUNS=2
CONCURRENCY="1,2,4"
WARMUP=1
LOAD_TIMEOUT=120   # seconds to wait for model to load
BENCH_TIMEOUT=300  # seconds for benchmark to complete

echo "=== Continuous Batching Benchmark ===" | tee "$OUTPUT"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$OUTPUT"
echo "Hardware: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')" | tee -a "$OUTPUT"
echo "Memory: $(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f GB", $1/1073741824}')" | tee -a "$OUTPUT"
echo "OS: $(sw_vers -productName 2>/dev/null) $(sw_vers -productVersion 2>/dev/null)" | tee -a "$OUTPUT"
echo "Max tokens: $MAX_TOKENS, Runs: $RUNS, Concurrency: $CONCURRENCY" | tee -a "$OUTPUT"
echo "========================================" | tee -a "$OUTPUT"
echo "" | tee -a "$OUTPUT"

TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0
FAILED_LIST=""

for MODEL_DIR in "$MODELS_DIR"/*/; do
    MODEL_NAME=$(basename "$MODEL_DIR")
    TOTAL=$((TOTAL + 1))

    echo "--- [$TOTAL] Testing: $MODEL_NAME ---" | tee -a "$OUTPUT"

    # Kill any existing server on the port
    lsof -ti:$PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
    sleep 1

    # Start server
    $MLXCEL serve -m "$MODEL_DIR" --max-batch-size 4 --port $PORT 2>/dev/null &
    SERVER_PID=$!

    # Wait for server to be ready
    READY=false
    for i in $(seq 1 $LOAD_TIMEOUT); do
        if curl -s --max-time 2 "http://localhost:$PORT/health" | grep -q "ok"; then
            READY=true
            break
        fi
        # Check if server process died
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            break
        fi
        sleep 1
    done

    if [ "$READY" = false ]; then
        echo "  SKIP: Failed to load within ${LOAD_TIMEOUT}s" | tee -a "$OUTPUT"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        SKIPPED=$((SKIPPED + 1))
        echo "" | tee -a "$OUTPUT"
        continue
    fi

    # Run benchmark with timeout
    BENCH_OUTPUT=$(timeout $BENCH_TIMEOUT $BENCH \
        --server "http://localhost:$PORT" \
        --concurrent "$CONCURRENCY" \
        --max-tokens $MAX_TOKENS \
        --runs $RUNS \
        --warmup $WARMUP \
        --format table 2>&1) || true

    if echo "$BENCH_OUTPUT" | grep -q "Concurrency"; then
        # Extract result table
        RESULT_TABLE=$(echo "$BENCH_OUTPUT" | sed -n '/^Concurrency/,/^$/p')
        echo "$RESULT_TABLE" | tee -a "$OUTPUT"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  FAIL: Benchmark did not produce results" | tee -a "$OUTPUT"
        echo "  Output: $(echo "$BENCH_OUTPUT" | tail -3)" | tee -a "$OUTPUT"
        FAILED=$((FAILED + 1))
        FAILED_LIST="$FAILED_LIST $MODEL_NAME"
    fi

    # Stop server
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 1

    echo "" | tee -a "$OUTPUT"
done

echo "========================================" | tee -a "$OUTPUT"
echo "Summary: $SUCCESS succeeded, $FAILED failed, $SKIPPED skipped out of $TOTAL models" | tee -a "$OUTPUT"
if [ -n "$FAILED_LIST" ]; then
    echo "Failed models:$FAILED_LIST" | tee -a "$OUTPUT"
fi
echo "Results saved to: $OUTPUT"
