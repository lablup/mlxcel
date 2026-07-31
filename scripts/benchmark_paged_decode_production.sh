#!/usr/bin/env bash
#
# Server-level before/after benchmark for the production paged decode v2 path
# (issue #899).
#
# Runs the issue's mandatory matrix twice against the same binary and the same
# model, changing only which decode path serves the batch:
#
#   after  = the fused v2 kernel (the new default)
#   before = gather-then-SDPA, pinned with MLXCEL_PAGED_ATTENTION_NATIVE=0
#            (the kill switch, which restores the pre-#899 behaviour end to end)
#
# Matrix:
#   * 4 concurrent clients at 1K / 4K / 16K prompt tokens (the documented v0.4
#     --parallel default), reporting aggregate decode throughput and TTFT.
#   * single-sequence long-context decode at 16K and 32K.
#
# The two arms are run serially against a freshly started server each time, so
# no measurement overlaps another. Results land as CSV plus a markdown skeleton
# ready to become docs/benchmark_results/paged-decode-production-<hw>-<date>.md.
#
# Usage:
#   MODEL=models/qwen2.5-7b-instruct-4bit ./scripts/benchmark_paged_decode_production.sh
#
# Environment:
#   MODEL          model directory (required)
#   SERVER_BIN     default target/release/mlxcel-server
#   PARALLEL       concurrent slots, default 4
#   MAX_TOKENS     decode tokens per request, default 128
#   CTX_SIZE       total --ctx-size; default is PARALLEL * 32768 so the per-slot
#                  window covers the 32K single-sequence case and the paged slab
#                  is sized for it. Undersizing this is the single most likely
#                  way to accidentally measure the gather path in both arms.
#   PORT           default 18991
#   OUT_DIR        default benchmarks/

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-}"
SERVER_BIN="${SERVER_BIN:-$ROOT_DIR/target/release/mlxcel-server}"
PARALLEL="${PARALLEL:-4}"
MAX_TOKENS="${MAX_TOKENS:-128}"
CTX_SIZE="${CTX_SIZE:-$((PARALLEL * 32768))}"
PORT="${PORT:-18991}"
HOST="${HOST:-127.0.0.1}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/benchmarks}"
DATE_TAG="${DATE_TAG:-$(date +%F)}"
HW_TAG="${HW_TAG:-$(uname -m)}"

if [[ -z "$MODEL" ]]; then
  echo "MODEL is required, e.g. MODEL=models/qwen2.5-7b-instruct-4bit $0" >&2
  exit 2
fi
if [[ ! -x "$SERVER_BIN" ]]; then
  echo "server binary not found at $SERVER_BIN; build with:" >&2
  echo "  cargo build --release --features metal,accelerate" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/paged_decode_production_${HW_TAG}_${DATE_TAG}.csv"
LOG_DIR="$OUT_DIR/paged_decode_production_logs_${DATE_TAG}"
mkdir -p "$LOG_DIR"
echo "arm,scenario,concurrency,prompt_tokens,max_tokens,raw_log" > "$CSV"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
    SERVER_PID=""
  fi
}
trap cleanup EXIT

wait_for_health() {
  for _ in $(seq 1 300); do
    if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "server did not become healthy on port $PORT" >&2
  return 1
}

start_server() {
  local arm="$1"
  local log="$LOG_DIR/server_${arm}.log"
  echo "--- starting server (arm: $arm), log $log"
  if [[ "$arm" == "before" ]]; then
    MLXCEL_PAGED_ATTENTION_NATIVE=0 "$SERVER_BIN" \
      --model "$MODEL" --host "$HOST" --port "$PORT" \
      --parallel "$PARALLEL" --ctx-size "$CTX_SIZE" \
      >"$log" 2>&1 &
  else
    "$SERVER_BIN" \
      --model "$MODEL" --host "$HOST" --port "$PORT" \
      --parallel "$PARALLEL" --ctx-size "$CTX_SIZE" \
      >"$log" 2>&1 &
  fi
  SERVER_PID=$!
  wait_for_health
  # Surface the resolved slab size: if this line is missing, the fused path is
  # not eligible and the "after" arm will silently measure gather.
  grep -m1 "Paged KV slab size" "$log" || \
    echo "NOTE: no slab-size line in the server log (see MLXCEL_PAGED_SLAB_BLOCKS)"
}

run_case() {
  local arm="$1" scenario="$2" concurrency="$3" prompt_tokens="$4"
  local log="$LOG_DIR/${arm}_${scenario}.txt"
  echo "--- $arm / $scenario: concurrency=$concurrency prompt_tokens=$prompt_tokens"
  python3 "$ROOT_DIR/scripts/bench_serving_concurrency.py" \
    --host "$HOST" --port "$PORT" \
    --concurrency "$concurrency" \
    --prompt-tokens "$prompt_tokens" \
    --max-tokens "$MAX_TOKENS" \
    | tee "$log"
  echo "$arm,$scenario,$concurrency,$prompt_tokens,$MAX_TOKENS,$log" >> "$CSV"
}

run_arm() {
  local arm="$1"
  start_server "$arm"
  # 4 concurrent clients, the v0.4 default scenario.
  run_case "$arm" "batch${PARALLEL}_ctx1k"  "$PARALLEL" 1024
  run_case "$arm" "batch${PARALLEL}_ctx4k"  "$PARALLEL" 4096
  run_case "$arm" "batch${PARALLEL}_ctx16k" "$PARALLEL" 16384
  # Single-sequence long context.
  run_case "$arm" "batch1_ctx16k" 1 16384
  run_case "$arm" "batch1_ctx32k" 1 32768
  cleanup
  # Let the allocator settle before the next arm so the two see the same
  # starting conditions.
  sleep 5
}

echo "model      : $MODEL"
echo "server     : $SERVER_BIN"
echo "parallel   : $PARALLEL"
echo "ctx-size   : $CTX_SIZE (per slot $((CTX_SIZE / PARALLEL)))"
echo "max-tokens : $MAX_TOKENS"
echo "csv        : $CSV"
echo

run_arm "before"
run_arm "after"

MD="$OUT_DIR/paged-decode-production-${HW_TAG}-${DATE_TAG}.md"
{
  echo "# Production paged decode v2: ${HW_TAG}, ${DATE_TAG}"
  echo
  echo "Issue #899. Before = gather-then-SDPA (\`MLXCEL_PAGED_ATTENTION_NATIVE=0\`),"
  echo "after = the fused v2 kernel (the new default). Same binary, same model,"
  echo "same host, arms run serially."
  echo
  echo "| Field | Value |"
  echo "|---|---|"
  echo "| Model | \`$MODEL\` |"
  echo "| Parallel slots | $PARALLEL |"
  echo "| \`--ctx-size\` | $CTX_SIZE (per slot $((CTX_SIZE / PARALLEL))) |"
  echo "| Decode tokens per request | $MAX_TOKENS |"
  echo "| Raw logs | \`$LOG_DIR\` |"
  echo
  echo "## Results"
  echo
  echo "Fill from the per-case logs listed in \`$CSV\`."
  echo
  echo "| scenario | before aggregate tok/s | after aggregate tok/s | ratio | before TTFT p50 | after TTFT p50 |"
  echo "|---|---|---|---|---|---|"
  for scenario in "batch${PARALLEL}_ctx1k" "batch${PARALLEL}_ctx4k" "batch${PARALLEL}_ctx16k" batch1_ctx16k batch1_ctx32k; do
    echo "| $scenario | | | | | |"
  done
} > "$MD"

echo
echo "CSV      : $CSV"
echo "Markdown : $MD (skeleton; fill from the logs)"
echo "Logs     : $LOG_DIR"
