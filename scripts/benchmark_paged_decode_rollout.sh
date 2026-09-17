#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRIMARY_WORKTREE="$(git -C "$ROOT_DIR" worktree list --porcelain | sed -n 's/^worktree //p' | head -n1)"
MODEL_ROOT="${MODEL_ROOT:-$ROOT_DIR/models}"
if [[ ! -d "$MODEL_ROOT" && -n "$PRIMARY_WORKTREE" && -d "$PRIMARY_WORKTREE/models" ]]; then
  MODEL_ROOT="$PRIMARY_WORKTREE/models"
fi
DATE_TAG="${DATE_TAG:-$(date +%F)}"
OUTPUT_CSV="${OUTPUT_CSV:-$ROOT_DIR/benchmarks/paged_decode_rollout_matrix_${DATE_TAG}.csv}"
SERVER_BIN="${SERVER_BIN:-$ROOT_DIR/target/release/mlxcel-server}"
KERNEL_BENCH_BIN="${KERNEL_BENCH_BIN:-$ROOT_DIR/target/release/examples/profile_paged_decode_kernel}"

MODEL_LLAMA="${MODEL_LLAMA:-$MODEL_ROOT/llama-3.2-1b-instruct-4bit}"
MODEL_QWEN3="${MODEL_QWEN3:-$MODEL_ROOT/qwen3-0.6b-4bit}"
MODEL_QWEN35="${MODEL_QWEN35:-$MODEL_ROOT/qwen3.5-0.8b-4bit}"
MODEL_GEMMA3="${MODEL_GEMMA3:-$MODEL_ROOT/gemma-3-1b-it-4bit}"
MODEL_LLAMA4="${MODEL_LLAMA4:-$MODEL_ROOT/llama-4-scout-17b-16e-instruct-4bit}"
MODEL_EXAONE4="${MODEL_EXAONE4:-$MODEL_ROOT/exaone-4.0-1.2b-4bit}"

BATCH_SIZE="${BATCH_SIZE:-2}"
PROMPT_LEN="${PROMPT_LEN:-128}"
WARMUP_STEPS="${WARMUP_STEPS:-6}"
DECODE_STEPS="${DECODE_STEPS:-24}"
RUNS="${RUNS:-2}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT_AUTO="${SERVER_PORT_AUTO:-18125}"
SERVER_PORT_FORCED="${SERVER_PORT_FORCED:-18126}"
REQUEST_MAX_TOKENS="${REQUEST_MAX_TOKENS:-16}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

wait_for_health() {
  local url="$1"
  for _ in $(seq 1 120); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "timed out waiting for $url" >&2
  return 1
}

extract_json_number() {
  local key="$1"
  sed -n "s/.*\"${key}\":\\([0-9.]*\\).*/\\1/p"
}

run_kernel_profile_row() {
  local family="$1"
  local model_path="$2"
  local output
  output="$("$KERNEL_BENCH_BIN" \
    -m "$model_path" \
    --batch-size "$BATCH_SIZE" \
    --prompt-len "$PROMPT_LEN" \
    --warmup "$WARMUP_STEPS" \
    --decode-steps "$DECODE_STEPS" \
    --runs "$RUNS" \
    --block-size "$BLOCK_SIZE")"

  local fallback_tps native_tps speedup
  fallback_tps="$(printf '%s\n' "$output" | sed -n 's/^fallback_tok_per_sec=//p')"
  native_tps="$(printf '%s\n' "$output" | sed -n 's/^native_tok_per_sec=//p')"
  speedup="$(printf '%s\n' "$output" | sed -n 's/^speedup=//p')"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$DATE_TAG" \
    "Apple M1 Ultra 128GB" \
    "model_kernel" \
    "$family" \
    "$(basename "$model_path")" \
    "paged" \
    "$BATCH_SIZE" \
    "$PROMPT_LEN" \
    "$DECODE_STEPS" \
    "$BLOCK_SIZE" \
    "$fallback_tps" \
    "$native_tps" \
    "$speedup" \
    "" \
    "" \
    "" \
    "release example profile_paged_decode_kernel"
}

run_server_policy_row() {
  local family="$1"
  local model_path="$2"
  local decode_storage="$3"
  local port="$4"
  local note="$5"
  local health_url="http://${SERVER_HOST}:${port}/health"
  local request_url="http://${SERVER_HOST}:${port}/v1/completions"

  cleanup
  if [[ "$decode_storage" == "paged" ]]; then
    MLXCEL_SERVER_DECODE_STORAGE=paged RUST_LOG=info \
      "$SERVER_BIN" -m "$model_path" --host "$SERVER_HOST" --port "$port" \
      --parallel 2 --max-batch-size 2 --metrics --verbose \
      >/tmp/mlxcel-paged-benchmark-${port}.log 2>&1 &
  else
    RUST_LOG=info \
      "$SERVER_BIN" -m "$model_path" --host "$SERVER_HOST" --port "$port" \
      --parallel 2 --max-batch-size 2 --metrics --verbose \
      >/tmp/mlxcel-paged-benchmark-${port}.log 2>&1 &
  fi
  SERVER_PID=$!

  wait_for_health "$health_url"

  local health_before health_after block_size fallback_count request_time
  health_before="$(curl -fsS "$health_url")"
  request_time="$(
    curl -fsS -o /dev/null -w '%{time_total}' "$request_url" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$(basename "$model_path")\",\"prompt\":\"Hello\",\"max_tokens\":${REQUEST_MAX_TOKENS},\"temperature\":0}"
  )"
  health_after="$(curl -fsS "$health_url")"

  block_size="$(printf '%s\n' "$health_after" | extract_json_number cache_pool_paged_block_size)"
  fallback_count="$(printf '%s\n' "$health_after" | extract_json_number decode_storage_fallbacks)"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$DATE_TAG" \
    "Apple M1 Ultra 128GB" \
    "server_policy" \
    "$family" \
    "$(basename "$model_path")" \
    "$decode_storage" \
    "" \
    "" \
    "" \
    "" \
    "" \
    "" \
    "" \
    "$request_time" \
    "$block_size" \
    "$fallback_count" \
    "$note"
}

mkdir -p "$(dirname "$OUTPUT_CSV")"

printf '%s\n' \
  'date,hardware,scenario,family,model,decode_storage,batch_size,prompt_len,decode_steps,block_size,fallback_tok_per_sec,native_tok_per_sec,speedup,request_time_sec,cache_pool_paged_block_size,decode_storage_fallbacks,notes' \
  >"$OUTPUT_CSV"

run_kernel_profile_row "Llama text" "$MODEL_LLAMA" >>"$OUTPUT_CSV"
run_kernel_profile_row "Qwen3 text" "$MODEL_QWEN3" >>"$OUTPUT_CSV"
run_kernel_profile_row "Qwen3.5 text" "$MODEL_QWEN35" >>"$OUTPUT_CSV"
run_kernel_profile_row "Gemma 3 text" "$MODEL_GEMMA3" >>"$OUTPUT_CSV"
run_kernel_profile_row "Llama 4 text" "$MODEL_LLAMA4" >>"$OUTPUT_CSV"
run_server_policy_row "ExaOne4" "$MODEL_EXAONE4" "auto" "$SERVER_PORT_AUTO" \
  "dense-default policy probe; auto keeps dense decode" >>"$OUTPUT_CSV"
run_server_policy_row "ExaOne4" "$MODEL_EXAONE4" "paged" "$SERVER_PORT_FORCED" \
  "forced paged request falls back to dense and increments decode_storage_fallbacks" >>"$OUTPUT_CSV"

cleanup
SERVER_PID=""

echo "wrote $OUTPUT_CSV"
