#!/usr/bin/env bash
# M5 INT8 KV-cache benchmark with thermal-aware sequential A/B runs.
#
# Goals:
# - never run benchmark jobs concurrently
# - alternate fp16/int8 order to reduce drift bias
# - warm up before each measured run
# - insert cooldown periods between measured runs and models
# - record pmset thermal/performance warning state before and after each run
#
# Usage:
#   ./scripts/bench_m5_kv_int8.sh
#   ./scripts/bench_m5_kv_int8.sh models/qwen3-1.7b-4bit models/llama-3.1-8b-4bit
#   ./scripts/bench_m5_kv_int8.sh --output benchmarks/m5_kv_int8_2026-04-04.csv

set -euo pipefail

MLXCEL="./target/release/mlxcel"
DATE="$(date '+%Y-%m-%d')"
OUTPUT="benchmarks/m5_kv_int8_${DATE}.csv"
TIMEOUT=600
MAX_TOKENS=100
WARMUP_TOKENS=24
REPEATS=4
COOLDOWN_BETWEEN_RUNS=45
COOLDOWN_BETWEEN_MODELS=120
THERMAL_POLL_SECONDS=20

PROMPT="M5 decode benchmark for INT8 KV cache research. Repeat this paragraph to create a longer prompt and a meaningful cache footprint. M5 decode benchmark for INT8 KV cache research. Repeat this paragraph to create a longer prompt and a meaningful cache footprint. M5 decode benchmark for INT8 KV cache research. Repeat this paragraph to create a longer prompt and a meaningful cache footprint. M5 decode benchmark for INT8 KV cache research. Repeat this paragraph to create a longer prompt and a meaningful cache footprint."

DEFAULT_MODELS=(
  "models/qwen3-1.7b-4bit"
  "models/meta-llama-3.1-8b-instruct-4bit"
  "models/qwen3.5-4b-4bit"
  "models/gemma-3-4b-it-4bit"
  "models/gemma-4-e4b-it-4bit"
  "models/ai21-jamba-reasoning-3b-4bit"
)

MODELS=()

usage() {
  cat <<'EOF'
Usage: bench_m5_kv_int8.sh [model_path ...] [options]

Options:
  --output PATH            Output CSV path
  --max-tokens N           Generated tokens per measured run (default: 100)
  --warmup-tokens N        Warmup tokens before each measured run (default: 24)
  --repeats N              Number of measured A/B pairs per model (default: 4)
  --timeout N              Timeout per run in seconds (default: 600)
  --run-cooldown N         Cooldown between measured runs in seconds (default: 45)
  --model-cooldown N       Cooldown between models in seconds (default: 120)
  --help                   Show this help
EOF
}

thermal_snapshot() {
  pmset -g therm 2>/dev/null || true
}

thermal_summary() {
  local raw="$1"
  local thermal perf
  thermal=$(echo "$raw" | sed -n 's/^.*Thermal warning level: *//p' | head -1)
  perf=$(echo "$raw" | sed -n 's/^.*Performance warning level: *//p' | head -1)
  if [[ -z "$thermal" ]]; then
    if echo "$raw" | grep -q "No thermal warning level has been recorded"; then
      thermal="none"
    else
      thermal="unknown"
    fi
  fi
  if [[ -z "$perf" ]]; then
    if echo "$raw" | grep -q "No performance warning level has been recorded"; then
      perf="none"
    else
      perf="unknown"
    fi
  fi
  echo "${thermal}|${perf}"
}

wait_for_cool_state() {
  local max_checks=18
  local checks=0
  while true; do
    local raw summary
    raw="$(thermal_snapshot)"
    summary="$(thermal_summary "$raw")"
    if [[ "$summary" == "none|none" || "$summary" == "unknown|unknown" ]]; then
      return 0
    fi
    checks=$((checks + 1))
    if [[ "$checks" -ge "$max_checks" ]]; then
      >&2 echo "thermal state did not return to cool baseline; proceeding with recorded warning state: $summary"
      return 0
    fi
    >&2 echo "waiting for thermal cooldown: $summary"
    sleep "$THERMAL_POLL_SECONDS"
  done
}

parse_profile() {
  local output="$1"
  local prompt_tok gen_tok prefill_ms prefill_tps decode_ms decode_tps

  prompt_tok=$(echo "$output" | sed -n 's/.*Prompt tokens:[[:space:]]*\([0-9]*\).*/\1/p' | head -1)
  gen_tok=$(echo "$output" | sed -n 's/.*Generated tokens:[[:space:]]*\([0-9]*\).*/\1/p' | head -1)
  prefill_ms=$(echo "$output" | sed -n 's/.*Prefill:[[:space:]]*\([0-9.]*\) ms.*/\1/p' | head -1)
  prefill_tps=$(echo "$output" | sed -n 's/.*Prefill:.*(\([0-9.]*\) tok\/s).*/\1/p' | head -1)
  decode_ms=$(echo "$output" | sed -n 's/.*Decode:[[:space:]]*\([0-9.]*\) ms.*/\1/p' | head -1)
  decode_tps=$(echo "$output" | sed -n 's/.*Decode:.*(\([0-9.]*\) tok\/s).*/\1/p' | head -1)

  echo "${prompt_tok:-},${gen_tok:-},${prefill_ms:-},${prefill_tps:-},${decode_ms:-},${decode_tps:-}"
}

run_one() {
  local model_path="$1"
  local mode="$2"
  local pair_index="$3"
  local order_label="$4"
  local run_index="$5"
  local model_name start_raw end_raw start_summary end_summary raw fields

  model_name="$(basename "$model_path")"
  wait_for_cool_state
  start_raw="$(thermal_snapshot)"
  start_summary="$(thermal_summary "$start_raw")"

  >&2 echo ">>> [warmup] ${model_name} mode=${mode} pair=${pair_index} order=${order_label}"
  timeout "$TIMEOUT" "$MLXCEL" generate \
    -m "$model_path" \
    -p "$PROMPT" \
    -n "$WARMUP_TOKENS" \
    --kv-cache-mode "$mode" \
    --profile >/dev/null 2>&1

  >&2 echo ">>> [bench]   ${model_name} mode=${mode} pair=${pair_index} order=${order_label}"
  raw="$(timeout "$TIMEOUT" "$MLXCEL" generate \
    -m "$model_path" \
    -p "$PROMPT" \
    -n "$MAX_TOKENS" \
    --kv-cache-mode "$mode" \
    --profile 2>&1)"

  end_raw="$(thermal_snapshot)"
  end_summary="$(thermal_summary "$end_raw")"
  fields="$(parse_profile "$raw")"
  echo "${model_name},${model_path},${mode},${pair_index},${order_label},${run_index},${fields},${start_summary},${end_summary},${DATE}"
}

append_header() {
  mkdir -p "$(dirname "$OUTPUT")"
  echo "model,model_path,kv_cache_mode,pair_index,order_label,run_index,prompt_tokens,generated_tokens,prefill_ms,prefill_tok_s,decode_ms,decode_tok_s,thermal_start,thermal_end,date" > "$OUTPUT"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --output) OUTPUT="$2"; shift 2 ;;
      --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
      --warmup-tokens) WARMUP_TOKENS="$2"; shift 2 ;;
      --repeats) REPEATS="$2"; shift 2 ;;
      --timeout) TIMEOUT="$2"; shift 2 ;;
      --run-cooldown) COOLDOWN_BETWEEN_RUNS="$2"; shift 2 ;;
      --model-cooldown) COOLDOWN_BETWEEN_MODELS="$2"; shift 2 ;;
      --help) usage; exit 0 ;;
      -*) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
      *) MODELS+=("$1"); shift ;;
    esac
  done
}

main() {
  parse_args "$@"

  if [[ ! -x "$MLXCEL" ]]; then
    echo "Error: $MLXCEL not found. Run 'cargo build --release' first." >&2
    exit 1
  fi

  if [[ "$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)" != *"M5"* ]]; then
    >&2 echo "Warning: this script is intended for M5-family thermal-aware benchmarking."
  fi

  if [[ "${#MODELS[@]}" -eq 0 ]]; then
    MODELS=("${DEFAULT_MODELS[@]}")
  fi

  append_header
  >&2 echo "Output: $OUTPUT"
  >&2 echo "Thermal baseline:"
  >&2 thermal_snapshot

  local run_index=0
  local pair_index
  local order_a order_b

  for model_path in "${MODELS[@]}"; do
    if [[ ! -d "$model_path" ]]; then
      >&2 echo "skip missing model: $model_path"
      continue
    fi

    >&2 echo ""
    >&2 echo "=== $(basename "$model_path") ==="
    for ((pair_index = 1; pair_index <= REPEATS; pair_index++)); do
      if (( pair_index % 2 == 1 )); then
        order_a="fp16"
        order_b="int8"
      else
        order_a="int8"
        order_b="fp16"
      fi

      run_index=$((run_index + 1))
      run_one "$model_path" "$order_a" "$pair_index" "${order_a}_first" "$run_index" | tee -a "$OUTPUT"
      sleep "$COOLDOWN_BETWEEN_RUNS"

      run_index=$((run_index + 1))
      run_one "$model_path" "$order_b" "$pair_index" "${order_b}_second" "$run_index" | tee -a "$OUTPUT"

      if (( pair_index < REPEATS )); then
        sleep "$COOLDOWN_BETWEEN_RUNS"
      fi
    done

    >&2 echo "model cooldown: ${COOLDOWN_BETWEEN_MODELS}s"
    sleep "$COOLDOWN_BETWEEN_MODELS"
  done
}

main "$@"
