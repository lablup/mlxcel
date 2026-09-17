#!/usr/bin/env bash
# M5 warmed sequential A/B benchmark for softcap grouped decode path.
#
# A/B toggle:
# - OFF: default behavior
# - ON:  MLXCEL_ENABLE_SOFTCAP_GQA_DECODE_GROUPED=1

set -euo pipefail

MLXCEL="./target/release/mlxcel"
DATE="$(date '+%Y-%m-%d')"
OUTPUT="benchmarks/m5_softcap_grouped_models_ab_${DATE}.csv"
TIMEOUT=600
MAX_TOKENS=100
WARMUP_TOKENS=24
REPEATS=4
COOLDOWN_BETWEEN_RUNS=15
COOLDOWN_BETWEEN_MODELS=30

PROMPT="M5 softcap grouped decode benchmark. Repeat this paragraph for a stable decode workload and consistent KV-cache behavior. M5 softcap grouped decode benchmark. Repeat this paragraph for a stable decode workload and consistent KV-cache behavior."

DEFAULT_MODELS=(
  "models/gemma-2-2b-it-4bit"
  "models/qwen3-1.7b-4bit"
  "models/ministral-3-3b-instruct-2512-4bit"
)

MODELS=()

usage() {
  cat <<'EOF'
Usage: bench_m5_softcap_grouped.sh [model_path ...] [options]

Options:
  --output PATH            Output CSV path
  --max-tokens N           Generated tokens per measured run (default: 100)
  --warmup-tokens N        Warmup tokens before each measured run (default: 24)
  --repeats N              Number of measured A/B pairs per model (default: 4)
  --timeout N              Timeout per run in seconds (default: 600)
  --run-cooldown N         Cooldown between measured runs in seconds (default: 15)
  --model-cooldown N       Cooldown between models in seconds (default: 30)
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
  local mode="$2"          # off | on
  local pair_index="$3"
  local order_label="$4"
  local run_index="$5"

  local model_name env_prefix start_raw start_summary end_raw end_summary raw fields
  model_name="$(basename "$model_path")"
  env_prefix=()
  if [[ "$mode" == "on" ]]; then
    env_prefix=(MLXCEL_ENABLE_SOFTCAP_GQA_DECODE_GROUPED=1)
  fi

  start_raw="$(thermal_snapshot)"
  start_summary="$(thermal_summary "$start_raw")"

  >&2 echo ">>> [warmup] ${model_name} mode=${mode} pair=${pair_index} order=${order_label}"
  timeout "$TIMEOUT" env \
    ${env_prefix[@]+"${env_prefix[@]}"} \
    "$MLXCEL" generate \
      -m "$model_path" \
      -p "$PROMPT" \
      -n "$WARMUP_TOKENS" \
      --no-chat-template \
      --profile >/dev/null 2>&1

  >&2 echo ">>> [bench]   ${model_name} mode=${mode} pair=${pair_index} order=${order_label}"
  raw="$(timeout "$TIMEOUT" env \
    ${env_prefix[@]+"${env_prefix[@]}"} \
    "$MLXCEL" generate \
      -m "$model_path" \
      -p "$PROMPT" \
      -n "$MAX_TOKENS" \
      --no-chat-template \
      --profile 2>&1)"

  end_raw="$(thermal_snapshot)"
  end_summary="$(thermal_summary "$end_raw")"
  fields="$(parse_profile "$raw")"
  echo "${model_name},${model_path},${mode},${pair_index},${order_label},${run_index},${fields},${start_summary},${end_summary},${DATE}" >> "$OUTPUT"
}

append_header() {
  mkdir -p "$(dirname "$OUTPUT")"
  echo "model,model_path,mode,pair_index,order_label,run_index,prompt_tokens,generated_tokens,prefill_ms,prefill_tok_s,decode_ms,decode_tok_s,thermal_start,thermal_end,date" > "$OUTPUT"
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
    >&2 echo "Warning: this benchmark is intended for M5-family hosts."
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
        order_a="off"
        order_b="on"
      else
        order_a="on"
        order_b="off"
      fi

      run_index=$((run_index + 1))
      run_one "$model_path" "$order_a" "$pair_index" "${order_a}_first" "$run_index"
      sleep "$COOLDOWN_BETWEEN_RUNS"

      run_index=$((run_index + 1))
      run_one "$model_path" "$order_b" "$pair_index" "${order_b}_second" "$run_index"

      if (( pair_index < REPEATS )); then
        sleep "$COOLDOWN_BETWEEN_RUNS"
      fi
    done

    >&2 echo "model cooldown: ${COOLDOWN_BETWEEN_MODELS}s"
    sleep "$COOLDOWN_BETWEEN_MODELS"
  done

  echo "done: $OUTPUT"
}

main "$@"
