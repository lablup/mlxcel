#!/usr/bin/env bash
# M5 Neural Accelerator attention benchmark with dispatch instrumentation.
#
# Usage:
#   ./scripts/bench_na_attention.sh models/qwen3-0.6b-4bit
#   ./scripts/bench_na_attention.sh all
#   ./scripts/bench_na_attention.sh all --output benchmarks/na_attention.csv
#
# The script performs:
#   1. Warmup run with MLXCEL_LOG_NA_ATTENTION=1 (discarded)
#   2. Measured run with MLXCEL_LOG_NA_ATTENTION=all
#   3. CSV extraction of throughput + dispatch metadata

set -euo pipefail

MLXCEL="./target/release/mlxcel"
MODELS_DIR="${MODELS_DIR:-./models}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/store_guard.sh"

BENCHMARKS_DIR="./benchmarks"
PROMPT="Hello, how are you today?"
MAX_TOKENS=50
WARMUP_TOKENS=20
TIMEOUT=300
DATE=$(date '+%Y-%m-%d')
OUTPUT=""
MODEL_ARG=""
NO_PADDED_PREFILL=0

detect_hardware_full() {
  local chip mem
  chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
  mem=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0fGB", $1/1073741824}')
  echo "${chip}_${mem}" | tr ' ' '_'
}

detect_hardware_short() {
  local full
  full=$(detect_hardware_full)
  case "$full" in
    *M1_Ultra*)  echo "m1ultra" ;;
    *M1_Max*)    echo "m1max" ;;
    *M2_Ultra*)  echo "m2ultra" ;;
    *M2_Max*)    echo "m2max" ;;
    *M3_Ultra*)  echo "m3ultra" ;;
    *M3_Max*)    echo "m3max" ;;
    *M4_Ultra*)  echo "m4ultra" ;;
    *M4_Max*)    echo "m4max" ;;
    *M5_Ultra*)  echo "m5ultra" ;;
    *M5_Max*)    echo "m5max" ;;
    *)           echo "unknown" ;;
  esac
}

HARDWARE_FULL=$(detect_hardware_full)
HARDWARE_SHORT=$(detect_hardware_short)

usage() {
  cat <<'EOF'
Usage: bench_na_attention.sh <model_path|all> [options]

Options:
  --prompt TEXT        Override prompt
  --max-tokens N       Max tokens to generate (default: 50)
  --warmup-tokens N    Warmup tokens (default: 20)
  --timeout N          Timeout in seconds (default: 300)
  --output PATH        Output CSV path
  --no-padded-prefill  Disable M5 padded prefill for A/B comparison
  --help               Show this help
EOF
}

parse_metrics() {
  local output="$1"
  local generated decode_tps metrics_line
  local na_dispatches native_causal array_mask padded_prefill fast_path_eligible nax_eligible
  local prefill_dispatches decode_dispatches prefill_nax decode_nax
  local windowed softcapped max_window unique_windows fast_reason nax_reason

  generated=$(echo "$output" | grep -oE '\[Generated [0-9]+ tokens in' | grep -oE '[0-9]+' | tail -1 || true)
  decode_tps=$(echo "$output" | grep -oE '= [0-9.]+ tok/s' | grep -oE '[0-9.]+' | tail -1 || true)

  metrics_line=$(printf '%s\n' "$output" | awk '
    /\[mlxcel\]\[na-attention\]/ {
      total++
      route=""
      phase="unknown"
      fast="false"
      nax="false"
      window="0"
      softcap="0.000"
      fast_reason="none"
      nax_reason="none"
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^route=/) {
          split($i, a, "=")
          route = a[2]
        } else if ($i ~ /^phase=/) {
          split($i, a, "=")
          phase = a[2]
        } else if ($i ~ /^fast_path_eligible=/) {
          split($i, a, "=")
          fast = a[2]
        } else if ($i ~ /^fast_path_reason=/) {
          split($i, a, "=")
          fast_reason = a[2]
        } else if ($i ~ /^nax_eligible=/ || $i ~ /^na_eligible=/) {
          split($i, a, "=")
          nax = a[2]
        } else if ($i ~ /^nax_reason=/) {
          split($i, a, "=")
          nax_reason = a[2]
        } else if ($i ~ /^window_size=/) {
          split($i, a, "=")
          window = a[2] + 0
          windows[window] = 1
          if (window > max_window) {
            max_window = window
          }
          if (window > 0) {
            windowed++
          }
        } else if ($i ~ /^softcap=/) {
          split($i, a, "=")
          softcap = a[2]
          if (softcap != "0.000") {
            softcapped++
          }
        }
      }
      if (route == "native_causal") native++
      else if (route == "array_mask") array_mask++
      else if (route == "padded_prefill") padded++
      if (fast == "true") fast_count++
      if (nax == "true") nax_count++
      if (phase == "prefill") prefill_count++
      else if (phase == "decode") decode_count++
      if (phase == "prefill" && nax == "true") prefill_nax_count++
      else if (phase == "decode" && nax == "true") decode_nax_count++
      if (fast != "true") fast_blocked[fast_reason]++
      if (nax != "true") nax_blocked[nax_reason]++
    }
    END {
      if (max_window == "") max_window = 0
      unique = ""
      for (w in windows) {
        unique = unique (unique == "" ? "" : "|") w
      }
      if (unique == "") unique = "none"
      top_fast = "none"
      top_fast_count = -1
      for (r in fast_blocked) {
        if (fast_blocked[r] > top_fast_count) {
          top_fast = r
          top_fast_count = fast_blocked[r]
        }
      }
      top_nax = "none"
      top_nax_count = -1
      for (r in nax_blocked) {
        if (nax_blocked[r] > top_nax_count) {
          top_nax = r
          top_nax_count = nax_blocked[r]
        }
      }
      print total + 0 "," native + 0 "," array_mask + 0 "," padded + 0 "," fast_count + 0 "," nax_count + 0 "," prefill_count + 0 "," decode_count + 0 "," prefill_nax_count + 0 "," decode_nax_count + 0 "," windowed + 0 "," softcapped + 0 "," max_window + 0 "," unique "," top_fast "," top_nax
    }')

  IFS=',' read -r na_dispatches native_causal array_mask padded_prefill fast_path_eligible nax_eligible prefill_dispatches decode_dispatches prefill_nax decode_nax windowed softcapped max_window unique_windows fast_reason nax_reason <<< "$metrics_line"

  echo "${generated:-},${decode_tps:-},${na_dispatches:-0},${native_causal:-0},${array_mask:-0},${padded_prefill:-0},${fast_path_eligible:-0},${nax_eligible:-0},${prefill_dispatches:-0},${decode_dispatches:-0},${prefill_nax:-0},${decode_nax:-0},${windowed:-0},${softcapped:-0},${max_window},\"${unique_windows}\",${fast_reason},${nax_reason}"
}

bench_one() {
  local model_path="$1"
  local model_name
  model_name=$(basename "$model_path")

  >&2 printf '>>> [warmup] %s ...\n' "$model_name"
  if ! MLXCEL_LOG_NA_ATTENTION=all MLXCEL_NO_PADDED_PREFILL="$NO_PADDED_PREFILL" timeout "$TIMEOUT" "$MLXCEL" generate \
      -m "$model_path" -p "$PROMPT" -n "$WARMUP_TOKENS" >/dev/null 2>&1; then
    echo "${model_name},${model_path},,,,,,,${DATE},${HARDWARE_FULL},FAIL:warmup"
    return
  fi

  >&2 printf '>>> [bench]  %s ...\n' "$model_name"
  local raw
  if ! raw=$(MLXCEL_LOG_NA_ATTENTION=all MLXCEL_NO_PADDED_PREFILL="$NO_PADDED_PREFILL" timeout "$TIMEOUT" "$MLXCEL" generate \
      -m "$model_path" -p "$PROMPT" -n "$MAX_TOKENS" 2>&1); then
    echo "${model_name},${model_path},,,,,,,${DATE},${HARDWARE_FULL},FAIL:bench"
    return
  fi

  local metrics
  metrics=$(parse_metrics "$raw")
  echo "${model_name},${model_path},${metrics},${DATE},${HARDWARE_FULL},OK"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)         PROMPT="$2"; shift 2 ;;
    --max-tokens)     MAX_TOKENS="$2"; shift 2 ;;
    --warmup-tokens)  WARMUP_TOKENS="$2"; shift 2 ;;
    --timeout)        TIMEOUT="$2"; shift 2 ;;
    --output)         OUTPUT="$2"; shift 2 ;;
    --no-padded-prefill) NO_PADDED_PREFILL=1; shift ;;
    --help)           usage; exit 0 ;;
    -*)               echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    *)                MODEL_ARG="$1"; shift ;;
  esac
done

if [[ -z "$MODEL_ARG" ]]; then
  echo "Error: model path or 'all' required" >&2
  usage >&2
  exit 1
fi

if [[ ! -x "$MLXCEL" ]]; then
  echo "Error: $MLXCEL not found. Run 'cargo build --release' first." >&2
  exit 1
fi

if [[ -z "$OUTPUT" ]]; then
  OUTPUT="${BENCHMARKS_DIR}/na_attention_${HARDWARE_SHORT}_${DATE}.csv"
fi

mkdir -p "$(dirname "$OUTPUT")"
if [[ ! -f "$OUTPUT" || ! -s "$OUTPUT" ]]; then
  echo 'model,model_path,generated_tokens,decode_tok_s,na_dispatches,native_causal_dispatches,array_mask_dispatches,padded_prefill_dispatches,fast_path_eligible_dispatches,nax_eligible_dispatches,prefill_dispatches,decode_dispatches,prefill_nax_dispatches,decode_nax_dispatches,windowed_dispatches,softcap_dispatches,max_window_size,unique_window_sizes,top_fast_path_block_reason,top_nax_block_reason,date,hardware,status' > "$OUTPUT"
fi
>&2 echo "Output: $OUTPUT"

emit() {
  echo "$1" | tee -a "$OUTPUT"
}

if [[ "$MODEL_ARG" == "all" ]]; then
  require_checkpoints "$MODELS_DIR" "NA-attention sweep"
  while IFS= read -r dir; do
    [[ -d "$dir" ]] || continue
    emit "$(bench_one "$dir")"
  done < <(find "$MODELS_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
else
  emit "$(bench_one "$MODEL_ARG")"
fi
