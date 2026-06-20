#!/usr/bin/env bash
# TurboQuant KV cache speed gate matrix runner.
#
# Sweeps the configured set of KVCacheModes against a model at the listed
# decode contexts plus a prefill@8K reading. Thermal-aware sequential A/B
# pattern modelled on bench_m5_kv_int8.sh:
#   - never run benchmark jobs concurrently
#   - warm up before each measured run
#   - cooldown between measured runs and modes
#   - record pmset thermal/performance warning state per run
#
# Measured baselines and regression gates (M1 Ultra, post-#369; see the
# 2026-06-19 sweep under benchmarks/turbo_kv/). Quantized KV trades cache
# footprint for memory, NOT decode speed: every mode below decodes slower than
# fp16. The earlier "decode >=0.97x FP16" targets were aspirational and never
# achievable on this path; they are replaced with the measured ranges as
# regression backstops. Decode ratios are vs the same model's fp16 at 4K.
#   int8:            decode 0.61-0.78x (dense), 0.28-0.48x (MoE); prefill
#                    0.98-1.29x. Fastest quantized mode, 2x compression.
#   turbo4-delegated decode 0.63-0.80x (dense), 0.21-0.43x (MoE); prefill
#                    0.99-1.27x. Fastest Turbo codec, ~4x V compression.
#   turbo4-asym:     decode 0.18-0.44x; prefill 0.61-0.92x. K stays fp16
#                    (exact). The #369 dequant-SDPA path is for memory +
#                    exactness, not speed, and MUST stay parity-exact vs the
#                    graph SDPA reference (RMS ~0). Fused kernel tracked in #370.
#   turbo4 (sym):    decode 0.10-0.27x; prefill 0.40-0.73x. Max compression,
#                    quality-sensitive; allowlist families only.
#   turbo3-asym:     decode 0.02-0.07x (near-unusable, 32K ~0.4 tok/s).
#                    Memory-extremis only; tracking, not a recommended mode.
# Regression rule: flag any mode whose 4K decode drops below half the measured
# baseline above on the same model, or any turbo4-asym parity-RMS > 5e-3.
#
# Usage:
#   ./scripts/bench_kv_cache.sh                              # default model + full matrix
#   ./scripts/bench_kv_cache.sh models/Meta-Llama-3.1-8B-Instruct-4bit
#   ./scripts/bench_kv_cache.sh --modes fp16,turbo4-asym --contexts 4096
#   ./scripts/bench_kv_cache.sh --output benchmarks/turbo_kv/2026-04-29_M5Max_llama31.csv

set -euo pipefail

MLXCEL="./target/release/mlxcel"
DATE="$(date '+%Y-%m-%d')"
HARDWARE_TAG="$(sysctl -n machdep.cpu.brand_string 2>/dev/null | tr ' ' '_' | tr -d '()' || echo "unknown")"
DEFAULT_MODEL="models/Meta-Llama-3.1-8B-Instruct-4bit"
DEFAULT_OUTPUT_DIR="benchmarks/turbo_kv"
TIMEOUT=900
WARMUP_TOKENS=24
DECODE_TOKENS=100        # tokens generated for the decode-throughput measurement
PREFILL_PROBE_TOKENS=1   # 1 token forces just the prefill stage to be timed
COOLDOWN_BETWEEN_RUNS=30
COOLDOWN_BETWEEN_MODES=60
THERMAL_POLL_SECONDS=15

# Decode contexts: prompt length seeded to ~target_tokens via repeated paragraph.
# 4K, 16K, 32K decode + 8K prefill are the gate cells.
DECODE_CONTEXTS=(4096 16384 32768)
PREFILL_CONTEXTS=(8192)

# Default mode sweep covers the full gate matrix.
DEFAULT_MODES=(fp16 int8 turbo4-asym turbo4 turbo4-delegated turbo3-asym)
MODES=()
CONTEXTS_OVERRIDE=()
PREFILL_OVERRIDE=()
MODEL=""
OUTPUT=""

# A ~50-token paragraph; we repeat it to approach the target prompt length.
PARAGRAPH="The TurboQuant KV cache speed gate matrix benchmarks decode and prefill throughput across cache quantization modes on Apple Silicon to validate the per-config compression ratio targets from epic 458 against a fixed-prompt fixed-decode workload that mirrors production-scale inference."

usage() {
  cat <<'EOF'
Usage: bench_kv_cache.sh [model_path] [options]

Options:
  --modes csv              Comma-separated KVCacheModes (default: fp16,int8,turbo4-asym,turbo4,turbo4-delegated,turbo3-asym)
  --contexts csv           Comma-separated decode prompt-token targets (default: 4096,16384,32768)
  --prefill-contexts csv   Comma-separated prefill-only prompt-token targets (default: 8192)
  --decode-tokens N        Generated tokens for decode runs (default: 100)
  --prefill-tokens N       Generated tokens for prefill-only probe (default: 1)
  --warmup-tokens N        Warmup tokens before each measured run (default: 24)
  --output PATH            Output CSV path (default: benchmarks/turbo_kv/<date>_<hw>_<model>.csv)
  --timeout N              Timeout per run, seconds (default: 900)
  --run-cooldown N         Cooldown between runs (default: 30)
  --mode-cooldown N        Cooldown between modes (default: 60)
  --help                   Show this help
EOF
}

thermal_snapshot() { pmset -g therm 2>/dev/null || true; }

thermal_summary() {
  local raw="$1" thermal perf
  thermal=$(echo "$raw" | sed -n 's/^.*Thermal warning level: *//p' | head -1)
  perf=$(echo "$raw" | sed -n 's/^.*Performance warning level: *//p' | head -1)
  [[ -z "$thermal" ]] && thermal="none"
  [[ -z "$perf" ]] && perf="none"
  echo "${thermal}|${perf}"
}

wait_for_cool_state() {
  local checks=0
  while (( checks < 12 )); do
    local raw summary
    raw="$(thermal_snapshot)"
    summary="$(thermal_summary "$raw")"
    if [[ "$summary" == "none|none" ]]; then return 0; fi
    >&2 echo "  thermal cooldown wait: $summary"
    checks=$((checks + 1))
    sleep "$THERMAL_POLL_SECONDS"
  done
  >&2 echo "  thermal not fully cooled; proceeding"
}

build_prompt() {
  # Repeat $PARAGRAPH until token count approximates target. ~50 tokens / repetition,
  # so reps = ceil(target / 50). The mlxcel tokenizer reports the actual count.
  local target="$1"
  local reps=$(( (target + 49) / 50 ))
  local out=""
  for ((i = 0; i < reps; i++)); do out+="${PARAGRAPH} "; done
  echo "$out"
}

parse_profile() {
  local out="$1"
  local prompt_tok gen_tok prefill_ms prefill_tps decode_ms decode_tps
  prompt_tok=$(echo "$out" | sed -n 's/.*Prompt tokens:[[:space:]]*\([0-9]*\).*/\1/p' | head -1)
  gen_tok=$(echo "$out" | sed -n 's/.*Generated tokens:[[:space:]]*\([0-9]*\).*/\1/p' | head -1)
  prefill_ms=$(echo "$out" | sed -n 's/.*Prefill:[[:space:]]*\([0-9.]*\) ms.*/\1/p' | head -1)
  prefill_tps=$(echo "$out" | sed -n 's/.*Prefill:.*(\([0-9.]*\) tok\/s).*/\1/p' | head -1)
  decode_ms=$(echo "$out" | sed -n 's/.*Decode:[[:space:]]*\([0-9.]*\) ms.*/\1/p' | head -1)
  decode_tps=$(echo "$out" | sed -n 's/.*Decode:.*(\([0-9.]*\) tok\/s).*/\1/p' | head -1)
  echo "${prompt_tok:-},${gen_tok:-},${prefill_ms:-},${prefill_tps:-},${decode_ms:-},${decode_tps:-}"
}

run_one() {
  local model_path="$1" mode="$2" context_target="$3" stage="$4" gen_tokens="$5" run_idx="$6"
  local model_name prompt start_raw end_raw start_summary end_summary raw fields

  model_name="$(basename "$model_path")"
  prompt="$(build_prompt "$context_target")"
  wait_for_cool_state
  start_raw="$(thermal_snapshot)"
  start_summary="$(thermal_summary "$start_raw")"

  >&2 echo "  [warmup] mode=${mode} ctx≈${context_target} stage=${stage}"
  timeout "$TIMEOUT" "$MLXCEL" generate \
    -m "$model_path" \
    -p "$prompt" \
    -n "$WARMUP_TOKENS" \
    --kv-cache-mode "$mode" \
    --profile >/dev/null 2>&1 || true

  >&2 echo "  [bench]  mode=${mode} ctx≈${context_target} stage=${stage}"
  raw="$(timeout "$TIMEOUT" "$MLXCEL" generate \
    -m "$model_path" \
    -p "$prompt" \
    -n "$gen_tokens" \
    --kv-cache-mode "$mode" \
    --profile 2>&1 || true)"

  end_raw="$(thermal_snapshot)"
  end_summary="$(thermal_summary "$end_raw")"
  fields="$(parse_profile "$raw")"
  echo "${model_name},${model_path},${mode},${context_target},${stage},${run_idx},${fields},${start_summary},${end_summary},${HARDWARE_TAG},${DATE}"
}

append_header() {
  mkdir -p "$(dirname "$OUTPUT")"
  echo "model,model_path,kv_cache_mode,context_target,stage,run_index,prompt_tokens,generated_tokens,prefill_ms,prefill_tok_s,decode_ms,decode_tok_s,thermal_start,thermal_end,hardware,date" > "$OUTPUT"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --modes) IFS=',' read -ra MODES <<< "$2"; shift 2 ;;
      --contexts) IFS=',' read -ra CONTEXTS_OVERRIDE <<< "$2"; shift 2 ;;
      --prefill-contexts) IFS=',' read -ra PREFILL_OVERRIDE <<< "$2"; shift 2 ;;
      --decode-tokens) DECODE_TOKENS="$2"; shift 2 ;;
      --prefill-tokens) PREFILL_PROBE_TOKENS="$2"; shift 2 ;;
      --warmup-tokens) WARMUP_TOKENS="$2"; shift 2 ;;
      --output) OUTPUT="$2"; shift 2 ;;
      --timeout) TIMEOUT="$2"; shift 2 ;;
      --run-cooldown) COOLDOWN_BETWEEN_RUNS="$2"; shift 2 ;;
      --mode-cooldown) COOLDOWN_BETWEEN_MODES="$2"; shift 2 ;;
      --help) usage; exit 0 ;;
      -*) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
      *) MODEL="$1"; shift ;;
    esac
  done
}

main() {
  parse_args "$@"

  [[ -z "$MODEL" ]] && MODEL="$DEFAULT_MODEL"
  [[ "${#MODES[@]}" -eq 0 ]] && MODES=("${DEFAULT_MODES[@]}")
  [[ "${#CONTEXTS_OVERRIDE[@]}" -gt 0 ]] && DECODE_CONTEXTS=("${CONTEXTS_OVERRIDE[@]}")
  [[ "${#PREFILL_OVERRIDE[@]}" -gt 0 ]] && PREFILL_CONTEXTS=("${PREFILL_OVERRIDE[@]}")
  if [[ -z "$OUTPUT" ]]; then
    local model_short
    model_short="$(basename "$MODEL" | tr '/' '_')"
    OUTPUT="${DEFAULT_OUTPUT_DIR}/${DATE}_${HARDWARE_TAG}_${model_short}.csv"
  fi

  if [[ ! -x "$MLXCEL" ]]; then
    echo "Error: $MLXCEL not found. Run 'cargo build --release' first." >&2
    exit 1
  fi
  if [[ ! -d "$MODEL" ]]; then
    echo "Error: model dir $MODEL not present." >&2
    exit 1
  fi

  append_header
  >&2 echo "Output: $OUTPUT"
  >&2 echo "Model: $MODEL"
  >&2 echo "Modes: ${MODES[*]}"
  >&2 echo "Decode contexts: ${DECODE_CONTEXTS[*]}"
  >&2 echo "Prefill contexts: ${PREFILL_CONTEXTS[*]}"
  >&2 echo "Hardware: $HARDWARE_TAG"
  >&2 echo "Thermal baseline:"
  >&2 thermal_snapshot

  local run_idx=0
  for mode in "${MODES[@]}"; do
    >&2 echo ""
    >&2 echo "=== mode=${mode} ==="
    for ctx in "${DECODE_CONTEXTS[@]}"; do
      run_idx=$((run_idx + 1))
      run_one "$MODEL" "$mode" "$ctx" "decode" "$DECODE_TOKENS" "$run_idx" | tee -a "$OUTPUT"
      sleep "$COOLDOWN_BETWEEN_RUNS"
    done
    for ctx in "${PREFILL_CONTEXTS[@]}"; do
      run_idx=$((run_idx + 1))
      run_one "$MODEL" "$mode" "$ctx" "prefill" "$PREFILL_PROBE_TOKENS" "$run_idx" | tee -a "$OUTPUT"
      sleep "$COOLDOWN_BETWEEN_RUNS"
    done
    >&2 echo "  mode cooldown ${COOLDOWN_BETWEEN_MODES}s"
    sleep "$COOLDOWN_BETWEEN_MODES"
  done
}

main "$@"
