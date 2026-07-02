#!/usr/bin/env bash
# Long-prompt prefill benchmark sweep (epic #623 #624).
#
# Runs a representative subset of models across a prompt-length ladder so that
# prefill throughput is measured in the matmul-bound regime rather than the
# launch-overhead-bound regime that short (8-66 token) prompts produce. Each
# cell reuses scripts/bench_decode.sh --prompt-tokens N, so warmup, OOM
# classification, and the CSV schema are identical to the standard harness; the
# only added column is prompt_target_len.
#
# Usage:
#   ./scripts/bench_longprompt.sh
#   ./scripts/bench_longprompt.sh --lengths "512 2048 8192"
#   ./scripts/bench_longprompt.sh --models "llama-3.1-8b-4bit qwen2.5-7b-4bit"
#   ./scripts/bench_longprompt.sh --output benchmarks/custom.csv
#
# Default output: benchmarks/{backend}_{hardware}_longprompt_{YYYY-MM-DD}.csv
#   e.g. benchmarks/cuda_gb10_longprompt_2026-07-03.csv
#
# A model+length cell that OOMs (common at 32768 for large MoE models) is
# recorded with the usual SKIP:oom / SKIP:oom_estimate classification and the
# sweep continues.

set -euo pipefail

trap 'echo "Interrupted (signal received)" >&2; exit 130' INT TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DECODE="${SCRIPT_DIR}/bench_decode.sh"
MODELS_DIR="./models"
BENCHMARKS_DIR="./benchmarks"
DATE=$(date '+%Y-%m-%d')

# Representative subset: mix of dense (llama, qwen2.5), MoE (qwen3-a3b,
# mixtral), and a large multimodal-capable text model (gemma-4).
MODELS_DEFAULT="llama-3.1-8b-4bit qwen2.5-7b-4bit qwen3-30b-a3b-4bit mixtral-8x7b-4bit gemma-4-31b-it-4bit"
LADDER_DEFAULT="512 2048 8192 32768"

MODELS="$MODELS_DEFAULT"
LADDER="$LADDER_DEFAULT"
OUTPUT=""
MAX_TOKENS=32
WARMUP_TOKENS=4
COOLDOWN=0
BIG_COOLDOWN=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: bench_longprompt.sh [options]

Options:
  --models "A B C"    Space-separated model basenames under ./models
                      (default: llama-3.1-8b-4bit qwen2.5-7b-4bit
                      qwen3-30b-a3b-4bit mixtral-8x7b-4bit gemma-4-31b-it-4bit)
  --lengths "N N N"   Space-separated prompt-token ladder
                      (default: 512 2048 8192 32768)
  --max-tokens N      Decode tokens per cell (default: 32; kept small so the
                      run measures prefill, not decode)
  --warmup-tokens N   Warmup decode tokens per cell (default: 4)
  --cooldown N        Seconds to sleep after each cell (default: 0)
  --big-cooldown N    Extra seconds after a >10GB model (default: 0)
  --output PATH       Aggregate CSV path (default: auto-named under benchmarks/)
  --help              Show this help

All cells share one aggregate CSV with the 15-column bench_decode.sh schema
(prompt_target_len appended). Cell order is model-major, length-minor.
EOF
}

# ---------------------------------------------------------------------------
# Minimal hardware/backend detection for the default filename (mirrors
# bench_decode.sh so the two share a naming convention).
# ---------------------------------------------------------------------------
detect_backend() {
  if [[ "$(uname)" == "Linux" ]] && nvidia-smi &>/dev/null; then
    echo "cuda"
  else
    echo "metal"
  fi
}

detect_hardware_short() {
  local chip=""
  if [[ "$(uname)" == "Darwin" ]]; then
    chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
  else
    chip=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
  fi
  case "$chip" in
    *M1\ Ultra*) echo "m1ultra" ;;
    *M5\ Max*)   echo "m5max" ;;
    *GB10*)      echo "gb10" ;;
    *)           echo "$chip" | tr '[:upper:] ' '[:lower:]_' | tr ',' '_' | cut -c1-20 ;;
  esac
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)        MODELS="$2"; shift 2 ;;
    --lengths)       LADDER="$2"; shift 2 ;;
    --max-tokens)    MAX_TOKENS="$2"; shift 2 ;;
    --warmup-tokens) WARMUP_TOKENS="$2"; shift 2 ;;
    --cooldown)      COOLDOWN="$2"; shift 2 ;;
    --big-cooldown)  BIG_COOLDOWN="$2"; shift 2 ;;
    --output)        OUTPUT="$2"; shift 2 ;;
    --help)          usage; exit 0 ;;
    *)               echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ ! -x "$BENCH_DECODE" ]]; then
  echo "Error: $BENCH_DECODE not found or not executable" >&2
  exit 1
fi

if [[ -z "$OUTPUT" ]]; then
  BACKEND=$(detect_backend)
  HW=$(detect_hardware_short)
  OUTPUT="${BENCHMARKS_DIR}/${BACKEND}_${HW}_longprompt_${DATE}.csv"
fi

mkdir -p "$(dirname "$OUTPUT")"
: > "$OUTPUT"   # truncate; header is copied from the first cell

TMP_CELL=$(mktemp -t bench_longprompt_cell.XXXXXX.csv)
cleanup() { rm -f "$TMP_CELL"; }
trap 'cleanup; echo "Interrupted (signal received)" >&2; exit 130' INT TERM
trap cleanup EXIT

>&2 echo "Long-prompt sweep"
>&2 echo "  output:  $OUTPUT"
>&2 echo "  models:  $MODELS"
>&2 echo "  lengths: $LADDER"
>&2 echo "  max-tokens=$MAX_TOKENS warmup-tokens=$WARMUP_TOKENS"
>&2 echo ""

header_written=0
for model_name in $MODELS; do
  model_path="${MODELS_DIR}/${model_name}"
  if [[ ! -d "$model_path" ]]; then
    >&2 echo ">>> [miss]   $model_name (not found under $MODELS_DIR, skipping)"
    continue
  fi
  for len in $LADDER; do
    >&2 echo "=== $model_name @ ${len} tokens ==="
    # Reuse the standard runner for one (model, length) cell. --output isolates
    # the cell so the aggregate CSV is never truncated mid-sweep.
    "$BENCH_DECODE" "$model_path" \
      --prompt-tokens "$len" \
      --max-tokens "$MAX_TOKENS" \
      --warmup-tokens "$WARMUP_TOKENS" \
      --cooldown "$COOLDOWN" \
      --big-cooldown "$BIG_COOLDOWN" \
      --output "$TMP_CELL" \
      "${EXTRA_ARGS[@]}" >/dev/null || {
        >&2 echo "    cell runner exited non-zero (continuing)"
      }
    if [[ ! -s "$TMP_CELL" ]]; then
      >&2 echo "    no output produced for this cell (continuing)"
      continue
    fi
    if [[ "$header_written" -eq 0 ]]; then
      cat "$TMP_CELL" >> "$OUTPUT"
      header_written=1
    else
      tail -n +2 "$TMP_CELL" >> "$OUTPUT"
    fi
  done
done

>&2 echo ""
>&2 echo "Long-prompt results saved to: $OUTPUT"
