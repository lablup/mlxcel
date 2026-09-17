#!/usr/bin/env bash
# Capture Metal System Trace for mlxcel decode/prefill path validation on M5.
#
# Purpose:
# - warm up once to reduce cold-start noise
# - record a single measured run with xctrace Metal System Trace template
# - export table-of-contents XML for quick post-processing
#
# Example:
#   ./scripts/capture_m5_metal_trace.sh \
#     --model models/gemma2-2b-4bit \
#     --tokens 100 \
#     --time-limit 25s

set -euo pipefail

MLXCEL="./target/release/mlxcel"
MODEL=""
PROMPT="M5 metal trace capture for fused attention verification. Repeat this paragraph for stable decode behavior. M5 metal trace capture for fused attention verification. Repeat this paragraph for stable decode behavior."
TOKENS=100
WARMUP_TOKENS=24
TIME_LIMIT="25s"
OUTPUT_DIR="traces/xctrace"
TRACE_NAME=""
ENABLE_BOOL_MASK=0
ENABLE_SOFTCAP_GQA_GROUPED=0
DISABLE_SOFTCAP_GQA_GROUPED=0
ENV_ARGS=()

usage() {
  cat <<'EOF'
Usage: capture_m5_metal_trace.sh --model PATH [options]

Required:
  --model PATH                    Model directory (e.g. models/gemma-2-2b-it-4bit)

Options:
  --prompt TEXT                   Prompt string
  --tokens N                      Measured generation tokens (default: 100)
  --warmup-tokens N               Warmup tokens before trace capture (default: 24)
  --time-limit T                  xctrace record limit (default: 25s, e.g. 10s, 2m)
  --output-dir DIR                Output directory for trace artifacts
                                  (default: traces/xctrace)
  --trace-name NAME               Optional trace file prefix (default: auto)
  --enable-bool-mask              Set MLXCEL_EXPERIMENTAL_BOOL_CAUSAL_MASK=1
  --enable-softcap-gqa-grouped    Set MLXCEL_ENABLE_SOFTCAP_GQA_DECODE_GROUPED=1
  --disable-softcap-gqa-grouped   Set MLXCEL_DISABLE_SOFTCAP_GQA_DECODE_GROUPED=1
  --help                          Show this help
EOF
}

require_cmd() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "Error: required command not found: $name" >&2
    exit 1
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model) MODEL="$2"; shift 2 ;;
      --prompt) PROMPT="$2"; shift 2 ;;
      --tokens) TOKENS="$2"; shift 2 ;;
      --warmup-tokens) WARMUP_TOKENS="$2"; shift 2 ;;
      --time-limit) TIME_LIMIT="$2"; shift 2 ;;
      --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
      --trace-name) TRACE_NAME="$2"; shift 2 ;;
      --enable-bool-mask) ENABLE_BOOL_MASK=1; shift ;;
      --enable-softcap-gqa-grouped) ENABLE_SOFTCAP_GQA_GROUPED=1; shift ;;
      --disable-softcap-gqa-grouped) DISABLE_SOFTCAP_GQA_GROUPED=1; shift ;;
      --help) usage; exit 0 ;;
      -*) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
      *) echo "Unexpected positional argument: $1" >&2; usage >&2; exit 1 ;;
    esac
  done
}

setup_env_args() {
  ENV_ARGS=()
  if [[ "$ENABLE_BOOL_MASK" -eq 1 ]]; then
    ENV_ARGS+=(--env MLXCEL_EXPERIMENTAL_BOOL_CAUSAL_MASK=1)
  fi
  if [[ "$ENABLE_SOFTCAP_GQA_GROUPED" -eq 1 ]]; then
    ENV_ARGS+=(--env MLXCEL_ENABLE_SOFTCAP_GQA_DECODE_GROUPED=1)
  fi
  if [[ "$DISABLE_SOFTCAP_GQA_GROUPED" -eq 1 ]]; then
    ENV_ARGS+=(--env MLXCEL_DISABLE_SOFTCAP_GQA_DECODE_GROUPED=1)
  fi
}

main() {
  parse_args "$@"
  require_cmd xcrun
  require_cmd timeout

  if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required" >&2
    usage >&2
    exit 1
  fi
  if [[ ! -x "$MLXCEL" ]]; then
    echo "Error: $MLXCEL not found. Run 'cargo build --release' first." >&2
    exit 1
  fi
  if [[ ! -d "$MODEL" ]]; then
    echo "Error: model directory does not exist: $MODEL" >&2
    exit 1
  fi

  local template_list
  template_list="$(xcrun xctrace list templates 2>/dev/null || true)"
  if [[ "$template_list" != *"Metal System Trace"* ]]; then
    echo "Error: 'Metal System Trace' template not available in xctrace." >&2
    exit 1
  fi

  mkdir -p "$OUTPUT_DIR"
  local now model_name trace_base
  now="$(date '+%Y%m%d-%H%M%S')"
  model_name="$(basename "$MODEL")"
  if [[ -n "$TRACE_NAME" ]]; then
    trace_base="$TRACE_NAME"
  else
    trace_base="metal_trace_${model_name}_${now}"
  fi

  local trace_path toc_path run_stdout warmup_log
  trace_path="${OUTPUT_DIR}/${trace_base}.trace"
  toc_path="${OUTPUT_DIR}/${trace_base}_toc.xml"
  run_stdout="${OUTPUT_DIR}/${trace_base}_target_stdout.log"
  warmup_log="${OUTPUT_DIR}/${trace_base}_warmup.log"

  setup_env_args

  echo "== Trace Capture Configuration =="
  echo "model:        $MODEL"
  echo "tokens:       $TOKENS"
  echo "warmup:       $WARMUP_TOKENS"
  echo "time-limit:   $TIME_LIMIT"
  echo "trace output: $trace_path"
  echo "toc output:   $toc_path"
  if [[ "${#ENV_ARGS[@]}" -gt 0 ]]; then
    echo "env flags:    ${ENV_ARGS[*]}"
  else
    echo "env flags:    (none)"
  fi
  echo

  echo ">>> warmup run"
  timeout 600 env \
    ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} \
    "$MLXCEL" generate \
      -m "$MODEL" \
      -p "$PROMPT" \
      -n "$WARMUP_TOKENS" \
      --no-chat-template \
      --profile >"$warmup_log" 2>&1

  echo ">>> xctrace record"
  xcrun xctrace record \
    --template "Metal System Trace" \
    --output "$trace_path" \
    --time-limit "$TIME_LIMIT" \
    --target-stdout "$run_stdout" \
    ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} \
    --no-prompt \
    --launch -- \
    "$MLXCEL" generate \
      -m "$MODEL" \
      -p "$PROMPT" \
      -n "$TOKENS" \
      --no-chat-template \
      --profile

  echo ">>> xctrace export --toc"
  xcrun xctrace export --input "$trace_path" --toc --output "$toc_path"

  echo
  echo "Done."
  echo "- trace:        $trace_path"
  echo "- toc xml:      $toc_path"
  echo "- target stdout $run_stdout"
  echo "- warmup log:   $warmup_log"
}

main "$@"
