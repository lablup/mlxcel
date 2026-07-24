#!/usr/bin/env bash
# End-to-end LLaVA reference validation for the OpenXLA / IREE backend (#862).
#
# This gate uses the pinned, independently maintained Hugging Face model as the
# oracle and the converted MLX checkpoint as the production input. Both local
# directories are verified by SHA-256 before inference; no weights or generated
# captures belong in Git.
#
# Source:
#   llava-hf/llava-interleave-qwen-0.5b-hf
#   revision 1090956dd1c79bc93ae98dcf395590369435ec91
# Converted:
#   mlx-community/llava-interleave-qwen-0.5b-bf16
#   revision ba7385935f69c5417bfbe29c3809858a98afc22f
# License:
#   Tongyi Qianwen Research License (research/evaluation use; review its terms)
#
# Usage:
#   scripts/xla/validate_llava_reference.sh \
#     --image tests/fixtures/test_image.png \
#     --out /tmp/mlxcel-llava-reference \
#     [--source-model /path/to/pinned-hf-snapshot] \
#     [--model /path/to/pinned-mlx-snapshot] \
#     [--device local-task|local-sync|cuda|metal]
#     [--text-model /path/to/text-checkpoint]
#
# The selected IREE runtime must already be configured. See
# src/lib/mlxcel-xla/README.md for IREE_DIST, IREE_CUDA_HOME, and Metal setup.
# If a constrained Linux host rejects local-task worker creation, local-sync is
# the documented single-threaded CPU execution fallback.
# `--text-model` runs validate_arch.sh after the multimodal gates to provide the
# ordinary text/batch regression companion requested by the architecture gate.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${MLXCEL_ORACLE_PYTHON:-$REPO_ROOT/spike/openxla/.venv/bin/python}"
ORACLE="$REPO_ROOT/spike/openxla/llava_reference_oracle.py"
ORACLE_TEST="$REPO_ROOT/spike/openxla/test_llava_reference_oracle.py"
SURFACE_CHECK="$REPO_ROOT/spike/openxla/llava_cli_server_check.py"
CACHE_ROOT="${MLXCEL_REFERENCE_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/mlxcel/reference-models}"
SOURCE_REVISION="1090956dd1c79bc93ae98dcf395590369435ec91"
CONVERTED_REVISION="ba7385935f69c5417bfbe29c3809858a98afc22f"
FIXTURE_SHA256="5e7d54e8a7d21802378c87d2d70cf551e29739fe27599ddf129ebccdad1e6261"
SOURCE_ARTIFACTS=(
  added_tokens.json chat_template.json config.json generation_config.json
  merges.txt model.safetensors preprocessor_config.json processor_config.json
  special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json
)
CONVERTED_ARTIFACTS=(
  added_tokens.json chat_template.json config.json generation_config.json
  merges.txt model.safetensors model.safetensors.index.json
  preprocessor_config.json processor_config.json special_tokens_map.json
  tokenizer.json tokenizer_config.json vocab.json
)

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; $d'; }

SOURCE_MODEL="$CACHE_ROOT/llava-interleave-qwen-0.5b-hf-$SOURCE_REVISION"
MODEL="$CACHE_ROOT/llava-interleave-qwen-0.5b-bf16-$CONVERTED_REVISION"
IMAGE=""
OUT=""
DEVICE="${MLXCEL_XLA_DEVICE:-local-task}"
TEXT_MODEL=""
CONTEXT_CAPACITY=1536
PORT=18062
SKIP_STRUCTURAL=0
SKIP_SURFACES=0
STRUCTURAL_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --source-model) SOURCE_MODEL="${2:?}"; shift 2 ;;
    --model) MODEL="${2:?}"; shift 2 ;;
    --image) IMAGE="${2:?}"; shift 2 ;;
    --out) OUT="${2:?}"; shift 2 ;;
    --device) DEVICE="${2:?}"; shift 2 ;;
    --text-model) TEXT_MODEL="${2:?}"; shift 2 ;;
    --context-capacity) CONTEXT_CAPACITY="${2:?}"; shift 2 ;;
    --port) PORT="${2:?}"; shift 2 ;;
    --skip-structural) SKIP_STRUCTURAL=1; shift ;;
    --skip-surfaces) SKIP_SURFACES=1; shift ;;
    --structural-only) STRUCTURAL_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

cd "$REPO_ROOT"
# The reference policy is true F32. MLX CUDA enables FAST_TF32 by default,
# which is a useful production throughput mode but a different compute dtype.
export MLX_ENABLE_TF32=0

[ -x "$PYTHON" ] || { echo "error: oracle Python not found: $PYTHON" >&2; exit 3; }

if [ "$SKIP_STRUCTURAL" -eq 0 ]; then
  echo "== [structural] LLaVA diagnostic seam and chat-template selection =="
  "$PYTHON" -m unittest -v "$ORACLE_TEST"
  cargo test -p mlxcel-xla --lib llava_diagnostics_share_the_production_embeddings_prefill
  cargo test --lib server::chat_template
  echo "[structural] PASS"
fi
if [ "$STRUCTURAL_ONLY" -eq 1 ]; then
  echo "RESULT: PASS (structural only)"
  exit 0
fi

for value in IMAGE OUT; do
  [ -n "${!value}" ] || { echo "error: --$(echo "$value" | tr '[:upper:]_' '[:lower:]-') is required" >&2; usage; exit 2; }
done
[ -f "$IMAGE" ] || { echo "error: image not found: $IMAGE" >&2; exit 2; }
ACTUAL_FIXTURE_SHA256="$(sha256sum "$IMAGE" | cut -d' ' -f1)"
[ "$ACTUAL_FIXTURE_SHA256" = "$FIXTURE_SHA256" ] || {
  echo "error: image fixture SHA-256 mismatch: expected $FIXTURE_SHA256, got $ACTUAL_FIXTURE_SHA256" >&2
  exit 3
}

ensure_snapshot() {
  local repo="$1"
  local revision="$2"
  local directory="$3"
  shift 3
  local complete=1
  local artifact
  for artifact in "$@"; do
    if [ ! -f "$directory/$artifact" ]; then
      complete=0
      break
    fi
  done
  if [ "$complete" -eq 1 ]; then
    return
  fi
  command -v hf >/dev/null 2>&1 || {
    echo "error: pinned snapshot unavailable at $directory and the 'hf' CLI is not installed" >&2
    exit 3
  }
  mkdir -p "$directory"
  echo "== [download] $repo@$revision -> $directory =="
  hf download "$repo" --revision "$revision" --local-dir "$directory"
  for artifact in "$@"; do
    [ -f "$directory/$artifact" ] || {
      echo "error: pinned snapshot is missing required artifact: $directory/$artifact" >&2
      exit 3
    }
  done
}

ensure_snapshot \
  "llava-hf/llava-interleave-qwen-0.5b-hf" \
  "$SOURCE_REVISION" \
  "$SOURCE_MODEL" \
  "${SOURCE_ARTIFACTS[@]}"
ensure_snapshot \
  "mlx-community/llava-interleave-qwen-0.5b-bf16" \
  "$CONVERTED_REVISION" \
  "$MODEL" \
  "${CONVERTED_ARTIFACTS[@]}"

mkdir -p "$OUT"
REFERENCE="$OUT/hf-reference"
ACTUAL="$OUT/xla-$DEVICE"
REPORT="$OUT/comparison-$DEVICE.json"

echo "== [oracle] pinned Hugging Face capture =="
"$PYTHON" "$ORACLE" capture \
  --source-model "$SOURCE_MODEL" \
  --converted-model "$MODEL" \
  --image "$IMAGE" \
  --out "$REFERENCE" \
  --device cpu

echo "== [runtime] production host preprocessor + IREE $DEVICE =="
cargo run --release --features xla-diagnostics --example xla_llava_reference_check -- \
  --model "$MODEL" \
  --reference "$REFERENCE" \
  --image "$IMAGE" \
  --out "$ACTUAL" \
  --device "$DEVICE" \
  --context-capacity "$CONTEXT_CAPACITY"

echo "== [comparison] ordered first-divergence gate =="
"$PYTHON" "$ORACLE" compare \
  --reference "$REFERENCE" \
  --actual "$ACTUAL" \
  --report "$REPORT"

if [ "$SKIP_SURFACES" -eq 0 ]; then
  echo "== [surfaces] non-streaming CLI + streaming OpenAI API + metrics =="
  cargo build --release --features xla-diagnostics --bin mlxcel
  "$PYTHON" "$SURFACE_CHECK" \
    --mlxcel-bin "$REPO_ROOT/target/release/mlxcel" \
    --model "$MODEL" \
    --reference "$REFERENCE" \
    --image "$IMAGE" \
    --device "$DEVICE" \
    --port "$PORT" \
    --context-capacity "$CONTEXT_CAPACITY"
fi

if [ -n "$TEXT_MODEL" ]; then
  echo "== [regression] existing text and batch architecture gates =="
  "$REPO_ROOT/scripts/xla/validate_arch.sh" \
    --model "$TEXT_MODEL" \
    --device "$DEVICE" \
    --skip-structural
fi

echo "RESULT: PASS"
echo "reference: $REFERENCE/manifest.json"
echo "actual:    $ACTUAL/manifest.json"
echo "report:    $REPORT"
