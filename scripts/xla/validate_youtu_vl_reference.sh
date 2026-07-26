#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
MODE="${1:---contract-only}"

cd "$ROOT"
"$PYTHON" -m unittest spike/openxla/test_youtu_vl_reference_oracle.py
"$PYTHON" -m py_compile \
  spike/openxla/youtu_vl_reference_oracle.py \
  spike/openxla/test_youtu_vl_reference_oracle.py

if [[ "$MODE" == "--contract-only" ]]; then
  exit 0
fi

if [[ "$MODE" != "--actual" || $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 [--contract-only | --actual OUTPUT_ROOT [IREE_DEVICE]]" >&2
  exit 2
fi

OUTPUT_ROOT="$2"
DEVICE="${3:-local-task}"
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "error: immutable output already exists: $OUTPUT_ROOT" >&2
  exit 2
fi

CACHE_ROOT="${MLXCEL_YOUTU_VL_ORACLE_CACHE:-$ROOT/.cache/youtu-vl-oracle}"
REVISION="8d30a0e49662a1d628a472b12df264dbcd768753"
MODEL_DIR="$CACHE_ROOT/Youtu-VL-4B-Instruct-$REVISION"
IMAGE="$ROOT/tests/fixtures/test_image.png"
REFERENCE="$OUTPUT_ROOT/hf-reference"
ACTUAL="$OUTPUT_ROOT/mlxcel-actual"
REPORT="$OUTPUT_ROOT/comparison.json"

mkdir -p "$CACHE_ROOT" "$OUTPUT_ROOT"
"$PYTHON" - "$MODEL_DIR" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, "spike/openxla")
import youtu_vl_reference_oracle as oracle

destination = Path(sys.argv[1])
snapshot_download(
    repo_id=oracle.CHECKPOINT_REPO,
    revision=oracle.CHECKPOINT_REVISION,
    local_dir=destination,
    allow_patterns=sorted(oracle.CHECKPOINT_ARTIFACTS),
)
PY

"$PYTHON" spike/openxla/youtu_vl_reference_oracle.py capture \
  --model "$MODEL_DIR" \
  --image "$IMAGE" \
  --out "$REFERENCE"

cargo run --release \
  --features xla-diagnostics \
  --example xla_youtu_vl_reference_check -- \
  --model "$MODEL_DIR" \
  --reference "$REFERENCE" \
  --image "$IMAGE" \
  --out "$ACTUAL" \
  --device "$DEVICE"

"$PYTHON" spike/openxla/youtu_vl_reference_oracle.py compare \
  --reference "$REFERENCE" \
  --actual "$ACTUAL" \
  --report "$REPORT"
