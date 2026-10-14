#!/usr/bin/env bash
# One-command per-architecture validation for the OpenXLA / IREE backend (issue #496).
#
# Given a checkpoint, this drives every gate an architecture must pass:
#
#   [structural] byte-exact emitter regression (pure Rust, no GPU) - a fast
#                pre-gate that the registered fixtures still emit their goldens.
#   [gate 1/2]   token-exact single-sequence vs an HF fp32 oracle (xla_oracle_check).
#   [gate 2/2]   serve reference-exact: every batched request equals its single-seq
#                reference (xla_batch_bench).
#
# The oracle is produced here too: spike/openxla/oracle_continuation.py loads the
# checkpoint in fp32 (dequantizing an MLX 4-bit/8-bit checkpoint offline first) and
# records the greedy continuation the token-exact gate diffs against.
#
# LLaVA's host vision/projector plus prepared-embeddings path has a separate
# independent HF stage oracle and CLI/server gate:
#   scripts/xla/validate_llava_reference.sh --help
#
# The two execution gates need a real IREE build (the `xla-iree` feature), so set
# the IREE environment first (see src/lib/mlxcel-xla/README.md):
#   - CPU (prebuilt dist):  export IREE_DIST=/path/to/iree-dist
#   - CUDA (GB10):          eval "$(scripts/iree/setup-cuda.sh --env)"    (--device cuda)
#   - ROCm (HIP):           eval "$(scripts/iree/setup-rocm.sh --env)"    (--device hip)
#   - macOS (Metal):        eval "$(scripts/iree/setup-macos.sh --env)"
# The structural pre-gate needs none of that; run it alone with --structural-only.
#
# Usage:
#   scripts/xla/validate_arch.sh --model <checkpoint> [options]
#
# Options:
#   --model <dir>       checkpoint directory (required)
#   --device <name>     IREE HAL device (default: $MLXCEL_XLA_DEVICE or local-task)
#   --prompt <text>     oracle prompt (default: "The capital of France is")
#   --max-new <n>       oracle continuation length / token-exact steps (default: 40)
#   --batch <n>         serve B_max slots (default: 4)
#   --requests <n>      serve request count (default: 2*batch)
#   --maxcap <n>        serve per-request token budget clamp (default: 24)
#   --oracle <json>     oracle JSON path (default: a temp file)
#   --skip-structural   skip the byte-exact pre-gate
#   --structural-only   run only the byte-exact pre-gate (no IREE build, no GPU)
#   -h, --help          this help
#
# Exit status: 0 only if every run gate passes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# The oracle venv (spike/openxla/README.md) is gitignored; override its python with
# MLXCEL_ORACLE_PYTHON when it lives outside this checkout (e.g. a shared setup).
VENV_PY="${MLXCEL_ORACLE_PYTHON:-$REPO_ROOT/spike/openxla/.venv/bin/python}"
ORACLE_PY="$REPO_ROOT/spike/openxla/oracle_continuation.py"

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; $d'; }

MODEL=""
DEVICE="${MLXCEL_XLA_DEVICE:-local-task}"
PROMPT="The capital of France is"
MAXNEW=40
BATCH=4
REQUESTS=""
MAXCAP=24
ORACLE=""
SKIP_STRUCTURAL=0
STRUCTURAL_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="${2:?}"; shift 2 ;;
    --device) DEVICE="${2:?}"; shift 2 ;;
    --prompt) PROMPT="${2:?}"; shift 2 ;;
    --max-new) MAXNEW="${2:?}"; shift 2 ;;
    --batch) BATCH="${2:?}"; shift 2 ;;
    --requests) REQUESTS="${2:?}"; shift 2 ;;
    --maxcap) MAXCAP="${2:?}"; shift 2 ;;
    --oracle) ORACLE="${2:?}"; shift 2 ;;
    --skip-structural) SKIP_STRUCTURAL=1; shift ;;
    --structural-only) STRUCTURAL_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

# --- structural pre-gate (fast, pure Rust, no GPU / IREE) ---
if [ "$SKIP_STRUCTURAL" -eq 0 ]; then
  echo "== [structural] byte-exact + per-family signature emitter gate (cargo test) =="
  # The whole validation tests module: the byte-exact goldens
  # (registered_fixtures_are_byte_exact) plus the golden-less dense-family
  # signatures (structural_families_emit_expected_signature, issue #497).
  cargo test -p mlxcel-xla --lib validation::tests -- --nocapture
  echo "[structural] PASS"
fi
if [ "$STRUCTURAL_ONLY" -eq 1 ]; then
  echo "RESULT: PASS (structural only)"
  exit 0
fi

# --- validation of the execution-tier inputs ---
[ -n "$MODEL" ] || { echo "error: --model is required" >&2; usage; exit 2; }
[ -d "$MODEL" ] || { echo "error: model directory not found: $MODEL" >&2; exit 2; }
REQUESTS="${REQUESTS:-$((2 * BATCH))}"

echo "== validate_arch: model=$MODEL device=$DEVICE =="

# --- Tier 1: produce or reuse the fp32 oracle ---
if [ -n "$ORACLE" ] && [ -f "$ORACLE" ]; then
  echo "== [oracle] reusing existing oracle -> $ORACLE =="
elif [ -x "$VENV_PY" ]; then
  ORACLE="${ORACLE:-$(mktemp -t xla_oracle.XXXXXX.json)}"
  echo "== [oracle] producing HF fp32 continuation -> $ORACLE =="
  "$VENV_PY" "$ORACLE_PY" --model "$MODEL" --out "$ORACLE" \
    --prompt "$PROMPT" --max-new "$MAXNEW"
else
  ORACLE="${ORACLE:-$(mktemp -t xla_oracle.XXXXXX.json)}"
  echo "== [oracle] oracle venv not found at $VENV_PY; producing XLA local-task f32 continuation -> $ORACLE =="
  MLXCEL_XLA_PRECISION=f32 cargo run --release --features xla-iree --example xla_traj_dump -- \
    --model "$MODEL" --device local-task --prompt "$PROMPT" --max-new "$MAXNEW" --out "$ORACLE"
fi

# --- Gate 1/2: token-exact single-sequence vs the oracle ---
echo "== [gate 1/2] token-exact single-seq (xla_oracle_check) =="
gate1=0
cargo run --release --features xla-iree --example xla_oracle_check -- \
  --model "$MODEL" --oracle "$ORACLE" --device "$DEVICE" || gate1=$?

# --- Gate 2/2: serve reference-exact (reuses the oracle's prompt_ids) ---
echo "== [gate 2/2] serve reference-exact (xla_batch_bench) =="
gate2=0
cargo run --release --features xla-iree --example xla_batch_bench -- \
  --model "$MODEL" --prompts "$ORACLE" --device "$DEVICE" \
  --batch "$BATCH" --requests "$REQUESTS" --maxcap "$MAXCAP" || gate2=$?

echo ""
echo "== summary =="
[ "$SKIP_STRUCTURAL" -eq 0 ] && echo "structural (byte-exact) : PASS" || echo "structural (byte-exact) : SKIPPED"
[ "$gate1" -eq 0 ] && echo "token-exact single-seq  : PASS" || echo "token-exact single-seq  : FAIL"
[ "$gate2" -eq 0 ] && echo "serve reference-exact   : PASS" || echo "serve reference-exact   : FAIL"
if [ "$gate1" -eq 0 ] && [ "$gate2" -eq 0 ]; then
  echo "RESULT: PASS (both execution gates)"
  exit 0
fi
echo "RESULT: FAIL"
exit 1
