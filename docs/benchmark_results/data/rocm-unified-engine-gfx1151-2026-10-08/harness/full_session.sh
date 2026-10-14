#!/usr/bin/env bash
# One lock hold for every GPU item of issue #2192 (coordinator request,
# 2026-10-09): the host-wide guard lock is taken once with flock, and the
# per-run guards inside (ROCM_GPU_GUARD_LOCK_HELD names this shell) still wait
# for an idle window and monitor each run without re-queueing on the lock.
#
# Usage: full_session.sh REPO_ROOT PRE_EPIC_ROOT OUT_DIR MODELS_DIR PROMPT_FILE
set -uo pipefail
ROOT="$1"; PRE="$2"; OUT="$3"; MODELS_DIR="$4"; PROMPT_FILE="$5"
H="$ROOT/docs/benchmark_results/data/rocm-unified-engine-gfx1151-2026-10-08/harness"
GUARD="${GUARD:-$ROOT/scripts/rocm_gpu_guard.sh}"
LOCK="${ROCM_GPU_GUARD_LOCK:-/tmp/mlxcel-rocm-gpu-guard.lock}"
mkdir -p "$OUT"
echo "waiting for $LOCK ($(date -u +%FT%TZ))"
flock "$LOCK" bash -c '
  export ROCM_GPU_GUARD_LOCK_HELD=$$
  export GUARD="'"$GUARD"'" GUARD_IDLE_SECS="${GUARD_IDLE_SECS:-10}" GUARD_ATTEMPTS="${GUARD_ATTEMPTS:-10}" COOLDOWN="${COOLDOWN:-5}"
  echo "lock held by $$ ($(date -u +%FT%TZ))"
  bash "'"$H"'/measure_session.sh" "'"$ROOT"'" "'"$PRE"'" "'"$OUT"'/measure" "'"$MODELS_DIR"'"
  echo "measure_session rc=$? ($(date -u +%FT%TZ))"
  mkdir -p "'"$OUT"'/parity-floor0"
  PAGED_FLOOR=0 "$GUARD" --idle-secs "$GUARD_IDLE_SECS" --max-attempts 5 --log "'"$OUT"'/parity-floor0/guard.log" -- \
    bash "'"$H"'/parity_session.sh" "'"$ROOT"'" "'"$OUT"'/parity-floor0" "'"$PROMPT_FILE"'" "'"$MODELS_DIR"'" Qwen3-0.6B-4bit Meta-Llama-3.1-8B-Instruct-4bit
  echo "parity floor0 rc=$? ($(date -u +%FT%TZ))"
'
echo "lock released ($(date -u +%FT%TZ))"
