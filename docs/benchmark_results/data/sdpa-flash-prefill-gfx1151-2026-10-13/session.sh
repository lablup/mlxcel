#!/usr/bin/env bash
# Two-arm measurement for issue #2251 on gfx1151: the fused SDPA prefill path
# at the chunked-prefill shapes, before and after the deferred-accumulator
# rescale + one-time Q staging fix.
#   nofix  the before build's binaries (saved aside before the fix rebuilt)
#   fix    the after build's binaries
# Each run is its own scripts/rocm_gpu_guard.sh command inside one host-wide
# lock hold (ROCM_GPU_GUARD_LOCK_HELD names the enclosing shell). Probe and
# bench-decode rounds alternate arms per round; the server matrix runs one
# fresh server per arm with 3 repetitions of batch-4 at 16K-token prompts.
# Usage: session.sh REPO_ROOT OUT_DIR MODELS_DIR NOFIX_BIN_ROOT FIX_BIN_ROOT
set -uo pipefail
ROOT="$1"; OUT="$2"; MODELS_DIR="$3"; NOFIX="$4"; FIX="$5"
GUARD="${GUARD:-$ROOT/scripts/rocm_gpu_guard.sh}"
HARNESS="$OUT/harness"
mkdir -p "$OUT/guard" "$OUT/raw"
export OUT GUARD MODELS_DIR NOFIX FIX
echo "waiting for $LOCK_SLOT ($(date -u +%FT%TZ))" >&2 || true
echo "waiting for the host-wide lock ($(date -u +%FT%TZ))"
flock "${ROCM_GPU_GUARD_LOCK:-/tmp/mlxcel-rocm-gpu-guard.lock}" bash -c '
  export ROCM_GPU_GUARD_LOCK_HELD=$$
  echo "lock held ($(date -u +%FT%TZ))"
  REPS=3

  echo "round,arm,keys,ms_per_call,tflops" > "$OUT/probe.csv"
  echo "round,arm,case,prefill_tok_s,decode_tok_s,status" > "$OUT/rounds.csv"
  echo "round,arm,decode_tok_s_per_request,occupancy,mean_decode_step_ms,status" > "$OUT/server.csv"

  # Probe + bench-decode rounds, arm order rotating by one per round.
  for r in 1 2 $REPS; do
    if [[ $((r % 2)) == 1 ]]; then order=(nofix fix); else order=(fix nofix); fi
    for arm in "${order[@]}"; do
      bin="$NOFIX"; [[ "$arm" == fix ]] && bin="$FIX"
      bash "$OUT/harness/run_bench_round.sh" "$OUT" "$arm" "$r" "$bin" "$MODELS_DIR" "$GUARD" "$REPS"
    done
  done

  # Batch-4 ~16K server decode, one fresh server per arm, REPS repetitions.
  for arm in nofix fix; do
    bin="$NOFIX"; [[ "$arm" == fix ]] && bin="$FIX"
    bash "$OUT/harness/run_server.sh" "$OUT" "$arm" "$REPS" "$bin" "$MODELS_DIR" "$ROOT/scripts"
  done

  echo "--- probe.csv ---"; cat "$OUT/probe.csv"
  echo "--- rounds.csv ---"; cat "$OUT/rounds.csv"
  echo "--- server.csv ---"; cat "$OUT/server.csv"
'
echo "lock released ($(date -u +%FT%TZ))"
