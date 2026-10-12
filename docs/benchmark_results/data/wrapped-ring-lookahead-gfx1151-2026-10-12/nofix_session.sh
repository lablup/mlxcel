#!/usr/bin/env bash
# Follow-up measurement for issue #2239 on gfx1151: the same gemma-3-4b-it-4bit
# pp2048 arms on the pre-fix main build (e24fb7b2, the origin/main tip the fix
# rebases on), to separate what the wrapped-window synchronous fallback
# recovers from what the commits merged into main since the Oct 8 session
# changed. 3 rounds of two arms, arm order rotating per round.
#   nofix      the e24fb7b2 binary, pipelined B=1 decode
#   nofix-sync the same binary with MLXCEL_FORCE_SYNC=1
# Every run is its own scripts/rocm_gpu_guard.sh command inside one host-wide
# lock hold (ROCM_GPU_GUARD_LOCK_HELD names the enclosing shell).
# Usage: nofix_session.sh REPO_ROOT NOFIX_ROOT OUT_DIR MODELS_DIR
set -uo pipefail
ROOT="$1"; NOFIX="$2"; OUT="$3"; MODELS_DIR="$4"
GUARD="${GUARD:-$ROOT/scripts/rocm_gpu_guard.sh}"
LOCK="${ROCM_GPU_GUARD_LOCK:-/tmp/mlxcel-rocm-gpu-guard.lock}"
mkdir -p "$OUT/guard" "$OUT/raw"
export ROOT NOFIX OUT MODELS_DIR GUARD
echo "waiting for $LOCK ($(date -u +%FT%TZ))"
flock "$LOCK" bash -c '
  export ROCM_GPU_GUARD_LOCK_HELD=$$
  echo "lock held ($(date -u +%FT%TZ))"
  g() { "$GUARD" --idle-secs 10 --max-attempts 10 --log "$OUT/guard/$1.log" -- "${@:2}"; }

  CSV="$OUT/rounds.csv"
  echo "round,arm,prefill_tok_s,decode_tok_s,status" > "$CSV"
  ARMS=(nofix nofix-sync)
  for r in 1 2 3; do
    shift_by=$(( (r - 1) % 2 ))
    order=("${ARMS[@]:$shift_by}" "${ARMS[@]:0:$shift_by}")
    for arm in "${order[@]}"; do
      bin="$NOFIX/target/release/mlxcel-bench-decode"; envs=()
      [[ "$arm" == "nofix-sync" ]] && envs=(MLXCEL_FORCE_SYNC=1)
      log="$OUT/raw/gemma3_pp2048_r${r}_${arm}.log"
      g "gemma3_pp2048_r${r}_${arm}" env "${envs[@]}" "$bin" -m "$MODELS_DIR/gemma-3-4b-it-4bit" -p x -n 128 --warmup-tokens 20 --ignore-eos --prompt-tokens 2048 > "$log" 2>&1
      rc=$?
      p=$(sed -n "s/.*Prefill:.*(\([0-9.]*\) tok\/s).*/\1/p" "$log" | head -1)
      d=$(sed -n "s/.*Decode:.*(\([0-9.]*\) tok\/s).*/\1/p" "$log" | head -1)
      echo "$r,$arm,$p,$d,$([[ $rc -eq 0 ]] && echo ok || echo rc=$rc)" >> "$CSV"
      sleep 5
    done
  done
  cat "$CSV"
'
echo "lock released ($(date -u +%FT%TZ))"
