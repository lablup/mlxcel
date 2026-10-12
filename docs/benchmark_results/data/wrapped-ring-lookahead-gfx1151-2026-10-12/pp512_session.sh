#!/usr/bin/env bash
# Follow-up measurement for issue #2239 on gfx1151: the pp512 cell of the
# #2192 session (inside the 1024-token sliding window, so the wrapped-window
# fallback does not run) on the fixed main build, to confirm the cell stays
# within ADR 0007's 1.0 percent threshold after the change. 3 interleaved
# rounds of three arms, arm order rotating per round.
#   base      the pre-epic CxxGenerator binary, built from a git archive of
#             4c44e317 with the same toolchain
#   new       the fixed main build
#   base-null the pre-epic binary again, the method's noise floor
# Usage: pp512_session.sh REPO_ROOT PRE_EPIC_ROOT OUT_DIR MODELS_DIR
set -uo pipefail
ROOT="$1"; PRE="$2"; OUT="$3"; MODELS_DIR="$4"
GUARD="${GUARD:-$ROOT/scripts/rocm_gpu_guard.sh}"
LOCK="${ROCM_GPU_GUARD_LOCK:-/tmp/mlxcel-rocm-gpu-guard.lock}"
mkdir -p "$OUT/guard" "$OUT/raw"
export ROOT PRE OUT MODELS_DIR GUARD
echo "waiting for $LOCK ($(date -u +%FT%TZ))"
flock "$LOCK" bash -c '
  export ROCM_GPU_GUARD_LOCK_HELD=$$
  echo "lock held ($(date -u +%FT%TZ))"
  g() { "$GUARD" --idle-secs 10 --max-attempts 10 --log "$OUT/guard/$1.log" -- "${@:2}"; }

  CSV="$OUT/rounds.csv"
  echo "round,arm,prefill_tok_s,decode_tok_s,status" > "$CSV"
  ARMS=(base new base-null)
  for r in 1 2 3; do
    shift_by=$(( (r - 1) % 3 ))
    order=("${ARMS[@]:$shift_by}" "${ARMS[@]:0:$shift_by}")
    for arm in "${order[@]}"; do
      bin="$ROOT/target/release/mlxcel-bench-decode"; envs=()
      [[ "$arm" == base* ]] && bin="$PRE/target/release/mlxcel-bench-decode"
      log="$OUT/raw/gemma3_pp512_r${r}_${arm}.log"
      g "gemma3_pp512_r${r}_${arm}" env "${envs[@]}" "$bin" -m "$MODELS_DIR/gemma-3-4b-it-4bit" -p x -n 128 --warmup-tokens 20 --ignore-eos --prompt-tokens 512 > "$log" 2>&1
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
