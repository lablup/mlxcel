#!/usr/bin/env bash
# Interleaved pre-epic vs post-epic `mlxcel-bench-decode` rounds on gfx1151
# (issue #2192). Shape: scripts/bench_decode.sh defaults (pp512 or the given
# prompt length, tg128, 20-token same-process warmup, --ignore-eos, greedy).
#
# Usage:
#   bench_rounds.sh OUT_DIR ROUNDS BASE_BIN NEW_BIN MODELS_DIR model[:prompt_tokens]...
#
# Arms per round: base (BASE_BIN), new (NEW_BIN), base-null (BASE_BIN again,
# the method's noise floor). The arm order rotates by one position each round.
# Every run is a fresh process. Raw output goes to OUT_DIR/raw/, one CSV row
# per run to OUT_DIR/rounds.csv. With GUARD set to the path of
# scripts/rocm_gpu_guard.sh, every run is its own guarded command (idle wait
# GUARD_IDLE_SECS, default 60; GUARD_ATTEMPTS, default 10) and its guard log
# lands in OUT_DIR/guard/; a run whose guard gave up (exit 75) is recorded with
# status guard_contended and its numbers are not used. A run already in
# rounds.csv with status ok is skipped, so an interrupted session resumes.
set -euo pipefail

OUT="$1"; ROUNDS="$2"; BASE_BIN="$3"; NEW_BIN="$4"; MODELS_DIR="$5"; shift 5
mkdir -p "$OUT/raw"
CSV="$OUT/rounds.csv"
[[ -f "$CSV" ]] || echo "model,prompt_tokens,round,arm,prompt_tokens_actual,generated,prefill_ms,prefill_tok_s,decode_ms,decode_tok_s,peak_gb,decode_path,status" > "$CSV"
COOLDOWN="${COOLDOWN:-10}"
GUARD="${GUARD:-}"
GUARD_IDLE_SECS="${GUARD_IDLE_SECS:-60}"
GUARD_ATTEMPTS="${GUARD_ATTEMPTS:-10}"
[[ -z "$GUARD" ]] || mkdir -p "$OUT/guard"
ARMS=(base new base-null)

bin_for() {
  case "$1" in
    base|base-null) echo "$BASE_BIN" ;;
    new) echo "$NEW_BIN" ;;
  esac
}

run_one() {
  local model="$1" ptok="$2" round="$3" arm="$4"
  if grep -q "^$model,$ptok,$round,$arm,.*,ok\$" "$CSV"; then
    >&2 echo ">>> round $round $arm $model pp$ptok already recorded, skipping"
    return 0
  fi
  local bin; bin=$(bin_for "$arm")
  local log="$OUT/raw/${model}_pp${ptok}_r${round}_${arm}.log"
  local status=ok
  >&2 echo ">>> round $round $arm $model pp$ptok ($(date -u +%H:%M:%S))"
  local -a guard=()
  [[ -z "$GUARD" ]] || guard=("$GUARD" --idle-secs "$GUARD_IDLE_SECS" --max-attempts "$GUARD_ATTEMPTS" --log "$OUT/guard/${model}_pp${ptok}_r${round}_${arm}.log" --)
  "${guard[@]}" "$bin" -m "$MODELS_DIR/$model" -p x -n 128 --warmup-tokens 20 --ignore-eos --prompt-tokens "$ptok" > "$log" 2>&1
  local rc=$?
  if [[ $rc -eq 75 ]]; then status=guard_contended; elif [[ $rc -ne 0 ]]; then status=fail; fi
  local prefill_ms prefill_tps decode_ms decode_tps peak path ptok_actual gen
  prefill_ms=$(sed -n 's/.*Prefill:[[:space:]]*\([0-9.]*\) ms.*/\1/p' "$log" | head -1)
  prefill_tps=$(sed -n 's/.*Prefill:.*(\([0-9.]*\) tok\/s).*/\1/p' "$log" | head -1)
  decode_ms=$(sed -n 's/.*Decode:[[:space:]]*\([0-9.]*\) ms.*/\1/p' "$log" | head -1)
  decode_tps=$(sed -n 's/.*Decode:.*(\([0-9.]*\) tok\/s).*/\1/p' "$log" | head -1)
  peak=$(sed -n 's/.*MLX peak memory:[[:space:]]*\([0-9.]*\) GB.*/\1/p' "$log" | head -1)
  path=$(sed -n 's/.*Decode path:[[:space:]]*\([a-z_-]*\).*/\1/p' "$log" | head -1)
  ptok_actual=$(sed -n 's/.*Prompt tokens:[[:space:]]*\([0-9]*\).*/\1/p' "$log" | head -1)
  gen=$(sed -n 's/.*Generated tokens:[[:space:]]*\([0-9]*\).*/\1/p' "$log" | head -1)
  [[ -n "$decode_tps" ]] || status="${status}:no_output"
  echo "$model,$ptok,$round,$arm,${ptok_actual},${gen},${prefill_ms},${prefill_tps},${decode_ms},${decode_tps},${peak},${path},$status" >> "$CSV"
  sleep "$COOLDOWN"
}

for ((r = 1; r <= ROUNDS; r++)); do
  shift_by=$(( (r - 1) % ${#ARMS[@]} ))
  order=("${ARMS[@]:$shift_by}" "${ARMS[@]:0:$shift_by}")
  for spec in "$@"; do
    model="${spec%%:*}"
    ptok=512
    [[ "$spec" == *:* ]] && ptok="${spec##*:}"
    for arm in "${order[@]}"; do
      run_one "$model" "$ptok" "$r" "$arm"
    done
  done
done
echo "done: $CSV"
