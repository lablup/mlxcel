#!/usr/bin/env bash
# Issue #2232 measurement session (run under scripts/rocm_gpu_guard.sh --hold).
# Before/after f16 decode at pp2048/tg128 on Meta-Llama-3.1-8B-Instruct-4bit
# (3 rounds, arms alternated, each run in its own idle-guarded window), the
# cache counters with the cache forced on, and the #2156 decode-like counter
# check (batch-4 forward_batched at decode-like row counts, short and long
# decode). One release binary: the before arm sets
# MLX_ROCM_QMM_DEQUANT_CACHE_SIZE=8, the after arm leaves the default (0, off)
# in place.
# Usage: session.sh REPO_ROOT OUT_DIR MODELS_DIR
set -uo pipefail
ROOT="$1"; OUT="$2"; MODELS_DIR="$3"
GUARD="$ROOT/scripts/rocm_gpu_guard.sh"
mkdir -p "$OUT/raw"
echo "session start $(date -u +%FT%TZ)"
MODEL="$MODELS_DIR/Meta-Llama-3.1-8B-Instruct-4bit"
BIN="$ROOT/target/release/mlxcel-bench-decode"
PBD="$ROOT/target/release/examples/profile_batched_decode"

CSV="$OUT/decode.csv"
echo "round,arm,prefill_tok_s,decode_tok_s,peak_gb,active_after_prefill_gb,active_end_gb,stats,status" > "$CSV"

# Extracts, from one bench log, the measured prefill and decode tok/s, the MLX
# peak, the active memory after the warmup prefill and after the measured
# pass, and the exit counter line, pipe-separated.
summarize() {
  local log="$1"
  local p d peak ap ae stats
  p=$(sed -n "s/.*Prefill:.*(\([0-9.]*\) tok\/s).*/\1/p" "$log" | head -1)
  d=$(sed -n "s/.*Decode:.*(\([0-9.]*\) tok\/s).*/\1/p" "$log" | head -1)
  peak=$(sed -n "s/.*MLX peak memory:  *\([0-9.]*\) GB.*/\1/p" "$log" | head -1)
  ap=$(sed -n "s/.*after warmup (prefill + warmup tokens): active \([0-9.]*\) GB.*/\1/p" "$log" | head -1)
  ae=$(sed -n "s/.*after measured pass: active \([0-9.]*\) GB.*/\1/p" "$log" | head -1)
  stats=$(grep -o "qmm dequant cache: .*" "$log" | head -1)
  echo "${p:-} | ${d:-} | ${peak:-} | ${ap:-} | ${ae:-} | ${stats:-none}"
}

# One idle-guarded bench run per arm; echoes "<rc> | <summarize>".
bench_run() {
  local arm="$1" round="$2" size="$3"
  local log="$OUT/raw/pp${size}_r${round}_${arm}.log"
  if [[ "$arm" == before ]]; then
    $GUARD --idle-secs 60 -- \
      env MLX_ROCM_QMM_DEQUANT_CACHE_SIZE=8 MLX_ROCM_QMM_DEQUANT_CACHE_STATS=1 \
      "$BIN" -m "$MODEL" -p x -n 128 --warmup-tokens 20 --ignore-eos --prompt-tokens "$size" \
      > "$log" 2>&1
  else
    $GUARD --idle-secs 60 -- \
      env MLX_ROCM_QMM_DEQUANT_CACHE_STATS=1 \
      "$BIN" -m "$MODEL" -p x -n 128 --warmup-tokens 20 --ignore-eos --prompt-tokens "$size" \
      > "$log" 2>&1
  fi
  local rc=$?
  echo "$rc | $(summarize "$log")"
}

# Before/after decode at the issue's evidence shape (pp2048/tg128), 3 rounds,
# arms alternated per round.
for r in 1 2 3; do
  if (( r % 2 == 1 )); then arms=(before after); else arms=(after before); fi
  for arm in "${arms[@]}"; do
    out=$(bench_run "$arm" "$r" 2048)
    rc=${out%% | *}
    fields=${out#* | }
    p=$(echo "$fields" | cut -d'|' -f1 | xargs)
    d=$(echo "$fields" | cut -d'|' -f2 | xargs)
    peak=$(echo "$fields" | cut -d'|' -f3 | xargs)
    ap=$(echo "$fields" | cut -d'|' -f4 | xargs)
    ae=$(echo "$fields" | cut -d'|' -f5 | xargs)
    stats=$(echo "$fields" | cut -d'|' -f6- | xargs)
    echo "$r,$arm,${p:--},${d:--},${peak:--},${ap:--},${ae:--},\"$stats\",$rc" >> "$CSV"
    echo "$(date -u +%T) pp2048 r$r $arm rc=$rc p=$p d=$d peak=$peak ap=$ap ae=$ae"
    echo "  stats: $stats"
    sleep 3
  done
done
cat "$CSV"

# The cache counters with the cache forced on, short decode: every lookup the
# prefill runs, none from decode steps at batch 1.
log="$OUT/raw/pp512_short.log"
$GUARD --idle-secs 60 -- \
  env MLX_ROCM_QMM_DEQUANT_CACHE_SIZE=8 MLX_ROCM_QMM_DEQUANT_CACHE_STATS=1 \
  "$BIN" -m "$MODEL" -p x -n 32 --warmup-tokens 4 --ignore-eos --prompt-tokens 512 \
  > "$log" 2>&1
echo "$(date -u +%T) pp512 tg32 rc=$? $(summarize "$log")"

# #2156 decode-like case: batch-4 forward_batched (the server's decode step)
# at decode-like row counts, cache forced on, with a short and a long decode.
# Identical lookup counts in the two exit lines mean the decode loop touches
# no cache entry.
for steps in 5 50; do
  log="$OUT/raw/pbd_b4_steps${steps}.log"
  MLX_ROCM_QMM_DEQUANT_CACHE_SIZE=8 MLX_ROCM_QMM_DEQUANT_CACHE_STATS=1 \
    "$PBD" -m "$MODEL" --batch-sizes 4 --decode-steps "$steps" --warmup 2 \
    --runs 1 --prompt-len 899 > "$log" 2>&1
  echo "$(date -u +%T) pbd b4 steps=$steps rc=$? $(grep -o "qmm dequant cache: .*" "$log" | head -1)"
done

# Same at batch 1 for the single-stream control.
for steps in 5 50; do
  log="$OUT/raw/pbd_b1_steps${steps}.log"
  MLX_ROCM_QMM_DEQUANT_CACHE_SIZE=8 MLX_ROCM_QMM_DEQUANT_CACHE_STATS=1 \
    "$PBD" -m "$MODEL" --batch-sizes 1 --decode-steps "$steps" --warmup 2 \
    --runs 1 --prompt-len 899 > "$log" 2>&1
  echo "$(date -u +%T) pbd b1 steps=$steps rc=$? $(grep -o "qmm dequant cache: .*" "$log" | head -1)"
done
echo "session end $(date -u +%FT%TZ)"
