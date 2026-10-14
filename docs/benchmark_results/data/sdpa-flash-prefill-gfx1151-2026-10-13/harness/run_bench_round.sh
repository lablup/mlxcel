#!/usr/bin/env bash
# One probe + bench-decode round for issue #2251 on gfx1151, one arm.
# Usage: run_bench_round.sh OUT_DIR ARM ROUND BIN_ROOT MODELS_DIR GUARD REPS
#   BIN_ROOT: directory holding target/release/{examples/sdpa_prefill_probe,
#             mlxcel-bench-decode} for this arm (before or after build).
# Writes raw logs to OUT_DIR/raw/ and appends CSV rows to
#   OUT_DIR/probe.csv   (round,arm,keys,ms_per_call,tflops)
#   OUT_DIR/rounds.csv  (round,arm,case,prefill_tok_s,decode_tok_s,status)
# Each run is its own scripts/rocm_gpu_guard.sh command; the caller holds the
# host-wide flock (ROCM_GPU_GUARD_LOCK_HELD names the enclosing shell).
set -uo pipefail
OUT="$1"; ARM="$2"; r="$3"; BIN="$4"; MODELS_DIR="$5"; GUARD="$6"
PROBE="$BIN/examples/sdpa_prefill_probe"
BENCH="$BIN/mlxcel-bench-decode"

g() { "$GUARD" --idle-secs 10 --max-attempts 10 --log "$OUT/raw/$1.log" -- "${@:2}"; }

log="$OUT/raw/probe_r${r}_${ARM}.log"
g "probe_r${r}_${ARM}" "$PROBE" --key-lens 2048,8192,16384 > "$log" 2>&1
# one row per key length; the probe prints its internal median of 5
awk -v arm="$ARM" -v r="$r" '/^ +[0-9]+ +[0-9.]+ +[0-9.]+ +[0-9.]+ +[0-9.]+$/ {
    print r "," arm "," $1 "," $2 "," $3 }' "$log" >> "$OUT/probe.csv"
sleep 5

for pt in 512 16384; do
  log="$OUT/raw/llama_pp${pt}_r${r}_${ARM}.log"
  g "llama_pp${pt}_r${r}_${ARM}" \
    "$BENCH" -m "$MODELS_DIR/Meta-Llama-3.1-8B-Instruct-4bit" -p profile -n 128 \
    --warmup-tokens 20 --ignore-eos --prompt-tokens "$pt" --temperature 0 --top-p 1.0 \
    > "$log" 2>&1
  rc=$?
  p=$(sed -n "s/.*Prefill:.*(\([0-9.]*\) tok\/s).*/\1/p" "$log" | head -1)
  d=$(sed -n "s/.*Decode:.*(\([0-9.]*\) tok\/s).*/\1/p" "$log" | head -1)
  echo "$r,$ARM,pp$pt,$p,$d,$([[ $rc -eq 0 ]] && echo ok || echo rc=$rc)" >> "$OUT/rounds.csv"
  sleep 5
done
