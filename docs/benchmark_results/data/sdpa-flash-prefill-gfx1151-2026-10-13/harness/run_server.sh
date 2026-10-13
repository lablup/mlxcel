#!/usr/bin/env bash
# Batch-4 ~16K server decode for issue #2251 on gfx1151, one arm: one fresh
# mlxcel-server, a discarded warmup level, then REPS runs of
# scripts/bench_serving_concurrency.py at concurrency 4, 16,384-token prompts.
# Usage: run_server.sh OUT_DIR ARM REPS BIN_ROOT MODELS_DIR SCRIPTS_DIR
#   BIN_ROOT: directory holding target/release/mlxcel-server for this arm.
#   SCRIPTS_DIR: the repository scripts/ directory (load generator).
# Writes raw outputs to OUT_DIR/raw/ and appends rows to OUT_DIR/server.csv
# (round,arm,decode_tok_s_per_request,occupancy,mean_decode_step_ms,status).
# Runs inside one scripts/rocm_gpu_guard.sh per level; the caller holds the
# host-wide flock (ROCM_GPU_GUARD_LOCK_HELD names the enclosing shell).
set -uo pipefail
OUT="$1"; ARM="$2"; REPS="${3:-3}"; BIN="$4"; MODELS_DIR="$5"; SCRIPTS_DIR="$6"
GUARD="${GUARD:?set GUARD to the repository scripts/rocm_gpu_guard.sh}"
mkdir -p "$OUT/raw"
SERVER="$BIN/mlxcel-server"
PORT="${PORT:-18997}"

mkdir -p "$OUT"
LOG="$OUT/raw/server_${ARM}.log"
"$SERVER" --model "$MODELS_DIR/Meta-Llama-3.1-8B-Instruct-4bit" --host 127.0.0.1 --port "$PORT" \
  --parallel 4 --ctx-size 131072 --no-prompt-cache --metrics > "$LOG" 2>&1 &
SERVER_PID=$!
echo "server pid $SERVER_PID arm $ARM" > "$OUT/raw/pid_${ARM}.txt"
stop_server() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -INT "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -0 "$SERVER_PID" 2>/dev/null && kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap stop_server EXIT
ok=0
for _ in $(seq 1 600); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
  kill -0 "$SERVER_PID" 2>/dev/null || break
  sleep 1
done
[[ $ok == 1 ]] || { echo "server not healthy" >&2; exit 1; }

H="python3 $SCRIPTS_DIR/bench_serving_concurrency.py --port $PORT --max-tokens 128 --metrics"
# warmup (discarded)
"$GUARD" --idle-secs 10 --max-attempts 10 --log "$OUT/raw/guard_${ARM}_warmup.log" -- \
  $H --concurrency 1 --prompt-tokens 64 > "$OUT/raw/${ARM}_warmup.txt" 2>&1 || true
"$GUARD" --idle-secs 10 --max-attempts 10 --log "$OUT/raw/guard_${ARM}_warmup2.log" -- \
  $H --concurrency 4 --prompt-tokens 256 >> "$OUT/raw/${ARM}_warmup.txt" 2>&1 || true

for rep in $(seq 1 "$REPS"); do
  log="$OUT/raw/${ARM}_c4_p16384_r${rep}.txt"
  "$GUARD" --idle-secs 10 --max-attempts 10 --log "$OUT/raw/guard_${ARM}_r${rep}.log" -- \
    $H --concurrency 4 --prompt-tokens 16384 > "$log" 2>&1
  rc=$?
  # final table row: conc ok fail ttft_mean ttft_p95 decode_tok_s(mean) aggregate
  tps=$(grep -A3 '^----' "$log" | grep -E '^ +4 ' | awk '{print $6}' | head -1)
  counters=$(grep -oE 'decode_steps=[+-]?[0-9]+|decode_tokens=[+-]?[0-9]+|tokens_predicted_seconds=[+-]?[0-9.]+' "$log" | tail -3 | tr '\n' ' ')
  steps=$(sed -n "s/.*decode_steps=\([+-][0-9]*\).*/\1/p" <<< "$counters" | tail -1)
  tokens=$(sed -n "s/.*decode_tokens=\([+-][0-9]*\).*/\1/p" <<< "$counters" | tail -1)
  psec=$(sed -n "s/.*tokens_predicted_seconds=\([+-][0-9.]*\).*/\1/p" <<< "$counters" | tail -1)
  occupancy=""
  stepms=""
  if [[ -n "${steps:-}" && -n "${tokens:-}" && "$steps" != 0 ]]; then
    occupancy=$(python3 -c "print(f'{$tokens/$steps:.2f}')")
  fi
  if [[ -n "${tokens:-}" && -n "${psec:-}" && "$tokens" != 0 ]]; then
    stepms=$(python3 -c "print(f'{$psec/$tokens*1000:.1f}')")
  fi
  echo "$rep,$ARM,${tps:-},occupancy=${occupancy:-},step_ms=${stepms:-},$([[ $rc -eq 0 ]] && echo ok || echo rc=$rc)" >> "$OUT/server.csv"
done
stop_server
trap - EXIT
