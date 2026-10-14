#!/usr/bin/env bash
# Measurement matrix for lablup/mlxcel#2156, run inside one rocm_gpu_guard.sh.
# Usage: run_matrix.sh OUT_DIR ARM [REPS] [CASES]
#   ARM: name of the server arm; extra server env comes from SERVER_ENV
#        (space-separated KEY=VALUE pairs).
#   CASES: space-separated conc:prompt pairs (default "1:1024 4:1024 1:16384 4:16384")
#   BENCH_DECODE=1 also runs single-stream mlxcel-bench-decode REPS times.
set -euo pipefail
WT=.
OUT="$1"; ARM="$2"; REPS="${3:-3}"; CASES="${4:-1:1024 4:1024 1:16384 4:16384}"
MODEL="$WT/models/mlx/Meta-Llama-3.1-8B-Instruct-4bit"
PORT="${PORT:-18996}"
SERVER_BIN="${SERVER_BIN:-$WT/target/release/mlxcel-server}"
mkdir -p "$OUT"
LOG="$OUT/server_${ARM}.log"
if [[ "${BENCH_DECODE:-0}" == 1 ]]; then
  for rep in $(seq 1 "$REPS"); do
    "${BENCH_BIN:-$WT/target/release/mlxcel-bench-decode}" -m "$MODEL" -p "profile" -n 128 --warmup-tokens 20 \
      --ignore-eos --prompt-tokens 512 --temperature 0 --top-p 1.0 > "$OUT/bench_decode_r${rep}.txt" 2>&1
  done
  echo done > "$OUT/DONE_${ARM}"
  exit 0
fi
read -r -a extra_env <<< "${SERVER_ENV:-}"
env ${extra_env[@]+"${extra_env[@]}"} "$SERVER_BIN" --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --parallel 4 --ctx-size 131072 --no-prompt-cache --metrics > "$LOG" 2>&1 &
SERVER_PID=$!
echo "server pid $SERVER_PID arm $ARM" | tee "$OUT/pid_${ARM}.txt"
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
[[ $ok == 1 ]] || { echo "server not healthy"; exit 1; }
H="python3 $WT/scripts/bench_serving_concurrency.py --port $PORT --max-tokens 128 --metrics"
# warmup (discarded)
$H --concurrency 1 --prompt-tokens 64 > "$OUT/${ARM}_warmup.txt" 2>&1
$H --concurrency 4 --prompt-tokens 256 >> "$OUT/${ARM}_warmup.txt" 2>&1
if [[ "${PROFILE:-0}" == 1 ]]; then
  rocprofv3 --attach "$SERVER_PID" --attach-duration-msec "${ATTACH_MS:-420000}" --attach-sync-output \
    --kernel-trace --stats -f csv -d "$OUT/trace_${ARM}" -o srv < /dev/null > "$OUT/rocprof_${ARM}.log" 2>&1 &
  PROF_PID=$!
  echo "rocprofv3 pid $PROF_PID" >> "$OUT/pid_${ARM}.txt"
  sleep 20
  python3 "$(dirname "$0")/profile_driver.py" "$PORT" "$OUT/profile_${ARM}.json" $CASES > "$OUT/profile_${ARM}.txt" 2>&1
  date +%s%N > "$OUT/profile_${ARM}_driver_end_ns"
  wait "$PROF_PID" || echo "rocprofv3 exit $?" >> "$OUT/rocprof_${ARM}.log"
  REPS=0
fi
for rep in $(seq 1 "$REPS"); do
  for c in $CASES; do
    conc="${c%%:*}"; pt="${c##*:}"
    $H --concurrency "$conc" --prompt-tokens "$pt" > "$OUT/${ARM}_c${conc}_p${pt}_r${rep}.txt" 2>&1
  done
done
stop_server
trap - EXIT
echo done > "$OUT/DONE_${ARM}"
