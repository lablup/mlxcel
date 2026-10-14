#!/usr/bin/env bash
# Session 6 for #2156: server-level kernel trace (launch mode, SIGINT stop) and a pre-fix 16K example check.
set -uo pipefail
WT=.
D=<meas>
OUT="$1"; mkdir -p "$OUT"; cd "$WT"
M=models/mlx/Meta-Llama-3.1-8B-Instruct-4bit
log() { echo "$1 $2 $(date -Is)" >> "$OUT/s.log"; }
$D/bin_before/profile_batched_decode -m $M --batch-sizes 1 --decode-steps 20 --warmup 3 --runs 1 --prompt-len 16384 > "$OUT/pbd_before_16k.txt" 2>&1; log pbd_before_16k $?
PORT=18997
rocprofv3 --kernel-trace -f csv -d "$OUT/trace_srv" -o srv -- target/release/mlxcel-server --model $M --host 127.0.0.1 --port $PORT \
  --parallel 4 --ctx-size 131072 --no-prompt-cache --metrics > "$OUT/server.log" 2>&1 &
SPID=$!
echo "server pid $SPID" > "$OUT/pid.txt"
ok=0; for _ in $(seq 1 600); do curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }; kill -0 $SPID 2>/dev/null || break; sleep 1; done
log health $ok
if [[ $ok == 1 ]]; then
  python3 $WT/scripts/bench_serving_concurrency.py --port $PORT --max-tokens 128 --concurrency 4 --prompt-tokens 256 > "$OUT/warmup.txt" 2>&1
  python3 $D/profile_driver.py $PORT "$OUT/profile.json" 1:1024 4:1024 1:16384 4:16384 > "$OUT/profile.txt" 2>&1; log driver $?
fi
kill -INT $SPID 2>/dev/null; for _ in $(seq 1 120); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
kill -0 $SPID 2>/dev/null && { kill -TERM $SPID; sleep 20; }
kill -0 $SPID 2>/dev/null && { kill -KILL $SPID; log server_killed 1; }
wait $SPID; log server_exit $?
