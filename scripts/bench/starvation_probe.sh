#!/bin/sh
# Does a chunked prefill make progress while decode streams are running?
#
# Issue #908 assumed tick alternation blocks decode. The implementation found
# the reverse: decide_action returned Decode while any sequence is active, so the
# prefill is what starved. This probe measures that directly by sampling
# counters rather than latencies, which makes it robust to the machine being
# busy with unrelated work.
#
#   scripts/bench/starvation_probe.sh baseline
#   PORT=18994 scripts/bench/starvation_probe.sh mixed
#
# GRANT sets --prefill-grant-interval (issue #1011), the fairness dial that
# bounds how long the parked prefill yields. GRANT=0 disables the grant and
# reproduces the pre-#1011 starvation; unset leaves the shipped default:
#
#   GRANT=0 scripts/bench/starvation_probe.sh baseline     # starved arm
#   scripts/bench/starvation_probe.sh baseline             # shipped default
#
# Read `prefill_grants` as the dispatch proof for the fairness policy the same
# way `mixed_steps` is for the #908 prototype: it can only move when the grant
# fired, so a flat column means the arm did not engage.
#
# Results: docs/benchmark_results/mixed-step-prototype-m1ultra-2026-08-03.md
#          docs/benchmark_results/prefill-fairness-m1ultra-2026-08-03.md
#
# Every exit path kills the server. Do not add `wait` on the background curls:
# an earlier version did and hung for eight hours after a stream stalled.

set -u

ARM=${1:-baseline}          # baseline | mixed
PORT=${PORT:-18993}
MODEL=${MODEL:-$HOME/.cache/mlxcel/models/mlx-community/Meta-Llama-3.1-8B-Instruct-4bit}
SERVER=${SERVER:-./target/release/mlxcel-server}
PARALLEL=${PARALLEL:-8}
CHUNK=${CHUNK:-512}
SAMPLES=${SAMPLES:-12}      # one every 5s
GRANT=${GRANT:-}            # --prefill-grant-interval; empty = shipped default

GRANT_ARGS=""
[ -n "$GRANT" ] && GRANT_ARGS="--prefill-grant-interval $GRANT"

case "$ARM" in
  baseline|mixed) ;;
  *) echo "usage: $0 [baseline|mixed]" >&2; exit 2 ;;
esac
[ -x "$SERVER" ] || { echo "no server binary at $SERVER (cargo build --release)" >&2; exit 2; }
[ -d "$MODEL" ] || { echo "no model at $MODEL (set MODEL=)" >&2; exit 2; }

SRV=""
cleanup() { [ -n "$SRV" ] && kill -9 "$SRV" 2>/dev/null; }
trap cleanup EXIT INT TERM

# shellcheck disable=SC2086  # GRANT_ARGS is an intentional word split
if [ "$ARM" = mixed ]; then
  MLXCEL_MIXED_STEP=1 "$SERVER" -m "$MODEL" --parallel "$PARALLEL" $GRANT_ARGS \
    --prefill-chunk-size "$CHUNK" --metrics --port "$PORT" >"/tmp/starv_$ARM.log" 2>&1 &
else
  "$SERVER" -m "$MODEL" --parallel "$PARALLEL" $GRANT_ARGS \
    --prefill-chunk-size "$CHUNK" --metrics --port "$PORT" >"/tmp/starv_$ARM.log" 2>&1 &
fi
SRV=$!

i=0
while [ "$i" -lt 90 ]; do
  sleep 2
  curl -s --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  i=$((i + 1))
done
curl -s --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || {
  echo "server failed to start; see /tmp/starv_$ARM.log" >&2; exit 1; }

counter() {
  curl -s --max-time 5 "http://127.0.0.1:$PORT/metrics" 2>/dev/null \
    | awk -v k="^mlxcel_batch_$1 " '$0 ~ k {print $2}'
}

# Four long decode streams, detached.
for _ in 1 2 3 4; do
  curl -s --max-time 180 -X POST "http://127.0.0.1:$PORT/v1/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"m","prompt":"Count slowly, describing each number.","max_tokens":400,"temperature":0}' \
    >/dev/null 2>&1 &
done
sleep 6

# Admit one long prompt while those four are decoding.
BODY=$(python3 -c "import json; print(json.dumps({'model':'m','max_tokens':8,'temperature':0,'prompt':'The quick brown fox jumps over the lazy dog. ' * 950}))")
curl -s --max-time 180 -X POST "http://127.0.0.1:$PORT/v1/completions" \
  -H 'Content-Type: application/json' -d "$BODY" >/dev/null 2>&1 &

# If the prefill is starved, chunks stops advancing while decode keeps climbing.
echo "arm=$ARM grant=${GRANT:-default}   t  prefill_chunks  prefill_grants  mixed_steps  decode_steps"
t=1
while [ "$t" -le "$SAMPLES" ]; do
  sleep 5
  printf "         %5ss  %13s  %14s  %11s  %12s\n" "$((t * 5))" \
    "$(counter prefill_chunks_total)" "$(counter prefill_grants_total)" \
    "$(counter mixed_steps_total)" "$(counter decode_steps_total)"
  t=$((t + 1))
done
