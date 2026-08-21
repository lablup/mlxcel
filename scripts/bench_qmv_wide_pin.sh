#!/usr/bin/env bash
# Measure the collateral cost of the process-wide qmv_wide narrow pin
# (issue #1261) on a generation 15+ host.
#
# Two modes, one per arm of the issue's Step 1:
#
#   ./scripts/bench_qmv_wide_pin.sh sweep    # batched-decode B-sweep, no drafter
#   ./scripts/bench_qmv_wide_pin.sh mixed    # one MTP stream + N classic streams
#
# sweep: boots mlxcel-server on the batch-capable target WITHOUT a drafter,
# alternating MLXCEL_QMV_WIDE=1 (wide) and =0 (narrow) boots in ABBA order.
# Pinning the env in BOTH arms is what keeps them on different kernels: any
# value of MLXCEL_QMV_WIDE counts as an operator pin
# (`qmv_wide_pinned_by_operator` in src/models/speculative_exactness.rs), so
# the exactness gate's retry can never flip an arm mid-run, and with no
# drafter loaded the gate has no probe to run in the first place. Each boot
# runs one discarded warm-up pass and MEASURE_PASSES measured passes; a pass
# is the B = 1,2,4,8 ladder plus a long-context B=4 cell.
#
# mixed: boots a target WITH its drafter, alternating the default env
# (arm A: the gate's retry pins the process narrow) and
# MLXCEL_QMV_WIDE=1 MLXCEL_MTP_ALLOW_INEXACT=1 (arm B: the operator pin skips
# the retry and the override engages MTP on the wide kernel). Note the arm B
# recipe needs BOTH variables: MLXCEL_MTP_ALLOW_INEXACT=1 alone does not
# leave the switch wide, because the gate's retry runs before the override is
# consulted and pins the process narrow itself. The pairing must be one whose
# narrow retry actually passes on the host, or arm A has no pin to price; on
# M3 Ultra that is the Qwen pairing, and the 2026-08-22 run used
#
#   TARGET=models/qwen3.8-27b-4bit DRAFTER=models/qwen3.8-27b-mtp-4bit \
#   DRAFT_BLOCK=3 ./scripts/bench_qmv_wide_pin.sh mixed
#
# (the default Gemma 31B + bf16 pairing probes non-identical under BOTH
# kernels there and declines MTP; see
# docs/benchmark_results/qmv-wide-pin-tax-m3ultra-2026-08-22.md).
# Every boot also runs with
# MLXCEL_MTP_SLICE_GRANT_ROUNDS=0 so the first stream holds the speculative
# slot for its whole generation and the N concurrent streams fall back to
# classic decode (the pre-#746 behaviour), which is what makes "one MTP
# stream plus N classic streams" a deterministic shape. After each boot the
# server log is grepped for the exactness-gate lines so the arm identity is
# evidenced, not assumed.
#
# Protocol (docs/benchmarks.md): ABBA boot order against thermal drift,
# warm-up passes discarded, spreads reported by the analysis step, and the
# whole invocation belongs under scripts/with_indexers_paused.sh:
#
#   INDEXER_RESUME_DEADLINE=7200 ./scripts/with_indexers_paused.sh \
#       ./scripts/bench_qmv_wide_pin.sh sweep
#
# Output: one directory per invocation under bench-results/qmv-wide-pin/,
# holding the raw per-pass tables, the server logs, the gate-line evidence
# and an environment record. Nothing is aggregated here; aggregation belongs
# to the write-up so that discarded samples stay visible.

set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO/target/release/mlxcel-server"
MODE="${1:-}"
if [ "$MODE" != "sweep" ] && [ "$MODE" != "mixed" ]; then
  echo "usage: $0 sweep|mixed" >&2
  exit 2
fi

TARGET="${TARGET:-$REPO/models/gemma-4-31b-it-4bit}"
DRAFTER="${DRAFTER:-$REPO/models/gemma-4-31b-it-assistant-bf16}"
DRAFT_BLOCK="${DRAFT_BLOCK:-4}"
PORT="${PORT:-8113}"
PARALLEL="${PARALLEL:-8}"
MEASURE_PASSES="${MEASURE_PASSES:-2}"
# ABBA blocks; sweep arms are qmv_wide pins, mixed arms are gate recipes.
SWEEP_ARMS="${SWEEP_ARMS:-1 0 0 1 0 1 1 0}"
MIXED_ARMS="${MIXED_ARMS:-A B B A}"
CLASSIC_STREAMS="${CLASSIC_STREAMS:-4}"

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="${OUT:-$REPO/bench-results/qmv-wide-pin/$MODE-$STAMP}"
mkdir -p "$OUT"

SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
  fi
}
trap cleanup EXIT INT TERM

record_env() {
  {
    echo "mode: $MODE"
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host: $(sysctl -n machdep.cpu.brand_string), $(sysctl -n hw.memsize | awk '{print $1/1073741824 " GB"}')"
    echo "macos: $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
    echo "branch: $(git -C "$REPO" rev-parse --abbrev-ref HEAD) at $(git -C "$REPO" rev-parse --short HEAD)"
    echo "binary: $BIN ($(stat -f %Sm "$BIN"))"
    echo "target: $TARGET"
    [ "$MODE" = "mixed" ] && echo "drafter: $DRAFTER"
    echo "parallel: $PARALLEL, port: $PORT, measured passes per boot: $MEASURE_PASSES"
    echo "arms: $([ "$MODE" = "sweep" ] && echo "$SWEEP_ARMS" || echo "$MIXED_ARMS")"
    echo "time machine running: $(tmutil status 2>/dev/null | grep Running | tr -cd '0-9')"
  } > "$OUT/env.txt"
}

wait_ready() {
  local deadline=$((SECONDS + 600))
  while [ $SECONDS -lt $deadline ]; do
    if curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    if [ -n "$SERVER_PID" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server died during startup; see its log" >&2
      return 1
    fi
    sleep 2
  done
  echo "server not ready after 600s" >&2
  return 1
}

stop_server() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
  fi
  SERVER_PID=""
  # Let the port close and the GPU settle between boots.
  sleep 5
}

run_sweep() {
  local boot=0 arm
  for arm in $SWEEP_ARMS; do
    boot=$((boot + 1))
    local label="boot${boot}-wide${arm}"
    echo "=== $label: MLXCEL_QMV_WIDE=$arm ==="
    MLXCEL_QMV_WIDE="$arm" "$BIN" -m "$TARGET" --port "$PORT" \
      --parallel "$PARALLEL" --metrics \
      > "$OUT/$label-server.log" 2>&1 &
    SERVER_PID=$!
    wait_ready || exit 1

    local pass
    for pass in $(seq 0 "$MEASURE_PASSES"); do
      local tag="$label-pass$pass"
      [ "$pass" = 0 ] && tag="$label-warmup"
      python3 "$REPO/scripts/bench_serving_concurrency.py" --port "$PORT" \
        --concurrency 1,2,4,8 --prompt-tokens 512 --max-tokens 256 --metrics \
        > "$OUT/$tag.txt" 2>&1
      python3 "$REPO/scripts/bench_serving_concurrency.py" --port "$PORT" \
        --concurrency 4 --prompt-tokens 4096 --max-tokens 256 --metrics \
        > "$OUT/$tag-long.txt" 2>&1
      echo "  pass $pass done"
    done
    stop_server
  done
}

run_mixed() {
  local boot=0 arm
  for arm in $MIXED_ARMS; do
    boot=$((boot + 1))
    local label="boot${boot}-arm${arm}"
    echo "=== $label ==="
    # Scheduler debug logging is on in BOTH arms (identical overhead) so the
    # "falls back to classic decode" lines evidence the N classic rows.
    local -a env_pairs=(
      MLXCEL_ENABLE_MTP_B1=1 MLXCEL_MTP_ADAPTIVE=0 MLXCEL_MTP_SLICE_GRANT_ROUNDS=0
      "RUST_LOG=info,mlxcel::server::batch::scheduler=debug"
    )
    if [ "$arm" = "B" ]; then
      env_pairs+=(MLXCEL_QMV_WIDE=1 MLXCEL_MTP_ALLOW_INEXACT=1)
    fi
    env "${env_pairs[@]}" "$BIN" -m "$TARGET" --model-draft "$DRAFTER" \
      --draft-block-size "$DRAFT_BLOCK" --port "$PORT" --parallel "$PARALLEL" --metrics \
      > "$OUT/$label-server.log" 2>&1 &
    SERVER_PID=$!
    wait_ready || exit 1

    # Window 0 is the warm-up (it also pays the one-time exactness probe);
    # the rest are the measured windows.
    python3 "$REPO/scripts/bench_qmv_pin_mixed.py" --port "$PORT" \
      --windows $((MEASURE_PASSES + 1)) --classic-streams "$CLASSIC_STREAMS" \
      > "$OUT/$label-windows.txt" 2>&1
    stop_server

    # Arm identity is evidenced by which exactness-gate line the boot logged.
    grep -E "exactness probe|qmv_wide|ALLOW_INEXACT|falls back to classic|falling back to classic|slot busy" \
      "$OUT/$label-server.log" > "$OUT/$label-gate.txt" || true
  done
}

record_env
if [ "$MODE" = "sweep" ]; then run_sweep; else run_mixed; fi
echo "results in $OUT"
