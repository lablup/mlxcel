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
# Seconds a server gets to honour SIGTERM before the harness escalates to SIGKILL.
SHUTDOWN_GRACE_S="${SHUTDOWN_GRACE_S:-60}"

# Preflight the binary. Without it the first symptom is record_env's stat
# failing, then every boot failing in turn, and the operator has to read a
# server log to find out that nothing was ever built.
if [ ! -x "$BIN" ]; then
  echo "server binary not found at $BIN; build with:" >&2
  echo "  cargo build --release --features metal,accelerate" >&2
  exit 2
fi

# The arms are defined by the environment, so an inherited value silently
# redefines them: run_mixed's arm A is "the default env" and run_sweep assumes
# these two are the only things that differ between boots. Testing a gate
# recipe by hand is exactly how this measurement's own recipes were checked
# (see the results record), and an export left behind by that would make both
# arms the same arm while the run still reports success.
if [ -n "${MLXCEL_QMV_WIDE+set}" ] || [ -n "${MLXCEL_MTP_ALLOW_INEXACT+set}" ]; then
  echo "the arms are defined by MLXCEL_QMV_WIDE and MLXCEL_MTP_ALLOW_INEXACT, but this shell already exports:" >&2
  echo "  MLXCEL_QMV_WIDE=${MLXCEL_QMV_WIDE-<unset>}  MLXCEL_MTP_ALLOW_INEXACT=${MLXCEL_MTP_ALLOW_INEXACT-<unset>}" >&2
  echo "an inherited value collapses both arms onto one kernel; unset them and rerun" >&2
  exit 2
fi

# Refuse to run against a server this script did not boot. If something already
# holds $PORT (typically a server leaked by an aborted run) every launch below
# dies on EADDRINUSE while wait_ready happily probes the survivor, so both arms
# measure the same kernel and the run still reports success. That silently
# destroys the only thing the ABBA design establishes, so abort instead.
if curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "something is already serving on port $PORT; stop it or run with PORT=<free port>" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="${OUT:-$REPO/bench-results/qmv-wide-pin/$MODE-$STAMP}"
# This script does not use `set -e`, so a failed mkdir would leave every
# redirect below failing silently while the servers still boot: hours of
# measurement discarded as it is produced.
mkdir -p "$OUT" || exit 1

SERVER_PID=""
# Counts python3 harness invocations that exited non-zero, so an unbalanced
# design cannot be mistaken for a complete one at aggregation time.
HARNESS_FAILURES=0

# Stop the current server and make sure it is really gone. SIGTERM first so
# the server can release the model cleanly; escalate to SIGKILL if it is
# wedged, because an unbounded wait here would hang the harness with tens of
# GB of unified memory still held. SERVER_PID is cleared before returning so
# a second call (signal handler, then the EXIT trap) can never signal a PID
# the OS has recycled.
kill_server() {
  local pid="$SERVER_PID"
  SERVER_PID=""
  [ -n "$pid" ] || return 0
  kill -0 "$pid" 2>/dev/null || { wait "$pid" 2>/dev/null; return 0; }
  kill -TERM "$pid" 2>/dev/null
  local waited=0
  while [ "$waited" -lt "$SHUTDOWN_GRACE_S" ] && kill -0 "$pid" 2>/dev/null; do
    sleep 1
    waited=$((waited + 1))
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "server $pid ignored SIGTERM after ${SHUTDOWN_GRACE_S}s; sending SIGKILL" >&2
    kill -KILL "$pid" 2>/dev/null
  fi
  wait "$pid" 2>/dev/null
}

cleanup() {
  kill_server
}
# The signal handlers exit rather than fall through. A bash handler that just
# returns resumes the script where the signal interrupted it, so a Ctrl-C part
# way through an ABBA run would kill the current server and then boot the next
# arm instead of aborting. HUP is trapped for the same reason an SSH drop or a
# closed terminal must not leave a 31B server holding tens of GB of unified
# memory.
trap cleanup EXIT
trap 'cleanup; echo "Interrupted (signal received)" >&2; exit 130' INT TERM HUP

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
    # Recorded rather than assumed: any of these inherited from the operator's
    # shell reaches every boot, so the record has to show what the arms
    # actually ran under.
    echo "inherited env: $(env | grep -E '^(MLXCEL_|MLX_|LLAMA_ARG_)' | sort | tr '\n' ' ')"
  } > "$OUT/env.txt"
}

wait_ready() {
  local deadline=$((SECONDS + 600))
  while [ $SECONDS -lt $deadline ]; do
    # Liveness before readiness. A curl success proves only that something is
    # answering on $PORT, not that it is the boot we just launched, so a boot
    # that died on EADDRINUSE against a stale server must be reported dead
    # here rather than silently measured as this arm.
    if [ -n "$SERVER_PID" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server died during startup; see the *-server.log files in $OUT" >&2
      return 1
    fi
    if curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  echo "server not ready after 600s" >&2
  return 1
}

stop_server() {
  kill_server
  # Let the port close and the GPU settle between boots.
  sleep 5
}

run_sweep() {
  local boot=0 arm rc
  for arm in $SWEEP_ARMS; do
    # SWEEP_ARMS is deliberately word-split, which also glob-expands, and each
    # token then lands in an output path below. Reject anything unexpected so a
    # typo or a stray `/` cannot write outside $OUT.
    case "$arm" in
      0|1) ;;
      *) echo "invalid SWEEP_ARMS entry '$arm' (expected 0 or 1)" >&2; exit 2 ;;
    esac
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
      # A harness that errors out leaves a traceback in its output file while
      # the ABBA loop marches on, which costs one arm a sample and is invisible
      # at aggregation time. Record the failure both in the file and in the
      # run-level counter so it cannot be missed.
      python3 "$REPO/scripts/bench_serving_concurrency.py" --port "$PORT" \
        --concurrency 1,2,4,8 --prompt-tokens 512 --max-tokens 256 --metrics \
        > "$OUT/$tag.txt" 2>&1
      rc=$?
      if [ "$rc" -ne 0 ]; then
        echo "  harness failed for $tag (rc=$rc)" >&2
        echo "HARNESS-FAILED rc=$rc" >> "$OUT/$tag.txt"
        HARNESS_FAILURES=$((HARNESS_FAILURES + 1))
      fi
      python3 "$REPO/scripts/bench_serving_concurrency.py" --port "$PORT" \
        --concurrency 4 --prompt-tokens 4096 --max-tokens 256 --metrics \
        > "$OUT/$tag-long.txt" 2>&1
      rc=$?
      if [ "$rc" -ne 0 ]; then
        echo "  harness failed for $tag-long (rc=$rc)" >&2
        echo "HARNESS-FAILED rc=$rc" >> "$OUT/$tag-long.txt"
        HARNESS_FAILURES=$((HARNESS_FAILURES + 1))
      fi
      echo "  pass $pass done"
    done
    stop_server
  done
}

run_mixed() {
  local boot=0 arm rc
  for arm in $MIXED_ARMS; do
    # MIXED_ARMS is deliberately word-split, which also glob-expands, and each
    # token then lands in an output path below. Reject anything unexpected so a
    # typo or a stray `/` cannot write outside $OUT.
    case "$arm" in
      A|B) ;;
      *) echo "invalid MIXED_ARMS entry '$arm' (expected A or B)" >&2; exit 2 ;;
    esac
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
    rc=$?
    # A non-zero exit means the harness gave up (no MTP token in time, or no
    # valid window), so this boot contributed nothing. Mark it in the file and
    # in the counter rather than letting the loop hide the lost sample.
    if [ "$rc" -ne 0 ]; then
      echo "  harness failed for $label (rc=$rc)" >&2
      echo "HARNESS-FAILED rc=$rc" >> "$OUT/$label-windows.txt"
      HARNESS_FAILURES=$((HARNESS_FAILURES + 1))
    fi
    stop_server

    # Arm identity is evidenced by which exactness-gate line the boot logged.
    grep -E "exactness probe|qmv_wide|ALLOW_INEXACT|falls back to classic|falling back to classic|slot busy" \
      "$OUT/$label-server.log" > "$OUT/$label-gate.txt" || true
  done
}

record_env
if [ "$MODE" = "sweep" ]; then run_sweep; else run_mixed; fi
echo "results in $OUT"
# run_sweep and run_mixed run in the current shell, not a subshell, so the
# counter they incremented is the one read here.
if [ "$HARNESS_FAILURES" -ne 0 ]; then
  echo "WARNING: $HARNESS_FAILURES harness invocation(s) failed; arms are not balanced, see the HARNESS-FAILED markers" >&2
fi
