#!/usr/bin/env bash
# Per-kernel ROCm decode profile of mlxcel-bench-decode (issue #2061).
#
# Usage:
#   scripts/rocm_decode_profile.sh [options] MODEL_DIR [MODEL_DIR...]
#
# For each model this runs, each under scripts/rocm_gpu_guard.sh (90 s of idle
# GPU and no compiler before, a 1 Hz monitor during, rerun on contention):
#
#   1. a plain `mlxcel-bench-decode` run, for tok/s without the profiler;
#   2. the same run under `rocprofv3 --kernel-trace --hip-graph-trace --stats
#      -f csv` with MLXCEL_BENCH_PHASE_MARKS=1;
#
# then scripts/rocm_decode_profile.py cuts the measured decode out of the
# kernel trace by the bench's phase marks and writes the per-kernel decode
# table and a summary. The shape is scripts/bench_decode.sh's default, pp512 /
# tg128 with a 20-token same-process warmup and --ignore-eos, greedy unless
# --temperature is given.
#
# Options:
#   --out DIR          output directory (default benchmarks/rocm_profiles/gfx1151_<commit>)
#   --trace-dir DIR    where full kernel traces go (default: a temp dir; they are
#                      tens to hundreds of MB and are not meant to be committed)
#   --temperature T    sampling temperature (default 0, greedy)
#   --top-p P          top-p (default 1.0, off)
#   --tag NAME         run name suffix (default greedy, or t<T>[-p<P>])
#   --no-plain         skip the unprofiled run (no overhead figure)
#   --idle-secs N      guard idle window (default 90)
#   --bench PATH       bench binary (default target/release/mlxcel-bench-decode)
#
# Build first: cargo build --release --features rocm --bin mlxcel-bench-decode
# Render the tables: python3 scripts/rocm_decode_profile.py report <out dir>

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="$ROOT/target/release/mlxcel-bench-decode"
ROCPROF="${ROCPROFV3:-$(command -v rocprofv3 || echo /opt/rocm/bin/rocprofv3)}"
OUT=""
TRACE_DIR=""
TEMPERATURE=0
TOP_P=1.0
TAG=""
PLAIN=1
IDLE_SECS=90
PROMPT_TOKENS=512
MAX_TOKENS=128
WARMUP_TOKENS=20

usage() { sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'; }

MODELS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)         OUT="$2"; shift 2 ;;
    --trace-dir)   TRACE_DIR="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --top-p)       TOP_P="$2"; shift 2 ;;
    --tag)         TAG="$2"; shift 2 ;;
    --no-plain)    PLAIN=0; shift ;;
    --idle-secs)   IDLE_SECS="$2"; shift 2 ;;
    --bench)       BENCH="$2"; shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    -*)            echo "unknown option $1" >&2; usage >&2; exit 2 ;;
    *)             MODELS+=("$1"); shift ;;
  esac
done
[[ ${#MODELS[@]} -gt 0 ]] || { usage >&2; exit 2; }
[[ -x "$BENCH" ]] || { echo "bench binary not found: $BENCH (build it first)" >&2; exit 2; }
[[ -x "$ROCPROF" ]] || { echo "rocprofv3 not found (set ROCPROFV3)" >&2; exit 2; }

COMMIT=$(git -C "$ROOT" rev-parse --short=8 HEAD)
if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no -- src Cargo.toml Cargo.lock)" ]]; then
  echo "note: the source tree differs from $COMMIT; record the diff with the results" >&2
fi
OUT="${OUT:-$ROOT/benchmarks/rocm_profiles/gfx1151_${COMMIT}}"
mkdir -p "$OUT"
if [[ -z "$TRACE_DIR" ]]; then
  TRACE_DIR=$(mktemp -d -t rocm-decode-trace.XXXXXX)
fi
mkdir -p "$TRACE_DIR"
if [[ -z "$TAG" ]]; then
  if [[ "$TEMPERATURE" == 0 || "$TEMPERATURE" == 0.0 ]]; then
    TAG=greedy
  else
    TAG="t${TEMPERATURE}"
    [[ "$TOP_P" == 1 || "$TOP_P" == 1.0 ]] || TAG="${TAG}-p${TOP_P}"
  fi
fi
GUARD_LOG="$OUT/guard.log"

# guard.log is published evidence; keep local paths out of it. Runs on every
# exit (guard exit 75, a rocprofv3 or cp failure, a signal), and replaces the
# paths literally so '#' and regex characters in them are harmless.
scrub_guard_log() {
  [[ -f "$GUARD_LOG" ]] || return 0
  ROOT_PFX="$ROOT/" TRACE_PFX="$TRACE_DIR" python3 - "$GUARD_LOG" <<'PY' || true
import os, sys
path = sys.argv[1]
text = open(path, errors="surrogateescape").read()
text = text.replace(os.environ["ROOT_PFX"], "").replace(os.environ["TRACE_PFX"], "<trace-dir>")
open(path, "w", errors="surrogateescape").write(text)
PY
}
trap scrub_guard_log EXIT

bench_args() {
  printf '%s\n' -m "$1" -p "profile" -n "$MAX_TOKENS" --warmup-tokens "$WARMUP_TOKENS" \
    --ignore-eos --prompt-tokens "$PROMPT_TOKENS" --temperature "$TEMPERATURE" --top-p "$TOP_P"
}

for model in "${MODELS[@]}"; do
  name="$(basename "$model")_${TAG}"
  mapfile -t args < <(bench_args "$model")
  echo ">>> $name" >&2

  run_dir="$TRACE_DIR/$name"
  mkdir -p "$run_dir"

  if [[ "$PLAIN" == 1 ]]; then
    echo "== $name plain" >> "$GUARD_LOG"
    "$ROOT/scripts/rocm_gpu_guard.sh" --idle-secs "$IDLE_SECS" --log "$GUARD_LOG" -- \
      "$BENCH" "${args[@]}" > "$run_dir/plain_bench.log" 2>&1
    # The guard's own lines are in guard.log already.
    grep -v '^rocm_gpu_guard: ' "$run_dir/plain_bench.log" > "$OUT/${name}_plain_bench.log" || true
  fi

  echo "== $name profiled" >> "$GUARD_LOG"
  MLXCEL_BENCH_PHASE_MARKS=1 "$ROOT/scripts/rocm_gpu_guard.sh" --idle-secs "$IDLE_SECS" \
    --log "$GUARD_LOG" -- \
    "$ROCPROF" --kernel-trace --hip-graph-trace --stats -f csv -d "$run_dir" -o "$name" -- \
    "$BENCH" "${args[@]}" > "$run_dir/bench.log" 2>&1
  # Drop rocprofv3's own glog lines (they name local temp paths); keep the bench's.
  grep -Ev '^[WEI][0-9]{4} |^rocm_gpu_guard: ' "$run_dir/bench.log" > "$OUT/${name}_bench.log" || true
  cp "$run_dir/${name}_kernel_stats.csv" "$OUT/${name}_kernel_stats.csv"

  plain_args=()
  [[ "$PLAIN" == 1 ]] && plain_args=(--plain-log "$OUT/${name}_plain_bench.log")
  python3 "$ROOT/scripts/rocm_decode_profile.py" summarize \
    --trace "$run_dir/${name}_kernel_trace.csv" --log "$OUT/${name}_bench.log" \
    ${plain_args[@]+"${plain_args[@]}"} --model-dir "$model" --name "$name" --out-dir "$OUT" \
    --temperature "$TEMPERATURE" \
    || echo "summarize failed for $name; rerun it on $run_dir/${name}_kernel_trace.csv" >&2
done
echo "traces: $TRACE_DIR" >&2
echo "results: $OUT" >&2
