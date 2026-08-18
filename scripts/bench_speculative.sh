#!/usr/bin/env bash
# Speculative-decoding (MTP) throughput, measured the way the numbers in
# README.md and docs/benchmarks.md were measured.
#
# Usage:
#   ./scripts/bench_speculative.sh                 # every pairing it can find
#   ./scripts/bench_speculative.sh gemma           # one pairing
#   ./scripts/bench_speculative.sh --reps 4        # more samples per arm
#   ./scripts/bench_speculative.sh --no-wait       # do not wait for a quiet host
#
# Output: a markdown table row per pairing, ready to paste beside the existing
# rows, plus the acceptance diagnostics behind each one.
#
# ## Why this is a script and not a command line
#
# A speculative-decoding number is only meaningful with the things that were
# left out of the figure it replaced: the prompt, the host, and the protocol.
# Every one of them moved the answer by more than a code change would.
#
# - **The prompt.** The same pairing on the same host measured 1.95x on prose
#   and 2.80x on source code, because acceptance is a property of how
#   predictable the continuation is. The prompts are baked in here rather than
#   passed in, so a rerun is comparable to the row it updates.
# - **The host.** Whether a verify block pays depends on how the GPU generation
#   dispatches the quantized projections it runs. The table carries a host
#   column for that reason and this script fills it from `sysctl`.
# - **Contention.** Three sweeps were thrown away on a machine that was running
#   a Time Machine first backup, Spotlight indexing and Photos analysis. Every
#   arm halved together, no thermal warning fired, and the medians looked like a
#   regression. So this refuses to start until the host is quiet and says why.
# - **Drift.** Throughput climbs over the first several runs of a cold process
#   and sags under sustained load, so arms are alternated and a warm-up is
#   discarded. A block of runs per arm measures the ramp, not the arms.
#
# A run whose spread exceeds SPREAD_LIMIT is reported as untrustworthy rather
# than averaged, because a contaminated median is indistinguishable from a real
# regression once it reaches a document.

set -uo pipefail

REPS=3                 # ABBA blocks; each gives 2 samples per arm
SPREAD_LIMIT=4         # percent of the median, above which a run is suspect
QUIET_LIMIT=15         # percent CPU for the busiest non-shell process
QUIET_HOLD=90          # seconds the host must stay quiet before starting
WAIT_FOR_QUIET=1
ONLY=""

while [ $# -gt 0 ]; do
  case "$1" in
    --reps) REPS="$2"; shift 2 ;;
    --no-wait) WAIT_FOR_QUIET=0; shift ;;
    --spread-limit) SPREAD_LIMIT="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) ONLY="$1"; shift ;;
  esac
done

BIN=${MLXCEL_BIN:-target/release/mlxcel}
[ -x "$BIN" ] || BIN=target/test-fast/mlxcel
if [ ! -x "$BIN" ]; then
  echo "no mlxcel binary; build one first:" >&2
  echo "  cargo build --release --features metal,accelerate" >&2
  exit 1
fi

HOST=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)
MEM=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))

# The prompts are the protocol. Changing one makes a row incomparable to the
# row it replaces, so change them only alongside every published row.
PROMPT_CODE="Write a Python function that computes the nth Fibonacci number, with a docstring and type hints."
PROMPT_PROSE="Explain how speculative decoding accepts or rejects draft tokens."
PROMPT_LIST="Count from 1 to 200, one number per line, with no other text."

wait_for_quiet() {
  [ "$WAIT_FOR_QUIET" = "1" ] || return 0
  local streak=0 top
  echo "waiting for a quiet host (busiest process under ${QUIET_LIMIT}% for ${QUIET_HOLD}s)..." >&2
  while true; do
    top=$(ps -Ao %cpu,comm -r | awk 'NR==2 {print int($1)}')
    if [ "${top:-100}" -lt "$QUIET_LIMIT" ]; then
      streak=$((streak + 10))
    else
      if [ $((streak)) -gt 0 ]; then
        echo "  host became busy again (${top}%), restarting the hold" >&2
      fi
      streak=0
    fi
    [ "$streak" -ge "$QUIET_HOLD" ] && break
    sleep 10
  done
  echo "  quiet, starting" >&2
}

# $1 label  $2.. full mlxcel argv
sample() {
  local label="$1"; shift
  "$@" 2>/dev/null \
    | sed 's/\x1b\[[0-9;]*m//g' \
    | grep -oE '= [0-9.]+ tok/s' | sed 's/= //;s/ tok.s//'
}

# Collect one pairing. $1 name, $2 target, $3 drafter ("" for none), $4 block,
# $5 prompt label, $6 prompt, $7 tokens.
measure_pairing() {
  local name="$1" target="$2" drafter="$3" block="$4" plabel="$5" prompt="$6" ntok="$7"
  [ -d "$target" ] || { echo "skip $name: no $target" >&2; return; }
  [ -z "$drafter" ] || [ -d "$drafter" ] || { echo "skip $name: no $drafter" >&2; return; }

  local classic=() mtp=() i
  # Warm-up, discarded. Throughput climbs over the first runs of a cold host.
  for i in 1 2; do sample warm "$BIN" generate -m "$target" -p "$prompt" -n "$ntok" --temp 0 >/dev/null; sleep 4; done

  for ((i = 0; i < REPS; i++)); do
    # ABBA within each block, so a linear drift cancels instead of landing on
    # whichever arm happened to run later.
    classic+=("$(sample c "$BIN" generate -m "$target" -p "$prompt" -n "$ntok" --temp 0)"); sleep 4
    mtp+=("$(sample m "$BIN" generate -m "$target" --draft-model "$drafter" --draft-kind mtp --draft-block-size "$block" -p "$prompt" -n "$ntok" --temp 0)"); sleep 4
    mtp+=("$(sample m "$BIN" generate -m "$target" --draft-model "$drafter" --draft-kind mtp --draft-block-size "$block" -p "$prompt" -n "$ntok" --temp 0)"); sleep 4
    classic+=("$(sample c "$BIN" generate -m "$target" -p "$prompt" -n "$ntok" --temp 0)"); sleep 4
  done

  # One extra MTP run with diagnostics on, for the acceptance figures that
  # explain the ratio. Kept out of the timing samples: the log costs throughput.
  local diag
  diag=$(RUST_LOG=info "$BIN" generate -m "$target" --draft-model "$drafter" --draft-kind mtp \
           --draft-block-size "$block" -p "$prompt" -n "$ntok" --temp 0 2>&1 \
         | sed 's/\x1b\[[0-9;]*m//g' | grep -m1 "round-loop diagnostics")

  MEASURE_NAME="$name" MEASURE_HOST="$HOST ($MEM GB)" MEASURE_LABEL="$plabel" \
  MEASURE_BLOCK="$block" MEASURE_TOK="$ntok" MEASURE_LIMIT="$SPREAD_LIMIT" \
  MEASURE_DIAG="$diag" \
  python3 - "${#classic[@]}" "${classic[@]}" "${mtp[@]}" <<'PY'
import os, re, statistics, sys
n = int(sys.argv[1])
vals = [float(v) for v in sys.argv[2:] if v]
classic, mtp = sorted(vals[:n]), sorted(vals[n:])
if not classic or not mtp:
    print("  no samples collected", file=sys.stderr); raise SystemExit(1)
def spread(d): return 100 * (d[-1] - d[0]) / statistics.median(d)
mc, mm = statistics.median(classic), statistics.median(mtp)
limit = float(os.environ["MEASURE_LIMIT"])
sc, sm = spread(classic), spread(mtp)
bad = sc > limit or sm > limit
d = os.environ.get("MEASURE_DIAG", "")
g = dict(re.findall(r"(\w+)=([0-9.]+)", d))
print(f"\n## {os.environ['MEASURE_NAME']}  ({os.environ['MEASURE_LABEL']})")
print(f"| {os.environ['MEASURE_HOST']} | {os.environ['MEASURE_LABEL']} | "
      f"{os.environ['MEASURE_TOK']} | {os.environ['MEASURE_BLOCK']} | "
      f"{mc:.1f} | {mm:.1f} | **{mm/mc:.2f}x** |")
print(f"   classic n={len(classic)} median={mc:.2f} spread={sc:.1f}%")
print(f"   MTP     n={len(mtp)} median={mm:.2f} spread={sm:.1f}%")
if g:
    print(f"   effective block {g.get('effective_block_max','?')}, "
          f"acceptance {float(g.get('acceptance_rate', 0)):.3f}, "
          f"emitted per verify {float(g.get('emitted_per_verify', 0)):.3f}")
if bad:
    print(f"   !! spread above {limit:.0f}% of the median: the host was not quiet.")
    print(f"   !! do not publish this row. Re-run when nothing else is using the GPU.")
PY
}

wait_for_quiet
echo "host: $HOST ($MEM GB), binary: $BIN, reps: $REPS"

if [ -z "$ONLY" ] || [ "$ONLY" = "gemma" ]; then
  measure_pairing "Gemma 4 12B + 4-bit assistant" \
    models/gemma-4-12b-it-4bit models/gemma-4-12b-it-assistant-4bit 5 \
    "source code" "$PROMPT_CODE" 300
  measure_pairing "Gemma 4 12B + 4-bit assistant" \
    models/gemma-4-12b-it-4bit models/gemma-4-12b-it-assistant-4bit 5 \
    "prose" "$PROMPT_PROSE" 400
  measure_pairing "Gemma 4 12B + 4-bit assistant" \
    models/gemma-4-12b-it-4bit models/gemma-4-12b-it-assistant-4bit 4 \
    "enumeration" "$PROMPT_LIST" 400
fi

if [ -z "$ONLY" ] || [ "$ONLY" = "qwen" ]; then
  measure_pairing "Qwen 3.8 27B + its 4-bit MTP head" \
    models/qwen3.8-27b-4bit models/qwen3.8-27b-mtp-4bit 3 \
    "source code" "$PROMPT_CODE" 300
fi
