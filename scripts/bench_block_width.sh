#!/usr/bin/env bash
# Block-width sweep for one MTP pairing.
#
# Usage:
#   ./scripts/bench_block_width.sh qwen        # widths 2 3 4 5 6 8
#   ./scripts/bench_block_width.sh gemma       # widths 3 4 5 6 8 10 12
#   ./scripts/bench_block_width.sh gemma31b    # widths 2 3 4 5 6 8
#   ./scripts/bench_block_width.sh gemma 4 5 6 # explicit widths
#
# Run it through scripts/with_indexers_paused.sh, the same way the throughput
# sweep is run:
#
#   ./scripts/with_indexers_paused.sh ./scripts/bench_block_width.sh gemma
#
# The quiet gate is inside this script, not around it, and that ordering is
# load-bearing rather than stylistic. See scripts/lib/bench_quiet.sh: a gate
# placed outside the wrapper waits for the daemons the wrapper exists to
# suspend, and never releases. Do not add an outer runner.
#
# ## Why the widths are interleaved
#
# Measuring one width to completion before starting the next puts any drift
# over the run — thermal, or a background task that arrives partway — entirely
# on whichever widths happened to be measured late. That is indistinguishable
# from those widths being slower, which is the exact question the sweep is
# asking. So a round visits every width once, and the starting width rotates
# per round, which spreads a drift across the whole table instead of pooling
# it at one end.
#
# ## Why acceptance is reported beside throughput
#
# A width changes two things at once and they pull in opposite directions.
# Tokens emitted per verify rise towards `1 / (1 - acceptance)` and saturate,
# while the verify forward keeps costing more per position. Throughput alone
# says a width is worse; throughput next to `emitted per verify` says whether
# it was the saturation or the cost that did it. Both columns come from the
# same runs the throughput median does.
set -uo pipefail

. "$(dirname "$0")/lib/bench_quiet.sh"
QUIET_IGNORE="$QUIET_IGNORE|bench_block_width"

BIN=${MLXCEL_BIN:-target/release/mlxcel}
if [ ! -x "$BIN" ]; then
  echo "no mlxcel binary; build one first:" >&2
  echo "  cargo build --release --features metal,accelerate" >&2
  exit 1
fi

ROUNDS=${ROUNDS:-8}
# The prompt is the protocol; it must match the row the sweep is read against.
PROMPT="Write a Python function that computes the nth Fibonacci number, with a docstring and type hints."
NTOK=300

case "${1:-}" in
  qwen)
    TARGET=models/qwen3.8-27b-4bit; DRAFTER=models/qwen3.8-27b-mtp-4bit
    DEFAULT_WIDTHS=(2 3 4 5 6 8) ;;
  gemma)
    TARGET=models/gemma-4-12b-it-4bit; DRAFTER=models/gemma-4-12b-it-assistant-4bit
    DEFAULT_WIDTHS=(3 4 5 6 8 10 12) ;;
  gemma31b)
    # The batch-capable pairing the B=1 static gate governs (issue #1217). Its
    # declared drafter width is 4; the sweep brackets it the way the qwen one
    # does rather than reaching for the 12B pairing's wide tail, because a bf16
    # drafter costs far more per position than a 4-bit one.
    TARGET=models/gemma-4-31b-it-4bit; DRAFTER=models/gemma-4-31b-it-assistant-bf16
    DEFAULT_WIDTHS=(2 3 4 5 6 8) ;;
  *)
    sed -n '2,10p' "$0"; exit 1 ;;
esac
shift
WIDTHS=("$@"); [ ${#WIDTHS[@]} -eq 0 ] && WIDTHS=("${DEFAULT_WIDTHS[@]}")

[ -d "$TARGET" ]  || { echo "no $TARGET" >&2; exit 1; }
[ -d "$DRAFTER" ] || { echo "no $DRAFTER" >&2; exit 1; }

HOST=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)
MEM=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
echo "host: $HOST ($MEM GB), widths: ${WIDTHS[*]}, rounds: $ROUNDS" >&2

OUT=$(mktemp); DIAG=$(mktemp)
trap 'command rm -f "$OUT" "$DIAG"' EXIT

sample() {
  "$BIN" generate -m "$TARGET" --draft-model "$DRAFTER" --draft-kind mtp \
    --draft-block-size "$1" -p "$PROMPT" -n "$NTOK" --temp 0 2>/dev/null \
    | sed 's/\x1b\[[0-9;]*m//g' | grep -oE '= [0-9.]+ tok/s' | sed 's/= //;s/ tok.s//'
}

wait_for_quiet
start_contention_watch

# Discarded: throughput climbs over the first runs of a cold process.
for i in 1 2; do sample "${WIDTHS[0]}" >/dev/null; sleep 3; done

n=${#WIDTHS[@]}
for ((r = 0; r < ROUNDS; r++)); do
  for ((k = 0; k < n; k++)); do
    w=${WIDTHS[$(( (r + k) % n ))]}
    v=$(sample "$w"); [ -n "$v" ] && echo "$w $v" >> "$OUT"
    sleep 3
  done
  echo "  round $((r+1))/$ROUNDS" >&2
done

WATCH=$(stop_contention_watch)

# Diagnostics run separately: the log costs throughput, so it stays out of the
# timing samples.
for w in "${WIDTHS[@]}"; do
  line=$(RUST_LOG=info "$BIN" generate -m "$TARGET" --draft-model "$DRAFTER" --draft-kind mtp \
           --draft-block-size "$w" -p "$PROMPT" -n "$NTOK" --temp 0 2>&1 \
         | sed 's/\x1b\[[0-9;]*m//g' | grep -m1 "round-loop diagnostics")
  echo "$w $(echo "$line" | grep -oE 'effective_block_max=[0-9]+' | cut -d= -f2)" \
       "$(echo "$line" | grep -oE 'acceptance_rate=[0-9.]+' | cut -d= -f2)" \
       "$(echo "$line" | grep -oE 'emitted_per_verify=[0-9.]+' | cut -d= -f2)" >> "$DIAG"
done

MEASURE_HOST="$HOST ($MEM GB)" MEASURE_WATCH="$WATCH" MEASURE_QUIET="$QUIET_LIMIT" \
python3 - "$OUT" "$DIAG" <<'PY'
import os, sys, statistics, collections
vals = collections.defaultdict(list)
for ln in open(sys.argv[1]):
    w, v = ln.split(); vals[int(w)].append(float(v))
diag = {}
for ln in open(sys.argv[2]):
    p = (ln.split() + ['?', '0', '0'])[:4]
    diag[int(p[0])] = (p[1], float(p[2] or 0), float(p[3] or 0))
if not vals:
    print("no samples collected", file=sys.stderr); raise SystemExit(1)
peak_w = max(vals, key=lambda w: statistics.median(vals[w]))
peak = statistics.median(vals[peak_w])
print(f"\n{os.environ['MEASURE_HOST']}")
print("| width | decode tok/s | spread | acceptance | emitted per verify | vs peak |")
print("| ---: | ---: | ---: | ---: | ---: | ---: |")
for w in sorted(vals):
    v = sorted(vals[w]); m = statistics.median(v)
    sp = 100 * (v[-1] - v[0]) / m
    eb, ar, ev = diag.get(w, ('?', 0, 0))
    mark = "**peak**" if w == peak_w else f"{100*(m-peak)/peak:+.1f}%"
    warn = "  <-- spread over 4%, re-measure" if sp > 4 else ""
    print(f"| {w} | {m:.1f} | {sp:.1f}% | {ar:.3f} | {ev:.3f} | {mark} |{warn}")
    if eb != str(w) and eb != '?':
        print(f"   note: width {w} ran at effective block {eb}", file=sys.stderr)
watch = (os.environ.get("MEASURE_WATCH") or "0 0 0").split()
n_s, n_busy, peak = (float(x) for x in (watch + ['0', '0', '0'])[:3])
quiet = float(os.environ.get("MEASURE_QUIET") or 15)
share = n_busy / n_s if n_s else 0.0
# A fifth of the run under load is enough to move a median; one sample is a
# desktop drawing itself. Same threshold the throughput sweep uses.
if share > 0.2:
    print(f"\n!! another process was over {quiet:.0f}% for {100*share:.0f}% of this sweep "
          f"({n_busy:.0f} of {n_s:.0f} samples, peak {peak:.0f}%).")
    print("!! a steady load depresses every width together, so the ranking may survive")
    print("!! while none of the absolute numbers do. Re-run before publishing these.")
elif peak > quiet:
    print(f"\nbrief spikes to {peak:.0f}% in {n_busy:.0f} of {n_s:.0f} samples, not enough to flag")
PY
