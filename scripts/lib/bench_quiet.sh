#!/usr/bin/env bash
# Quiet-host gate and in-run contention watch, shared by the benchmark sweeps.
#
# Source it, do not execute it:
#
#   . "$(dirname "$0")/lib/bench_quiet.sh"
#   QUIET_IGNORE="$QUIET_IGNORE|bench_my_thing"
#   wait_for_quiet
#
# ## The gate belongs INSIDE the wrapper, never around it
#
# `scripts/with_indexers_paused.sh` suspends the daemons that make a Mac noisy.
# This gate waits for the host to go quiet. Their order is not a matter of
# taste:
#
#   ./scripts/with_indexers_paused.sh ./scripts/bench_thing.sh   # correct
#   ./wait_for_quiet_then.sh ./scripts/with_indexers_paused.sh ...   # deadlock
#
# In the second form the gate waits for `mediaanalysisd` to stop using the CPU
# *before* running the thing that would stop it. It never releases. That is not
# hypothetical: it cost 2.5 hours on 2026-08-19 before anyone noticed the log
# had been sitting at "waiting for a quiet host" the whole time, because the
# hold message only prints while the streak is above zero and a permanently
# busy host keeps it pinned at zero, printing nothing.
#
# Keeping the gate inside the sweep script, as every sweep here does, means
# there is no outer runner to get wrong. Write a new sweep the same way.
#
# ## What the two halves are for
#
# The gate guards the start. The watch guards the rest, because a load that
# arrives after the gate releases is otherwise left to the spread check alone,
# and a *steady* load evades that check by construction: it depresses every
# sample equally, so the median drops while the spread stays small. A video
# call measured one table 1.2% low with every spread under 4% and nothing
# flagged.
#
# What the watch reports is therefore how much of the run was contended, not
# its worst instant. A desktop spikes WindowServer past any threshold whenever
# it draws, and rejecting a row for one such sample rejects every row ever
# measured on a machine with a display. Sustained load, the kind that actually
# moves a median, shows up in most samples instead of one.

QUIET_LIMIT=${QUIET_LIMIT:-15}   # percent CPU for the busiest process that is not ours
QUIET_HOLD=${QUIET_HOLD:-90}     # seconds the host must stay quiet before starting
WAIT_FOR_QUIET=${WAIT_FOR_QUIET:-1}

# Processes that drive the measurement rather than compete with it. The agent
# or shell supervising a run spikes well over the limit every time it emits
# output, so counting it as contention means an interactively supervised sweep
# never leaves the hold at all. Anchored with `(^|/)` because `ps -o comm`
# reports a bare name on some hosts and a full path on others; a caller adds
# its own script name to this.
QUIET_IGNORE='(^|/)(claude|node|mlxcel|mlxcel-server|bash|zsh|sh|ps|awk|sleep|tmux)$'

# Busiest process that is not one of ours, as an integer percent. Matching on
# the whole command rather than a field keeps paths with spaces intact.
busiest() {
  ps -Ao %cpu,comm -r | awk -v skip="$QUIET_IGNORE" '
    NR > 1 {
      name = $0
      sub(/^[[:space:]]*[0-9.]+[[:space:]]+/, "", name)
      if (name ~ skip) next
      print int($1); exit
    }'
}

# Block until the host has been quiet for QUIET_HOLD seconds. Give up after
# QUIET_GATE_TIMEOUT seconds (0 disables the timeout) and say so, rather than
# waiting forever on a host that is never going to settle: an unattended run
# that says "measured under load, judge it by the spread" beats one that
# silently waits all night.
wait_for_quiet() {
  [ "$WAIT_FOR_QUIET" = "1" ] || return 0
  local streak=0 waited=0 timeout=${QUIET_GATE_TIMEOUT:-1800} top
  echo "waiting for a quiet host (busiest other process under ${QUIET_LIMIT}% for ${QUIET_HOLD}s)..." >&2
  while true; do
    top=$(busiest)
    if [ "${top:-100}" -lt "$QUIET_LIMIT" ]; then
      streak=$((streak + 10))
    else
      # A blip costs progress rather than all of it. Sustained load still
      # drives the streak to zero within a few samples and pins it there,
      # while one transient spike no longer discards a minute and a half.
      # Printed unconditionally, so a host pinned at zero still says what is
      # holding it up instead of going silent.
      echo "  busy (${top}%: $(busiest_name)), holding at ${streak}s" >&2
      streak=$((streak - 10))
      [ "$streak" -lt 0 ] && streak=0
    fi
    [ "$streak" -ge "$QUIET_HOLD" ] && break
    if [ "$timeout" -gt 0 ] && [ "$waited" -ge "$timeout" ]; then
      echo "  no quiet window in ${timeout}s; starting anyway." >&2
      echo "  Judge these rows by their spread and contention lines, not their absolute numbers." >&2
      return 0
    fi
    waited=$((waited + 10))
    sleep 10
  done
  echo "  quiet, starting" >&2
}

# Name of the process `busiest` is reporting, for the hold message. Knowing it
# is `mediaanalysisd` is what tells a reader to add it to the wrapper's list
# rather than keep waiting.
busiest_name() {
  ps -Ao %cpu,comm -r | awk -v skip="$QUIET_IGNORE" '
    NR > 1 {
      name = $0
      sub(/^[[:space:]]*[0-9.]+[[:space:]]+/, "", name)
      if (name ~ skip) next
      n = split(name, parts, "/")
      print parts[n]; exit
    }'
}

CONTENTION_FILE=""
CONTENTION_PID=""

start_contention_watch() {
  CONTENTION_FILE=$(mktemp)
  echo "0 0 0" > "$CONTENTION_FILE"   # samples, busy samples, peak
  (
    while :; do
      t=$(busiest)
      read -r n b pk < "$CONTENTION_FILE" 2>/dev/null || { n=0; b=0; pk=0; }
      n=$((n + 1))
      [ "${t:-0}" -ge "$QUIET_LIMIT" ] && b=$((b + 1))
      [ "${t:-0}" -gt "${pk:-0}" ] && pk=${t:-0}
      echo "$n $b $pk" > "$CONTENTION_FILE"
      sleep 5
    done
  ) &
  CONTENTION_PID=$!
  # Detached so killing it later cannot print a job notice into the output
  # this script is being read from.
  disown "$CONTENTION_PID" 2>/dev/null || true
}

# Prints "samples busy_samples peak" on stdout for the caller to judge.
stop_contention_watch() {
  [ -n "$CONTENTION_PID" ] && kill "$CONTENTION_PID" 2>/dev/null
  cat "$CONTENTION_FILE" 2>/dev/null || echo "0 0 0"
  # `command` so an interactive profile that wraps rm cannot print into the
  # captured output. This is a mktemp scratch file, not anything of value.
  command rm -f "$CONTENTION_FILE"
  CONTENTION_PID=""
}
