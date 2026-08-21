#!/usr/bin/env bash
# Run a command with macOS's background indexers suspended, then resume them
# however the command ends.
#
# Usage:
#   ./scripts/with_indexers_paused.sh ./scripts/bench_speculative.sh --reps 4
#
# ## Why this exists
#
# Spotlight indexing, Photos analysis and cloud sync are the reason a benchmark
# on a developer's Mac is hard to trust. They are idle-triggered, so they wake
# up precisely when a machine is left alone to measure something; they are
# bursty enough to reach several hundred percent CPU; and `pmset -g therm`
# reports nothing while they run. Four speculative-decoding sweeps were thrown
# away to them before this existed, and a fifth was measured 1.2% low with
# every spread under the limit, because a steady background load depresses
# both arms of a comparison equally.
#
# Waiting them out is not a plan. Photo analysis walks an entire library and
# resumes at every idle moment until it is finished, which can take days.
#
# ## What it does and does not do
#
# SIGSTOP and SIGCONT only. Nothing is killed, no preference is written, no
# service is disabled, and the suspended work resumes from where it stopped
# rather than starting over. The visible cost while a command runs is that
# Spotlight results go stale and cloud sync pauses.
#
# Three independent paths resume the daemons, because leaving one stopped is
# worse than any measurement is worth:
#
#   1. a trap on EXIT, INT, TERM, HUP and QUIT, covering a normal end, a
#      Ctrl-C and a Ctrl-\ quit;
#   2. that same trap on a failed command, since the trap is on EXIT;
#   3. a detached deadline resume, so even SIGKILL of this wrapper cannot
#      leave anything stopped. It is retired by path 1 on any ending that
#      reaches the trap, so a clean run does not leave it resident.
#
# A daemon that was already suspended before this ran is left alone, so this
# never takes over a suspension someone else owns. That guard is at suspend
# time only: a long-lived daemon that several sequential runs each stopped is
# on each of their resume lists, so a later run's resume does reach an earlier
# run's entry. That direction is harmless, since SIGCONT to a running process
# is a no-op, but it is not an isolation guarantee and should not be read as
# one.

set -uo pipefail

if [ $# -eq 0 ]; then
  sed -n '2,8p' "$0"
  exit 1
fi

DEADLINE=${INDEXER_RESUME_DEADLINE:-2700}   # seconds; the backstop resume
# Validated before it is used anywhere. The value reaches a detached `bash -c`
# below, so anything but a plain count of seconds is either a shell injection
# or a backstop that never fires. Zero is rejected too: it would resume the
# daemons the instant they were suspended.
case "$DEADLINE" in
  ''|*[!0-9]*) echo "INDEXER_RESUME_DEADLINE must be a whole number of seconds, got '$DEADLINE'" >&2; exit 2 ;;
esac
[ "$DEADLINE" -gt 0 ] || { echo "INDEXER_RESUME_DEADLINE must be greater than 0" >&2; exit 2; }

# One process name per line, because a desktop app's name can contain a space.
#
# `corespotlightd` coordinates and `mdworker` does the indexing, so suspending
# only the first leaves the load running: an mdworker reached 1612% during the
# 2026-08-19 M1 Ultra sweep and cost that row. New workers spawn on demand and
# this cannot catch those, so it reduces the load rather than removing it.
NAMES="mediaanalysisd
photoanalysisd
photolibraryd
corespotlightd
mdworker
mdworker_shared
OneDrive"

# Deliberately not here: Time Machine's backupd. A first backup is the single
# largest contender a Mac produces, but suspending it holds a backup session
# and its destination volume open for the length of the run. `tmutil stopbackup`
# ends the current pass cleanly instead, and Time Machine resumes incrementally
# at its next scheduled attempt, so stop it before the sweep rather than
# freezing it during one.

# Anything else this host needs quiet, one name per line. The indexers above
# are on every Mac; which chat and mail clients sit on top of them is a
# property of the machine, not of the benchmark, so the list is passed in:
#
#   INDEXER_EXTRA_NAMES=$'Microsoft Outlook\nMSTeams\nTelegram' \
#     ./scripts/with_indexers_paused.sh ./scripts/bench_speculative.sh
#
# Suspending an interactive app freezes its window and queues its network work
# until the run ends. Nothing is lost, but the host is visibly stopped, so this
# is opt-in rather than a default.
if [ -n "${INDEXER_EXTRA_NAMES:-}" ]; then
  NAMES="$NAMES
$INDEXER_EXTRA_NAMES"
fi

# Without a resume list nothing can undo a suspension: the trap and the
# detached backstop both read this file. An unchecked mktemp leaves PIDFILE
# empty and the daemons stopped with their pids recorded nowhere, so refuse to
# suspend anything at all rather than risk that.
PIDFILE=$(mktemp "${TMPDIR:-/tmp}/paused_indexers.XXXXXX") || { echo "cannot create the resume list; refusing to suspend anything" >&2; exit 1; }

BACKSTOP_PID=""

resume() {
  local pid
  while read -r pid; do
    [ -n "$pid" ] && kill -CONT "$pid" 2>/dev/null && echo "resumed $pid" >&2
  done < "$PIDFILE"
  command rm -f "$PIDFILE"
  # The backstop exists only for an ending that never reaches this trap, and
  # this trap just ran, so retire it instead of leaving a bash and a sleep
  # resident for the rest of the deadline. Eight of them were found on the
  # 2026-08-22 measurement host, one per run, the oldest 2h41m old. A stale one
  # is not only litter: it wakes hours later and signals pids the OS may have
  # recycled by then. The sleep goes first because it is a child of the
  # backstop shell and would outlive it, and the pid list is already unlinked
  # above, so whichever half wins the race has nothing left to act on.
  if [ -n "$BACKSTOP_PID" ]; then
    pkill -P "$BACKSTOP_PID" 2>/dev/null
    kill "$BACKSTOP_PID" 2>/dev/null
    BACKSTOP_PID=""
  fi
}
trap resume EXIT INT TERM HUP QUIT

while IFS= read -r n; do
  [ -n "$n" ] || continue
  for pid in $(pgrep -x "$n" 2>/dev/null); do
    # Already stopped by something else: not ours to resume.
    case "$(ps -o state= -p "$pid" 2>/dev/null)" in
      T*) echo "$n ($pid) was already suspended, leaving it alone" >&2; continue ;;
    esac
    # Record first, suspend second. A write that failed after the STOP would
    # leave a suspended daemon whose pid is on no list, and neither the trap
    # nor the detached backstop could reach it. The reverse ordering is safe:
    # a pid recorded for a STOP that then failed only earns a SIGCONT to a
    # process that was already running, and daemons someone else suspended are
    # skipped above, so this cannot resume another run's work.
    if ! echo "$pid" >> "$PIDFILE"; then
      echo "cannot record $n ($pid) in the resume list; aborting before suspending it" >&2
      exit 1
    fi
    if kill -STOP "$pid" 2>/dev/null; then
      echo "suspended $n ($pid)" >&2
    else
      echo "cannot signal $n ($pid), leaving it alone" >&2
    fi
  done
done <<EOF
$NAMES
EOF

# Backstop. Detached and holding only the file path, so a SIGKILL of this
# shell still leaves something that will resume the list. The deadline and the
# path are passed as arguments rather than spliced into the program text, so
# neither an operator-supplied deadline nor a TMPDIR containing a quote can
# become code in a process that outlives this shell.
nohup bash -c 'sleep "$1"
while read -r p; do kill -CONT "$p" 2>/dev/null; done < "$2" 2>/dev/null' \
  _ "$DEADLINE" "$PIDFILE" >/dev/null 2>&1 &
BACKSTOP_PID=$!

"$@"
