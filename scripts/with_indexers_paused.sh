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
#   1. a trap on EXIT, INT, TERM and HUP, covering a normal end and a Ctrl-C;
#   2. that same trap on a failed command, since the trap is on EXIT;
#   3. a detached deadline resume, so even SIGKILL of this wrapper cannot
#      leave anything stopped.
#
# A daemon that was already suspended before this ran is left alone, so two
# nested invocations cannot resume each other's work early.

set -uo pipefail

DEADLINE=${INDEXER_RESUME_DEADLINE:-2700}   # seconds; the backstop resume

if [ $# -eq 0 ]; then
  sed -n '2,8p' "$0"
  exit 1
fi

NAMES="mediaanalysisd photoanalysisd photolibraryd corespotlightd OneDrive"
PIDFILE=$(mktemp "${TMPDIR:-/tmp}/paused_indexers.XXXXXX")

resume() {
  local pid
  while read -r pid; do
    [ -n "$pid" ] && kill -CONT "$pid" 2>/dev/null && echo "resumed $pid" >&2
  done < "$PIDFILE"
  command rm -f "$PIDFILE"
}
trap resume EXIT INT TERM HUP

for n in $NAMES; do
  for pid in $(pgrep -x "$n" 2>/dev/null); do
    # Already stopped by something else: not ours to resume.
    case "$(ps -o state= -p "$pid" 2>/dev/null)" in
      T*) echo "$n ($pid) was already suspended, leaving it alone" >&2; continue ;;
    esac
    if kill -STOP "$pid" 2>/dev/null; then
      echo "$pid" >> "$PIDFILE"
      echo "suspended $n ($pid)" >&2
    else
      echo "cannot signal $n ($pid), leaving it alone" >&2
    fi
  done
done

# Backstop. Detached and holding only the file path, so a SIGKILL of this
# shell still leaves something that will resume the list.
nohup bash -c "sleep $DEADLINE
while read -r p; do kill -CONT \"\$p\" 2>/dev/null; done < '$PIDFILE' 2>/dev/null" \
  >/dev/null 2>&1 &

"$@"
