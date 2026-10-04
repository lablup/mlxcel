#!/usr/bin/env bash
# Run a command only while the ROCm GPU is otherwise idle, and prove it was.
#
# Usage:
#   scripts/rocm_gpu_guard.sh [--idle-secs N] [--max-attempts N] [--max-wait SECS]
#                             [--log FILE] -- COMMAND [ARGS...]
#
# A benchmark that shares the GPU with another process, or the UMA memory bus
# with a compiler, is not a measurement. This is the guard the gfx1151 baseline
# (docs/benchmark_results/rocm-baseline-gfx1151-2026-09-30.md, #2056) describes,
# as a script so every ROCm measurement can use the same one (#2061):
#
#   1. wait until the host has gone --idle-secs (default 90) consecutive seconds
#      with /sys/class/kfd/kfd/proc empty (the process list `rocm-smi --showpids`
#      reads) and no compiler process running;
#   2. run COMMAND while a monitor samples both once per second;
#   3. reject the attempt if any sample shows a GPU process that is not COMMAND
#      or one of its descendants, or a compiler, and go back to step 1.
#
# Every sample is appended to --log (default: stderr only), so the published
# evidence is the log itself. Exit status: COMMAND's status from the first clean
# attempt; 75 when every attempt was contended or --max-wait ran out, in which
# case nothing COMMAND printed should be used. COMMAND's stdout and stderr pass
# through unchanged, so callers capture them as usual. INT and TERM stop COMMAND
# and the monitor and exit 130 / 143. The three numeric options take plain
# non-negative integers.
#
# The sampling interval is one second: a GPU job shorter than that can in
# principle be missed. The compiler list matches /proc/<pid>/comm exactly.

set -euo pipefail

IDLE_SECS=90
MAX_ATTEMPTS=5
MAX_WAIT=0
LOG=""
KFD_PROC_DIR="${ROCM_GPU_GUARD_KFD_DIR:-/sys/class/kfd/kfd/proc}"
# Build tools whose memory traffic or CPU load would distort a UMA measurement.
# Driver processes (make, cmake, ninja, build scripts) are left out: they are
# idle while their compiler children, which are listed, do the work.
# ROCM_GPU_GUARD_COMPILER_RE overrides the list (the unit tests use it).
COMPILER_RE="${ROCM_GPU_GUARD_COMPILER_RE:-^(cargo|rustc|clang|clang\+\+|clang-[0-9]+|hipcc|nvcc|cc1|cc1plus|ld|ld\.lld|ld\.gold|ld\.bfd|lld|collect2)$}"

usage() {
  sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --idle-secs|--max-attempts|--max-wait|--log)
      [[ $# -ge 2 ]] || { echo "rocm_gpu_guard: $1 needs a value" >&2; exit 2; }
      case "$1" in
        --idle-secs)    IDLE_SECS="$2" ;;
        --max-attempts) MAX_ATTEMPTS="$2" ;;
        --max-wait)     MAX_WAIT="$2" ;;
        --log)          LOG="$2" ;;
      esac
      shift 2 ;;
    -h|--help)      usage; exit 0 ;;
    --)             shift; break ;;
    *)              echo "rocm_gpu_guard: unknown option $1" >&2; usage >&2; exit 2 ;;
  esac
done
for opt in IDLE_SECS MAX_ATTEMPTS MAX_WAIT; do
  if ! [[ "${!opt}" =~ ^[0-9]+$ ]]; then
    flag="--$(tr 'A-Z_' 'a-z-' <<<"$opt")"
    echo "rocm_gpu_guard: $flag needs a non-negative integer, got '${!opt}'" >&2
    exit 2
  fi
done
# Force base 10 so a value such as 08 is not read as octal.
IDLE_SECS=$((10#$IDLE_SECS)); MAX_ATTEMPTS=$((10#$MAX_ATTEMPTS)); MAX_WAIT=$((10#$MAX_WAIT))
[[ $# -gt 0 ]] || { echo "rocm_gpu_guard: no command given" >&2; usage >&2; exit 2; }
[[ -d "$KFD_PROC_DIR" ]] || { echo "rocm_gpu_guard: $KFD_PROC_DIR not found (no ROCm KFD driver?)" >&2; exit 2; }

note() {
  local line
  line="$(date '+%Y-%m-%dT%H:%M:%S%z') $*"
  [[ -n "$LOG" ]] && printf '%s\n' "$line" >> "$LOG"
  printf 'rocm_gpu_guard: %s\n' "$line" >&2
}

# pid:comm of every process holding the GPU, comma separated.
kfd_holders() {
  local d pid comm out=()
  for d in "$KFD_PROC_DIR"/*; do
    [[ -e "$d" ]] || continue
    pid="${d##*/}"
    comm=$(cat "/proc/$pid/comm" 2>/dev/null || echo '?')
    out+=("$pid:$comm")
  done
  local IFS=,
  echo "${out[*]}"
}

# pid:comm of every running compiler, comma separated.
compilers() {
  local f pid comm out=()
  for f in /proc/[0-9]*/comm; do
    comm=$(cat "$f" 2>/dev/null) || continue
    if [[ "$comm" =~ $COMPILER_RE ]]; then
      pid="${f#/proc/}"; pid="${pid%/comm}"
      out+=("$pid:$comm")
    fi
  done
  local IFS=,
  echo "${out[*]}"
}

# True when $1 is $2 or a descendant of it.
descends_from() {
  local pid="$1" root="$2" ppid stat
  while [[ -n "$pid" && "$pid" != 0 && "$pid" != 1 ]]; do
    [[ "$pid" == "$root" ]] && return 0
    # comm (field 2) may contain spaces and parentheses; the fields that
    # follow the last ')' are "state ppid ...".
    stat=$(cat "/proc/$pid/stat" 2>/dev/null) || return 1
    stat="${stat##*) }"
    read -r _ ppid _ <<<"$stat"
    pid="$ppid"
  done
  return 1
}

# Holders in $2 (a kfd_holders reading) that are not $1 or its descendants.
#
# A holder whose /proc entry is gone has already exited and been reaped: the
# kernel tears a process's /sys/class/kfd/kfd/proc entry down from a deferred
# work item, so it outlives the process by a moment. Its parentage can no
# longer be read, and it is not using the GPU, so it is not contention. Without
# this, the command's own children (bench_decode.sh's rocminfo probe, the bench
# binary itself) read as foreign in the sample taken just after they exit, and
# most attempts were rejected (#2065).
foreign_holders() {
  local root="$1" entry pid out=() entries
  IFS=, read -r -a entries <<<"$2"
  for entry in ${entries[@]+"${entries[@]}"}; do
    pid="${entry%%:*}"
    [[ -e "/proc/$pid" ]] || continue
    descends_from "$pid" "$root" || out+=("$entry")
  done
  local IFS=,
  echo "${out[*]}"
}

wait_idle() {
  local quiet=0 waited=0 holders comps
  note "waiting for ${IDLE_SECS}s of idle GPU and no compiler"
  while (( quiet < IDLE_SECS )); do
    holders=$(kfd_holders); comps=$(compilers)
    if [[ -z "$holders" && -z "$comps" ]]; then
      quiet=$((quiet + 1))
    else
      (( quiet > 0 )) && note "idle streak reset after ${quiet}s: gpu=[${holders}] compilers=[${comps}]"
      quiet=0
    fi
    if (( MAX_WAIT > 0 && waited >= MAX_WAIT )); then
      note "gave up after ${waited}s without ${IDLE_SECS}s of idle"
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
  note "idle for ${quiet}s: kfd proc empty and no compiler in every 1 Hz sample"
}

cmd_pid="" mon_pid=""
# On INT or TERM stop the command and the monitor instead of orphaning them.
on_signal() {
  local sig="$1" code="$2"
  trap - INT TERM
  note "caught SIG${sig}: stopping command and monitor"
  [[ -n "$mon_pid" ]] && kill "$mon_pid" 2>/dev/null || true
  [[ -n "$cmd_pid" ]] && kill -TERM "$cmd_pid" 2>/dev/null || true
  wait 2>/dev/null || true
  exit "$code"
}
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM

attempt=0
while (( attempt < MAX_ATTEMPTS )); do
  attempt=$((attempt + 1))
  wait_idle || exit 75
  note "attempt ${attempt}/${MAX_ATTEMPTS}: start: $*"
  "$@" &
  cmd_pid=$!
  # The monitor exits 1 if any of its samples was contended.
  (
    n=0 contended=0
    while kill -0 "$cmd_pid" 2>/dev/null; do
      holders=$(kfd_holders); comps=$(compilers)
      foreign=$(foreign_holders "$cmd_pid" "$holders")
      n=$((n + 1))
      if [[ -n "$foreign" || -n "$comps" ]]; then
        note "sample ${n}: CONTENDED foreign_gpu=[${foreign}] compilers=[${comps}]"
        contended=1
      elif [[ -n "$LOG" ]]; then
        printf '%s sample %d: clean gpu=[%s]\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$n" "$holders" >> "$LOG"
      fi
      sleep 1
    done
    note "monitor: ${n} samples"
    exit "$contended"
  ) &
  mon_pid=$!
  rc=0
  wait "$cmd_pid" || rc=$?
  mon_rc=0
  wait "$mon_pid" || mon_rc=$?
  if (( mon_rc != 0 )); then
    note "attempt ${attempt}: REJECTED (contended), exit ${rc}; rerunning"
    continue
  fi
  note "attempt ${attempt}: CLEAN, exit ${rc}"
  exit "$rc"
done
note "every attempt (${MAX_ATTEMPTS}) was contended"
exit 75
