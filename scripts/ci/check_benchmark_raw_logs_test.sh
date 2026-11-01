#!/usr/bin/env bash
# Deterministic coverage for scripts/ci/check_benchmark_raw_logs.py. Each case
# builds a throwaway repository, plants one state, and asserts the exit status
# and the line the operator is supposed to read.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="$(mktemp -d "${TMPDIR:-/tmp}/benchmark-raw-logs-test.XXXXXX")"
trap 'rm -rf "$work"' EXIT
failures=0

run_case() {
  local name="$1" expected_status="$2" expected_text="$3" file="$4" content="$5" allow="$6"
  local repo="$work/$name" data="docs/benchmark_results/data/run"
  mkdir -p "$repo/scripts/ci" "$repo/$data"
  cp "$script_dir/check_benchmark_raw_logs.py" "$repo/scripts/ci/"
  printf '%s' "$allow" > "$repo/scripts/ci/benchmark_raw_logs_allowlist.txt"
  printf '%s' "$content" > "$repo/$data/$file"
  git -C "$repo" init -q
  git -C "$repo" add -A
  local output status=0
  output="$(python3 "$repo/scripts/ci/check_benchmark_raw_logs.py" 2>&1)" || status=$?
  if [ "$status" -ne "$expected_status" ] || ! grep -qF "$expected_text" <<<"$output"; then
    echo "FAIL: $name (status $status): $output"
    failures=$((failures + 1))
  else
    echo "ok: $name"
  fi
}

data="docs/benchmark_results/data/run"
run_case results-pass 0 "OK" results.json '{}' ""
run_case server-log-fails 1 "server log under benchmark data" server-a-r1.log "line" ""
run_case build-log-fails 1 "build log under benchmark data" final-clippy.log "line" ""
run_case empty-file-fails 1 "empty file under benchmark data" greedy.err "" ""
run_case cited-log-passes 0 "OK" server.log "line" "$data/server.log"$'\n'
run_case stale-entry-fails 1 "stale allowlist entry" results.json '{}' "$data/server.log"$'\n'

if [ "$failures" -ne 0 ]; then
  echo "$failures case(s) failed"
  exit 1
fi
