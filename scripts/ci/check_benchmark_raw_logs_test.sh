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
  local name="$1" expected_status="$2" expected_text="$3" file="$4" content="$5" allow="$6" baseline="$7"
  local repo="$work/$name" data="docs/benchmark_results/data/run"
  mkdir -p "$repo/scripts/ci" "$repo/$data"
  cp "$script_dir/check_benchmark_raw_logs.py" "$repo/scripts/ci/"
  printf '%s' "$allow" > "$repo/scripts/ci/benchmark_raw_logs_allowlist.txt"
  printf '%s' "$baseline" > "$repo/scripts/ci/benchmark_raw_data_baseline.txt"
  mkdir -p "$(dirname "$repo/$data/$file")"
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
run_case results-csv-passes 0 "OK" results.csv $'arm,decode\none,10\n' "" ""
run_case manifest-json-passes 0 "OK" manifest.json '{}' "" ""
run_case server-log-fails 1 "server log under benchmark data" server-a-r1.log "line" "" ""
run_case build-log-fails 1 "build log under benchmark data" final-clippy.log "line" "" ""
run_case empty-file-fails 1 "empty file under benchmark data" greedy.err "" "" ""
run_case cited-log-cannot-return 1 "exception lists must stay empty" server.log "line" "$data/server.log"$'\n' ""
run_case stale-entry-fails 1 "stale allowlist entry" results.csv $'arm,decode\n' "$data/server.log"$'\n' ""
run_case raw-jsonl-fails 1 "raw record stream" run.jsonl '{}' "" ""
run_case harness-jsonl-fails 1 "raw record stream" harness/prompts.jsonl '{}' "" ""
run_case stdout-capture-fails 1 "raw output capture" driver.out 'output' "" ""
run_case stderr-capture-fails 1 "raw output capture" driver.stderr 'error' "" ""
# The result root and sibling campaign directories must not bypass the data/ gate.
run_case root-jsonl-fails 1 "raw record stream" ../../root.jsonl '{}' "" ""
run_case sibling-log-fails 1 "raw output capture" ../../campaign/trace.log 'trace' "" ""
run_case raw-jsonl-baseline-cannot-grow 1 "exception lists must stay empty" run.jsonl '{}' "" "$data/run.jsonl"$'\n'
run_case raw-jsonl-cannot-be-allowlisted 1 "raw record stream" run.jsonl '{}' "$data/run.jsonl"$'\n' ""
run_case stale-baseline-fails 1 "stale baseline entry" results.csv $'arm,decode\n' "" "$data/run.jsonl"$'\n'

if [ "$failures" -ne 0 ]; then
  echo "$failures case(s) failed"
  exit 1
fi
