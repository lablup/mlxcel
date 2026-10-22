#!/usr/bin/env bash
# Deterministic coverage for scripts/ci/check_windows_reserved_paths.py.
#
# A guard that only ever passes proves nothing: the useful question is whether
# it fails on the case it exists for. Each case below builds a throwaway
# repository, copies the checker in, tracks one path, and asserts both the exit
# status and the line the operator is supposed to read. `src/aux.rs` is the one
# that matters most, because it is the regression that
# src/lib/mlxcel-xla/src/aux.rs actually was.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
under_test="$script_dir/check_windows_reserved_paths.py"

work="$(mktemp -d "${TMPDIR:-/tmp}/windows-reserved-paths-test.XXXXXX")"
trap 'rm -rf "$work"' EXIT

failures=0

# Builds a repo holding the checker plus one tracked file at `path`.
setup_repo() {
  local repo="$1" path="$2"
  mkdir -p "$repo/scripts/ci"
  git init -q "$repo"
  git -C "$repo" config user.name "mlxcel test"
  git -C "$repo" config user.email "mlxcel-test@example.com"
  cp "$under_test" "$repo/scripts/ci/check_windows_reserved_paths.py"
  mkdir -p "$repo/$(dirname "$path")"
  : > "$repo/$path"
  git -C "$repo" add -f -- "$path"
}

run_case() {
  local name="$1" repo="$2" expected_status="$3" expected_text="$4"
  local stdout="$work/$name.stdout"
  set +e
  (cd "$repo" && python3 scripts/ci/check_windows_reserved_paths.py) > "$stdout" 2>&1
  local status=$?
  set -e

  if [[ "$status" -ne "$expected_status" ]]; then
    echo "FAIL [$name]: expected exit $expected_status, got $status"
    sed 's/^/    /' "$stdout"
    failures=$((failures + 1))
    return
  fi
  if ! grep -qF "$expected_text" "$stdout"; then
    echo "FAIL [$name]: output did not contain '$expected_text'"
    sed 's/^/    /' "$stdout"
    failures=$((failures + 1))
    return
  fi
  echo "ok   [$name]"
}

# expect_fail NAME PATH: the checker must exit 1 and name PATH.
expect_fail() {
  local name="$1" path="$2" repo="$work/$1"
  setup_repo "$repo" "$path"
  run_case "$name" "$repo" 1 "reserved: $path"
}

# expect_pass NAME PATH: the checker must exit 0.
expect_pass() {
  local name="$1" path="$2" repo="$work/$1"
  setup_repo "$repo" "$path"
  run_case "$name" "$repo" 0 "windows-reserved-paths: OK"
}

# The regression this guard exists for, then the other component positions,
# case folding, a bare name with no extension, and the superscript digits.
expect_fail "aux-file"          "src/aux.rs"
expect_fail "mixed-case-file"   "docs/Con.md"
expect_fail "reserved-dir"      "a/nul/b.txt"
expect_fail "bare-lpt1"         "lpt1"
expect_fail "multi-dot-name"    "src/prn.tar.gz"
expect_fail "com-digit"         "tools/COM9.txt"
expect_fail "com-superscript"   "tools/com¹.txt"
expect_fail "lpt-superscript"   "tools/LPT³"
expect_fail "trailing-space"    "docs/aux .txt"

# Names that only start with a reserved word, or sit one digit outside the
# reserved range, are legal on Windows and must not trip the guard.
expect_pass "aux-prefix"        "src/aux_manifest.rs"
expect_pass "console-prefix"    "src/console.rs"
expect_pass "nullable-prefix"   "src/nullable.rs"
expect_pass "com10"             "src/com10.rs"
expect_pass "reserved-in-ext"   "src/data.aux"
expect_pass "auxiliary"         "src/auxiliary.rs"

# The real tree must pass, or the rules above do not describe it.
run_case "repository-as-committed" "$(cd "$script_dir/../.." && pwd)" 0 "windows-reserved-paths: OK"

if [[ "$failures" -ne 0 ]]; then
  echo "windows-reserved-paths self-test: $failures case(s) failed"
  exit 1
fi
echo "windows-reserved-paths self-test: all cases passed"
