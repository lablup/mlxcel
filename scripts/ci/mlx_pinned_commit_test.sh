#!/usr/bin/env bash
# Negative-path coverage for scripts/ci/mlx_pinned_commit.sh.
#
# The Rust half of the MLX-pin parser is unit-tested by mlxcel-mlx-pin. The awk
# half had no automated coverage at all, and the two must accept the same shape:
# release.yml purges every MLX build cache that does not match whatever this
# script prints, so a value this script accepts and the Rust parser rejects
# means a release run that purges its caches and then fails the build.
#
# Each case asserts what the script does with a synthetic CMakeLists, fed
# through MLX_PIN_CMAKE_FILE. Fixtures are written to a mktemp directory that is
# removed on exit; nothing is written inside the repository.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
under_test="$script_dir/mlx_pinned_commit.sh"

work="$(mktemp -d "${TMPDIR:-/tmp}/mlx-pin-test.XXXXXX")"
trap 'rm -rf "$work"' EXIT

GOOD=2c46b953db88965c4270cc7306eda6887a3247f2
OTHER=b7c3dd6d27f45b5365b08a840310187dc503f1db

failures=0

# expect_ok <name> <expected-sha> <fixture-body>
expect_ok() {
  local name="$1" expected="$2" body="$3" actual status
  printf '%s' "$body" > "$work/$name"
  set +e
  actual="$(MLX_PIN_CMAKE_FILE="$work/$name" "$under_test" 2>/dev/null)"
  status=$?
  set -e
  if [ "$status" -ne 0 ] || [ "$actual" != "$expected" ]; then
    echo "FAIL $name: expected exit 0 and '$expected', got exit $status and '$actual'" >&2
    failures=$((failures + 1))
  else
    echo "ok   $name -> $actual"
  fi
}

# expect_fail <name> <fixture-body>
expect_fail() {
  local name="$1" body="$2" actual status
  printf '%s' "$body" > "$work/$name"
  set +e
  actual="$(MLX_PIN_CMAKE_FILE="$work/$name" "$under_test" 2>/dev/null)"
  status=$?
  set -e
  if [ "$status" -eq 0 ]; then
    echo "FAIL $name: expected a non-zero exit, got 0 and '$actual'" >&2
    failures=$((failures + 1))
  elif [ -n "$actual" ]; then
    echo "FAIL $name: exited $status but still printed '$actual' on stdout" >&2
    failures=$((failures + 1))
  else
    echo "ok   $name -> rejected"
  fi
}

mlx_block() {
  printf 'FetchContent_Declare(\n  mlx\n  GIT_REPOSITORY "https://github.com/ml-explore/mlx.git"\n  %s)\n' "$1"
}

# The shape the repository actually uses, including a comment between
# GIT_REPOSITORY and GIT_TAG.
expect_ok canonical "$GOOD" "$(mlx_block "GIT_TAG $GOOD")"
expect_ok commented "$GOOD" "$(printf 'FetchContent_Declare(\n  mlx\n  GIT_REPOSITORY "https://github.com/ml-explore/mlx.git"\n  # single source of truth, issue #1047\n  GIT_TAG %s)\n' "$GOOD")"

# An unrelated dependency's GIT_TAG must never be mistaken for the pin.
expect_ok other_dependency "$GOOD" "$(printf 'FetchContent_Declare(\n  fmt\n  GIT_REPOSITORY "https://github.com/fmtlib/fmt.git"\n  GIT_TAG %s)\n%s' "$OTHER" "$(mlx_block "GIT_TAG $GOOD")")"

# A commented-out declaration is not the live pin.
expect_ok commented_out_declaration "$GOOD" "$(printf '# FetchContent_Declare(mlx GIT_REPOSITORY "https://github.com/ml-explore/mlx.git" GIT_TAG %s)\n%s' "$OTHER" "$(mlx_block "GIT_TAG $GOOD")")"

# Two blocks naming the MLX repository: which one supplies the pin is not well
# defined, and the Rust parser rejects it, so this must too. Regression guard:
# when only the second block carried a GIT_TAG, the tag count was 1 and this
# script returned that second block's SHA.
expect_fail two_mlx_declarations "$(printf '%s%s' "$(mlx_block "GIT_TAG $GOOD")" "$(mlx_block "GIT_TAG $OTHER")")"
expect_fail two_mlx_declarations_one_tag "$(printf 'FetchContent_Declare(\n  mlx\n  GIT_REPOSITORY "https://github.com/ml-explore/mlx.git"\n)\n%s' "$(mlx_block "GIT_TAG $OTHER")")"

# A repository whose path merely contains the marker is still a second
# declaration, not a licence to pick either one.
expect_fail lookalike_repository "$(printf 'FetchContent_Declare(\n  evil\n  GIT_REPOSITORY "https://github.com/attacker/ml-explore/mlx.git"\n  GIT_TAG %s)\n%s' "$OTHER" "$(mlx_block "GIT_TAG $GOOD")")"

# Nothing to resolve.
expect_fail no_mlx_declaration "$(printf 'FetchContent_Declare(\n  fmt\n  GIT_REPOSITORY "https://github.com/fmtlib/fmt.git"\n  GIT_TAG %s)\n' "$OTHER")"
expect_fail no_git_tag "$(printf 'FetchContent_Declare(\n  mlx\n  GIT_REPOSITORY "https://github.com/ml-explore/mlx.git"\n)\n')"

# Only a full lowercase hex SHA is a pin. A moving ref would make the cache
# marker and the fetched-HEAD check meaningless.
expect_fail branch_name "$(mlx_block "GIT_TAG main")"
expect_fail short_sha "$(mlx_block "GIT_TAG 2c46b95")"
expect_fail uppercase_sha "$(mlx_block "GIT_TAG $(printf '%s' "$GOOD" | tr 'a-f' 'A-F')")"

# A missing file must fail loudly rather than resolve to an empty pin, which
# release.yml would read as "every cache is stale".
if MLX_PIN_CMAKE_FILE="$work/does-not-exist" "$under_test" >/dev/null 2>&1; then
  echo "FAIL missing_file: expected a non-zero exit" >&2
  failures=$((failures + 1))
else
  echo "ok   missing_file -> rejected"
fi

if [ "$failures" -ne 0 ]; then
  echo "$failures case(s) failed" >&2
  exit 1
fi
echo "all cases passed"
