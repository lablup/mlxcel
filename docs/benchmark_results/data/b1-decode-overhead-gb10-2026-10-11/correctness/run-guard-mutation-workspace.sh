#!/usr/bin/env bash
set -euo pipefail

worktree=${WORKTREE:-/home/inureyes/Development/backend.ai/wt-auto-20261011T0224-issue-2227}
patch=/tmp/mlxcel-auto-plan/2227-acceptance/mutate-terminal-guard.patch
file=src/server/batch/scheduler/decode_tick.rs
mutated=0

cd "$worktree"
git diff --quiet -- "$file" || {
    echo "refusing to mutate a dirty $file" >&2
    exit 2
}
git apply --check "$patch"

restore_guard() {
    if (( mutated )); then
        git apply -R "$patch"
        mutated=0
    fi
}
trap restore_guard EXIT INT TERM

git apply "$patch"
mutated=1

cargo test --workspace --profile test-fast --features cuda --no-run
set +e
gpu-lock run --tag issue-2227-guard-mutation-test -- target/test-fast/deps/mlxcel-e4114342978c2f44 \
    b1_terminal_length_skips_two_forwards_and_leaves_b2_unchanged \
    --test-threads=1
test_rc=$?
set -e

restore_guard
trap - EXIT INT TERM
git diff --exit-code -- "$file"

if (( test_rc != 101 )); then
    echo "mutation did not produce the required assertion failure" >&2
    exit 1
fi

echo "mutation killed as expected (cargo test exit $test_rc)"
