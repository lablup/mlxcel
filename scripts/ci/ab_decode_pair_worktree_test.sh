#!/usr/bin/env bash
# Regression coverage for the A/B harness worktree lifetime and for the round
# count its summary reports (issue #3108).
#
# `scripts/xla/ab_decode_pair.sh` builds each git-ref arm in a temporary
# worktree whose `target/` is about 2 GB. The arms were resolved in a command
# substitution, so the registration array was appended inside a subshell and
# the EXIT trap never saw a path: every run left two worktrees behind, and
# repeated runs filled /tmp. The measurement values were likewise kept in two
# fixed /tmp files that the summary read back without noticing a lost round, so
# a ten-round run could report `n=3`.
#
# This test drives the real script inside a throwaway repository with a `cargo`
# stub, so it needs no toolchain, no GPU and no model. It checks the normal
# exit, a SIGTERM during a build, and the round-count guard.

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
ab="$repo_root/scripts/xla/ab_decode_pair.sh"

work="$(mktemp -d "${TMPDIR:-/tmp}/ab-worktree-test.XXXXXX")"
trap 'rm -rf "$work"' EXIT

failures=0

fail() {
  echo "FAIL [$1]: $2" >&2
  failures=$((failures + 1))
}

check() {
  local name=$1 condition=$2 message=$3
  if eval "$condition"; then
    echo "ok   [$name]"
  else
    fail "$name" "$message"
  fi
}

# The script records its worktrees under $TMPDIR, so pinning TMPDIR to the
# scratch directory keeps every leftover inside it and makes one visible.
export TMPDIR="$work/tmp"
mkdir -p "$TMPDIR" "$work/bin" "$work/repo/model"

# --- throwaway repository ---------------------------------------------------
repo="$work/repo"
git -C "$repo" init -q
printf 'fixture\n' > "$repo/README"
git -C "$repo" -c user.email=test@example.com -c user.name=test add README
git -C "$repo" -c user.email=test@example.com -c user.name=test commit -qm "fixture"

# --- fake mlxcel and cargo --------------------------------------------------
# The fake binary prints the two profile lines `measure` parses. AB_TEST_RUN_LOG
# counts invocations so a scenario can drop the decode rate after a given call,
# and AB_TEST_DROP_AFTER selects that call.
fake_mlxcel="$work/bin/mlxcel-fake"
cat > "$fake_mlxcel" <<'INNER'
#!/usr/bin/env bash
if [[ -n "${AB_TEST_RUN_LOG:-}" ]]; then
  calls=$(( $(cat "$AB_TEST_RUN_LOG" 2>/dev/null || echo 0) + 1 ))
  printf '%s\n' "$calls" > "$AB_TEST_RUN_LOG"
  if [[ -n "${AB_TEST_DROP_AFTER:-}" ]] && (( calls > AB_TEST_DROP_AFTER )); then
    exit 0
  fi
fi
printf 'Prefill: 10.00 ms\n'
printf 'Decode: 92 tokens in 460.00 ms (200.00 tok/s)\n'
INNER
chmod +x "$fake_mlxcel"

cargo_stub="$work/bin/cargo"
cat > "$cargo_stub" <<'INNER'
#!/usr/bin/env bash
if [[ -n "${AB_TEST_SLOW_CARGO:-}" ]]; then
  sleep 8
fi
mkdir -p target/release
cp "$AB_TEST_FAKE_BIN" target/release/mlxcel
chmod +x target/release/mlxcel
INNER
chmod +x "$cargo_stub"

export AB_TEST_FAKE_BIN="$fake_mlxcel"
export PATH="$work/bin:$PATH"

run_ab() {
  # run_ab <log> [extra args...]
  local log=$1
  shift
  ( cd "$repo" && bash "$ab" --model "$repo/model" --device local-task \
      --old HEAD --new HEAD --tokens 4 --allow-identical --skip-equality "$@" ) \
    >"$log" 2>&1
  echo $?
}

worktree_count() {
  git -C "$repo" worktree list | wc -l
}

ab_dirs() {
  find "$TMPDIR" -maxdepth 1 -name 'mlxcel-ab-*' -type d | wc -l
}

# --- 1. normal exit ---------------------------------------------------------
rc=$(run_ab "$work/normal.log" --rounds 2)
check normal-exit "[[ $rc -eq 0 ]]" "expected exit 0, got $rc"
check normal-removed "grep -q 'removed:' '$work/normal.log'" "no removal was reported"
check normal-count "grep -q 'cleaned 2 worktree(s)' '$work/normal.log'" "cleanup did not account for both worktrees"
check normal-registry "[[ $(worktree_count) -eq 1 ]]" "worktrees are still registered"
check normal-dirs "[[ $(ab_dirs) -eq 0 ]]" "worktree directories remain in \$TMPDIR"
check normal-summary "grep -q 'n=2' '$work/normal.log'" "the summary did not cover both rounds"

# --- 2. SIGTERM during a build ----------------------------------------------
# `exec` so the recorded pid is the script itself and the signal reaches its
# own traps, not an intermediate subshell.
(
  cd "$repo" && AB_TEST_SLOW_CARGO=1 exec bash "$ab" --model "$repo/model" \
    --device local-task --old HEAD --new HEAD --tokens 4 --rounds 1 \
    --allow-identical --skip-equality
) >"$work/term.log" 2>&1 &
term_pid=$!
sleep 3
kill -TERM "$term_pid" 2>/dev/null || true
wait "$term_pid"
rc=$?
check term-exit "[[ $rc -eq 143 ]]" "expected exit 143 after SIGTERM, got $rc"
check term-cleanup "grep -q 'removed:' '$work/term.log'" "the interrupted arm was not removed"
check term-registry "[[ $(worktree_count) -eq 1 ]]" "the interrupted worktree is still registered"
check term-dirs "[[ $(ab_dirs) -eq 0 ]]" "the interrupted worktree directory remains"

# --- 3. a round that produced no rate is not reportable ---------------------
export AB_TEST_RUN_LOG="$work/runs.log"
export AB_TEST_DROP_AFTER=4
fresh_log="$work/lost.log"
rm -f "$AB_TEST_RUN_LOG"
rc=$(run_ab "$fresh_log" --rounds 2)
unset AB_TEST_RUN_LOG AB_TEST_DROP_AFTER
check lost-round-exit "[[ $rc -ne 0 ]]" "a run missing rounds exited 0"
check lost-round-guard "grep -q 'not reportable' '$fresh_log'" "the summary did not reject the missing rounds"
check lost-round-cleanup "[[ $(worktree_count) -eq 1 ]]" "the failed run left its worktrees registered"
check lost-round-dirs "[[ $(ab_dirs) -eq 0 ]]" "the failed run left worktree directories behind"

if (( failures > 0 )); then
  echo "ab_decode_pair_worktree_test: $failures failure(s)" >&2
  exit 1
fi
echo "ab_decode_pair_worktree_test: all checks passed"
