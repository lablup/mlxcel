# Technical Report: PR #2161 - Serialize rocm_gpu_guard.sh Runs with a Host-Wide Lock

**Date**: 2026-10-06

**Status**: Implemented and verified on the gfx1151 host; head `06e0abff` (up to date with origin/main `1e561f1e`), PR open, pending merge.

**Languages**: Bash (`scripts/rocm_gpu_guard.sh`), Python (`tests/test_rocm_decode_profile.py`), Markdown (`docs/benchmarks.md`)

**Risk Level**: Low (benchmark tooling only; no Rust, HIP, Metal or CUDA code changes. One behavior change for callers: `--max-wait` is now a single budget for the whole guard run)

## Executive Summary

Issue #2146 (part of epic #1801) reported that two `scripts/rocm_gpu_guard.sh` runs started together reject each other until both exit 75. The guard, added in #2061, waits for `--idle-secs` of idle GPU, runs COMMAND, and rejects the attempt if a 1 Hz monitor sees any GPU process that is not COMMAND or its descendant. Two guards with the same idle window finish the wait in the same second, start their commands together, and each monitor sees the other's command as a foreign GPU user. During the #1814 ROCm port run, parallel units worked around it by choosing distinct `--idle-secs` values by hand (45, 55, 60, 75, 90, 120 s).

The PR serializes guards host-wide with `flock`. A guard takes the lock before its first idle wait and holds it until it exits, so guards run one after another. The design closes the edge cases a naive lock would open: the lock file is created only if missing and opened read-only, a non-regular lock path is refused, `--max-wait` becomes one budget for the lock wait and the idle waits, INT and TERM still work while waiting, COMMAND runs without the lock descriptor, and a nested guard skips the lock only when `ROCM_GPU_GUARD_LOCK_HELD` names one of its ancestors.

Eight new tests cover the lock. Seven of them fail against main's script; the eighth (a daemon left by COMMAND must not keep the lock) passes on main, which has no lock, and fails on this branch when the descriptor close is removed. On the real KFD list, two guards started in the same second both finished `CLEAN` with no `CONTENDED` or `REJECTED` line; the second acquired the lock after 779 s.

## 1. Problem Statement

### 1.1 Why two guards rejected each other

`wait_idle` counts consecutive idle samples from `/sys/class/kfd/kfd/proc` and the process list. Nothing coordinated two guards: the guards are bash processes and never appear in the KFD list, so each sees an idle GPU while the other is also waiting. With equal `--idle-secs`, both streaks complete in the same second, both start COMMAND, and both monitors then see a KFD entry that is not their own COMMAND's tree. Both attempts are `REJECTED (contended)`, both go back to the same idle wait with the same window, and the lockstep repeats until `--max-attempts` (default 5) runs out and both exit 75. Nothing either COMMAND printed is usable.

### 1.2 The workaround it forced

The 2026-10 ROCm port run (#1814) ran several benchmark units in parallel. Each unit had to pick its own `--idle-secs` (45, 55, 60, 75, 90, 120 s) so their streaks would end at different times. That only lowers the collision rate, couples every caller to a host-wide naming scheme for windows, and lengthens every run that picks a longer window.

## 2. Change Summary

| Area | Change |
|---|---|
| `scripts/rocm_gpu_guard.sh` | `acquire_lock` with `flock` on `ROCM_GPU_GUARD_LOCK` (default `/tmp/mlxcel-rocm-gpu-guard.lock`); single `--max-wait` budget via bash `SECONDS` minus COMMAND run time; ancestor check for `ROCM_GPU_GUARD_LOCK_HELD`; lock descriptor closed for COMMAND and the monitor; `flock` missing exits 2; header and `usage()` range updated |
| `tests/test_rocm_decode_profile.py` | `run_guard` points `ROCM_GPU_GUARD_LOCK` at a per-test temp path and drops any inherited `ROCM_GPU_GUARD_LOCK_HELD`; `hold_lock` helper; 8 new `GuardTests` |
| `docs/benchmarks.md` | One paragraph on the lock, the budget, nesting, `sudo`, and pointing the lock elsewhere on hosts shared with untrusted users |

Three commits: the lock (`fc0bc3c2`), trusting `ROCM_GPU_GUARD_LOCK_HELD` only from an ancestor guard (`1f6ff1ad`), and refusing a non-regular lock path (`06e0abff`). The diff against origin/main is 3 files, 274 insertions and 15 deletions.

## 3. Design

### 3.1 Why a lock and not backoff

The issue rejected randomized backoff. It lowers the collision rate but every collision still burns a whole attempt, and runs started together stay correlated. A lock removes the race: only one guard is ever between "idle wait" and "COMMAND done". The lock is taken before the first idle wait and held across every attempt, so a guard whose attempt is rejected by a non-guard GPU user keeps its turn instead of racing the next guard again.

### 3.2 Creating and opening the lock file

```bash
[[ -e "$LOCK" ]] || { (umask 000; : >>"$LOCK") 2>/dev/null || true; }
[[ -f "$LOCK" ]] || { echo "... is not a regular file" >&2; exit 2; }
{ exec {LOCK_FD}<"$LOCK"; } 2>/dev/null || { echo "... cannot open guard lock" >&2; exit 2; }
```

- **Create only if missing, world-writable.** Any user's guard can share the default file. A failed create (another guard made it first) is ignored as long as the open succeeds.
- **Open read-only.** `flock` works on a read-only descriptor. An `O_CREAT` open of a file another user owns in sticky `/tmp` fails under `fs.protected_regular`, so the guard never opens with `O_CREAT` once the file exists.
- **Refuse a non-regular path.** Opening a FIFO for reading blocks until a writer appears, which would hang past any `--max-wait` before the budget code ever runs. The guard exits 2 instead.

### 3.3 One `--max-wait` budget

`SECONDS=0` is set after argument parsing, and `run_secs` accumulates the time COMMAND runs. Spent budget is `SECONDS - run_secs`, so the clock covers the lock wait and every idle wait but not the measurement itself.

- The lock wait uses `flock -w <remaining>` when `--max-wait > 0` and a blocking `flock` otherwise. A timeout logs `gave up waiting for guard lock <path> after <N>s` and exits 75 without running COMMAND.
- After acquiring, a guard with less budget left than `--idle-secs` exits 75 rather than holding the lock through an idle wait it cannot finish, which would only delay the guards queued behind it.
- `wait_idle` checks the same budget, and only while the streak is incomplete, so a streak that completes exactly at the deadline still runs.

### 3.4 Signals during the lock wait

Bash defers a trap until a foreground child exits, so a foreground `flock` would leave the guard deaf to INT and TERM for the whole lock wait. `acquire_lock` runs `flock` in the background on the inherited descriptor and calls `wait`, which the trapped signal interrupts. Because the lock belongs to the open file description, the background `flock` locks it for the guard shell too. `on_signal` kills `lock_pid` along with COMMAND and the monitor, so no `flock` outlives the guard, and the guard exits 130 or 143.

### 3.5 The descriptor is closed for COMMAND and the monitor

COMMAND starts as `"$@" {LOCK_FD}<&- &` and the monitor subshell with the same redirection. A daemon that COMMAND leaves behind would otherwise inherit the descriptor and hold the lock after the guard exits, blocking every later guard indefinitely. The same applies to the monitor's last `sleep 1`.

### 3.6 Nesting

A guard exports `ROCM_GPU_GUARD_LOCK_HELD=$$` to COMMAND. A guard that starts with that variable skips the lock, logging `lock held by outer guard <pid>`, only if the value is a positive integer, is not its own pid, and names one of its ancestors per a `/proc/<pid>/stat` parent walk that stops before pid 1. Without the ancestor check (the first commit trusted any value), a reused pid, an export carried by a daemon, or pid 1 would let an unrelated guard run unlocked. With it, nesting cannot deadlock and a stale value cannot bypass the lock. When the outer guard holds the lock, the inner guard opens `/dev/null` as a placeholder descriptor so the closing redirections on COMMAND and the monitor stay unconditional.

Two limits are documented in the header and `docs/benchmarks.md`: guards nested under one outer guard are not serialized among themselves, and an environment reset (for example `sudo` without `--preserve-env=ROCM_GPU_GUARD_LOCK_HELD`) makes a nested guard wait on its own ancestor's lock.

### 3.7 `flock` missing

When no outer guard holds the lock and `flock` is absent, the guard exits 2 with `rocm_gpu_guard: flock not found (util-linux)`. Running unlocked would bring the lockstep back without warning.

## 4. Behavior Change: `--max-wait`

Before this PR, `wait_idle` kept its own counter, so `--max-wait` limited each attempt's idle wait separately, and a guard with 5 attempts could wait up to 5 times `--max-wait`. Now `--max-wait` limits all waiting in the run: the lock wait plus every idle wait, from guard start, minus the time COMMAND runs. The give-up message changed to `gave up after <MAX_WAIT>s of waiting without <IDLE_SECS>s of idle`. No script in the repository passes `--max-wait`, so no in-tree caller changes behavior; an external caller that relied on the per-attempt meaning gets a shorter total wait.

## 5. Verification

### 5.1 New tests and which fail against main

The new tests run against a fake KFD directory and a per-test lock path, so they never touch the host lock. Running the branch's test file against main's `rocm_gpu_guard.sh` (in a scratch copy, for this report):

| Test | Against main |
|---|---|
| `test_two_guards_started_together_run_one_after_the_other` | Fails: both guards exit 75 (the issue's bug) |
| `test_a_held_lock_counts_against_max_wait_and_the_command_never_runs` | Fails |
| `test_a_guard_inside_a_guard_does_not_wait_for_the_lock` | Fails |
| `test_a_held_variable_naming_a_non_ancestor_does_not_skip_the_lock` | Fails: main exits 0 where 75 is expected; per the PR, also fails against the first commit |
| `test_a_lock_path_that_is_not_a_regular_file_is_refused` | Fails |
| `test_sigterm_while_waiting_for_the_lock_stops_the_guard` | Fails |
| `test_a_nested_guard_command_completes` | Fails on the `lock held by outer guard` assertion (the nested command itself completes on main) |
| `test_a_daemon_left_by_the_command_does_not_keep_the_lock` | Passes (main has no lock); per the PR, fails on this branch with the `{LOCK_FD}<&-` close removed |

Result on main's script: 26 tests run, 7 failures. The two-guard test asserts both exit 0, no `CONTENDED` line, and four ordered events (start, CLEAN, start, CLEAN) with one guard logging `waiting for guard lock`.

### 5.2 Gates

From the orchestrator on head `06e0abff` (up to date with origin/main):

- `bash -n scripts/rocm_gpu_guard.sh`: OK.
- `python3 -m unittest tests/test_rocm_decode_profile.py`: 26 tests OK (17 guard tests).
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-rocm-overlay verify-fmt`: exit 0.

From the PR: `cargo test --features rocm --test dead_doc_pointers` passed; SIGTERM during a manual lock wait exits 143 with no `flock` left; with `flock` hidden from `PATH` the guard exits 2 with the message above. The full `make verify-rocm` gate was not run because the change is the script, its tests and docs only.

### 5.3 End to end on gfx1151

Two guards (at commit `1f6ff1ad`; the last commit only adds the regular-file check and comments), default lock path, `--idle-secs 20 --max-attempts 2`, one shared `--log`, started in the same second, each around a 1-token `mlxcel generate -m models/mlx/Qwen3-0.6B-4bit` on the real KFD list:

- Guard A logged `acquired guard lock ... after 0s`; guard B logged `waiting for guard lock`.
- A's idle wait lasted until another unit's GPU job and a cargo build finished, then it logged `attempt 1: CLEAN, exit 0` at 22:05:25.
- B logged `acquired guard lock ... after 779s` in the same second, waited its own 20 s idle window, and logged `attempt 1: CLEAN, exit 0` at 22:05:57.
- Neither run has a `CONTENDED` or `REJECTED` line.

The 779 s also shows the cost of the design: B queued behind A for as long as A waited on an unrelated GPU user, because the lock is held across the idle wait.

## 6. Technical Decisions

- **Serialize rather than tolerate.** Guards cannot tell each other's commands from any other GPU user through the KFD list, so the fix coordinates them outside it rather than teaching the monitor about other guards.
- **Hold the lock from the first idle wait to exit.** Taking it only around COMMAND would let the second guard's idle window end before the first COMMAND ran, so the second would measure right after the first without a fresh idle window.
- **Read-only open of a create-if-missing file.** This is what makes one default path work for every user under `fs.protected_regular`.
- **One budget instead of per-attempt limits.** With a lock wait added, per-phase limits would make the real bound the sum of several limits times the attempt count; one clock answers "how long may this guard wait" directly.
- **Background `flock` plus `wait`.** Keeps the existing INT and TERM contract during the new wait at the cost of one tracked pid.
- **Ancestor check for nesting.** An environment variable alone is forgeable by accident (pid reuse, daemons); a parent walk ties the skip to the actual process tree.
- **Fail closed without `flock`.** Exit 2 rather than silently reintroducing the race.

## 7. Residual Risks and Follow-ups

- **Shared default path.** Any local user can open `/tmp/mlxcel-rocm-gpu-guard.lock` and hold it, or pre-create it unreadable, so a hostile user can block every guard. The docs tell hosts shared with untrusted users to set `ROCM_GPU_GUARD_LOCK` to a path only benchmarking users can reach.
- **No fairness among waiters.** `flock` does not guarantee FIFO order, so with three or more guards queued, start order is not acquisition order. Only the two-guard case was run end to end.
- **Head-of-line blocking.** A guard waiting on a non-guard GPU user holds the lock the whole time (779 s in the run above). Callers that need a bound should pass `--max-wait`, now a total.
- **Cross-user sharing not exercised.** The `fs.protected_regular` reasoning was not tested with two different users on the host.
- **Nested guards under one outer guard** are not serialized among themselves; this is documented, not enforced.
- **Metal and CUDA** were not run; the guard is ROCm-only and the change touches no Metal or CUDA path.

## 8. Learning Points

- **A guard can be its own contention.** A monitor that rejects any foreign GPU user needs a way to keep peers from becoming that user; process-list filtering alone cannot do it.
- **Bash traps wait for foreground children.** Any long blocking call in a script that promises signal handling should run in the background and be `wait`ed on.
- **Inherited descriptors outlive their owner.** A lock held through a file descriptor must be closed in every child that may outlive the holder, or a stray daemon owns the lock.
- **`/tmp` has its own open semantics.** `fs.protected_regular` turns a harmless `>>` into a failure for another user's file; create once, then open without `O_CREAT`.
- **Opening a path can block.** Checking `-f` before `open` keeps a FIFO from hanging the script past its own timeout.

Refs: #2146, #2061, #2065, #1814, #1801.
