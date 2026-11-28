# PR #2946: Serialize mlxcel-server env tests and stop lock-poison cascades

**Date**: 2026-11-10
**Status**: Completed (PR open as a draft; closes #2937; the full `make verify-test` run was still pending when this report was written)

## Executive Summary

Issue #2937 reported 8 failures from `cargo test --workspace --profile test-fast --features metal,accelerate --no-fail-fast` on an M1 Ultra. That command is the one the local agent instructions document, and it runs libtest threaded. The CI-faithful gate, `make verify-test` and the nightly workflow, adds `-- --test-threads=1` on purpose, because #1092 and #1210 traced Metal device faults to many MLX-backed tests running concurrently inside one binary. Most of the reported failures are therefore outside the gate by design. The PR fixes the two defects that made failures harder to read whatever the threading: an environment race among the `mlxcel-server` argument tests, and a lock-poison cascade in the dense batched decode tests. Every one of the 52 tests in the `mlxcel-server` test module now takes a single poison-tolerant `cli_env_lock()` guard, and the parse helpers and the `ScopedEnv` writers assert that it is held. On the untouched tree, 5 of 6 threaded runs of `--bin mlxcel-server` failed. With the fix, 15 of 15 threaded runs of the whole binary passed (10 in the first commit, 5 more after the review follow-up, each 51 passed and 1 ignored), and 5 of 5 runs of the issue's four-test filter passed. No tolerance or snapshot changed. 3 files, +154/-85.

## 1. Problem Statement

### 1.1 The documented command and the real gate differ

The "Build and test" block in the local agent instructions gives `cargo test --workspace --profile test-fast --features metal,accelerate` without `-- --test-threads=1`. `make verify-test` and the nightly workflow serialize test execution within each binary because concurrent MLX-backed tests fault the Metal device (#1092, #1210). `CONTRIBUTING.md` repeated the threaded form for macOS while its own Linux/CUDA line already carried `--test-threads=1`. A contributor following either document saw failures the gate never produces: the glm4 case in `src/models/popular_dense_batched_decode_tests.rs`, one test each in `qwen4_exp_snapshot_tests.rs`, `dots1_tests.rs` and gemma_unified nvfp4, and once a layer_norm test in `src/imagegen/mage_flow/vae/vae_tests.rs`. All of them pass alone and under `--test-threads=1`, and the same set reproduced on `main` with #2931's changes stashed.

### 1.2 The env race in `mlxcel-server`

The issue named four argument tests that raced: `legacy_server_mode_*` (two tests), `no_prompt_cache_flag_disables_prompt_cache` and `image_model_without_model_leaves_the_chat_path_empty`. They parsed argv without taking `CLI_ENV_LOCK`, while sibling tests set `LLAMA_ARG_*` and `MLXCEL_*` variables under that lock through `ScopedEnv`.

The root cause is broader than four tests. clap 4 reads each `env = "..."` binding when it builds the command (`Arg::env` calls `env::var_os`), so any test that builds `Cli`, renders its help, parses argv, or calls `build_startup_input` (which has its own env fallbacks) reads process env. The audit found 40 unlocked tests doing this, including the help and flag-introspection tests and the parent of the CORS child-process test, whose child inherits whatever env the parent has at spawn time. The four named tests were only the ones whose assertions happened to notice a sibling's value. The race also left a stray zero-byte `legacy.log` in the checkout: a test built a logging configuration while a sibling held `LLAMA_LOG_FILE=legacy.log`.

### 1.3 Poison cascades

Seven of the 12 existing `CLI_ENV_LOCK` sites took the lock with `.expect("CLI env lock")`, and the dense batched decode tests took `test_guard()` with `.expect("dense test lock")`. When one test panicked while holding the mutex, every later test that took it failed with a `PoisonError`, so a single real failure was reported as many and the original was hard to find.

## 2. What Changed

### 2.1 `src/bin/mlx_server.rs` test module

- **One guard.** `cli_env_lock()` returns a `CliEnvGuard` that wraps the `MutexGuard` and recovers from poison with `unwrap_or_else(PoisonError::into_inner)`. It replaces the 12 hand-rolled lock sites, which had mixed `expect` and poison-tolerant forms.
- **A held-lock flag.** A thread-local `CLI_ENV_LOCK_HELD` is set when the guard is taken and cleared in its `Drop`. `cli_env_lock()` asserts the flag is clear first, so taking the lock twice on one thread panics with a message instead of deadlocking.
- **Asserting helpers.** `cli_command()` wraps `Cli::command()`, `try_parse_cli()` wraps `Cli::try_parse_from()`, and `parse_server_args()` goes through `try_parse_cli()`. All three call `assert_cli_env_lock_held()`, whose message names the fix (`let _lock = cli_env_lock();`). Direct `Cli::command()` and `Cli::try_parse_from()` calls in the tests were replaced, so the `use clap::CommandFactory` imports in individual tests went away.
- **Asserting writers.** The review follow-up (`e7ec7451`) added the same assertion to `ScopedEnv::set` and `ScopedEnv::unset`, which covers the writer side of the race. The `SAFETY` comments on `set_var` and `remove_var` now point at that assertion instead of a convention.
- **Coverage.** All 52 `#[test]` functions in the module take the guard exactly once, including `both_cors_origin_envs_without_a_flag_fail_startup` (the parent, which spawns the child) and the ignored `both_cors_origin_envs_child`. The comment on the parent was rewritten: it used to say that other tests call `build_startup_input` unlocked, which is no longer true, and it now explains why the parent holds the lock while spawning.
- **Drop order.** The guard doc tells tests to bind `_lock` before any `ScopedEnv`, so locals drop in reverse order and the env is restored before the lock is released.

### 2.2 `src/models/popular_dense_batched_decode_tests.rs`

The `dense_contract!` macro takes `test_guard()` with `unwrap_or_else(PoisonError::into_inner)`, with a comment explaining why. A failing case now fails alone.

### 2.3 `CONTRIBUTING.md`

The macOS test command is now the serial form, annotated as the same command `make verify-test` runs. One sentence explains why it must stay serial and cites #1092 and #1210. The wording avoids calling it "the gate" because step 4 of the same document says tests are not gated at PR time; it says the command is what `make verify-test` and the nightly workflow run.

## 3. Validation

| Run | Before | After |
|-----|--------|-------|
| `--bin mlxcel-server`, threaded, whole binary | 5 of 6 failed (`image_model_without_model_*` 3 times, `legacy_server_mode_*` and `no_prompt_cache_flag_*` 1 to 2 times each) | 15 of 15 passed (10 in the first commit, 5 after the follow-up; each 51 passed, 1 ignored) |
| Issue's four-test filter, threaded | failed intermittently | 5 of 5 passed |
| `--lib popular_dense_batched_decode`, threaded | not measured | passed (15 passed) |
| Stray `legacy.log` after threaded runs | created | none in 5 runs |

- Negative check: removing the guard from one test makes it fail with the assertion message, so the check is live and does not just document a convention.
- `cargo clippy --lib --tests --bin mlxcel-server --features metal,accelerate -- -D warnings` and `cargo fmt --check` are clean.
- No numeric tolerance was widened and no snapshot was changed. None of the MLX-backed failures reproduced under `--test-threads=1`, so the issue's condition for touching one was never met.
- The full `make verify-test` pass, the issue's last acceptance criterion, was running in the worktree when this report was written. Its result is not recorded here.

## 4. Technical Decisions

### 4.1 Serialize the env tests instead of removing env reads

**Context:** clap reads env whenever `Cli` is built, so a test cannot opt out of env reads short of not building `Cli`.

| Option | Pros | Cons |
|--------|------|------|
| Lock every test that builds, parses, or spawns (chosen) | Small, mechanical change; matches the existing `CLI_ENV_LOCK` design; the tests keep exercising real clap env bindings | Serializes 52 tests within the binary; relies on every new test taking the guard |
| Strip env bindings in tests (for example, parse with a `Command` that has env disabled) | Tests run in parallel | Tests would stop covering the env bindings, which are part of the `LLAMA_ARG_*` compatibility surface |
| Move env-mutating tests into child processes, as the CORS test does | True isolation | A process spawn per test, and a much larger diff |

The tests are argument parsing only and finish quickly, so serializing them costs little. The gate already runs them serially anyway.

### 4.2 Assertions that fail instead of a convention

A lock enforced only by convention is what produced 40 unlocked tests. The helpers now assert, so a new test that forgets the guard fails deterministically on its first run, threaded or not, with a message naming the fix. Without the assertion it would pass under the serial gate and race only in threaded runs. The thread-local flag also turns a second acquisition into a clear panic instead of a hang, which matters because the guard is easy to take inside a helper that a test already guarded.

### 4.3 Tolerate poison in both locks

The env lock protects no data, only exclusivity. A test that panics while holding it has already restored the env through its `ScopedEnv` drop, because the `ScopedEnv` was bound after the guard and drops first. Recovering the guard is therefore safe, and it keeps one real failure from turning into dozens of `PoisonError` failures. The dense test lock protects only ordering on the Metal device, so the same reasoning applies.

### 4.4 Fix the documentation, not the test threading

The issue explicitly ruled out widening tolerances or changing snapshots to make threaded runs pass. Making MLX-backed tests safe under concurrent execution is the #1092 and #1210 problem, and the project already answered it with the serial gate. The fix for that part is to document the gate correctly.

## 5. Limits and Open Items

- **`CLAUDE.md` is not in the PR.** The issue's first acceptance criterion names `CLAUDE.md`, but in this checkout it is a gitignored local symlink to an untracked `AGENTS.md`. Its "Build and test" line still shows the threaded command and has to be fixed outside the tree. `CONTRIBUTING.md` carries the tracked fix.
- **The MLX-backed parallel-only failures remain.** The glm4 dense case, the qwen4_exp snapshot, dots1, gemma_unified nvfp4 and the vae layer_norm test can still fail under a threaded run. The PR makes only the poison cascade in the dense tests go away; the underlying concurrency faults are out of scope by design.
- **`build_startup_input` is not itself asserted.** It reads env through its fallbacks, but it takes a `ServerArgs`, and in the tests every `ServerArgs` comes from the asserting `parse_server_args`, inside a test that holds the guard for its whole body. A future test that constructs `ServerArgs` some other way would bypass the check.
- **Scope of the guard.** It covers this one test module. Other binaries or modules that read `LLAMA_ARG_*` through clap in tests are not covered by it; this matters only if they share a process with env-mutating tests.
- **`make verify-test` is unconfirmed** as of this report (see section 3).

## 6. Learning Points

- clap's `Arg::env` reads the environment when the command is built, not only when a value is missing. Rendering help or introspecting flags reads env as well, so "this test only reads help text" is not proof it is env-independent.
- A child process inherits the parent's environment at spawn time, so isolating a test in a child does not protect it from a sibling's `ScopedEnv` unless the parent holds the lock while spawning.
- When a mutex guards exclusivity rather than data, prefer `unwrap_or_else(PoisonError::into_inner)` in tests. `expect` multiplies one failure into many.
- When a command is documented in more than one place, check each copy against what CI actually runs. Here the Linux line in `CONTRIBUTING.md` was already serial and the macOS line was not.
