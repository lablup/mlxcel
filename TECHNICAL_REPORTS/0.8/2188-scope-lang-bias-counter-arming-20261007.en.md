# Technical Report: PR #2188 - Scope the Lang-Bias Counter Arming to the Test Thread

**Date**: 2026-10-07

**Status**: Implemented and verified on the gfx1151 host (ROCm/HIP); head `c0a70f60` (rebased onto origin/main `97f35bca`), PR open, pending merge. Closes #2187.

**Languages**: Rust (mlxcel-core new `lang_bias_counters.rs`, `lib.rs`, `sampling.rs`; mlxcel `sampling_observability_tests.rs`), Markdown (`docs/environment-variables.md`)

**Risk Level**: Low. The production gate keeps its exact semantics: the same variable, the same parse, the same process-wide `OnceLock`. The only new runtime behavior is a thread-local override that is `None` unless a test sets it, so production reads one thread-local `Cell` before the cached value. All other changes are in tests and docs.

## Executive Summary

After PR #2182 restored the Gemma 3 decode lookahead, five of its scheduler tests failed in the full single-threaded `-p mlxcel --lib` run on ROCm, and only there. Each passed alone and alongside any single module's tests, so `make verify-rocm` broke at 5 failures with no obvious culprit.

A 13-step bisection over the 5,236 tests that sort before the victims ended at `sampling_observability_tests`. Every test in that file calls `counter_lock()`, which armed the opt-in lang-bias suppression counters with `set_var("MLXCEL_LANG_BIAS_COUNTERS", "1")`. The gate caches that variable in a process-wide `OnceLock`, so from then on every biased row in the process failed `row_supports_fused_batch_except_bias`. The victims set `ignore_eos`, which installs a token bias, so `lookahead_params` returned `None` and the lookahead never primed.

PR #2188 moves the gate into a new `mlxcel_core::lang_bias_counters` module and adds a thread-local RAII `scoped_override`. The observability tests hold that guard for their whole body, and the `set_var` is gone. The variable, which was undocumented, now has a row in `docs/environment-variables.md`. There is no production impact. The leak is pure Rust process state, so Metal and CUDA lib runs could hit it too.

The orchestrator's `make verify-rocm` on `c0a70f60` passed every step: 11,993 tests passed, 0 failed, 383 ignored, smoke OK. This returns the ROCm gate to 0 failures after #2182.

## 1. Problem Statement

### 1.1 The symptom

On origin/main `9c0ae2e9`, `cargo test -p mlxcel --lib --profile test-fast --features rocm -- --test-threads=1` reported 8,844 passed, 5 failed, 156 ignored, reproducibly. All five failures were tests added or re-enabled by #2182 in `src/server/batch/scheduler/mod.rs`:

- `scheduler_completion_snapshot_tests::model_owned_sequence_primes_the_decode_lookahead_only_with_a_rewind` ("Gemma 3 rewinds its own state, so it pipelines")
- `scheduler_model_owned_lookahead_tests::batched_lookahead_matches_force_sync_past_the_window` and `single_sequence_lookahead_matches_force_sync_past_the_window` ("the lookahead pipeline ran (0 ticks)")
- `scheduler_model_owned_lookahead_tests::failed_rewind_fails_the_request` ("the request is still running")
- `scheduler_model_owned_lookahead_tests::lookahead_is_gated_on_the_rewind_capability` ("Dense: FP16 Gemma 3 pipelines")

They passed alone and with only the `models`, `execution`, `loading` or `server` module tests. The issue was filed during PR #2186's verification, which does not touch the scheduler; the first hypotheses (`MLXCEL_FORCE_SYNC`, a global KV cache mode) were guesses.

### 1.2 The bisection

The victims run late in a single-threaded run, so the interfering test had to sort before them. The 5,236 tests that do were halved repeatedly. Each step ran one half plus `lookahead_is_gated_on_the_rewind_capability` with `--exact --test-threads=1` on the prebuilt test-fast binary, so no step needed a rebuild. Thirteen steps (log2 of 5,236 rounds up to 13) ended at `sampling_observability_tests::byte_fragment_suppression_counter_tracks_opt_in_entries`.

That test was only the first in sort order. Any test in the file triggers the failure, because each one takes `counter_lock()`, and `counter_lock()` armed the counters. As an independent check, running a victim alone with `MLXCEL_LANG_BIAS_COUNTERS=1` exported reproduces the same failure, which ties it to the variable and not to some other side effect of the observability tests.

### 1.3 The leaked state

The B9 lang-bias observability counters come in two kinds. `mlxcel_lang_bias_applied_total` is free and always counted. `mlxcel_lang_bias_tokens_suppressed_total` and `mlxcel_lang_bias_byte_fragment_suppressions_total` need the pre-bias argmax, which costs an `eval` plus a device-to-host read on every biased decode step (measured at 13% to 78% of decode). Those two are opt-in through `MLXCEL_LANG_BIAS_COUNTERS`, and while they are on, a biased row cannot use the batched fused sampler:

```rust
if !config.token_bias.is_empty() && lang_bias_counters_enabled() {
    return false;
}
```

`lang_bias_counters_enabled()` read the environment once into a `static OnceLock<bool>`. The observability tests needed the counters on, so `counter_lock()` called `arm_counters()`, which did `set_var("MLXCEL_LANG_BIAS_COUNTERS", "1")` under a `Once`. Its comment said nothing else in the test binary samples with a non-empty bias. #2182 made that false.

The chain in the victims:

1. The scheduler tests build requests with `ignore_eos`, which installs a bias on the EOS token, so `config.token_bias` is non-empty.
2. `lookahead_params` (`src/server/batch/scheduler/decode_tick.rs`) calls `batched_decode_fused_params`, which rejects any row for which `row_supports_fused_batch_except_bias` is false.
3. With the gate cached as `true`, every biased row is rejected, `lookahead_params` returns `None`, and the scheduler falls back to the synchronous tick. The tests then see 0 lookahead ticks.

In the single-threaded run, `sampling_observability_tests` sorts before `server::batch::scheduler`, so the arming always came first. In a parallel run it would depend on which test touched the gate first.

## 2. Change Summary

| Area | Change |
|---|---|
| `mlxcel-core/src/lang_bias_counters.rs` (new) | `enabled()`: a thread-local override wins, otherwise the process-wide `OnceLock` read of `MLXCEL_LANG_BIAS_COUNTERS` (same parse as before). `scoped_override(bool) -> ScopedOverride` RAII guard, `#[must_use]`, `!Send`, `#[doc(hidden)]`. Unit test `override_is_scoped_nested_and_thread_local` |
| `mlxcel-core/src/lib.rs` | `pub mod lang_bias_counters;` |
| `mlxcel-core/src/sampling.rs` | Removes `lang_bias_counters_enabled()`; `apply_token_bias_stage` and `row_supports_fused_batch_except_bias` call `crate::lang_bias_counters::enabled()`. `config_supports_fused_batch_false_for_token_bias` pins the gate off and asserts that a biased row leaves the fused path while it is on |
| `src/sampling_observability_tests.rs` | `counter_lock()` returns a `CounterScope { _counters: ScopedOverride, _lock: MutexGuard }`; `arm_counters()` and its `unsafe set_var` are removed; module docs updated |
| `docs/environment-variables.md` | New `MLXCEL_LANG_BIAS_COUNTERS` row |

Two commits: the fix (`e4ddc1fc`) and review follow-ups that pin the gate in the predicate test and tighten docs (`c0a70f60`). 5 files, 172 insertions and 40 deletions.

## 3. Design

### 3.1 A thread-local override, not a resettable cache

The root problem is a test mutating process state that outlives it. Several fixes were possible:

- **Unset the variable after the test.** Does nothing: the `OnceLock` has already cached `true`.
- **Make the cache resettable** (an `AtomicU8` with a reset hook, or reading the environment on every call). This changes production behavior, adds a per-step `env::var` or an atomic that tests can still flip for the whole process, and keeps the race between a parallel test arming the gate and a scheduler test on another thread.
- **Move the observability tests to their own test binary.** Isolates this case but leaves the trap for the next test that sets the variable, and costs another link.
- **Thread-local override (chosen).** The override lives in `thread_local! { static OVERRIDE: Cell<Option<bool>> }`. `enabled()` returns the override when it is `Some`, otherwise the cached environment value. Nothing outside the calling thread can see it, so no other test, in a serial or parallel run, is affected.

The design relies on one fact checked in the tests: the observability tests' sampling calls run on the test thread. `sample_token_optimized` is called directly, not through a scheduler thread, so the override on the test thread covers it. The PR makes that load-bearing in a way that fails loudly: with the variable unset, the counter assertions in `sampling_observability_tests` would fail if the override did not reach the sampling call.

### 3.2 The RAII guard

`scoped_override(enabled)` swaps `Some(enabled)` into the cell and returns a guard that stores the previous value; `Drop` restores it. That gives nesting for free (an inner `scoped_override(false)` inside an outer `true` restores `true`), which the unit test asserts along with the thread isolation (a spawned thread sees the baseline, not the override).

Two type-level choices keep the guard honest:

- **`!Send`** via `PhantomData<*const ()>`. Dropping the guard on another thread would restore that thread's cell, not the one it modified. The compiler now refuses to move it.
- **`#[must_use]`.** `let _ = scoped_override(true);` would drop immediately and arm nothing.

Out-of-order drops restore a stale value; the doc comment says so rather than adding a stack. `#[doc(hidden)]` keeps the function out of the public docs, since production configures the counters only through the environment.

### 3.3 Lock and override drop order

`CounterScope` holds the override and the mutex guard. Rust drops struct fields in declaration order, so `_counters` is declared first: the override is cleared before the lock is released, and the next observability test cannot start while this thread still has the counters armed. Since the override is thread-local the order is not strictly needed for correctness today, but it keeps the lock covering the full armed window if a test ever spawns work.

### 3.4 Pinning the predicate test

`config_supports_fused_batch_false_for_token_bias` asserted that a biased row stays on the fused path. If a developer exported `MLXCEL_LANG_BIAS_COUNTERS=1`, it would have failed for an unrelated reason. It now holds `scoped_override(false)` for its body and adds a nested `scoped_override(true)` block that asserts the row leaves the fused path. The test now covers both branches of the gate and is independent of the shell environment.

### 3.5 Why the module move

The gate was a private function inside `sampling.rs`, which tests in the `mlxcel` crate cannot reach. Moving it to its own public module gives `sampling_observability_tests` a supported way to arm the counters and gives the gate's contract (cached, opt-in, why it leaves the fused path, how tests must arm it) a single home in the module docs.

## 4. Production Impact

None. In production, `MLXCEL_LANG_BIAS_COUNTERS` is set (or not) before the server starts, and nothing changes it afterward, so a server that serves one model family and then another sees the same setting throughout. `scoped_override` is never called from production code, so the thread-local is always `None` and `enabled()` returns the same cached value as before.

The cost is one thread-local `Cell::get` per call, on the two call sites that already ran per biased decode step. That is negligible next to the sampling work around it.

The leak is backend-independent. It lives in Rust process state, not in any kernel or device path, so a single-threaded Metal or CUDA lib run would hit the same five failures, and a parallel run on any backend could hit them whenever an observability test armed the gate before the scheduler tests reached the check. It surfaced on ROCm only because `make verify-rocm` is the gate that runs the lib tests with `--test-threads=1`.

## 5. Documentation

`MLXCEL_LANG_BIAS_COUNTERS` existed since the counters became opt-in but had no entry in `docs/environment-variables.md`. The new row records:

- The accepted values: any non-empty value other than `0` enables; default off.
- What it collects: the two suppression counters; `mlxcel_lang_bias_applied_total` is counted either way.
- The cost and the side effect operators need to know: requests with a token bias (`logit_bias`, `ignore_eos`, `--lang-bias`) leave the batched fused sampler and therefore the lookahead decode pipeline; unbiased requests are unaffected.
- That it is read once per process, so it must be set before the server starts.
- That tests arm the counters with `scoped_override`, never by setting the variable (issue #2187).

The side effect on lookahead was the part most likely to surprise someone enabling the counters to diagnose a suppression issue.

## 6. Verification

On gfx1151 (`--features rocm`, test-fast profile, `--test-threads=1`):

- **Minimal pair.** `byte_fragment_suppression_counter_tracks_opt_in_entries` plus each of `model_owned_sequence_primes_the_decode_lookahead_only_with_a_rewind` and `lookahead_is_gated_on_the_rewind_capability`: 1 failed before, 2 passed after.
- **Full lib run.** `cargo test -p mlxcel --lib --profile test-fast --features rocm --no-fail-fast -- --test-threads=1`: 8,849 passed, 0 failed, 156 ignored (before: 8,844 passed, 5 failed).
- **Targeted tests.** `sampling_observability_tests`: 6 passed. `lang_bias_counters::tests::override_is_scoped_nested_and_thread_local` passes. `config_supports_fused_batch_false_for_token_bias` passes with and without `MLXCEL_LANG_BIAS_COUNTERS=1` exported.
- **Unit's `make verify-rocm`** with `MLXCEL_ROCM_SMOKE_MODEL=models/mlx/Qwen3-0.6B-4bit`: script gates, fmt, clippy `-D warnings` and the 32-token smoke pass; 11,987 passed, 0 failed, 382 ignored across 152 test binaries.
- **Orchestrator's `make verify-rocm`** on head `c0a70f60` (rebased onto `97f35bca`): every step passed, 11,993 passed, 0 failed, 383 ignored, smoke OK. This is the first green ROCm gate since #2182.

Not verified: Metal and CUDA, which are not available on this host. No Metal or CUDA path is touched, and the change is backend-independent Rust.

## 7. Technical Decisions

- **Scope the override to the thread.** It is the only option that cannot leak into another test, in either a serial or a parallel run, and it leaves the production gate unchanged.
- **Keep the process-wide `OnceLock`.** Reading the environment per decode step or making the cache mutable would change production for a test-only need.
- **RAII guard, `!Send`, `#[must_use]`.** The type rules out the misuse that would recreate the leak or silently arm nothing.
- **Fail loudly if the override stops reaching the sampling call.** With the variable unset, the observability assertions would fail if the override did not take effect, so a future refactor that moves sampling off the test thread will show up.
- **Pin the predicate test both ways.** It no longer depends on the developer's shell, and it now tests the counters-on branch that caused this issue.
- **Document the variable with its side effect.** The fused-path and lookahead exit is the behavior that turned a diagnostic switch into a test failure; operators get the same warning.

## 8. Residual Risks and Follow-ups

- **`metric_not_incremented_when_bias_empty` can flake under parallel `cargo test`.** It asserts `lang_bias_applied_total() == 0` after sampling with an empty bias. That counter is always on and process-global, and `counter_lock()` serializes only the tests in this file. Under the default parallel harness, any other test sampling with a bias at the same moment (the scheduler tests with `ignore_eos`, for example) increments it between `reset_counters()` and the assertion. This is pre-existing and unchanged by the PR; the single-threaded gates do not see it. Possible fixes: assert on a delta under a lock that every biased sampler respects, or make the counter assertions thread-scoped as well.
- **Other cached environment gates.** The same pattern (a test calling `set_var` on a variable cached in a `OnceLock`) can exist elsewhere, for example `MLXCEL_APC_TRACE` and other `OnceLock`-read switches. This PR fixes the one that broke the gate; an audit for other `set_var` calls on cached variables was not done.
- **Process lesson: `pkill -f` on a shared host.** During this unit, a `pkill -f` with a broad pattern was used on the shared development host, where other worktrees and agents run their own cargo and test processes. A pattern match can kill processes that belong to someone else's run. The orchestrator's brief now includes this as a rule: stop only processes you started, by PID.

## 9. Learning Points

- **Test-order failures point at leaked process state.** A test that passes alone and with every module subset but fails in the full serial run almost always has a predecessor that changed something global. Bisection over the preceding tests finds it in log2(n) runs.
- **Bisect on the prebuilt binary.** Running halves with `--exact` against the already-built test binary made each of the 13 steps a run, not a rebuild.
- **The first hit is not the whole cause.** The bisection named one test, but the mechanism was in a helper every test in the file calls. Confirming with the environment variable alone separated the mechanism from the test that happened to sort first.
- **`set_var` plus `OnceLock` is a one-way door.** Any test that sets a variable read through a process-wide cache changes it for the rest of the binary, and unsetting it later does nothing. A scoped, thread-local override is the safe shape for test-only switches.
- **Comments that state an invariant about other code go stale.** `arm_counters()` was safe only while "nothing else in this test binary samples with a non-empty bias". #2182 broke that without touching the file.
