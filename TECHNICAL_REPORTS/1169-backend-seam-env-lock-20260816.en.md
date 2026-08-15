# Technical Report: PR #1169 - test: acquire the env lock in backend seam tests for xla-backend builds

**Date**: 2026-08-16
**Author**: AI Code Reviewer
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low

---

## Executive Summary

Four tests in `src/backend/tests.rs` called `select_backend()` without holding the crate-wide env lock. Under the opt-in `xla-backend` / `experimental-backend` features that function reads `MLXCEL_BACKEND` (`src/backend/mod.rs:316`), so a parallel full-suite run could interleave with `muse_glimmer_startup_rejects_xla_backend_selection`, which transiently sets `MLXCEL_BACKEND=xla` while holding that same lock, and hand a backend test the wrong backend. This PR takes the lock in all four tests using the pattern PR #1157 established. Test-only change, 26 insertions and no deletions in a single `#[cfg(test)]` module, no behavior change under the feature sets CI builds.

The security pass surfaced a second justification the original issue did not anticipate: the guarded call paths perform their own environment access independent of `MLXCEL_BACKEND`, so the lock is correct under the Rust 2024 `setenv`/`getenv` rules regardless of the race that motivated it.

---

## 1. Problem Statement

### 1.1 Background

Issue #1159 was filed from the security review of PR #1157 (implementation of #1155), which added the crate-wide env lock to `src/server/muse_glimmer_startup_guard_tests.rs` and documented this residual race as deliberately out of scope. The env mutator itself arrived earlier with #1116; the backend tests predate the guard-test hardening.

### 1.2 Existing Issues

- **Issue 1**: `mlx_session_threads_the_token_bias_through` (`src/backend/tests.rs:104` pre-change) called `select_backend()` with no lock. Under `xla-backend` a racing `MLXCEL_BACKEND=xla` yields a `Session::Xla`, hitting the `unreachable!("select_backend defaults to MLX without MLXCEL_BACKEND=xla")` arm.
- **Issue 2**: The same exposure applied to three sibling tests at lines 44, 62 and 81, which the issue flagged for evaluation rather than prescribing a fix.
- **Issue 3** (found during the security pass): the guarded paths read and write the environment on their own account. `create_session` reaches `boundary_v_layers_from_env()` (`src/lib/mlxcel-core/src/cache/turbo/boundary.rs:83`, two `std::env::var` reads) and `load_model` on a real checkpoint can call `set_var("MLX_USE_CUDA_GRAPHS")` via `maybe_disable_cuda_graphs_for_model` (`src/loading/mod.rs:509`). This crate is edition 2024, where `set_var` is `unsafe` precisely because concurrent reads are unsound.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Spurious test failure under an opt-in feature build, misread as a real backend regression | Low | Low (no CI workflow builds `xla-backend` or `experimental-backend`) |
| Silent validation of the wrong backend in tests that accept any `Err` | Low | Low (same feature gating) |
| Unsynchronized `getenv` against a concurrent `setenv` under edition 2024 | Low | Low (test binary only) |

---

## 2. Technical Review

### 2.1 Security

No security surface: the diff is entirely inside `#[cfg(test)]` and `test_support` is `cfg(test)`-gated at `src/lib.rs:51-52`, so no production code can reach `env_lock()`. The security pass returned zero findings at every severity and settled the concurrency questions by tracing the full call tree under each guard:

- `select_backend()` is one `std::env::var` plus a zero-sized-type construction.
- `load_model` bails at `get_model_type` for the nonexistent path, with no lock and no network.
- `create_session` reaches only struct construction (`KVCache::new_with_mode`) and thread-local stream setup.
- The single `static Mutex` in that subtree (`sparse_v_count_file`, `src/lib/mlxcel-core/src/cache/turbo/sparse_v.rs:335`) is reachable only from the attention path, never from cache construction.

Re-entrancy is structurally impossible, and `mlxcel-core` carries its own separate `ENV_LOCK` in a separate test binary, so there is no cross-crate lock interaction.

**Issues Found:**
| Issue | Severity | Status |
|-------|----------|--------|
| None | - | - |

Two informational notes were recorded and deliberately not acted on: the explanatory comment names the `unreachable!` arm as the failure mode when `.expect("session creation must succeed")` one line earlier would in practice fire first (both are loud failures, and the framing came from the issue body), and the new rationale is appended to the pre-existing test-purpose comment without a separating blank line in two tests.

### 2.2 Performance

Negligible. Every guarded test completes in under 10 ms including process startup, and nothing under a guard does network I/O, sleeps, or joins a thread. Added contention against the other `env_lock` call sites in the test binary is immaterial.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none
- **New Dependencies**: none
- **Compatibility**: no production path affected; the lock is a no-op under default features, where `select_backend` const-folds to the single MLX variant with no environment read

### 2.4 Code Quality

- **Test Coverage**: unchanged in count; the four existing tests become deterministic under opt-in feature builds
- **Code Complexity**: one added binding and one explanatory comment per test
- **Technical Debt**: decreased; the last unguarded `select_backend()` call sites in the crate are now covered

---

## 3. Technical Decisions

### 3.1 Guard all four call sites, not only the one named in the issue

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Guard only `mlx_session_threads_the_token_bias_through` | Smallest diff, matches the issue title literally | Leaves three tests with the identical exposure, so the issue would recur |
| **Chosen: guard all four `select_backend()` call sites** | Closes the class of defect rather than one instance | Slightly larger diff |

**Rationale:** The issue explicitly asked whether lines 44, 62 and 81 warranted the same guard. Each was evaluated on its own failure mode rather than guarded reflexively. `select_backend_resolves_to_mlx_under_default_features` would fail its `matches!(backend, Backend::Mlx(_))` assertion outright under the race. `mlx_backend_creates_a_session_and_advertises_batched_serving` would fail its first assertion, because `XlaBackend::supports_batched_serving()` is `cfg!(feature = "xla-iree")` (`src/backend/xla.rs:117`), false under `xla-backend` alone. `seam_delegates_to_real_mlx_loader_on_missing_dir` is the interesting case: it would NOT fail, because `XlaBackend::load_model` routes to `load_unsupported()` and also returns `Err` (`src/backend/xla.rs:75`), so the race would silently validate the XLA scaffold while the test's docstring claims it proves the seam reaches the real MLX loader. That silent-pass case is the strongest argument for guarding it.

### 3.2 Bind the guard to a named variable

**Rationale:** `let _ = env_lock();` drops the `MutexGuard` at the end of the statement and silently reinstates the exact race. All four sites use `let _env_guard = ...` as the first binding in the function, so reverse drop order retires the guard last, after the backend, session and capability values it protects.

### 3.3 Reuse the existing lock rather than add a backend-local one

**Rationale:** The race is between two different modules' tests, so the lock has to be the crate-wide one from `src/test_support/env_lock.rs`. It already recovers from poisoning via `unwrap_or_else(|p| p.into_inner())`, so a panicking holder cannot cascade into the newly guarded tests.

---

## 4. Implementation Details

### 4.1 Key Code Changes

**File: `src/backend/tests.rs`**
```rust
// Before
#[test]
fn mlx_session_threads_the_token_bias_through() {
    let mut bias = TokenBiasMap::new();
    bias.insert(5, -2.0);
    let session = select_backend()

// After
#[test]
fn mlx_session_threads_the_token_bias_through() {
    // Under the opt-in `xla-backend` feature, `select_backend()` reads
    // `MLXCEL_BACKEND`, and `muse_glimmer_startup_rejects_xla_backend_selection`
    // transiently sets that var to "xla" while holding this same lock. Without
    // taking it here too, a parallel full-suite run could hand this test an
    // XLA session and hit the `unreachable!` arm below.
    let _env_guard = crate::test_support::env_lock::env_lock();
    let mut bias = TokenBiasMap::new();
    bias.insert(5, -2.0);
    let session = select_backend()
```

**Reason for change:** The guard must be acquired before `select_backend()` and must outlive every inspection of the resulting backend or session. Each of the four tests carries a comment explaining why the lock is needed, because under default features it looks superfluous and would otherwise be a plausible deletion for a future reader.

---

## 7. Change Summary

### Statistics
| Item | Value |
|------|-------|
| Files changed | 1 |
| Lines added | +26 |
| Lines deleted | -0 |
| Tests added | 0 (4 existing tests made deterministic) |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Test Correctness | 4 | Env lock acquired before `select_backend()` in each test |

### Related Commits
| Hash | Type | Message |
|------|------|---------|
| `5de2bc60` | test | acquire the env lock in backend seam tests for xla-backend builds |

---

## 8. Follow-up Actions

### Required
- [ ] None; all three acceptance criteria are met

### Future Improvements
- `seam_delegates_to_real_mlx_loader_on_missing_dir` accepts any `Err` and discards the message, so it cannot itself distinguish the MLX loader's error from another backend's. The lock makes it deterministic without making the assertion stronger. An `assert_eq!(backend.name(), "mlx")` before the `load_model` call would enforce what the docstring claims. Pre-existing property, out of scope for this PR.
- No CI workflow builds `xla-backend` or `experimental-backend`, so these arms are only ever exercised locally. A periodic opt-in feature build would catch this class of defect without a manual run.

---

## Appendix

### A. Test Results

Default features:

| Command | Result |
|---------|--------|
| `cargo fmt --check` | clean |
| `cargo check --lib --tests` | clean |
| `cargo test --lib backend::tests` | 5 passed, 0 failed |
| `cargo clippy --lib --tests -- -D warnings` | clean |

The issue's second acceptance criterion named `--features metal,accelerate,xla-backend`. `metal` and `accelerate` are Apple-targeted and not buildable on the Linux/CUDA machine used here, so verification used `--features xla-backend`, which is the feature that actually makes `select_backend()` read the environment and therefore the one the race depends on. It builds without `IREE_DIST`, since only `xla-iree` requires that.

| Command | Result |
|---------|--------|
| `cargo test --features xla-backend --lib backend::tests` | 7 passed, 0 failed (includes the two XLA-gated tests) |

This configuration compiles the `Session::Xla` arm, so the `unreachable!` arm in `mlx_session_threads_the_token_bias_through` is live code rather than compiled out.

Contention run, both modules in one binary on 8 threads, repeated five times:

```
for i in 1 2 3 4 5; do
  cargo test --features xla-backend --lib -- \
    backend::tests server::startup::muse_glimmer_startup_guard_tests --test-threads=8
done
```

All five runs reported `test result: ok. 12 passed; 0 failed` (7 backend tests plus 5 startup-guard tests). This also covers the main risk the change itself introduces, four additional tests contending on the same crate-wide mutex as the env-mutating guard test, and shows no deadlock.

Scope note: this verifies the criterion as written, that the tests pass under a parallel run with the feature enabled. It is not a reproduction of the original failure. The race is timing-dependent and was never observed failing here, consistent with the issue describing it as a latent interleaving rather than a reliably reproducing one.

### C. References
- Issue #1159 (specification), PR #1157 (source of the `env_lock` pattern and of the review finding that filed this issue), #1116 (introduced the env mutation)
- `src/backend/mod.rs:316` (the `MLXCEL_BACKEND` read), `src/test_support/env_lock.rs:60` (`env_lock`), `src/server/muse_glimmer_startup_guard_tests.rs:150` (the contending mutator)
- PR #1169 review, security and verification comments
