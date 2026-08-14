# Technical Report: PR #1157 - test: skip pinned muse_glimmer checkpoint tests and fix MLXCEL_BACKEND race

**Date**: 2026-08-15
**Author**: AI Code Reviewer
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low

---

## Executive Summary

Three muse_glimmer tests introduced with #1116 broke the full `cargo test --release --features metal,accelerate --lib` run on any machine without the pinned `models/mlx/muse-glimmer-30b` checkpoint: two panicked on the missing index file and one failed intermittently because a sibling test mutates `MLXCEL_BACKEND` process-wide. This PR makes the two checkpoint-pinned tests skip gracefully when the index is absent and serializes the racing test through the crate-wide `test_support::env_lock`, restoring a green suite (5621 passed / 0 failed) on checkpoint-less machines. Test-only change; no production code is touched.

---

## 1. Problem Statement

### 1.1 Background

Issue #1155 was filed during the epic #1148 integration verification. A two-arm measurement on main `a206f089` and the pre-epic baseline showed the three failures reproduce identically on both arms and no epic commit touches muse files, so they are pre-existing defects from #1116, not an epic regression.

### 1.2 Existing Issues

- **Issue 1**: `loading::vlm::muse_glimmer::tests::pinned_weight_index_classifies_each_source_weight_once` panicked with "Failed to read ... No such file or directory" when `models/mlx/muse-glimmer-30b/model.safetensors.index.json` is absent.
- **Issue 2**: `vision::encoders::muse_glimmer_fusion::tests::pinned_post_tower_weight_roots_and_shapes_match_published_contract` asserted on the same missing index ("pinned checkpoint index is required for this shape contract").
- **Issue 3**: `server::startup::muse_glimmer_startup_guard_tests::muse_glimmer_startup_allows_baseline_and_keeps_video_disabled` failed under default parallel execution but passed in isolation: the sibling `muse_glimmer_startup_rejects_xla_backend_selection` sets `MLXCEL_BACKEND=xla` process-wide (set_var/restore), and the baseline test's validator reads the env var and can observe the transient value.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Red full suite on contributor machines masks real regressions | Medium | High (any machine without the 30B checkpoint) |
| Ordering-dependent flake erodes trust in the suite | Medium | Medium (scheduler-dependent) |

---

## 2. Technical Review

### 2.1 Security

No security surface: all three changed files are `#[cfg(test)]` test modules; nothing ships in production binaries. The security pass confirmed the index-path TOCTOU is inert (hardcoded literal path, no attacker input; worst case is the pre-PR panic) and found no lock-ordering cycle with the existing per-module MLX guards.

**Issues Found:**
| Issue | Severity | Status |
|-------|----------|--------|
| Unlocked `MLXCEL_BACKEND` reader in `backend::tests` under opt-in `xla-backend`/`experimental-backend` features | Medium | Open (pre-existing from #1116, unreachable in default and CI feature sets, left as follow-up to keep the PR scoped) |
| Fusion test skip guard checks only the index; a partially materialized checkpoint still panics on `config.json`/shard `unwrap()`s | Low | Open (repo-wide skip-guard convention; follow-up candidate) |

### 2.2 Performance

The added mutex acquisition serializes exactly two tests in one module; the effect on suite wall time is unmeasurable (module runs in well under a second).

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none
- **New Dependencies**: none (reuses the existing crate-internal `test_support::env_lock`)
- **Compatibility**: behavior is unchanged on machines that do have the pinned checkpoint; both contract tests still run their full assertions there

### 2.4 Code Quality

- **Test Coverage**: unchanged in count; effective coverage improves because the suite is now runnable end-to-end on checkpoint-less machines
- **Code Complexity**: two early-return guards and one lock acquisition; trivially reviewable
- **Technical Debt**: decreased (removes an ordering dependency; aligns with the established skip convention in `tests/prompt_cache_e2e.rs`)

---

## 3. Technical Decisions

### 3.1 Graceful skip (`eprintln` + early return) over `#[ignore]`

**Context:** The two checkpoint-pinned contract tests must still run automatically on the machine that owns the checkpoint, but must not fail elsewhere.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| `#[ignore = "..."]` | Visible as ignored in test output | Never runs automatically, even where the checkpoint exists; needs `--ignored` |
| **Chosen: guard + eprintln + return** | Runs fully when the checkpoint is present, skips with a printed reason otherwise; matches `tests/prompt_cache_e2e.rs` | Skipped runs report `ok`, so coverage erosion is silent (noted as a follow-up idea: an opt-in `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1`) |

**Rationale:** Issue #1155 explicitly prefers the repo's existing skip convention, and the checkpoint-owning box keeps its contract coverage with zero workflow change.

### 3.2 Crate-wide `env_lock` over per-module mutex, `serial_test`, or a parameter refactor

**Context:** The race is between one mutator and one reader of `MLXCEL_BACKEND` in the same module, but env vars are process-wide state.

**Rationale:** `crate::test_support::env_lock` (declared at `src/lib.rs:52`) is the established pattern with 18 acquisition sites and documented poison recovery (`unwrap_or_else(|p| p.into_inner())`); the mutating sibling already held it, so adding the acquisition on the reader side closes the race with one line. A `serial_test` dependency or a startup-guard signature refactor would be strictly larger changes for the same guarantee. The review pass proved the lock is load-bearing differentially: with the lock removed the module failed 25/25 stress runs; with it, 10/10 passed.

---

## 4. Implementation Details

### 4.2 Key Code Changes

**File: `src/vision/encoders/muse_glimmer_fusion_tests.rs`**
```rust
// Before
assert!(
    index_path.exists(),
    "Muse Glimmer pinned checkpoint index is required for this shape contract"
);

// After
if !index_path.exists() {
    eprintln!(
        "Skipping pinned_post_tower_weight_roots_and_shapes_match_published_contract: \
         pinned Muse Glimmer checkpoint index not present at {}",
        index_path.display()
    );
    return;
}
```

**Reason for change:** converts a hard environment dependency into a graceful skip while preserving the full shape-contract assertions when the checkpoint is present. `src/loading/vlm_muse_glimmer_tests.rs` receives the same guard shape ahead of its index read.

**File: `src/server/muse_glimmer_startup_guard_tests.rs`**
```rust
// Added at the top of muse_glimmer_startup_allows_baseline_and_keeps_video_disabled
let _env_guard = crate::test_support::env_lock::env_lock();
```

**Reason for change:** `validate_muse_glimmer_unsupported_startup` reads `MLXCEL_BACKEND`; the sibling XLA-rejection test mutates it under the same lock, so both sides now serialize and the baseline test can never observe the transient `xla` value. Guard order matters and is correct: `_env_guard` is declared before the `TempDir`, so it drops after it.

---

## 7. Change Summary

### Statistics
| Item | Value |
|------|-------|
| Files changed | 3 |
| Lines added | +25 |
| Lines deleted | -4 |
| Tests added | 0 (3 existing tests repaired) |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Code Quality | 3 | Two graceful checkpoint-skip guards, one env-lock acquisition |

### Related Commits
| Hash | Type | Message |
|------|------|---------|
| `79bc61a4` | test | skip pinned muse_glimmer checkpoint tests and fix MLXCEL_BACKEND race |

---

## 8. Follow-up Actions

### Required
- [ ] None; the issue's acceptance criteria are met for the default feature set

### Future Improvements
- Guard the unlocked `select_backend()` env read in `backend::tests::mlx_session_threads_the_token_bias_through` under the opt-in `xla-backend`/`experimental-backend` features (pre-existing, unreachable in CI)
- Consider extending the fusion test's skip guard to cover a partially materialized checkpoint (currently only the index file is checked before `unwrap()`s on `config.json` and shards)
- Consider an opt-in `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1` escape hatch so the checkpoint-owning box can turn silent skips into failures

---

## Appendix

### A. Test Results
- Full suite on a machine without the checkpoint: `cargo test --release --features metal,accelerate --lib` → 5621 passed / 0 failed / 117 ignored
- Targeted: `--lib muse_glimmer` → 80 passed / 0 failed; `--lib muse_glimmer_startup_guard` repeated (5x by the implementer, 3x by the security pass, 10x by the reviewer) → no flakes
- Differential proof of the race fix: lock removed → 25/25 module runs failed; lock present → 10/10 passed
- The checkpoint-present arm of the two skip guards was verified by code inspection only (no local checkpoint), stated honestly in the PR body

### C. References
- Issue #1155 (specification), #1116 (introduced the tests), epic #1148 (integration verification that surfaced the failures)
- PR review comments: implementation review and security review on PR #1157
