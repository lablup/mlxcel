# Technical Report: PR #1180 - fix(test): gate the loading-side pinned Muse Glimmer guard on MLXCEL_REQUIRE_PINNED_CHECKPOINTS

**Date**: 2026-08-16
**Author**: AI Code Reviewer
**Status**: Completed
**Languages**: Rust, Markdown

**Risk Level**: Low

---

## Executive Summary

PR #1173 introduced `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1`, which turns a pinned-checkpoint test's graceful skip into a hard failure so a corrupted checkpoint cannot silently disable the only coverage that exists on the machine owning it. That gate reached only one of the two pinned Muse Glimmer guards. This PR extends it to the second, moves the gate into a shared test-support helper so both call sites use one implementation, and documents the variable.

The documentation is half the value here rather than an afterthought. The gate was set nowhere in the tree: no workflow under `.github/`, no file under `scripts/`, no shell profile. An opt-in knob nobody turns protects nothing, and a repository change cannot export a variable into a maintainer's shell, so the issue gained a fourth deliverable during implementation: document when and how to actually enable it, not just what it does.

---

## 1. Problem Statement

### 1.1 Background

Issue #1177 was filed from the follow-up section of PR #1173's technical report. The gap it names is narrow but real: the loading-side guard skips unconditionally, so on the checkpoint-owning machine it can go quiet forever without anyone noticing.

### 1.2 Existing Issues

- **Issue 1**: `pinned_weight_index_classifies_each_source_weight_once` (`src/loading/vlm_muse_glimmer_tests.rs`) used the original unconditional skip on an absent index. It asserts a real contract, the 1436-weight `MuseWeightInventory` breakdown, so a silent skip hides something worth catching.
- **Issue 2**: the gate lived as a private function inside the vision test module, so extending it meant either duplicating the environment read and its lock discipline or extracting it.
- **Issue 3**: the variable was undocumented and enabled nowhere, which made the whole mechanism inert in practice.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Loading-side contract check silently disabled on the checkpoint-owning machine | Low | Low |
| Two divergent copies of the gate drift apart | Low | Medium (if extended by copy) |
| The gate remains a knob nobody turns, so none of the above is actually prevented | Medium | High without documentation |

---

## 2. Technical Review

### 2.1 Security

Test-only and docs-only. `test_support` is `#[cfg(test)] pub(crate)` at the crate root (`src/lib.rs:51-52`), the new module is `pub(crate)` inside it, there is no re-export, and the only references in the tree are the two test call sites. The helper is unreachable from production code, the binaries, and `tests/`.

The security pass verified the environment handling in detail. The guard is bound as `let required = { let _env_guard = env_lock(); ... };`, so it drops at the end of the block rather than living to the end of the statement, and the assertion runs unlocked. No deadlock is possible: `env_lock` is a non-reentrant `std::sync::Mutex`, none of the call sites holds it, and no holder elsewhere in the crate can reach the helper. Poisoning is handled twice over, by the early drop and by `env_lock()`'s own `unwrap_or_else(|p| p.into_inner())`. Every env-mutating site in the crate was checked against lock usage and the three that mutate without it were each shown benign.

The persistent-export advice in the documentation was checked for safety rather than taken on faith: nothing outside the test files reads the variable, no code enumerates `std::env::vars()`, and there is no unknown-`MLXCEL_*` rejection, so exporting it cannot affect a running `mlxcel` or `mlxcel-server`.

**Issues Found:**
| Issue | Severity | Status |
|-------|----------|--------|
| PR body described a table shape that the docs restructure had made stale, and it becomes the squash commit message | Low | Fixed (review pass) |
| PR body claimed the vision-side failure message was unchanged; it now names the test | Low | Fixed (review pass) |
| Docs claimed there was no availability pre-check before #1173 | Low | Fixed (`d2b4f419`, then corrected again by `758fe64c`) |

### 2.2 Performance

Nil. The lock is taken only on the skip or failure path, held for exactly one `std::env::var` call, and released before both the assertion and any file I/O. In `assert_post_tower_contract` the shard header read happens entirely outside the lock. On a machine with a complete checkpoint the helper is never called.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none
- **New Dependencies**: none
- **Compatibility**: no production path affected

### 2.4 Code Quality

- **Test Coverage**: unchanged in count; the loading-side guard gains the gate, and both guards keep naming themselves in their skip and failure messages
- **Code Complexity**: one shared helper replaces one private copy, with the test name threaded through as a parameter
- **Technical Debt**: decreased; the follow-up recorded in PR #1173's report is closed, and a second copy of the gate was avoided

---

## 3. Technical Decisions

### 3.1 Extract to `src/test_support/`, and thread the test name through

**Rationale:** `test_support` is declared at the crate root precisely so test modules at any depth can name it, which is what a helper shared between `src/vision/encoders/` and `src/loading/` needs. The original function hardcoded the vision test's name in its skip message; the shared version takes `test_name: &str` so each call site still identifies itself. A skip message that does not say which test skipped would have been a regression, and with two callers it would have been actively confusing.

### 3.2 Keep the loading-side `panic!` on a classification error

**Rationale:** `pinned_weight_index_classifies_each_source_weight_once` panics when `read_muse_weight_inventory_from_index` returns `Err`. That is a contract signal, not an availability signal, so it stays a panic. The gate was inserted only on the `!index_path.exists()` branch above it. The helper only ever converts a skip into a failure, never the reverse, and that direction was verified at every call site.

### 3.3 Fold the doc entry into the existing table rather than adding a second one

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| A second `Variable / Values / Default / Notes` table in the same section | Expresses value semantics in dedicated columns | Two tables in one section, introduced by a sentence explaining the formatting, which is commentary about the document rather than about the variable |
| **Chosen: one row in the existing `Variable / Purpose` table** | Matches the section's established shape | Value and default semantics have to be stated inline |

**Rationale:** The section's own precedent settles it. `MLXCEL_ALLOW_PARALLEL_CUDA_TESTS` and `MLXCEL_ALLOW_CONCURRENT_GPU_TESTS` already carry multi-sentence Purpose cells with issue references, so a long cell is normal there and a meta-sentence about table formatting is not.

### 3.4 Document when to turn it on, not only what it does

**Rationale:** This was added to the issue during implementation after confirming the variable was set nowhere. The documentation therefore states that it is test-only and never a runtime knob, that it belongs on a machine owning the pinned checkpoint and why a skip there is a real loss, how to enable it for one run and how to export it persistently, and its second already-realized use: with the gate on, a pinned test reporting `ok` can only mean the checkpoint was read and the contract asserted, because an unusable checkpoint would have failed. That is how acceptance criterion 2 of issue #1161 was proven.

---

## 4. Implementation Details

### 4.1 Key Code Changes

**File: `src/test_support/pinned_checkpoint.rs`** (new)
```rust
pub(crate) fn skip_or_fail_pinned_checkpoint(test_name: &str, reason: &str) {
    // The crate-wide env lock serializes this read against tests that mutate
    // the process environment with `unsafe set_var`; on Rust 2024 an
    // unsynchronized concurrent read of the env block is undefined behavior.
    // Hold the guard only for the read, and drop it before the assertion so
    // a failing assertion here cannot poison the mutex for later tests.
    let required = {
        let _env_guard = crate::test_support::env_lock::env_lock();
        std::env::var("MLXCEL_REQUIRE_PINNED_CHECKPOINTS").is_ok_and(|value| value == "1")
    };
    assert!(!required, "...{test_name}...{reason}");
    eprintln!("Skipping {test_name}: {reason}");
}
```

**Reason for change:** one implementation for both guards, with the caller's identity preserved in both the skip and the failure message.

---

## 7. Change Summary

### Statistics
| Item | Value |
|------|-------|
| Files changed | 5 |
| Lines added | +88 |
| Lines deleted | -29 |
| Tests added | 0 (behavior of 2 existing guards extended) |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Test Correctness | 1 | Loading-side guard now honors the gate |
| Refactor | 1 | Gate extracted to a shared test-support helper |
| Documentation | 1 | Variable documented, including when and how to enable it |

### Related Commits
| Hash | Type | Message |
|------|------|---------|
| `3a1c014e` | fix(test) | gate the loading-side pinned Muse Glimmer guard on MLXCEL_REQUIRE_PINNED_CHECKPOINTS |
| `d3993635` | docs | fold the pinned-checkpoint gate into the existing test-variable table |
| `d2b4f419` | docs | correct the pre-#1173 behavior described for the pinned gate |
| `758fe64c` | docs | name PR #1157 as the pre-#1173 pinned availability guard |

---

## 8. Follow-up Actions

### Required
- [ ] None; all five acceptance criteria are met

### Future Improvements
- `skip_or_fail_pinned_checkpoint` self-deadlocks if a future call site is added inside a test that already holds `env_lock()`. No current call site does, and the helper explains why it takes the lock, but it does not warn callers against holding it. Worth remembering if a third pinned guard is added.
- `src/lib.rs:47-49` still describes `test_support` as providing "the single shared `ENV_LOCK`", which is now one of two modules there.
- `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=true` silently does nothing. Strict equality on `1` is what the issue mandated and the documentation calls it out, but it is a sharp edge.

---

## Appendix

### A. Test Results

| Command | Result |
|---------|--------|
| `cargo test --lib -- vision::encoders::muse_glimmer_fusion loading::vlm::muse_glimmer` | 37 passed, 0 failed |
| Same with `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1` | 37 passed, 0 failed |
| `cargo clippy --lib --tests -- -D warnings` | clean |
| `cargo fmt --check` | clean |

Skip and gated-failure paths, exercised by running the compiled test binary from a temp directory containing no `models/`, so the real checkpoint was never touched:

| Condition | Result |
|-----------|--------|
| Gate unset | Both guards skip and report `ok`, each naming itself: `Skipping pinned_weight_index_classifies_each_source_weight_once: ...` and `Skipping pinned_post_tower_weight_roots_and_shapes_match_published_contract: ...` |
| `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1` | Both guards FAIL, each naming itself: `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1 but the pinned checkpoint <test_name> needs is not usable: ...` |

The second row of the first table is the load-bearing evidence for the criterion that the full checkpoint still validates its contract. Because the gate converts every skip into a failure, both guards passing under it means both genuinely read the real 60 GB checkpoint rather than skipping.

### B. Note on the documentation correction

The paragraph describing pre-#1173 behavior was wrong twice before it was right, which is worth recording because the sequence is easy to get wrong. The first version said the availability pre-check panicked before #1173; there was no general pre-check then. The correction then said there was no pre-check at all, which overcorrected: PR #1157 had already replaced the vision-side `assert!(index_path.exists(), ...)` with a silent skip and added the same index-absent skip on the loading side. The accurate sequence, now in the document, is that #1157 made the index-absent case quiet, #1173 made the index-present-but-incomplete case quiet, and the gate restores loudness for both.

### C. References
- Issue #1177 and its scope-addition comment, PR #1173 and its technical report (which recorded this as a follow-up), issue #1161, PR #1157
- `src/test_support/pinned_checkpoint.rs` (the shared gate), `src/test_support/env_lock.rs` (the lock it takes), `docs/environment-variables.md` (the documentation)
- PR #1180 review and security comments
