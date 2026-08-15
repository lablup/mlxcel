# Technical Report: PR #1173 - fix(test): skip partial Muse Glimmer checkpoints instead of panicking

**Date**: 2026-08-16
**Author**: AI Code Reviewer
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low

---

## Executive Summary

`pinned_post_tower_weight_roots_and_shapes_match_published_contract` skipped gracefully only when the pinned checkpoint's index file was absent. With the index present but anything else missing or corrupt, an interrupted `hf download` or `mlxcel download`, it panicked on an opaque `unwrap` instead of skipping with a reason. This PR adds an availability pre-check that skips with a message naming the offending file, while keeping every genuine contract violation a hard failure.

The governing constraint was that turning a wrong checkpoint into a silent skip would be strictly worse than the panic being fixed, because it would permanently disable the contract this test exists to enforce. The implementation separates the two categories structurally rather than by convention: `load_pinned_checkpoint` may only veto on availability, and `assert_post_tower_contract` holds the assertions with no skip path except for an unreadable shard header.

Two defects were found in review and fixed before merge, both of them cases where the first implementation put something on the wrong side of that line or left a hazard behind it.

---

## 1. Problem Statement

### 1.1 Background

Issue #1161 came from the security review LOW findings on PR #1157 (implementation of #1155). That PR added the index-absent guard and deliberately left the partial-checkpoint case out of scope. The sibling loading test `pinned_weight_index_classifies_each_source_weight_once` does not share the gap, because it reads only the index file itself.

### 1.2 Existing Issues

- **Issue 1**: past the index-existence guard, the test unwrapped every read: `config.json`, the index contents, `index["weight_map"].as_object()`, and every referenced shard through `safetensors_shape`, which itself unwrapped `File::open`, two `read_exact` calls, and the header parse.
- **Issue 2** (found in review): the availability pre-check deserialized `config.json` straight into `MuseGlimmerConfig`, so a config that was valid JSON but did not fit the schema was reported as unavailable and skipped. Schema drift in the pinned config, or a different model dropped into the pinned directory, is precisely the divergence this contract test exists to catch, so it belongs on the failing side.
- **Issue 3** (found in the security pass): the new header-length guard was file-relative only. `header_len > file_len.saturating_sub(8)` permits a declared header as large as the whole file, and the pinned shards are 50 GB and 9.6 GB, so a corrupt eight-byte prefix passed the check and reached `vec![0u8; header_len as usize]`. The outcome is either a `handle_alloc_error` abort, which is uncatchable and prints no reason, or gigabytes of tensor payload pulled into host memory, on the one machine that owns the checkpoint. An abort is strictly worse than the opaque unwrap this PR removes.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Opaque panic on a partially materialized checkpoint, misread as a code defect | Low | Medium (any interrupted download) |
| A wrong checkpoint silently skipping, disabling the contract check permanently | Medium | Low (closed by the boundary design and by issue 2's fix) |
| Uncatchable allocation abort from a corrupt header on the checkpoint-owning machine | Medium | Low (closed by the absolute ceiling) |

---

## 2. Technical Review

### 2.1 Security

The security pass filed one MEDIUM (issue 3 above) and fixed it by mirroring the guard the production reader already uses: `read_safetensors_header_bytes` (`src/lib/mlxcel-core/src/weights.rs:196`) caps the identical eight-byte prefix at `MAX_HEADER_BYTES = 256 * 1024 * 1024` with the rationale "reject absurdly large headers to avoid OOM". This test reader was the only header reader in the tree without such a cap. The mirrored constant is checked before the file-relative bound, and `safetensors_shape_refuses_a_header_over_the_allocation_ceiling` declares `u64::MAX` in a tiny temp file so the test itself allocates nothing large. The real pinned shard records an 11,720-byte header against a 9.6 GB file, roughly four orders of magnitude of margin, so no genuine checkpoint can be rejected.

Everything else was verified clean. The boundary holds on every traced path: an absent or non-object `weight_map` becomes an empty map that fails the weight-root assertion; a pinned key missing from a well-formed map is skipped past in the loop and reported by that same assertion; a key mapped to a non-string hits the "must name a shard" panic; a header that parses without a shaped entry for the key panics rather than returning `Err`. The env-var read holds the crate-wide env lock in a scoped block and drops it before the `assert!`, so a failing assertion cannot poison it, and nothing reachable from that function re-acquires the non-reentrant mutex. Only file headers are read, never tensor payloads, and every `TempDir` is bound to a live local so the `should_panic` tests still clean up on unwind.

**Issues Found:**
| Issue | Severity | Status |
|-------|----------|--------|
| Schema-mismatched `config.json` skipped instead of failing | Medium | Fixed (`3336732c`) |
| Header allocation bounded only by file size, so a corrupt length in a 50 GB shard could abort the process | Medium | Fixed (`3c61dd21`) |
| `contract_assertion_still_fails_on_a_wrong_recorded_shape` expected a bare key name that three different panics emit | Low | Fixed (`3336732c`) |
| `pinned_precheck_leaves_an_absent_weight_map_...` had no `should_panic` companion, unlike its missing-weight-root sibling | Low | Fixed (`3336732c`) |

### 2.2 Performance

No production path is touched. The pinned check reads only file headers, so it stays fast despite the 60 GB checkpoint: the full module runs in about 0.01 s.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none
- **New Dependencies**: none; `tempfile` was already a dev-dependency (`Cargo.toml:308`)
- **Compatibility**: no production loading path is affected

### 2.4 Code Quality

- **Test Coverage**: 5 tests became 23 in the module, adding synthetic coverage for every skip branch plus executable proof that contract violations still fail
- **Code Complexity**: the pinned check is now three named functions with one responsibility each instead of one unwrap-laden body
- **Technical Debt**: decreased

---

## 3. Technical Decisions

### 3.1 Hybrid pre-check rather than either option the issue offered

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Option 1: existence pre-check only | Simple | Cannot see a shard that exists but whose header was truncated mid-download |
| Option 2: convert every unwrap into skip-with-reason | Covers corruption | Reads the index twice, which the issue explicitly ruled out |
| **Chosen: hybrid** | Single index parse shared with the assertions, and corruption coverage | Slightly more structure |

**Rationale:** `load_pinned_checkpoint` parses the index once and hands the parsed `weight_map` to the assertion body, and `safetensors_shape` returns `Result` so header corruption is covered too.

### 3.2 Separate availability from contract structurally, not by convention

**Rationale:** The assertions live in `assert_post_tower_contract`, which takes an already-validated `PinnedCheckpoint`. Reaching it means every file it needs was confirmed readable, so anything failing there is the checkpoint disagreeing with the contract. The one skip inside it, an unreadable shard header, is scoped to availability by construction: the only `Err` producers in `safetensors_shape` are open failure, metadata failure, truncation, an over-ceiling or over-file declared length, and a non-JSON header. A header that parses but records no shape, or a non-integer dimension, panics.

### 3.3 Defer `MuseGlimmerConfig` deserialization into the assertion body

**Rationale:** This is the fix for issue 2. The pre-check now parses `config.json` as plain JSON only, so a truncated config, which is a syntax error and a real availability problem, still skips, while a config that parses and does not fit the schema reaches `serde_json::from_value::<MuseGlimmerConfig>` in the assertion body and panics. `PinnedCheckpoint` carries the raw `Value` to make the split explicit. The fix has teeth rather than passing vacuously: `MuseGlimmerConfig.text_config` and its inner fields carry no serde defaults (`src/models/muse_glimmer_config.rs:26`), so a mismatched config genuinely fails to deserialize.

### 3.4 Implement the optional env gate

**Rationale:** `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1` turns every skip into a failure, so a corrupted checkpoint cannot quietly disable the contract check forever on the machine positioned to enforce it. It is read under the crate-wide env lock, matching the convention `muse_glimmer_startup_guard_tests` established, because this crate is edition 2024 and a sibling test mutates the environment with `unsafe set_var` under that same lock. It also turned out to be the cleanest way to prove the happy path: see the appendix.

### 3.5 Split into a separate file

**Rationale:** the pinned check plus its synthetic coverage is a distinct concern from the fusion math tests, and the `#[cfg(test)] #[path = "..."] mod` sibling pattern is already used throughout `src/vision/`. Note that the PR and commit bodies justify the split with a "500-line limit"; the repo's own code-structure guidance actually puts the threshold at 800 lines, so that stated reason is wrong even though the split itself is sound.

---

## 4. Implementation Details

### 4.1 Key Code Changes

**File: `src/vision/encoders/muse_glimmer_fusion_pinned_tests.rs`** (new)
```rust
// The availability veto, which may never see a contract violation.
fn load_pinned_checkpoint(model_dir: &Path) -> Result<PinnedCheckpoint, String> {
    let index_text = read_checkpoint_file(&index_path)?;
    let index: Value = serde_json::from_str(&index_text)
        .map_err(|err| format!("{} does not parse as JSON: {err}", index_path.display()))?;
    // ... config parsed as plain JSON only ...
    // A missing or non-object `weight_map` is a malformed index rather than a
    // missing file: hand the caller an empty map so the weight-root assertion
    // reports it as the contract violation it is.
    let weight_map = index["weight_map"].as_object().cloned().unwrap_or_default();
    // ... each referenced shard checked for presence, naming any that is missing ...
}
```

```rust
// Bound the declared header before allocating.
if header_len > MAX_SAFETENSORS_HEADER_BYTES { /* Err */ }
if header_len > file_len.saturating_sub(8) { /* Err */ }
let mut header = vec![0u8; header_len as usize];
```

**Reason for change:** the original `safetensors_shape` did `vec![0u8; header_len]` straight from an unvalidated eight-byte little-endian read.

---

## 7. Change Summary

### Statistics
| Item | Value |
|------|-------|
| Files changed | 3 |
| Lines added | +609 |
| Lines deleted | -79 |
| Tests added | 18 (module went from 5 to 23) |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Test Robustness | 1 | Partial checkpoints skip with a named reason instead of panicking |
| Test Correctness | 4 | Contract violations proven still failing; two `should_panic` tests tightened |
| Security | 1 | Absolute ceiling on the declared safetensors header before allocation |
| Tooling | 1 | `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1` turns skips into failures |

### Related Commits
| Hash | Type | Message |
|------|------|---------|
| `5274e7e5` | fix(test) | skip partial Muse Glimmer checkpoints instead of panicking |
| `3336732c` | test | fail rather than skip on a schema-mismatched pinned config |
| `3c61dd21` | fix(test) | cap the safetensors header this reader will allocate |

---

## 8. Follow-up Actions

### Required
- [ ] None; all four acceptance criteria are met, including the optional one

### Future Improvements
- `pinned_weight_index_classifies_each_source_weight_once` (`src/loading/vlm_muse_glimmer_tests.rs:473`) does not honor `MLXCEL_REQUIRE_PINNED_CHECKPOINTS`, so only one of the two pinned guards is hardened on the checkpoint-owning machine. Worth filing separately; explicitly out of scope for #1161.
- The mid-loop skip on an unreadable shard returns early over a `BTreeMap`, so an unreadable `fc1` shard hides a wrong shape on `fc2` or `vision_projection`. Inherent to skip semantics and bounded by the env gate. Asserting readable keys first and skipping at the end would be strictly better.
- Two `should_panic` tests still expect a bare key name that the "must name a shard" panic also emits. Both alternatives are failures rather than skips, so the invariant holds; tightening would mean depending on more of `assert_eq!`'s output formatting.

---

## Appendix

### A. Test Results

| Command | Result |
|---------|--------|
| `cargo test --lib vision::encoders::muse_glimmer_fusion` | 23 passed, 0 failed |
| `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1 cargo test --lib vision::encoders::muse_glimmer_fusion` | 23 passed, 0 failed |
| `cargo clippy --lib --tests -- -D warnings` | clean |
| `cargo fmt --check` | clean |

### B. How the acceptance criteria were proven

The pinned checkpoint is fully materialized on the machine used here (`models` is a gitignored symlink to `/home/inureyes/models`; `config.json`, the index, and both shards at 50 GB and 9.6 GB are all present), which made every criterion directly testable.

- **Criterion 2, full checkpoint still validates the contract.** The env gate is the proof. Running with `MLXCEL_REQUIRE_PINNED_CHECKPOINTS=1` converts any skip into a failure, so the pinned test passing under that gate means it actually read the genuine checkpoint and asserted the weight-root and shape contract rather than skipping.
- **Criterion 3, index absent behaves as before.** The test binary was run from a directory containing no `models/`, and skipped with `models/mlx/muse-glimmer-30b/model.safetensors.index.json is not present`.
- **Optional criterion 4.** The same run with the gate set failed instead of skipping.
- **Criterion 1, partial checkpoint skips with a named reason.** Covered by the synthetic tests, which build throwaway checkpoints in `tempfile::tempdir()`. Nothing under `models/` was moved, renamed, deleted, truncated, or written at any point; both shards were verified intact with their original timestamps afterwards.

### C. References
- Issue #1161 (specification), PR #1157 (added the index-absent guard, and whose review filed this issue), #1116
- `src/vision/encoders/muse_glimmer_fusion_pinned_tests.rs` (the new module), `src/lib/mlxcel-core/src/weights.rs:196` (the production header cap this mirrors), `src/models/muse_glimmer_config.rs:26` (why the schema check has teeth)
- PR #1173 review and security comments
