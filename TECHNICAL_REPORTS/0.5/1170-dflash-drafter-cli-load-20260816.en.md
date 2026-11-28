# Technical Report: PR #1170 - fix(cli): reject a DFlash drafter before the offline standalone load

**Date**: 2026-08-16
**Author**: AI Code Reviewer
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low

---

## Executive Summary

`mlxcel generate --draft-model <dflash-drafter>` failed for every DFlash drafter and every target with `Error: Weight not found: model.embed_tokens.weight`, a message that names a tensor instead of the actual problem. A DFlash checkpoint borrows `embed_tokens` and `lm_head` from its target when it binds, so it ships neither, but its `config.json` still declares an ordinary `model_type` (`qwen3`), which routed it through the full standalone `LoadedModel` loader. This PR adds a structural probe, keyed on the checkpoint's own DFlash markers (a nested `dflash_config` object and/or `architectures: ["DFlashDraftModel"]`) rather than on the resolved `DrafterKind` (which defaults to `Dflash` for any non-Gemma-4-assistant drafter, including ordinary full models used as classic drafters), and rejects a DFlash drafter before both the `-m/--model` detection path and the offline `--draft-model` load, with a message that names the real problem and points at `mlxcel-server --draft-kind dflash`. The server path, which never reaches `get_model_type` for a drafter, is unaffected and pinned by a non-regression test. The finalization pass additionally fixed a predictable-temp-directory test hygiene finding (M3) in the new test fixtures.

---

## 1. Problem Statement

### 1.1 Background

Issue #1168: offline speculative decoding with a DFlash drafter could never succeed, with or without `--draft-kind`, because the offline entry point has no `DFlashGenerator` round loop and instead drives the drafter through the same loader used for a full standalone model.

### 1.2 Existing Issues

- **Issue 1**: `get_model_type` classified a DFlash drafter's `config.json` as `ModelType::Qwen3` (or whatever ordinary `model_type` the checkpoint's config declares), so `model_metadata` routed it to the matching family loader, which failed on the first missing tensor (`UnifiedEmbedding::from_weights` looking for `model.embed_tokens.weight`) rather than reporting that the directory is a drafter.
- **Issue 2**: The obvious fix, rejecting on the resolved `DrafterKind`, would have broken a working workflow: `DEFAULT_DRAFTER_KIND` is `Dflash` and `drafter_kind_by_model_type()` maps only the two Gemma 4 assistant model types, so an ordinary small full model used as a classic `--draft-model` drafter (a Qwen3 0.6B, for example) also auto-resolves to `Dflash` and runs fine today on the classic `SpeculativeGenerator` path.
- **Issue 3** (found during this finalization pass, security finding M3): `drafter_fixture_dir` in the new `src/commands/generate_tests.rs` tests built a predictable directory name under `env::temp_dir()` and created it with `create_dir_all`, which succeeds against a pre-existing directory or symlink there (CWE-377/379 class), and cleaned up only on the success path, leaking the fixture on an assertion failure. CI-relevant on Linux, where `TMPDIR` is shared across jobs.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Offline DFlash `--draft-model` stays permanently broken with a misleading error | Medium (every DFlash drafter, every target) | Certain before this fix |
| Rejecting on the wrong discriminator breaks the classic small-model drafter workflow | High if mis-keyed | Avoided by keying on checkpoint structure, confirmed by a dedicated control test |
| Predictable temp-dir fixture is squatted or symlinked in a shared `TMPDIR` (M3) | Low (test-only, no production surface) | Low, but real on shared Linux CI runners |

---

## 2. Technical Review

### 2.1 Security

Review and security passes were completed by the orchestrator and reviewer before finalization: no CRITICAL, no HIGH findings. One MEDIUM finding (M3) was raised against the test-only fixture helper added in this PR.

**Issues Found:**

| Issue | Severity | Status |
|-------|----------|--------|
| `drafter_fixture_dir` built a predictable path under `env::temp_dir()` with `create_dir_all` (CWE-377/379) and leaked on assertion failure | Medium | Fixed (`37136aa6`) |

The fix replaces the helper, and the identical inline pattern in the missing-config fixture, with `tempfile::tempdir()`, which mints a randomly-named, securely-permissioned directory and removes it on drop regardless of how the test exits. This is the same pattern the `mlxcel-core` `drafter::dflash::config` tests already use, so the PR is now internally consistent on this point.

### 2.2 Performance

None. This is a pre-load guard on the config-inspection path (a `serde_json` parse of `config.json`), not a hot inference path. No benchmark was required or run.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: an offline `--draft-model <dflash-drafter>` invocation that previously failed with a misleading `Weight not found` error now fails earlier with an explanatory one. No previously-working invocation is affected: no test or documented workflow constructed the offline DFlash round loop, since it does not exist.
- **New Dependencies**: none in the fix itself. `tempfile` (already a dev-dependency, used elsewhere in this same PR) backs the corrected test fixtures.
- **Compatibility**: the server path (`DFlashDrafter::load`, `resolve_drafter_kind`) never calls `get_model_type` for a drafter path, confirmed by re-checking every `get_model_type` call site in the tree; a new integration test pins that `SpeculativeDispatch::resolve` still returns the DFlash variant for the same fixture directory that detection now rejects.

### 2.4 Code Quality

- **Test Coverage**: new tests across four locations (`src/commands/generate_tests.rs`, `src/models/detection_tests.rs`, `src/lib/mlxcel-core/src/drafter/dflash/config.rs`, `tests/speculative_dispatch.rs`), each carrying an ordinary-full-model control that must keep resolving through the classic path. A mutation test (guarding the new detection arm as `if false && ...`) confirmed the new tests are real assertions rather than tautologies: both the `-m` and `--draft-model` rejection tests failed with the pre-fix classification.
- **Code Complexity**: the structural probe is two small pure functions (`is_dflash_drafter_config`, `is_dflash_drafter_dir`) plus one shared error constructor; no control flow changes to any existing load path.
- **Technical Debt**: decreased on both counts addressed here: the CI-relevant test hygiene inconsistency (M3) is closed, and the offline entry point no longer silently misroutes a class of checkpoint it was never built to run.

---

## 3. Technical Decisions

### 3.1 Key on checkpoint structure, not on the resolved `DrafterKind`

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Reject when `DrafterKind::Dflash` is resolved | Minimal code, reuses existing resolution | Breaks the classic small-full-model drafter workflow, which also resolves to `Dflash` by default |
| **Chosen: probe `config.json` for `dflash_config` and/or `architectures: ["DFlashDraftModel"]`** | Matches exactly what `DFlashConfig::from_json` and HuggingFace `AutoModel` dispatch each read; an ordinary full model has neither marker | Two independent markers to check instead of one resolved enum value |

**Rationale:** the resolved kind is a fallback default, not a structural fact about the checkpoint. The checkpoint's own markers are what the DFlash loader and the HuggingFace ecosystem both already treat as authoritative, so keying on them cannot misclassify an ordinary drafter.

### 3.2 Fix M3 by adopting `tempfile::tempdir()` rather than tightening the ad hoc path

**Rationale:** the PR already establishes the `tempfile::tempdir()` pattern in `mlxcel-core`'s own new tests. Hardening the ad hoc `env::temp_dir()` path (unique suffix, permission checks, explicit cleanup on every exit path including panics) would have reimplemented what `tempfile` already provides as an RAII guard, at higher risk of missing an exit path than reusing the proven crate.

---

## 4. Implementation Details

### 4.1 Detection (`src/models/detection.rs`, `src/lib/mlxcel-core/src/drafter/dflash/config.rs`)

`get_model_type` now rejects a structurally-DFlash directory before any `model_type` dispatch, through a shared `dflash_drafter_not_standalone_error`, covering the `-m` case, server startup, and the distributed stage loaders from one call site.

### 4.2 Offline CLI (`src/commands/generate.rs`)

New `reject_dflash_drafter_offline`, called in `run_generation_mode` immediately before `load_model(draft_model_path)`. Fires with no `--draft-kind` and with `--draft-kind dflash`; `--draft-kind mtp` is a separate, already-handled request routed to `run_offline_mtp`.

### 4.3 Test fixture hygiene fix (`src/commands/generate_tests.rs`, commit `37136aa6`)

```rust
// Before
fn drafter_fixture_dir(name: &str, config: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "mlxcel_generate_drafter_{name}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("config.json"), config).unwrap();
    dir
}
// ... each call site manually `fs::remove_dir_all(dir).unwrap()`s on the success path only

// After
fn drafter_fixture_dir(config: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("temp dir");
    fs::write(dir.path().join("config.json"), config).unwrap();
    dir
}
// call sites use dir.path(); TempDir removes the directory on drop, on every exit path
```

The `name` parameter was dropped: it existed only to build a unique-ish path segment, which `tempfile::tempdir()` already guarantees by construction. The identical inline pattern in `offline_draft_model_check_defers_missing_or_broken_configs_to_the_loader`'s `empty` fixture was fixed the same way.

---

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed (feature commit `82096193`) | 9 |
| Files changed (M3 fix commit `37136aa6`) | 1 |
| Lines added / removed (feature) | +645 / -5 |
| Lines added / removed (M3 fix) | +19 / -34 |
| Tests added | 4 (`commands::generate`) + 3 (`models::detection`) + 3 (`drafter::dflash::config`, plus existing coverage extended) + 2 (`speculative_dispatch`) |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Detection / loading | 3 | `is_dflash_drafter_config`, `is_dflash_drafter_dir`, `get_model_type` rejection, shared error text |
| CLI | 1 | `reject_dflash_drafter_offline` pre-load guard in `run_generation_mode` |
| Docs | 2 | `docs/supported-models.md`, `docs/speculative-acceptance.md` corrected to state the offline limitation |
| Test hygiene (this finalization pass) | 1 | `drafter_fixture_dir` and the inline `empty` fixture moved to `tempfile::tempdir()` |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `82096193` | fix | reject a DFlash drafter before the offline standalone load |
| `37136aa6` | fix | back drafter test fixtures with `tempfile::tempdir()` |

---

## 8. Follow-up Actions

### Required

- [ ] None; the security and review passes reported no CRITICAL or HIGH findings, and the one MEDIUM finding (M3) is fixed in this pass.

### Future Improvements (recorded as known limitations, not fixed here)

- The precheck fires only after the target model finishes loading, so an operator with a large target waits through the full load before seeing the error. Moving it earlier would also intercept `--draft-kind mtp`, which is deliberately routed elsewhere.
- `--draft-model <dir>/config.json` (pointing at the config file rather than the directory) bypasses the precheck and falls through to the detection error with `-m` phrasing, because the precheck does not apply `resolve_model_dir`.
- The probe uses raw `serde_json::from_slice`, while `get_model_type` applies `sanitize_config_json` first; a config with bare `NaN`/`Infinity` literals would fail the precheck open. No current checkpoint is affected, and closing this gap is a cross-crate change (the sanitizer lives in the root crate, the probe in `mlxcel-core`).
- Building the actual offline `DFlashGenerator` round loop remains out of scope, as stated in the PR summary; the CLI directs an operator who needs this today at `mlxcel-server --draft-kind dflash`.

---

## Appendix

### A. Test Results

- `cargo test --release --workspace --features metal,accelerate --no-fail-fast`: every target passes except `-p mlxcel --lib`, which reports 5621 passed / 3 failed. All 3 failures are `multimodal::video::tests::*`, proven unrelated to this PR (0 lines touched in `src/multimodal/video.rs`; a PATH-controlled experiment on the same binary gives 34 passed / 0 failed with ffmpeg hidden versus 31 passed / 3 failed with ffmpeg 9.0.1 present; root cause is `src/multimodal/video.rs:1123` passing `-vsync`, which ffmpeg 9 removed). Filed separately as #1172.
- The four `commands::generate::tests::offline_draft_model_*` precheck tests compile into `--bin mlxcel`, not `--lib`, so they were not part of the `--lib` run above. Run separately (`cargo test --release --bin mlxcel --features metal,accelerate commands::generate`): 82 passed, including all four.
- `cargo test --release -p mlxcel-core --lib --features metal,accelerate drafter::dflash::config`: 16 passed.
- `cargo test --release --test speculative_dispatch --features metal,accelerate`: 22 passed.
- `cargo test --release --lib --features metal,accelerate models::detection`: 30 passed.
- `cargo clippy --release --workspace --features metal,accelerate --tests -- -D warnings`: clean.
- `cargo fmt --check`: clean.
- Classic-path non-regression, executed against a real checkpoint: `-m qwen3-0.6b-4bit --draft-model qwen3-0.6b-4bit` runs `SpeculativeGenerator`, 24 tokens, acceptance_rate 1.0000, 137.54 tok/s.
- Mutation test: guarding the new detection arm as `if false && ...` makes both the `-m` and `--draft-model` rejection tests fail with the pre-fix classification, on both markers.

### B. References

- Issue #1168 (specification)
- Issue #1172 (pre-existing, unrelated ffmpeg 9 `-vsync` regression surfaced by the full workspace test run)
- `src/lib/mlxcel-core/src/drafter/dflash/config.rs` (structural probe), `src/models/detection.rs` (`get_model_type` rejection), `src/commands/generate.rs` (`reject_dflash_drafter_offline`)
- PR #1170 review and security comments
