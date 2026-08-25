# Technical Report: PR #1407 - Disable ThinLTO in test-fast

**Date**: 2026-08-25
**Author**: mlxcel contributors
**Status**: Completed
**Languages**: Rust, TOML, Make, Markdown, YAML
**Risk Level**: Low

---

## Executive Summary

PR #1407 disables LTO in the optimized `test-fast` profile after the full macOS workspace test graph repeatedly failed during ThinLTO linking with unresolved internal Rust drop-glue symbols. The change preserves `opt-level = 3` for MLX numerical behavior and leaves the shipping `release` profile unchanged, while making the CI-faithful workspace gate link and execute successfully.

---

## 1. Problem Statement

### 1.1 Background

The repository runs its complete workspace test gate under `[profile.test-fast]` because release-profile fat LTO and one codegen unit made the many integration-test binaries too expensive to link. The fast profile still explicitly enabled ThinLTO, so every large test binary entered a cross-crate LTO link path.

### 1.2 Observed Failure

`cargo test --workspace --profile test-fast --features metal,accelerate --no-run` failed before test execution while linking targets including `molmo_parity` and `qwen3_omni_moe_parity`. The linker reported unresolved internal Rust symbols for drop glue such as `serde_json::Value`, `RotatingKVCache`, and `KVCache`, each with LLVM-private suffixes.

The failure reproduced with `CARGO_BUILD_JOBS=1`, excluding concurrent Cargo link jobs as a necessary condition. A targeted `qwen3_omni_moe_parity` no-run build passed under a smaller unit graph, while the full workspace graph failed, isolating the problem to the explicit ThinLTO workspace link shape rather than the test source itself.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood before fix |
|---|---|---|
| Workspace gate cannot produce a test verdict | High | High |
| Source-correct PRs appear broken because linking fails first | High | High |
| Release behavior changes accidentally while fixing tests | High | Low, guarded by profile assertions and diff review |

---

## 2. Technical Review

### 2.1 Correctness

`[profile.test-fast]` now sets `lto = false` while retaining `opt-level = 3`, `codegen-units = 16`, `incremental = true`, `strip = false`, and inherited unwind behavior. `[profile.release]` remains `lto = true`, `codegen-units = 1`, `strip = true`, `opt-level = 3`, and `panic = "unwind"`.

The Cargo comments, Makefile gate contract, contributor guide, installation guide, and nightly workflow now consistently describe the test profile as disabling cross-crate LTO. They retain the manual `cargo test --release --features metal,accelerate` escape hatch for defects that reproduce only under shipping code generation.

### 2.2 Security and Performance

No input, authentication, data-handling, or runtime inference surface changes. Independent correctness and security/performance reviews found no findings at any severity.

The intended performance trade-off is confined to test binaries: the gate gives up release-only LTO optimization in exchange for a simpler and reliable link graph. Shipping artifacts and benchmarks continue to use `[profile.release]`; `.github/workflows/release.yml` is unchanged.

### 2.3 Compatibility and Dependencies

- **Breaking changes**: None.
- **New dependencies**: None.
- **Release compatibility**: Unchanged.
- **Test artifact compatibility**: Test binaries may differ from release binaries in LTO-driven inlining and layout, which is documented and intentional.

---

## 3. Technical Decisions

### 3.1 Preserve optimized numerics, remove test-time LTO

| Option | Advantages | Disadvantages |
|---|---|---|
| Keep explicit ThinLTO | Retains cross-crate LTO in tests | Full workspace graph can fail before tests run |
| Disable incremental compilation | Changes another major build dimension | Does not directly remove the reproduced cross-crate LTO path and harms iteration speed |
| Use release profile for the gate | Matches shipping code generation | Reintroduces multi-hour link cost and prior nightly timeouts |
| **Chosen: set `test-fast.lto = false`** | Removes the failing link path, preserves optimized numerics, keeps fast codegen | Does not test release-only LTO behavior |

`opt-level = 3` is the requirement that keeps optimized MLX numerics representative. LTO is a shipping optimization rather than a prerequisite for those test semantics, so the two concerns are separated instead of weakening optimization globally.

### 3.2 Keep release verification as a distinct contract

The change does not reinterpret `test-fast` as release-equivalent. Release workflows still build shipping artifacts under fat LTO and a single codegen unit, and contributors retain a documented release-profile test command for investigating code-generation-specific defects.

---

## 4. Implementation Details

### 4.1 Profile Change

```toml
[profile.test-fast]
inherits = "release"
lto = false
codegen-units = 16
strip = false
incremental = true
opt-level = 3
```

### 4.2 Documentation and Workflow Alignment

- `Cargo.toml` records the reproduced missing-symbol failure and release isolation.
- `Makefile` describes the full gate as dropping cross-crate LTO rather than using ThinLTO.
- `CONTRIBUTING.md` and `docs/installation.md` explain the test/release trade-off and manual escape hatch.
- `.github/workflows/nightly-verify.yml` reflects the same operational contract used by the nightly gate.

---

## 5. Validation Evidence

### 5.1 Before the Fix

- Full workspace no-run failed while linking multiple integration-test binaries with unresolved LLVM-suffixed Rust drop-glue symbols.
- Serializing Cargo build jobs did not prevent the failure.
- A targeted integration-test build could pass, demonstrating that the failure depended on the larger workspace unit graph.

### 5.2 After the Fix

- `cargo test --profile test-fast --features metal,accelerate --test molmo_parity --no-run`: passed.
- `cargo test --profile test-fast --features metal,accelerate --test qwen3_omni_moe_parity --no-run`: passed.
- `cargo test --workspace --profile test-fast --features metal,accelerate --no-run`: passed; every workspace test target linked in 9m19s on the cold corrected profile.
- `cargo test --workspace --profile test-fast --features metal,accelerate --no-fail-fast -- --test-threads=1`: passed with zero failures across all workspace test and doctest binaries.
- `cargo clippy --workspace --all-targets --features metal,accelerate -- -D warnings`: passed.
- `cargo fmt --all -- --check`: passed.
- TOML assertions confirmed the corrected test profile and unchanged release profile.
- GitHub CI checks passed, including formatting, clippy, dependency policy, crate versions, kernel dtype keys, cross-repository references, and OpenXLA feature compilation.

---

## 6. Change Summary

| Item | Value before report commit |
|---|---|
| Implementation files changed | 5 |
| Lines added | 21 |
| Lines deleted | 19 |
| Runtime code changed | 0 files |
| Dependencies changed | 0 |

| Category | Summary |
|---|---|
| Build profile | Disable LTO only for `test-fast` |
| Release behavior | Unchanged |
| Documentation | Align Cargo, Makefile, contributor, installation, and nightly guidance |
| Review | Correctness, security/performance, and finalization reviews found no issues |

### Related Commit

| Hash | Type | Message |
|---|---|---|
| `b2204d5dc` | fix | Disable ThinLTO in test-fast |

---

## 7. Learning Points

- A targeted Rust test build and a full workspace build can produce different codegen-unit and link graphs; a targeted pass does not disprove a workspace-only linker defect.
- Build-job serialization tests concurrency, not whether explicit cross-crate LTO itself is the trigger.
- Optimized numerical testing and shipped-binary link optimization are separate contracts: preserve the former with `opt-level = 3`, while release LTO remains in the shipping profile.
- Profile changes require documentation updates wherever developers and CI obtain their operational contract, not only in `Cargo.toml`.

---

## 8. Follow-up and Monitoring

No blocking follow-up is required. Monitor the next nightly workspace gate for stable link time and continue to use the release-profile escape hatch when investigating defects that may depend on fat LTO or single-codegen-unit code generation.

### Related Work

- Issue #1406: Disable ThinLTO in the workspace test gate.
- PR #1407: Disable ThinLTO in `test-fast`.
