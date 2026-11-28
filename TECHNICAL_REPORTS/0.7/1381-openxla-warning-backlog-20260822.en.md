# Technical Report: PR #1381 - clear the OpenXLA warning backlog and deny all warnings in CI

**Date**: 2026-08-22
**Author**: Jeongkyu Shin
**Status**: Completed
**Languages**: Rust, YAML (GitHub Actions)
**Risk Level**: Low (lint and visibility change; no runtime behavior altered)

---

## Executive Summary

The `xla-compile` CI job denied a single lint, `unused_imports`, rather than all warnings. That was deliberate: the OpenXLA feature combinations carried a dead-code and clippy backlog, and a job that is red the day it lands gets routed around and then ignored.

This PR clears the backlog and widens the gate to `RUSTFLAGS: "-D warnings"`. Fourteen findings across two crates were resolved, one by deletion and the rest by cfg gates, scoped `#[allow]`s with comments naming what keeps each item alive, and three mechanical fixes. Four of the fourteen were not in the issue's list at all; they were found by re-measuring the backlog before starting, because the issue's list had drifted since it was filed.

The change is verified by the widened gate running green on the PR that widens it: `OpenXLA feature compile` did a real 7m44s rebuild under the new flag and passed.

## 1. Problem Statement

### 1.1 Background

PR #1282 added `xla-compile` with `RUSTFLAGS: "-D unused_imports"`. That lint was chosen because it is the exact class of one of the two defects that reached `main` through the OpenXLA coverage gap, a dead `load_weights_from_dir_with_filter` re-export; compile errors covered the other, a non-exhaustive `match` after a new `ModelRequest` variant landed. Both historical breaks were caught. New dead code under these features was not.

### 1.2 Existing Issues

Clearing the backlog is the precondition for widening the gate, and the backlog spanned both `mlxcel-xla` and `mlxcel`. Deletion was usually the wrong fix, because several items are live under feature sets other than the one that reports them dead.

### 1.3 Risk Assessment

Low in isolation, compounding over time. Every ungated warning is a place where the next real defect of the same class arrives unremarked, and the two defects that motivated #1282 are the existing proof.

## 2. Change Summary

Ten files: eight Rust, one workflow, and the report.

| File | Resolution |
|---|---|
| `mlxcel-xla/src/aux.rs` | `Float16`, `Uint32`: `cfg_attr(not(micro-oracle), allow(dead_code))` |
| `mlxcel-xla/src/aux_manifest.rs` | collapsible `if` rewritten as an edition-2024 let chain |
| `mlxcel-xla/src/iree.rs` | `decode_ragged_logits` cfg-gated to `diagnostics`; `decode_ragged_mrope_logits` deleted; stranded doc comment reattached |
| `mlxcel-xla/src/phi4_audio.rs` | scoped `allow(clippy::too_many_arguments)` |
| `src/loading/vlm.rs`, `src/models/sanitize.rs` | `cfg_attr(not(surgery), allow(clippy::needless_match))` |
| `src/server/batch/xla_audio_preprocess.rs` | seven `cfg_attr(not(test), allow(dead_code))`, each with its caller named |
| `src/server/batch/xla_worker_admission.rs` | `allow(clippy::while_let_loop)` with reason |
| `src/server/media.rs` | cfg_attr condition widened to `any(not(xla-iree), not(test))` |
| `.github/workflows/ci.yml` | `RUSTFLAGS` to `-D warnings`; policy comment replaced; two coverage gaps recorded |

## 3. Technical Decisions

### 3.1 Re-measure the backlog before working from it

The issue's list was measured when it was filed. Re-running both clippy commands on the current tree produced fourteen findings against the issue's ten, and the four extras were not incidental: two were plain mechanical fixes, and two were the `needless_match` pair that is the most dangerous item in the whole change (3.3). Working from the stale list would have shipped a PR that fails its own acceptance criteria.

### 3.2 One deletion, and only after proving the item is dead everywhere

`decode_ragged_mrope_logits` is a thin wrapper over `decode_ragged_mrope_logits_with_modes` and is dead under both measured feature sets. It was deleted rather than allowed, because the acceptance criterion requires every `#[allow]` to name a caller keeping the item alive, and there is none. The justification is a repo-wide search across all file types, extended during review to the paths a plain grep misses: macros (`mlxcel-xla` has one `macro_rules!`, for MLIR fixtures, and no `paste!` or `concat_idents!`), doc tests, `pub use` re-exports, trait impls, and feature-gated callers under `micro-oracle` and `xla-diagnostics-cpu`. What remains is the identically-named C function `xla_llama_decode_ragged_mrope_logits`, its extern declaration, and the `_with_modes` sibling, which keeps its live caller.

Its twin `decode_ragged_logits` was **not** deleted. It is dead under `cuda,xla-iree` and live under `xla-diagnostics`, so it was cfg-gated to `feature = "diagnostics"` instead, matching all four of its callers in `batch.rs`, which already carry that cfg. Since `diagnostics = ["iree"]`, the new gate is strictly narrower than the `#[cfg(feature = "iree")] mod iree` containing it and cannot break any combination. Deleting it would have turned a warning into a build break on the other feature set, which is the failure this issue explicitly warned against.

### 3.3 Two clippy suggestions that had to be refused

`src/loading/vlm.rs` and `src/models/sanitize.rs` both contain:

```rust
let resolved_transform = match transform {
    Some(t) => Some(t),
    None => {
        #[cfg(feature = "surgery")]
        { active_pipeline.as_deref().map(...) }
        #[cfg(not(feature = "surgery"))]
        ...
    }
};
```

Under `--no-default-features` the `surgery` arm compiles out, the `None` arm collapses to `None`, and clippy correctly reports the match as needless with the suggestion "replace it with `transform`". Taking that suggestion would delete active-pipeline resolution from the default build, where `default = ["surgery"]`.

The fix is `#[cfg_attr(not(feature = "surgery"), allow(clippy::needless_match))]` on the `let` statement: silenced exactly for the builds that see the collapsed shape, and still linted in the default build. This is the concrete instance of the issue's rule that a fix which alters behavior is the wrong fix.

### 3.4 `not(test)` rather than a bare allow

Every suppression in `xla_audio_preprocess.rs` and `media.rs` is conditioned on `not(test)`. `cargo check --all-targets` compiles the `mlxcel` lib twice, once under `cfg(test)`, so the suppression is self-enforcing: if the sole test caller is deleted, the lint returns rather than staying silent forever. Each attribute carries a comment naming the caller, which the review checked one by one.

A detail worth recording: `#[allow]` suppresses a lint without marking the symbol live, so the `healthy` field still warned on its own after `is_healthy` was allowed. Each item needed its own attribute.

### 3.5 Widen `RUSTFLAGS`, and record what it still does not cover

`RUSTFLAGS: "-D warnings"` on a `cargo check` job denies rustc lints only. Clippy's own lints, which are most of what this PR fixed, remain gated by nothing under these feature sets, because the `clippy` job builds default features where `mlxcel-xla` is default-off. Swapping the job's `cargo check` for `cargo clippy` would close that, and was deliberately left out of scope. The gap is recorded in the workflow's exclusion list rather than left implied, so the cleared half of the backlog cannot regrow behind a comment that claims full coverage.

## 4. Review Findings

No CRITICAL or HIGH findings survived either review. The follow-up commit fixed four accuracy defects:

- **Error messages naming functions the build does not contain.** Two arity-check messages in `iree.rs` named the thin wrappers rather than their `_with_modes` emitters. Since one wrapper is now deleted and the other is diagnostics-only, a production `cuda,xla-iree` build could return `"decode_ragged_mrope_logits expects ..."` naming a function it does not have.
- **An incomplete caller list.** `AudioPreprocessStage::spawn`'s comment named one test file; `xla_worker_tests.rs` calls it too. Since the acceptance criterion is that every allow names what keeps the item alive, an incomplete list is precisely what it exists to prevent.
- **A comment describing half of what it covered.** `drain_preprocessed` has two drain loops, image and audio, and the function-level allow covers both.
- **Two unrecorded properties of the widened gate**, added to the exclusion list (see 5.2).

## 5. Validation

### 5.1 What passed

All on the GB10 host with the IREE runtime provisioned, and re-run on the final commit.

| Command | Result |
|---|---|
| `cargo clippy --features cuda,xla-iree --all-targets -- -D warnings` | exit 0, clean |
| `cargo clippy --no-default-features --features xla-diagnostics --all-targets -- -D warnings` | exit 0, clean |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings` (default regression) | exit 0, clean |
| `RUSTFLAGS="-D warnings" cargo check --features cuda,xla-iree --all-targets` | exit 0 |
| `RUSTFLAGS="-D warnings" cargo check --no-default-features --features xla-diagnostics --all-targets` | exit 0 |
| `cargo clippy -p mlxcel-xla --lib --features iree,micro-oracle --all-targets -- -D warnings` | exit 0 |
| `OpenXLA feature compile` on this PR | success, 7m44s real rebuild under the new flag |

The two `cargo check` rows are not redundant with the clippy rows. `cargo clippy -- -D warnings` applies the flag to the primary package only, while `RUSTFLAGS` reaches every path-built unit, so a warning in `mlxcel-core` or `mlxcel-surgery` would red the job without appearing in any clippy run. Verifying the acceptance criteria alone would not have shown whether the job goes green.

### 5.2 What the gate covers, precisely

`RUSTFLAGS` reaches `mlxcel-core`, `mlxcel-surgery`, `mlxcel-xla` and the build scripts, so a CUDA-only warning in `mlxcel-core` now reds a job named for OpenXLA. Registry and git dependencies cannot, because cargo compiles them with `--cap-lints allow`. `rust-toolchain.toml` pins `1.97.1` exactly rather than tracking `stable`, so a new rustc release cannot turn the job red on its own; the cost moves to the pin bump, deliberately.

There is no `default-members`, so `cargo check` here resolves to `-p mlxcel` and `--all-targets` expands within that package only. `mlxcel-xla`'s 23 `#[cfg(test)]` modules are therefore gated by no job at all. This PR's test plan ran `-p mlxcel-xla --all-targets` by hand; CI will not.

### 5.3 What was not verified

`metal,accelerate` cannot be built on this Linux host, so it is unverified by execution. The static argument that it is unaffected: `make verify-clippy` runs `--features metal,accelerate` without `--no-default-features`, so `surgery` stays on and both `needless_match` attributes expand to nothing; `mlxcel-xla` builds with no features there, so every file changed in it sits behind `#[cfg(feature = "iree")]` and never compiles; and on the `mlxcel` side only `media.rs` and `xla_audio_preprocess.rs` compile, where every change is an `allow` attribute that cannot break a build.

## 6. Learning Points

1. **A backlog measured at filing time is a snapshot, not a specification.** Re-measure before working from it. Here the drift was 40 percent, and one of the drifted items was the trap that would have caused a behavior regression.
2. **Verifying the acceptance criteria is not the same as verifying the deliverable.** The criteria named clippy; the job runs `cargo check` with `RUSTFLAGS`, which has a wider blast radius. Run what the artifact will actually run.
3. **`#[allow]` suppresses a lint; it does not mark a symbol live.** Allowing an accessor does not stop the field it reads from warning on its own.
4. **Condition suppressions so they expire.** `not(test)` keeps the lint alive in the test build, so a suppression whose justification disappears turns red again instead of outliving its reason.
5. **A clippy suggestion is a suggestion about the configuration clippy was run in.** Under `cfg`-split code, applying it can delete behavior that exists only in a configuration clippy never saw.

## 7. Follow-up Actions

- **`mlxcel-xla`'s test modules are ungated by any CI job.** Recorded in the workflow's exclusion list; closing it means selecting the package explicitly or adding `default-members`.
- **Clippy-only lints under these feature sets remain ungated**, so that half of the backlog can regrow. Closing it means running clippy in this job.
- **`media_tests.rs` reaches `validate_xla_raw_counts_with_audio` only through the `supports_audio = false` wrapper**, so the `true` branch has no direct unit coverage, and the sign of that flag is the security-relevant bit.
- **A pre-existing stranded doc comment** at `iree.rs:2296-2305`, the same defect this PR fixed at `:1819`, leaves two broken intra-doc links. It predates this branch and emits no warning because no blank line separates it.

## References

- Issue #1304 (this work), #1282 (the job and its recorded exclusions), #1303 and PR #1305 (the link half of the same exclusion list)
- `.github/workflows/ci.yml`, `src/lib/mlxcel-xla/src/iree.rs`, `src/server/batch/xla_audio_preprocess.rs`
