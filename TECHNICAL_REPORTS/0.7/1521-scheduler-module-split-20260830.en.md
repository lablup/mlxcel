# Technical Report: PR #1521 - Scheduler Module Split

**Date**: 2026-08-30
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low

---

## Executive Summary

PR #1521 splits the batch scheduler implementation into a directory module while preserving existing behavior and public API boundaries outside `src/server/batch/`. The change reduces the latest-main 8724-line implementation file into concern-oriented modules, keeps the MTP dispatch source-inspection guard tied to real compiled code, and adds a structural test to prevent immediate size regression.

## 1. Problem Statement

### 1.1 Background

`src/server/batch/scheduler.rs` had grown into the central merge-conflict hotspot for scheduler, prompt-cache, paged-KV, handoff, and speculative-decoding work. The issue requested a zero-behavior refactor into natural scheduler submodules, with no existing scheduler test edits and no public API drift outside `src/server/batch/`. During finalization, `origin/main` advanced through PR #1503; its context-retention scheduler behavior and tests were transplanted into the split before the branch was republished.

### 1.2 Existing Issues

- **Oversized implementation unit**: The scheduler implementation was 8724 lines on the rebased latest-main base, exceeding the documented 2000-line anti-pattern threshold without a recorded exception.
- **Unclear concern boundaries**: Admission, prefill, decode, prompt-cache, paged layout, handoff, and speculative finalization code lived in one large file, increasing review and rebase cost.
- **Source-inspection coupling**: `speculative_burst_tests.rs` reads `src/server/batch/scheduler.rs` directly to verify MTP dispatch coverage, so deleting that path outright would either break the guard or tempt a duplicated fake source marker.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Continued scheduler merge conflicts | Medium | High |
| Accidental behavior change during split | High | Low after targeted tests |
| Source-inspection guard masking MTP dispatch drift | High | Low after compiling the scanned file as production code |

## 2. Technical Review

### 2.1 Security

No request parsing, authentication, authorization, filesystem access, network access, or secret-handling logic changed. The refactor moved existing scheduler methods and adjusted visibility only to the scope required by sibling modules.

### 2.2 Performance

Runtime behavior is unchanged. The additional module boundary has no dynamic dispatch, allocation, or runtime branching impact because the same inherent `BatchScheduler` methods are compiled into the same crate.

### 2.3 Compatibility and Dependencies

- **Breaking Changes**: No public API change outside `src/server/batch/`.
- **New Dependencies**: None.
- **Compatibility**: Existing scheduler tests and adjacent speculative/prompt-cache tests passed under the same build profile used before the refactor.

### 2.4 Code Quality

The split creates concern modules for admission, configuration, decode ticks, handoff helpers, paged layout, prefill, prompt-cache handling, run-loop helpers, speculative finalization, and the MTP dispatch source seam. A structural unit test now fails if any scheduler module file exceeds 2000 lines without a documented exception.

## 3. Technical Decisions

### 3.1 Keep `scheduler.rs` as compiled MTP dispatch source

**Context:** The existing `speculative_burst_tests.rs` source-inspection guard uses `include_str!("scheduler.rs")` and must remain byte-identical to base for this issue. A fake shim would make the guard pass without proving real dispatch coverage.

**Decision:** Keep `src/server/batch/scheduler.rs` as a 611-line production source file containing the actual MTP dispatch methods, and compile it into the directory module with `#[path = "../scheduler.rs"] mod mtp_dispatch;`.

**Rationale:** The unchanged source-inspection test continues reading the legacy path, while the scanned source is the exact code compiled into the scheduler module. This preserves the guard's ability to catch dispatch drift.

### 3.2 Split by scheduler state-machine concerns

**Context:** The issue requested natural submodules and zero behavior change rather than redesign.

**Decision:** Move method groups into directory-module siblings and keep the same `BatchScheduler` type and method names, widening only moved private methods to `pub(super)` where sibling modules need access.

**Rationale:** Multiple inherent `impl BatchScheduler` blocks preserve call sites and avoid new traits, wrapper types, or dispatch layers.

## 4. Change Summary

| Module | Lines | Responsibility |
|--------|-------|----------------|
| `src/server/batch/scheduler.rs` | 611 | Real compiled MTP dispatch methods retained at the legacy source-inspection path |
| `src/server/batch/scheduler/mod.rs` | 980 | Shared imports, helpers, constants, state type, test module declarations |
| `src/server/batch/scheduler/admission.rs` | 703 | Intake, enqueue, scheduler action selection, preemption admission, paged-block admission |
| `src/server/batch/scheduler/config.rs` | 672 | Constructors, builder methods, resolved scheduler configuration, MTP policy setup |
| `src/server/batch/scheduler/decode_tick.rs` | 1632 | Decode execution, lookahead, preemption eviction, completion, cancellation, abort handling |
| `src/server/batch/scheduler/handoff.rs` | 526 | Sequence handoff extraction, ingest, role-specific handoff helpers |
| `src/server/batch/scheduler/paged_layout.rs` | 312 | Sequence-state allocation, KV mode/layout resolution, storage sync |
| `src/server/batch/scheduler/prefill.rs` | 1374 | Full, chunked, and batched prefill paths plus prefill finalization |
| `src/server/batch/scheduler/prompt_cache.rs` | 1229 | Prompt-cache adoption, donation, warmup, release, observability bookkeeping |
| `src/server/batch/scheduler/run_loop.rs` | 310 | Run loop, structured-mask helpers, thinking budget, metrics publishing |
| `src/server/batch/scheduler/speculative_finalize.rs` | 657 | Legacy burst finalization, grantee promotion, slice failure/routing helpers |
| `src/server/batch/scheduler/structure_tests.rs` | 47 | File-size regression guard |

## 5. Validation

- `cargo test --lib scheduler_modules_stay_below_documented_anti_pattern_threshold`: passed, 1 passed, 7417 filtered out.
- `cargo test --lib server::batch::speculative_burst_tests::every_mtp_dispatch_site_covers_every_capable_variant`: passed, 1 passed, 7416 filtered out.
- `cargo test --lib server::batch::scheduler::`: passed, 99 passed, 7 ignored hardware/model tests, 7312 filtered out.
- `cargo test --lib server::batch::scheduler_prompt_cache_tests::`: passed, 24 passed, 7394 filtered out.
- `cargo test --lib server::batch::speculative_burst_tests::`: passed, 47 passed, 7371 filtered out.
- `cargo test --lib server::batch::speculative_slice_tests::`: passed, 12 passed, 7406 filtered out.
- `cargo build`: passed.
- `cargo clippy --lib --tests -- -D warnings`: passed.
- `cargo fmt --check`: passed.
- `git diff --check`: passed.
- `speculative_burst_tests.rs` hash check against `origin/main`: passed, base and working tree SHA-256 both `74704868ced3b3627fcd148252e4bbe39ac98ff2812c5ed1354da9ef3d5845b9`.

## 6. Follow-up Notes

The ignored scheduler tests require the `qwen3-0.6b-4bit` checkpoint and real GPU forwards. They were not run in this bounded refactor pass; the normal targeted suites exercised the compile and non-hardware behavior gates.
