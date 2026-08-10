# Technical Report: PR #1096 - Deterministic Autotune Profiling Tests

**Date**: 2026-08-10
**Status**: Completed with platform qualification pending
**Languages**: Rust
**Risk Level**: Medium

## Executive Summary

PR #1096 removes host-scheduler timing jitter from autotune tests that assert candidate ordering. Production profiling continues to measure real `TunableOp::run` and `sync` latency with `Instant`, while tests exercise the same warmup, interleaving, selection, persistence, and memoization logic through an internal manual microsecond clock.

Review also closed a production resource-bound gap: caller-supplied warmup repetitions are now capped by normalized `max_reps`, with CLI and API documentation updated to match.

## 1. Problem Statement

Two CPU-only autotune tests intermittently selected the wrong synthetic tactic under host load because `FakeOp` modeled costs with microsecond `thread::sleep` calls. Scheduler overshoot could exceed the 400-720 microsecond gaps being compared, so tests intended to verify deterministic selection instead measured operating-system wakeup behavior.

The same test double powered other profile and resolve assertions, so fixing only the two observed failures would have left equivalent latent flakes elsewhere in the module.

## 2. Technical Decisions

### 2.1 Inject time at the profiler boundary

An internal `ProfileTimer` abstraction supplies marks and elapsed microseconds to the existing profiling algorithm. `profile()` instantiates `RealTimer`, preserving the production path, while test-only helpers use `ManualTimer` and advance it by the selected tactic's declared cost. The seam remains internal and the public `TunableOp` contract is unchanged.

### 2.2 Retain real sleep only for a one-candidate budget test

`timed_repetitions_scale_inversely_with_launch_cost` still sleeps because its subject is adaptive wall-clock repetition budgeting. Each profile in that test has one candidate, so scheduler jitter can change the realized repetition count within documented bounds but cannot decide candidate ordering.

### 2.3 Prove selection sensitivity and cap caller work

The regression suite evaluates both cost directions: a faster tuned candidate must replace the default, while an inverted cost map must retain the faster default. A temporary wrong expectation was observed red before restoration. Separately, sanitized `warmup` is capped after `max_reps` is normalized above the timed repetition floor, preventing pathological CLI/API input from forcing an oversized warmup loop.

## 3. Change Summary

| Item | Value |
|------|-------|
| Files changed | 4 |
| Lines added | 210 |
| Lines deleted | 56 |
| Focused tests | 57 passed |

- Added internal real/manual timer implementations and routed the profiler's warmup and timed phases through them.
- Migrated all ordering-sensitive fake-op profile and resolver tests to deterministic synthetic time.
- Added inverted-cost and warmup-ceiling regression coverage.
- Added a profiling closure seam for test-only resolver execution without changing public APIs.
- Clarified `ProfileConfig` and `mlxcel tune` help/output to state that warmup is capped by `max_reps`.

## 4. Review Findings

| Finding | Severity | Resolution |
|---------|----------|------------|
| Caller-supplied `warmup` could exceed the documented `max_reps` ceiling | Medium | Clamp warmup after normalizing `max_reps`; add a seven-call regression proof |
| Public and CLI wording still described warmup as an unconditional minimum after the cap | Low | Update profiler docs, CLI help, and tune summary output |

No Critical or High findings remained. The timer abstraction is internal, production still uses `Instant`, and the deterministic resolver entry point is test-only.

## 5. Validation

- `cargo fmt --check -p mlxcel-core`: passed.
- `cargo fmt --check -p mlxcel`: passed.
- `cargo test --profile test-fast -p mlxcel-core --lib autotune:: -- --test-threads=1`: 57 passed, 0 failed, 1,355 filtered out.
- Inverted synthetic-cost selection test: passed; a temporary wrong expectation failed as required before restoration.
- Warmup-ceiling regression: passed, proving five capped warmups plus two timed repetitions.
- Hosted change detection, crate-version, kernel-dtype-key, cross-repository-reference, cargo-fmt, cargo-deny, and CLA checks passed; MLX pin extraction was skipped because the pin did not change.
- macOS `metal,accelerate` and loaded-host full-lib gates were unavailable and are not claimed. A current Linux full-lib attempt began 1,411 tests but aborted in an unrelated CUDA-backed cache test because no CUDA backend is exposed.

## 6. Related Work

- Issue #1079: sleep-accuracy flake and deterministic timing requirements.
- Issue #997: separate concurrent-load flakes with a different mechanism and platform gate.
