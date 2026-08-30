# Technical Report: PR #1512 - Deterministic TP Crossover Assertions

**Date**: 2026-08-30
**Status**: Open PR, ready for central merge
**Languages**: Rust
**Risk Level**: Low

## Executive Summary

PR #1512 removes host-scheduler timing noise from `e2e_crossover_larger_models_benefit_more`. The tensor-parallel benchmark path still measures real elapsed time for ordinary `run_tp_benchmark` calls, while crossover analysis now evaluates the hidden-size model through an injected nominal timer. The target test asserts exact model efficiencies and benefit flags, so it checks the property named by the test instead of comparing microsecond wall-clock noise.

## 1. Problem Statement

The previous test compared `large.scaling_efficiency >= small.scaling_efficiency` after `run_crossover_analysis` had measured microsecond `spin_wait` loops with `Instant`. Under loaded full-suite runs, scheduler delay could dominate the 2048-vs-8192 synthetic timing gap and intermittently invert the assertion. The issue thread narrowed live scope to this TP test only; the autotune tests originally mentioned in the same family were already fixed by PR #1096 through a manual timer.

## 2. Technical Decisions

### 2.1 Inject timing at the benchmark harness boundary

A private `BenchmarkTimer` seam now supplies `wait` and `measure`. `run_tp_benchmark` constructs `RealBenchmarkTimer`, preserving real `Instant` and `spin_wait` behavior. `run_crossover_analysis` constructs `NominalBenchmarkTimer`, which advances synthetic elapsed time exactly by requested durations.

### 2.2 Keep crossover analysis deterministic

Crossover analysis is a model-size comparison, not a wall-clock benchmark. It now uses nominal simulated durations for baseline and TP runs so throughput, scaling efficiency, all-reduce overhead, and benefit decisions are stable under host load.

### 2.3 Assert exact model outputs

The target e2e test now requires the 2048 hidden-size TP=2 entry to have scaling efficiency 0.5 and the 8192 hidden-size TP=2 entry to have scaling efficiency 0.8. It also verifies that the smaller model only breaks even while the larger model is beneficial. Missing entries now fail explicitly instead of silently bypassing the assertion.

## 3. Change Summary

| Item | Value |
|------|-------|
| Files changed | 2 |
| Rust lines changed | 133 insertions, 30 deletions |
| Public API additions | None |
| Production timing path | Preserved for `run_tp_benchmark` |
| Crossover timing path | Deterministic nominal timer |

- Added private real and nominal benchmark timers.
- Routed `run_tp_benchmark` through the real timer without changing its signature.
- Routed `run_crossover_analysis` through the nominal timer and documented that behavior.
- Replaced the flaky crossover assertion with exact nominal model expectations and benefit checks.

## 4. Validation

- `cargo fmt --check`: passed.
- `git diff --check -- src/distributed/tensor_parallel/benchmark.rs tests/tp_e2e.rs`: passed.
- `cargo test --profile test-fast --test tp_e2e e2e_crossover_larger_models_benefit_more -- --exact --nocapture`: passed, 1 passed, 35 filtered out.
- `cargo test --profile test-fast -p mlxcel distributed::tensor_parallel::benchmark::tests::crossover_analysis_basic -- --exact --nocapture`: passed, 1 matching unit test passed, remaining filtered tests reported 0 run.
- Mutation check: temporarily changed crossover compute scaling from quadratic to linear. The exact target test failed on `small-model scaling efficiency 0.6666666666666666 should match the injected nominal model 0.5`, then passed after restoration.
- `cargo clippy --profile test-fast --test tp_e2e -- -D warnings`: passed.

The focused local validation ran while `/proc/loadavg` reported `12.14 9.86 6.28`. A full workspace or full-suite loaded-host run was not executed because this auto-implementation unit explicitly forbids broad cargo, workspace, and serial all-test commands.

## 5. Review Findings

| Finding | Severity | Resolution |
|---------|----------|------------|
| The target assertion depended on microsecond host scheduling | Medium | Replaced the crossover analysis timing source with deterministic injected nominal time |
| The test could silently pass if either expected entry was absent | Low | Replaced optional `if let` with explicit `expect` checks |
| Expected constants could be misread as caller-configured timing values | Low | Added comments explaining that the constants come from the hidden-size timing model |

No security issue was introduced. The change does not parse external input, add I/O, expose new APIs, or add unbounded resource use beyond existing configured benchmark loops.

## 6. Remaining Limits

- Hosted CI results should be read from GitHub before central merge.
- No hardware/model validation is claimed; this path is a simulated tensor-parallel benchmark and does not load real models.
