# Technical Report: PR #2209 - SSM BF16 timestep regression

**Date**: 2026-10-12

**Status**: Local CUDA runtime and Clippy verification completed

**Languages**: Rust

**Risk Level**: Low

## Executive Summary

PR #2209 originally diagnosed the CUDA SSM parity failure and promoted `dt_bias` to float32 before the compiled timestep calculation. PR #2270 subsequently landed the same production correction on `main`, so after merging current `main` this PR preserves the contributor's original history and contributes an isolated regression test for the precision contract.

The regression reduces the recurrence to an analytically known value, covers float32 and BF16 activations plus BF16-to-float32-to-BF16 bias transitions, and keeps the existing float32 state tolerance unchanged. The test lives in a child module to keep the parent parity file below the repository workflow's 500-line limit.

## 1. Problem Statement

The Granite BF16 cases in `ssm_update_parity_tests` failed because the compiled preprocessing cast `dt` to float32 but left `dt_bias` in BF16. MLX selected BF16 for the mixed-type addition, so `softplus(dt + dt_bias)` was rounded before the backend recurrence kernel received it; storing the resulting state as float32 could not recover that precision.

The CUDA kernel body and the float32 MLX reference were correct. The defect was in the shared pre-dispatch `compute_dt_compiled` helper. Mainline PR #2270 already casts both operands to float32, leaving this PR to pin that behavior against regression.

## 2. Change Summary

The new case starts with zero state and sets `x = B = 1`, `C = D = 0`, and `A_log = 0`. Every next-state element therefore equals the processed timestep `softplus(-2.5)`. Both inputs are exactly representable in float32 and BF16, and the expected value is calculated with host float32 arithmetic, so the oracle does not depend on another GPU operation or on input quantization noise.

The case runs float32 and BF16 activation variants while cycling the stored bias through BF16, float32, and BF16. This checks the precision contract and compilation-cache dtype transitions while preserving the existing output/state dtype assertions and tolerances.

## 3. Technical Decisions

### 3.1 Preserve mainline production code

The merge conflict in `compute_dt_compiled` was resolved to the current `main` implementation because PR #2270 had already landed the same explicit float32 cast with measured evidence. No duplicate production change is claimed by the final diff, and the original PR commit remains in history.

### 3.2 Use an independent scalar oracle

Comparing the fused kernel only with another MLX graph could repeat the same promotion behavior. The isolated recurrence and host float32 expected value identify timestep narrowing directly and fail under the old mixed-type preprocessing without widening a tolerance.

## 4. Verification

| Check | Result |
|---|---|
| Published PR baseline | `33265bb76c0244b66d198e51667256b15ec33e1a` |
| Stable source candidate | `a9696bd8cb3d0550d312ce5b9c561c544f9886e4` |
| Focused CUDA no-run build | Passed with `-p mlxcel-core --profile test-fast --features cuda --lib --no-run` and two build jobs |
| Rust formatting and diff whitespace | Passed |
| Focused GB10 SSM runtime | 6 passed, 0 failed, 0 ignored under `gpu-lock`; all six cases executed in 9.65 seconds |
| Repository contract gate | Passed: crate-version consistency, kernel dtype/cache-key checks, kernel-port dispatch, llama-server compatibility manifest, ROCm overlay records, 123 Python tooling tests, and 49 WebUI fixtures/DTO/schema checks |
| Full CUDA workspace gate | Passed under `gpu-lock`: 12,247 passed, 0 failed, and 389 ignored across 163 suites, including doc tests, with `cargo test --locked --workspace --profile test-fast --features cuda -j2 --no-fail-fast -- --test-threads=1` |
| Broad CUDA Clippy | Passed: `cargo clippy --locked --workspace --all-targets --profile test-fast --features cuda -j 2 -- -D warnings` |

PR #2270's preserved GB10 evidence measured normalized state RMS/max changing from `2.532402801e-3 / 4.127457397e-2` to `2.838613609e-8 / 1.701009653e-6` without changing tolerances. Its deterministic Granite comparison produced identical 64-token fused-kernel and graph sequences.

## 5. Review Findings and Limitations

Implementation review found the regression independent and correctly integrated. Security review found no new input, memory, privilege, or data-exposure surface. Performance impact is confined to test execution; production code is unchanged relative to current `main`.

Runtime scope is GB10 CUDA. Metal and ROCm runtime execution, and real-checkpoint testing of FalconH1 and Plamo2, are outside this PR's current validation.

## 6. Related Records

- Issue #2142: CUDA SSM BF16 parity failure
- PR #2209: original diagnosis and regression contribution
- PR #2270: production correction and preserved GB10 numerical/checkpoint evidence
- `docs/benchmark_results/data/ssm-bf16-parity-gb10-2026-10-11/`: stage deviations and Granite kernel-versus-graph artifacts
