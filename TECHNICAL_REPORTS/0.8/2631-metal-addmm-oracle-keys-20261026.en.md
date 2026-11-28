# PR #2631: Key the bf16 addmm and vision oracles on what each backend computes

**Date**: 2026-10-26
**Status**: Complete on Metal (M1 Ultra); ROCm, CUDA and Metal M3+ expectations are source-derived and unmeasured
**Languages**: Rust, Markdown
**Risk Level**: Low (test-only)

## Executive Summary

PR #2631 (issue #2626) fixes about ten vision tests that failed on Apple Silicon. They were first logged as flaky. A serial run showed most were deterministic, and the investigation found three backend-dependent oracles plus one real test race. No production code was wrong. The oracles had been captured on ROCm gfx1151 and on x86_64 Linux CPU (PRs #2551 and #2578), and Metal legitimately computes some of those cases differently. Each oracle is now keyed on a property the test can detect. No assertion was loosened into a tolerance that would hide an `addmm` routing regression.

## 1. Problem Statement

On `main`, on an M1 Ultra:
- `--lib dinov2_layers::policy` failed 5 of 5, deterministically.
- `--lib vision::dense` failed 5 tests in both parallel and `--test-threads=1` mode. Four failures were stable, and the fifth changed between runs.

The tests prove that the vision linear layers route through `mlxcel_core::addmm`. They rely on a bf16 halfway case where one fused rounding of `x @ w.T + b` differs from rounding the product and then the sum.

## 2. Root Causes

| Failure group | Cause |
|---|---|
| addmm routing oracles (9 tests) | MLX's Metal `AddMM::eval_gpu` sends `min(M, N) == 1` to `gemv_axbpy`, whose `GEMVKernel` epilogue in `gemv.h` casts the F32 accumulator to bf16 before adding the bias. A single-row product therefore rounds twice on Metal, exactly like a separate add. Two or more rows use steel GEMM or `gemv_wide` and round once. CPU and ROCm round once at every row count. Upstream `mlx` 0.32.4 on the same machine shows the same split (1 row: 1.9375; 2 or more rows: 1.9296875), so this is MLX behavior and not an mlxcel defect. The kernel is byte-identical across the old and new MLX pins, so the #2614 bump did not cause it. |
| Sapiens2 compiled RoPE | The fixture value was captured on ROCm. On Metal and CPU, the compiled and eager graphs agree with each other, and the repository's own generator refuses to write a fixture on this host. |
| SAM 3D Body MHR CPU geometry | The fixture was captured with glibc. On macOS, MLX CPU computes atan2, sin and cos through Apple's math library, so 27 of the 388 transcendental elements differ by 1 or 2 ULP. For the single atan2 elements, upstream `mx.arctan2` on this host returns exactly mlxcel's values. |
| The varying fifth failure | The MHR test held a CPU `DefaultDeviceGuard`. Device-dependent tests that did not take `lock_default_device()` sometimes ran on the CPU during that span. Once MHR passed, its guard lasted longer, which also exposed flakes in the MoGe3 refiner and batched-decode wrapper tests: 13 of 24 parallel runs failed, versus 0 of 24 with the MHR test skipped. |

## 3. Technical Decisions

### 3.1 Prove routing at two rows and pin one row per backend

The routing proof needs a shape where `addmm` and `matmul + add` give different results. At one row on Metal they give the same result, so the oracle could not tell them apart there, and loosening the assertion would have hidden real regressions. The new `src/vision/addmm_oracle_test_support.rs` runs each oracle on two identical rows. At that shape every backend rounds once and the separate path still differs. Each test also pins the single-row value per backend: separate on Metal GPU, fused elsewhere. A change in MLX's kernel selection would then fail loudly instead of passing silently.

### 3.2 Bound only the transcendental fields on non-Linux hosts

The MHR oracle stays bit-exact on Linux and for every non-transcendental field everywhere. Recapturing it on macOS would allow bit-exactness on macOS too, but that requires running the upstream sam3d_body code. That was not permitted in this run, so on non-Linux hosts the four transcendental fields are bounded to 2 ULP. This is the one remaining bounded assertion.

### 3.3 Lock in the tests, not in the shared harness

Putting `lock_default_device()` inside `batched_decode_harness` would deadlock: callers already hold `mlx_test_guard()`, and the lock is not reentrant. The lock is therefore taken in each device-dependent test.

## 4. Validation

All runs used an M1 Ultra release build with `--features metal,accelerate`:
- `--lib vision::dense`: 192 passed in 3 of 3 runs in both modes, and in 40 of 40 runs of the built binary.
- `--lib dinov2_layers::policy`: 5 passed in 3 of 3 runs in each mode.
- After merging the latest `main`, the orchestrator repeated `vision::` three times. Workspace clippy and fmt were also run.

No real-checkpoint run was needed because only `#[cfg(test)]` code and test fixtures changed.

## 5. Learning Points

- "Flaky" should be tested with a serial run before it is accepted. Here the serial run separated deterministic backend differences from the one real race.
- An oracle captured on one backend encodes that backend's kernel choices. When a value is authentic only for some backends, state which ones in the test.
- Metal single-row bf16 `addmm` rounds twice. This also applies to production decode (M = 1) for non-quantized bf16 layers, and it matches upstream MLX on Metal. Parity with mlx-lm holds, but bit-exact comparisons against CPU or ROCm captures at M = 1 will not.

## 6. Follow-up and Limitations

- ROCm gfx1151, CUDA, Metal M3 or newer (`gemv_wide`), and non-x86_64 Linux were not run. Their expectations come from reading the source.
- A macOS capture of the MHR oracle with the upstream code would remove the 2-ULP bound.

## References

- Issue #2626, PRs #2551, #2578, #2614
- `src/vision/addmm_oracle_test_support.rs`, `src/vision/encoders/dinov2_policy_tests.rs`, `src/vision/dense/sam3d_body/mhr_tests.rs`
