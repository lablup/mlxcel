# PR #2972: Preserve float32 precision in CUDA multivariate_normal and Rust reference top_p_filter

- **Date**: 2026-11-12
- **Status**: Completed (closes #2971)

## Executive Summary

Following PR #2963 (closing #2958), a codebase-wide audit of random distribution routines and sampling helpers under CUDA's `promote_types(bfloat16, float32) -> bfloat16` type-promotion overlay (`src/lib/mlxcelverse/overlays/cuda/mlx/dtype.cpp`) identified two remaining precision and dtype defects and confirmed one intentional non-issue:

1. **`mlx::core::random::multivariate_normal` (`src/lib/mlxcelverse/overlays/cuda/mlx/random.cpp`)**: Despite explicitly enforcing `dtype == float32` (`if (dtype != float32) throw std::invalid_argument(...)`) and computing SVD factors and `scaled_out` in `float32`, the final `add(mean, scaled_out, stream)` expression demoted `scaled_out` (`float32`) to `bfloat16` on CUDA builds whenever `mean.dtype() == bfloat16` (including on the CPU stream required by `linalg::svd`, because `overlays/cuda/mlx/dtype.cpp` applies process-wide).
2. **Rust reference `sampling::top_p_filter` (`src/lib/mlxcel-core/src/sampling.rs`)**: Unlike `top_p_filter_min_keep` immediately below it and the C++ `top_p_filter` updated in PR #2963, the Rust reference `top_p_filter` ran `softmax` and inclusive `cumsum` directly in `logits`'s dtype and finished with `where_cond(&mask, &sorted_logits, &neg_inf_f32)`. On non-`float32` (`bfloat16` or `float16`) inputs at large vocabulary sizes (`[1, 262144]`), sequential `cumsum` swamped per-token probabilities smaller than 1 ULP once cumulative mass passed `0.5` (`2^-8 = 3.91e-3` in `bfloat16`), leaving tail tokens unmasked, while on CUDA builds `where_cond` also demoted the output tensor to `BFLOAT16` instead of `FLOAT32`.
3. **In-chain `min_p` and `XTC` `softmax(x, -1)` (`src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp`, intentionally left unchanged)**: Neither filter uses `cumsum`, MLX's CUDA `softmax.cu` already accumulates `max` and `sum(exp)` in `float32` (`AccT = float`), masked `-inf` entries map to `0.0_bf16 < threshold` without leakage, and keeping `probs` in `logit_dtype` preserves #636's zero-`AsType` decode graph and `copy_shared_buffer` reuse.

PR #2972 upcasts `bfloat16` `mean` in `multivariate_normal` before `add(mean_up, scaled_out, stream)`, upcasts non-`FLOAT32` `logits` at the entry of `sampling::top_p_filter` without adding any FFI call when `logits` is already `FLOAT32`, updates the CUDA overlay documentation, and adds regression tests verified to fail before the fix and pass after the fix.

## 1. Root Cause Analysis

### 1.1 `multivariate_normal` `bfloat16` output demotion (`random.cpp:311`)

In upstream `mlx/random.cpp`, `multivariate_normal(mean, cov, shape, dtype, key, s)` validates `dtype == float32` at entry, casts `cov` to `float32`, runs `linalg::svd(covariance, true, stream)`, draws `standard_normal` in `float32`, forms `scaled_out` (`output_shape`, `float32`), and returns `add(mean, scaled_out, stream)`.

Under standard MLX promotion rules (Metal, ROCm, CPU-only builds), `promote_types(bfloat16, float32)` evaluates to `float32`, so `mean` is upcast to `float32` and the returned sample array has `dtype == float32` with full 24-bit mantissa precision. On `--features cuda` builds, however, `src/lib/mlxcelverse/overlays/cuda/mlx/dtype.cpp` overrides `promote_types(bfloat16, float32) -> bfloat16` in the shared MLX type table across both GPU and CPU streams. Passing `mean` in `bfloat16` therefore caused `add(mean, scaled_out, stream)` to downcast the entire `[..., N]` `scaled_out` tensor from `float32` to `bfloat16`, truncating the low 16 mantissa bits and violating `multivariate_normal`'s `float32` return contract.

Furthermore, downcasting `scaled_out` (`output_shape`, e.g., `[S, N]`) touched `S` times more elements than upcasting `mean` (`[N]`) when broadcasting across `S` samples, and prevented `Add` from reusing `scaled_out`'s buffer in place because input (`float32`, 4 bytes) and output (`bfloat16`, 2 bytes) item sizes differed.

### 1.2 Rust reference `sampling::top_p_filter` tail swamping and return dtype (`sampling.rs:3050`)

`pub(crate) fn top_p_filter` in `src/lib/mlxcel-core/src/sampling.rs` serves as the unit-test reference implementation for nucleus filtering (`top_p_filter_*` and `fused_sample_probs_equals_softmax_of_filtered_over_t`). Previously it called `ffi::softmax(logits, -1)` and `ffi::cumsum(&sorted_probs, -1, false, true)` without normalizing input dtype:

- On a `[1, 262144]` `bfloat16` row where token `0` has probability `~0.601` and tokens `1..=10000` each have probability `~2.73e-5` (true cumulative probability `~0.874 > top_p = 0.80`), sequential `bfloat16` addition evaluates `0.601 + 2.73e-5 == 0.601` because `1 ULP` at `0.5` is `2^-8 = 3.91e-3`, leaving all `252,143` tail tokens past index `10,000` unmasked on sequential CPU scans.
- In step 6 (`ffi::where_cond(&mask, &sorted_logits, &neg_inf)` with `neg_inf` in `FLOAT32`), Metal and CPU builds returned `FLOAT32` whereas CUDA builds returned `BFLOAT16`, breaking backend dtype parity and `to_vec_f32` test consumers.

### 1.3 Why in-chain `min_p` and `XTC` `softmax` stay in `logit_dtype`

An explicit check of `mlx/backend/cuda/softmax.cu` and `mlx/backend/cuda/arg_reduce.cu` confirmed that in-chain `min_p` (`mlx_cxx_bridge.cpp:6289`) and `XTC` (`mlx_cxx_bridge.cpp:6103`) do not require `float32` tensor materialization:

- `softmax.cu` accumulates row `max` and `normalizer = sum(exp(x_i - max))` in `float32` (`AccT = float`) and only stores each final scalar probability in `bfloat16` (8-bit exponent, minimum normal `~1.17e-38`).
- Neither filter runs `cumsum`; `min_p` performs elementwise `probs >= max_prob * min_p`, and `XTC` masks non-candidates to `+inf_bf16` (`Limits<__nv_bfloat16>::max()`) before `ArgMin`.
- Keeping `probs` in `bfloat16` preserves #636's single-dtype decode chain and allows `Softmax::eval_gpu` (`softmax.cu:118`) to reuse donatable `[B, V]` buffers via `copy_shared_buffer` instead of allocating `1 MB/row` of `float32` scratch memory per step.

## 2. Changes Made

1. **`src/lib/mlxcelverse/overlays/cuda/mlx/random.cpp` and `src/lib/mlxcelverse/overlays/cuda/README.md`**:
   - Upcast `bfloat16` `mean` in `multivariate_normal` before the final addition (`auto mean_up = mean.dtype() == bfloat16 ? astype(mean, float32, stream) : mean; return add(mean_up, scaled_out, stream);`).
   - Updated the overlay provenance comment block and `overlays/cuda/README.md` to document `multivariate_normal` as the fifth patched distribution site alongside `bernoulli`, `categorical_inverse_cdf`, batched `categorical_impl`, and `laplace` (issues #2958, #2971).
2. **`src/lib/mlxcel-core/src/sampling.rs` (`top_p_filter`)**:
   - Upcast non-`FLOAT32` inputs to `dtype::FLOAT32` at entry while borrowing `logits` directly without an extra FFI call when `ffi::array_dtype(logits) == dtype::FLOAT32`.
   - Added unit test `top_p_filter_bf16_large_vocab_masks_tail_and_returns_f32` verifying `[1, 262144]` `BFLOAT16` nucleus filtering and `FLOAT32` return dtype on both the default device and explicit CPU stream.
3. **`tests/cuda_bf16_categorical_last_vocab.rs`**:
   - Added CUDA integration test `random_multivariate_normal_bf16_mean_returns_f32_samples` exercising `mlxcel_core::random_multivariate_normal` with `BFLOAT16` `mean` and both `BFLOAT16` and `FLOAT32` `cov` matrices, asserting `dtype::FLOAT32` output, `[512, 2]` shape, empirical means near `[3.5, -2.25]`, and non-zero low-16 `float32` mantissa bits.

## 3. Verification

All verification gates were executed on NVIDIA GB10 (`sm_121`, `--features cuda`, GPU execution serialized under `/usr/local/bin/gpu-lock`):

- **Fail-before-fix check**:
  - `sampling::tests::top_p_filter_bf16_large_vocab_masks_tail_and_returns_f32`: failed before fix (`left: 12` [`BFLOAT16`] vs `right: 10` [`FLOAT32`] on CUDA default stream, and `252,143` unmasked tail tokens on sequential CPU `cumsum`).
  - `random_multivariate_normal_bf16_mean_returns_f32_samples`: failed before fix (`left: 12` [`BFLOAT16`] vs `right: 10` [`FLOAT32`]).
- **Pass-after-fix check**:
  - `cargo test --profile test-fast --features cuda -p mlxcel-core --lib sampling::tests::top_p_filter -- --test-threads=1`: `4 passed; 0 failed` (`0.39s`).
  - `cargo test --profile test-fast --features cuda --test cuda_bf16_categorical_last_vocab -- --test-threads=1`: `5 passed; 0 failed` (`3.86s`).
- **Static & provenance checks**:
  - `CONTRIBUTING.md` MLX overlay provenance check against `9f806b3606eaa9f7c807238f6bd465185cd94d6d`: `0` drift / `0` missing headers.
  - `make verify-fmt verify-versions verify-kernel-dtype-keys verify-env-var-docs verify-kernel-port-dispatch verify-rocm-overlay`: passed.
  - `cargo clippy --workspace --all-targets --features cuda -- -D warnings`: passed with zero warnings.
