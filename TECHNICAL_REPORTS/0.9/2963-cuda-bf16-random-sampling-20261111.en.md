# PR #2963: Preserve float32 precision in CUDA bfloat16 random sampling

- **Date**: 2026-11-11
- **Status**: Completed (closes #2958)

## Executive Summary

On NVIDIA GB10 (`sm_121`, CUDA 13), `models/gemma-4-e2b-it-4bit` intermittently emitted the literal token `<unused6226>` (token id `262143`, the final index `vocab_size - 1` of its `262144`-entry vocabulary) roughly once every few hundred sampled tokens while surrounding text remained fluent. An initial small-sample A/B suggested a quantized matrix-vector (`qmv`) row-blocking defect, which contradicted `qmv.cu` selecting `default_rows_per_warp = 1` outside Volta (`sm_70`).

Direct GPU instrumentation, step-73 logit capture against an `f32` CPU reference, and a `5,540`-token-per-arm A/B settled both questions:

1. **Quantized matmul is innocent**: On `sm_121`, both unset and `MLXCEL_QMV_ROWS_PER_WARP=1` launch the exact same stock `qmv_kernel<8,16,64,1,0,cutlass::bfloat16_t>`, and all five arms (`unset`, `1`, `2`, `3`, `4`) produce `65/65` byte-identical outputs across `65` seeds with the same `12` `<unused6226>` hits (`0.217%`). On the captured emission step (`--seed 13`, step `73`), the LM-head `quantized_matmul` output matches the CPU `f32` `dequantize` + `matmul` reference within `rel_err = 0.004860` (`raw_gpu[262143] = -21.875000` vs `raw_ref[262143] = -21.859125`), and `top_k = 64, top_p = 0.95` masks `filtered[262143]` to `-inf`.
2. **Root cause in `mlx/random.cpp` under CUDA `bfloat16` type promotion**: On CUDA builds, `overlays/cuda/mlx/dtype.cpp` overrides `promote_types(bfloat16, float32) -> bfloat16` and `promote_types(uint32, bfloat16) -> bfloat16` to avoid `copy_v<bfloat16, float>` casts in the transformer graph. When `vocab = 262144` exceeds `REJECTION_JOINT_VOCAB_MAX = 32768`, `fused_sample` routes Gemma 4 (`top_k = 64, top_p = 0.95`) to `fused_sample_argpartition_chain` -> `mlx::core::random::categorical(filtered, -1)` -> `categorical_inverse_cdf`, which explains why `MLXCEL_SAMPLING_REJECTION=0` still reproduced the bug. Inside `categorical_inverse_cdf`, `promote_types(logits.dtype(), float32)` evaluated to `bfloat16`, running `exp`, exclusive `cumsum`, `sum`, and `u = uniform_f32 * sum(w)` in `bfloat16`. Whenever `uniform_f32 >= 1 - 2^-8` (`1/256 = 0.391%`, modulated to `~0.21%` by 1-ULP scan-vs-reduce rounding), `u` rounded up to `sum(w) == cdf[262143]`, `searchsorted(cdf, u, "right")` returned `262144`, and `subtract(..., 1u)` emitted `262143` despite `w[262143] == 0`.

Adding a CUDA-only `overlays/cuda/mlx/random.cpp` overlay (preserving `float32` precision in `categorical_inverse_cdf`, batched `categorical_impl` Gumbel-max, `bernoulli`, and `laplace`) and upcasting `bfloat16` `softmax` + `cumsum` in `top_p_filter` eliminates the defect across `mlxcel generate`, `mlxcel run`, and `mlxcel-server` (`0` hits in `6,750` seeded E2B tokens, `0` hits in `6,000` unseeded E2B tokens, `0` hits in `2,898` seeded E4B tokens) with flat decode throughput (`80.91` vs `80.67 tok/s`, `-0.29%`).

## 1. Problem Statement

Issue #2958 reported intermittent `<unused6226>` (`id = 262143 = vocab_size - 1`) emissions during sampled CUDA generation on `models/gemma-4-e2b-it-4bit` (`sm_121` GB10):

- Observed in both `mlxcel generate` and interactive `mlxcel run` while neighboring tokens stayed coherent.
- Reproduced with `MLXCEL_SAMPLING_REJECTION=0` and with a fresh `MLX_PTX_CACHE_DIR`, did not reproduce at `--temp 0`, and did not reproduce on Apple Silicon Metal.
- An early 5-run sample (`~410` tokens) with `MLXCEL_QMV_ROWS_PER_WARP=1` observed `0` occurrences against `4` in `~510` default tokens, contradicting `qmv.cu:1223` (`default_rows_per_warp = volta ? 2 : 1`).

The issue required resolving the kernel contradiction by runtime instrumentation and a multi-thousand-token A/B, capturing raw logits on a failing decode step against a CPU `f32` `dequantize` + `matmul` reference, checking other models and dtypes, fixing the shared runtime path without an opt-in switch, and adding both `tests/cuda_qmv_reference_parity.rs` and a regression test that fails on `main` and passes with the fix.

## 2. Root Cause Analysis

### 2.1 Why `MLXCEL_SAMPLING_REJECTION=0` did not rule out the sampler

In `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp`, `REJECTION_JOINT_VOCAB_MAX` is `32768` on non-ROCm builds. When both `top_k > 0` and `top_p < 1.0` are active and `vocab > 32768`, `sampling_rejection_routes` returns `false`. Because Gemma 4 uses `vocab_size = 262144`, `top_k = 64`, `top_p = 0.95`, and `bfloat16` activations, both default execution and `MLXCEL_SAMPLING_REJECTION=0` took the exact same fallback path: `fused_sample_argpartition_chain` -> `fused_sample_filter_logits` -> `mlx::core::random::categorical(filtered, -1)`.

On Metal, `single_dtype_scalars` is `false`, so scalar `-inf` masking promotes `filtered` to `float32` before `categorical`. On CUDA, `single_dtype_scalars` is `true` (issue #636), keeping `filtered` in `bfloat16`.

### 2.2 `bfloat16` demotion in `mlx/random.cpp` and `top_p_filter`

`src/lib/mlxcelverse/overlays/cuda/mlx/dtype.cpp` patches MLX's promotion table so `promote_types(bfloat16, float32) -> bfloat16` and `promote_types(uint32, bfloat16) -> bfloat16`. Four distribution routines in upstream `mlx/random.cpp` assume promotion with `float32` yields `float32`:

1. **`categorical_inverse_cdf` (single-row decode, `logits.size() == n`)**:
   ```cpp
   auto dtype = promote_types(logits.dtype(), float32); // evaluated to bfloat16 on CUDA!
   auto x = astype(logits, dtype, s);
   ...
   auto cdf = cumsum(w, 0, /* reverse = */ false, /* inclusive = */ false, s);
   auto u = multiply(uniform(shape, float32, key, s), sum(w, s), s);
   return subtract(searchsorted(cdf, u, "right", s), array(1u, uint32), s);
   ```
   With `dtype == bfloat16`, `w`, exclusive `cdf`, `sum(w)`, and `u` all run in `bfloat16`. Masked tail entries have `w[i] = 0`, so `cdf` plateaus at `cdf[vocab - 1]`. Whenever `uniform_f32` lands in `[1 - 2^-8, 1)` (`u` rounds up to `sum(w)`) or `bfloat16` scan rounding leaves `cdf[vocab - 1]` 1 ULP below `sum(w)`, `u >= cdf[vocab - 1]` holds, `searchsorted(..., "right")` (`upper_bound`) finds no element strictly greater than `u` and returns `vocab`, and `subtract(..., 1u)` returns `vocab - 1` (`262143`).
2. **`categorical_impl` (batched Gumbel-max decode, `B >= 2`)**: `add(g /*float32*/, logits /*bfloat16*/)` demotes `float32` Gumbel noise to `bfloat16`, collapsing fine logit differences and shift-invariance at large logit magnitudes into index-0 `argmax` ties.
3. **`bernoulli`**: `multiply(p_bf16, upper_f32)` overflows/rounds `upper` (`2^32`) and `less(bits_u32, bf16)` demotes `uint32` random bits to `bfloat16`, returning `false` ~`1/256` of draws even when `p == 1.0`.
4. **`laplace`**: `multiply(array(-1.0f, bfloat16), abs(samples_f32))` rounds draws near `1.0` to `-1.0_bf16`, causing `log1p(-1.0_bf16) = -inf`.

In addition, `top_p_filter` (`src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp`) previously ran `softmax` and inclusive `cumsum` in `bfloat16` when `single_dtype_scalars` was true. Once cumulative probability exceeds `0.5` (`bfloat16` ULP = `2^-8 = 3.91e-3`), tail probabilities below `2^-9` (`1.95e-3`) swamp to zero during `cumsum`, preventing `top_p`-only (`top_k = 0`) scans on large vocabularies from reaching the cutoff threshold.

## 3. Changes Made

1. **`src/lib/mlxcelverse/overlays/cuda/mlx/random.cpp` (new CUDA-only core overlay)**:
   - Derived from pinned upstream MLX commit `9f806b3606eaa9f7c807238f6bd465185cd94d6d` (`mlx/random.cpp`) with the required provenance header.
   - Preserves `float32` working precision when inputs or target dtypes are `bfloat16` across `categorical_inverse_cdf`, `categorical_impl`, `bernoulli`, and `laplace`, leaving `float16`, `float32`, and integer paths untouched.
2. **`src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp` (`top_p_filter`)**:
   - Upcasts `bfloat16` inputs to `float32` for `softmax` + `cumsum` when `single_dtype_scalars` is true, while keeping `sorted_logits` and `fill_scalar` in `x.dtype()` and leaving Metal unchanged.
3. **`src/lib/mlxcelverse/CMakeLists.txt` and `src/lib/mlxcelverse/overlays/cuda/README.md`**:
   - Added `CONFIGURE_DEPENDS` to the `overlays/cuda/mlx/*.cpp` glob so newly added or removed core overlays trigger automatic CMake reconfiguration.
   - Added automatic `git checkout --` restoration in `_deps/mlx-src` for any retired or inactive top-level `mlx/*.cpp` overlay on non-ROCm builds.
   - Updated `overlays/cuda/README.md` core file count (`3` -> `4`) and aligned the backend file count (`30` -> `33`) with the enumerated list.
4. **New CUDA integration tests**:
   - `tests/cuda_qmv_reference_parity.rs`: Tests 4-bit `M = 1` `quantized_matmul` in `[1, K]` and `[1, 1, K]` layouts across `N in {262144, 262147}`, `K in {512, 576}`, `float16` and `bfloat16`, checking all rows (`rel < 2e-2`) and explicitly checking the final 64 rows for accuracy and finiteness across fresh subprocesses with `MLXCEL_QMV_ROWS_PER_WARP` unset, `2`, and `4`.
   - `tests/cuda_bf16_categorical_last_vocab.rs`: Deterministically tests `fused_sample`, `fused_sample_categorical`, single-row `random_categorical`, batched `random_categorical` Gumbel-max, and `top_p`-only `cumsum` swamping at `vocab = 262144` in `bfloat16` (`0/3` passed on pre-fix `main`, `4/4` pass post-fix).

## 4. Empirical Measurements (GB10 `sm_121`)

### 4.1 AC 1: Runtime LM-head kernel instrumentation

Instrumented `QuantizedMatmul::eval_gpu` and `cu::qmv` during `mlxcel generate -m models/gemma-4-e2b-it-4bit -p "하이" -n 200 --show-reasoning --seed 13` (`M = 1, N = 262144, K = 1536, B = 1`, `201` LM-head calls):

| Arm | Dispatch path | Launched kernel | `rows_per_warp` | Grid / Block | `stock_qmv` / `rowblock` / `multirow` |
|-----|---------------|-----------------|-----------------|--------------|---------------------------------------|
| `unset` | `qmv` (`M*B=1 < 8`) | `qmv_kernel<8,16,64,1,0,cutlass::bfloat16_t>` | `1` | `(32768,1,1)` / `(32,8,1)` | `201` / `0` / `0` |
| `MLXCEL_QMV_ROWS_PER_WARP=1` | `qmv` | `qmv_kernel<8,16,64,1,0,cutlass::bfloat16_t>` | `1` | `(32768,1,1)` / `(32,8,1)` | `201` / `0` / `0` |
| `MLXCEL_QMV_ROWS_PER_WARP=2` | `qmv` | `qmv_rowblock_kernel` | `2` | `(16384,1,1)` / `(32,8,1)` | `0` / `201` / `0` |

### 4.2 AC 2: Failing step logit capture vs CPU `f32` reference

Captured on pre-fix `models/gemma-4-e2b-it-4bit` `--seed 13` at decode step `73` (where `262143` `<unused6226>` was sampled):

- **Pre-softcap LM-head `quantized_matmul` vs CPU `f32` `dequantize` + `matmul`**: `max_abs_diff = 0.160517` (index `172211`), `max_abs_ref = 33.026920`, `rel_err = 0.004860`; top-1 matches (`236764`, GPU `33.000000` vs CPU `33.026920`); row `262143` is strongly negative (`raw_gpu[262143] = -21.875000`, `raw_ref[262143] = -21.859125`, `abs_diff = 0.015875`).
- **Post-softcap and filter**: `max_abs_diff = 0.195107`, `rel_err = 0.008121`, `gpu[262143] = -18.750000` vs `ref[262143] = -18.666931`; `2` nucleus tokens remain and `filtered[262143] = -inf`.
- **`categorical_inverse_cdf` state**: `w[262143] = 0.0`, `cdf_bf16[262143] = 1.078125 (0x3f8a)`, `sum_bf16 = 1.078125 (0x3f8a)`, `uniform_f32 = 0.997557521 (0x3f7f5fee)`, `u_bf16 = 1.078125 >= cdf_bf16[262143] -> sampled 262143`; in `float32`, `cdf_f32[262143] = 1.082085013` and `u_f32 = 1.079442024 < cdf_f32[262143] -> sampled 236764`.

### 4.3 AC 3: Pre-fix QMV kernel A/B (`5,540` tokens per arm)

| Pre-fix arm | Seeds / Tokens | `<unused6226>` hits (rate) | Hit seeds | Byte-identical vs `unset` | Median decode |
|-------------|----------------|----------------------------|-----------|---------------------------|---------------|
| `unset` | `65` / `5,540` | `12` (`0.217%`) | `2(x2),7,13,25,27,29,35,37,51,62,65` | baseline | `81.09 tok/s` |
| `MLXCEL_QMV_ROWS_PER_WARP=1` | `65` / `5,540` | `12` (`0.217%`) | identical | `65 / 65` | `81.28 tok/s` |
| `MLXCEL_QMV_ROWS_PER_WARP=2` | `65` / `5,540` | `12` (`0.217%`) | identical | `65 / 65` | `80.87 tok/s` |
| `MLXCEL_QMV_ROWS_PER_WARP=3` | `65` / `5,540` | `12` (`0.217%`) | identical | `65 / 65` | `80.65 tok/s` |
| `MLXCEL_QMV_ROWS_PER_WARP=4` | `65` / `5,540` | `12` (`0.217%`) | identical | `65 / 65` | `80.77 tok/s` |

### 4.4 AC 4 & AC 8: Cross-model impact and post-fix validation

- **Other Gemma 4 size (`models/gemma-4-e4b-it-4bit`, `BF16`, `vocab = 262144`)**: `6` hits in `2,847` pre-fix tokens (`30` seeds, `0.211%`) -> **`0` hits in `2,898` post-fix tokens (`30` seeds)**.
- **Non-Gemma `BF16` vs `F16` 4-bit models (`vocab = 151936`, `--top-k 64 --top-p 0.95 --temp 0.8`, `1,608` tokens across `8` seeds)**: `models/qwen3-4b-4bit` (`BF16`) sampled `vocab - 1 = 151935` `3` times (`0.187%`, silent in console output because Qwen 3's tokenizer maps `0..=151668` and leaves `151669..=151935` unmapped), whereas `models/qwen2.5-1.5b-instruct-4bit` (`F16`) sampled `151935` `0` times (`promote_types(float16, float32) == float32`).
- **Post-fix `models/gemma-4-e2b-it-4bit`**: **`0` `<unused6226>` hits across `6,750` seeded tokens (`75` seeds `1..=75`)** plus **`0` hits across `30 x 200` unseeded reproduction runs (`~6,000` tokens)**.
- **Idle-host interleaved decode throughput (`seeds 1, 3, 6, 8, 9` under `gpu-lock`)**: pre-fix median `80.91 tok/s` (mean `80.85 tok/s`) vs post-fix median `80.67 tok/s` (mean `80.66 tok/s`, `-0.29%`).

## 5. Verification Summary

- `make verify-fmt`, `make verify-versions`, `make verify-kernel-dtype-keys`, `make verify-env-var-docs`, `make verify-kernel-port-dispatch`, `make verify-rocm-overlay`, `make verify-python-tooling`: all passed.
- `CONTRIBUTING.md` MLX overlay provenance and drift check (`PIN = 9f806b3606eaa9f7c807238f6bd465185cd94d6d`): `0` drift across all CPU/Metal/CUDA overlays.
- `cargo clippy --workspace --all-targets --features cuda -- -D warnings`: clean (`0` warnings).
- `cargo test --profile test-fast --features cuda --test cuda_bf16_categorical_last_vocab -- --test-threads=1`: `4` passed (`0/3` passed on pre-fix `main`).
- `cargo test --profile test-fast --features cuda --test cuda_qmv_reference_parity -- --test-threads=1` across `unset`, `MLXCEL_QMV_ROWS_PER_WARP=2`, and `MLXCEL_QMV_ROWS_PER_WARP=4`: `2` passed on each arm.
- `make verify-test-cuda-qmm-determinism`: `5` passed.

## 6. Change Summary

| Item | Value |
|------|-------|
| Files changed | 6 |
| Lines added | +1190 |
| Lines deleted | -10 |
| Commits | 2 (before report commit) |

Related: #2958 (issue), #2963 (PR), #636 (`single_dtype_scalars` CUDA scalar typing).
