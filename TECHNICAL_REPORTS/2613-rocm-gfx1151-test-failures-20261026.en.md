# PR #2613: Clear the nine gfx1151 test failures in verify-rocm

**Date**: 2026-10-26
**Status**: Implemented and verified locally on gfx1151 (draft PR, not yet merged)
**Languages**: C++ (HIP JIT preamble), Rust, Markdown
**Risk Level**: Low to Medium (the preamble change touches every fused ROCm kernel's source text; real-typed code paths are unchanged)

## Executive Summary

PR #2613 (issue #2608) clears the nine `make verify-rocm` failures on gfx1151. All nine pass on macOS, so every failure is ROCm-only. They fall into three unrelated causes, and each got a different kind of fix:

- **A real backend bug, fixed.** Fused (compiled) kernels on ROCm could not handle complex64 at all: the hiprtc preamble never defined the `complex64_t` type the shared codegen emits. The preamble now defines it, with complex64 overloads for the arithmetic and part-extraction ops, and makes float-only ops refuse complex64 at build time instead of computing on the real part. Two `mlxcel-core` tests.
- **A known backend gap, gated.** ROCm has no NVFP4 kernel and refuses MLX-native NVFP4 planes at load by design (#1806). Six `mlxcel` tests loaded such planes unconditionally. A new test helper, `expect_native_nvfp4_load`, asserts the refusal on ROCm, prints a skip naming the gap, and lets the backend-independent parts of each test keep running.
- **A test with a reference on its tolerance bound, fixed.** The PhiMoE fused-expert test compared the fused kernel against the bf16 graph path with a 2% bound. The fixture amplifies bf16 rounding to about 2.2%, so the bf16 reference itself was the inaccurate side. The reference now runs in f32.

## 1. Problem Statement

Issue #2608 listed nine failures from a `make verify-rocm-held` run, also reproduced on unmodified main. The cause was undiagnosed, and the issue asked that each test either pass on ROCm or be gated by an explicit skip that names a confirmed backend gap, with the root cause of each recorded in the PR.

The first diagnostic step the issue asked for was to separate ROCm-only failures from general regressions. Hosted CI on main currently runs no test jobs, so the PR used the macOS Nightly verify run (39703647781 at 414ae122) instead: all nine pass there. No CUDA data point exists.

### 1.1 Group B: compiled `real` / `imag` of complex64

`compiled_dtype_cast_tests::compiled_{real,imag}_of_complex64_matches_eager` failed because the fused kernel never compiled. The shared codegen in `overlays/rocm/mlx/backend/rocm/compiled.cpp` names a complex64 value `complex64_t` (`dtype_to_hip_type`), writes constants as `static_cast<complex64_t>(...)`, and emits `static_cast` in both directions for `AsType`. The hiprtc preamble (`g_jit_includes`) declared no such type, and hiprtc cannot include `<hip/hip_complex.h>`. Every fused graph touching complex64 therefore failed with "unknown type name 'complex64_t'". The preamble's `Real`, `Imag` and `Conjugate` functors were stubs for real inputs (identity, zero, identity), so even with a type defined they would have been wrong for complex input.

### 1.2 Group A: NVFP4 loads

Six tests (three in `switch_layers_global_scale_tests`, one each in gemma-unified loading, Mistral 4 and PhiMoE sanitize) build an MLX-native NVFP4 `SwitchLinear` / `SwitchGLU` plane and load it. On ROCm, `GpuBackendKind::quant_mode_support(Nvfp4)` is `ConvertTo(Affine)`, only ModelOpt checkpoints are converted, and `validate_quantization_mode_runnable` refuses a native plane at load. The ROCm `GatherQMM` also has no input for a per-expert global scale. The failures are the designed refusal, not a regression.

### 1.3 Group A: PhiMoE fused expert kernel

`phimoe_tests::fused_expert_kernel_keeps_sparsemixer_scores` guards against a fused single-token expert kernel renormalizing PhiMoE's sparsemixer scores. With `MLXCEL_FUSED_MOE_PARITY_CHECK=1` on gfx1151 the PR measured: fused vs an all-f32 truth 0.17%, the `gather_qmm` graph chain vs truth 2.2%, fused vs graph 2.2%. The fixture cancels heavily in the down projection (|activation| about 176 against |output| about 12): rounding the activation alone to bf16 moves the output 0.7%, and the bf16 gate/up/activation chain moves it several percent. A single `quantized_matmul` matches f32 to bf16 rounding (0.16%). Neither kernel is wrong. The bf16 reference sat on the 2% bound, and ROCm's rounding pushed it over.

## 2. Change Summary

| Area | Change |
|---|---|
| hiprtc preamble (`overlays/rocm/mlx/backend/rocm/compiled.cpp`) | New `struct alignas(8) complex64_t { float x, y; }` with the `hipFloatComplex` layout, an explicit templated constructor from real types (real part, imaginary zero), an explicit templated conversion to real types (real part), and a non-template explicit `operator bool` that is true when either part is nonzero. Free `+ - * /`, unary `-`, `==`, `!=`. |
| Op functors | complex64 overloads on `Add`, `Subtract`, `Multiply`, `Divide`, `Square`, `Real` (float32), `Imag` (float32), `Conjugate`, `Abs` (`hypotf`, returned as float) and `NaNEqual` (the MLX CUDA / Metal rule). `Abs` moves out of `UNARY_FLOAT_OP` into its own struct to carry the overload. |
| Real-only guard | New `MLX_ROCM_REAL_ONLY(T)` macro, a `static_assert(!__is_same(T, complex64_t), ...)`. Placed in `UNARY_FLOAT_OP` (Exp, Log, Sqrt, Rsqrt and the rest), `Power`, `ArcTan2`, `Remainder`, the float branch of `FloorDivide`, `LogAddExp`, `Sigmoid`, `Sign`, `Reciprocal`. |
| `LOCAL_FIXES.md` | Item 59 records the fix as fork-side. It is a backend file, so `CORE_RESIDUAL.diff` is unchanged. |
| `switch_layers.rs` | New `#[cfg(test)] pub(crate) fn expect_native_nvfp4_load<T>(loaded, prefix, what) -> Option<T>`. Native NVFP4: unwrap or panic with `what`. Otherwise: panic if the load succeeded, assert the error names the plane prefix and `nvfp4`, print a skip naming #1806, return `None`. |
| Six NVFP4 tests | Each load goes through the helper. Tests that use the layer afterwards `return` (or `continue` in the gemma-unified loop) on `None`; tests whose load was the last step just call it. Sidecar stacking into `inv_global_scale` and the "left unstacked" refusal still run on every backend. |
| `vlm_gemma_unified_tests.rs` | Before the gated load, the test now reads `<prefix>.<NATIVE_GLOBAL_SCALE_LEAF>` directly from the weight map and checks the per-expert scale values, so the split's output is still verified on ROCm. |
| `phimoe_tests.rs` | The graph reference casts `x` to f32 and runs `moe_weighted_sum` in f32. The 2% bound is unchanged. |

## 3. Technical Decisions

### 3.1 Fix the preamble rather than disable complex64 fusion

The alternative was to keep complex64 graphs out of fusion on ROCm and let them run eagerly. That would have hidden the missing type behind a performance difference and left `Real` / `Imag` / `Conjugate` stubs in place to bite later. Defining the type in the preamble matches what the codegen already assumes, and the struct mirrors `hipFloatComplex` and `mlx::core::complex64_t` (two floats, 8-byte aligned), so device buffers are read and written without conversion.

### 3.2 Make every conversion explicit

Both the constructor from real types and the conversion to real types are `explicit`. That keeps `complex64_t` out of overload resolution for real operands, so a real-typed `Add{}(a, b)` cannot become ambiguous with the new `Add{}(complex64_t, complex64_t)` overload, and real-typed kernels compile exactly as before. The codegen's `AsType` uses `static_cast`, which does see explicit conversions, so casts still work.

### 3.3 `bool` and `NaNEqual` follow MLX CPU and CUDA, not the generic path

The templated `operator T()` returns the real part, which is MLX's rule for complex to real casts. Applied to `bool` it would have made a fused `astype(z, bool)` false for a purely imaginary value. A non-template `operator bool` wins over the template and checks both parts. Likewise, the float `NaNEqual` template would have compared real parts only through `static_cast<float>`, so a fused `array_equal(..., equal_nan=True)` on complex input would have ignored the imaginary part. Both were found during review: before the type existed they could not be reached, and after it existed they would have been silently wrong rather than failing to build.

### 3.4 Fail at build time for unsupported complex ops

Once `complex64_t` converts to `float`, every op template that computes through `static_cast<float>` would compile for complex64 and quietly operate on the real part. `MLX_ROCM_REAL_ONLY` turns that into a hiprtc compile error with a readable message. A loud failure for an uncovered op is the same behaviour users had before this PR (the graph failed to build), so the guard narrows the change to the ops that were actually implemented.

### 3.5 Gate the NVFP4 tests on the capability, not on `cfg(rocm)`

The helper asks `quant_mode_support(QuantMode::Nvfp4) == Native` instead of checking for a ROCm build. Other tests already branch the same way, and the condition stays correct if a backend gains or loses a native kernel. On a non-native backend it does not merely skip: it requires the load to fail, and with an error that names both the plane and the mode. If ROCm ever started loading NVFP4 planes silently (for example by converting them in a way that drops the global scale), the test would fail.

### 3.6 Keep the backend-independent checks running

A blanket `#[ignore]` on ROCm would have dropped the sidecar stacking checks, which are pure weight-map transformations and run everywhere. The `Option` return lets each test stop exactly where the loaded layer is first needed. The gemma-unified test goes further and checks the stored per-expert native scale on the weight map before the load, so the split logic it exists to protect is still covered on ROCm.

### 3.7 Change the PhiMoE reference, not the tolerance

Widening the bound past 2% would have weakened the test for every backend. The property under test is that the fused kernel does not renormalize scores, and a renormalizing kernel (score sum 1.5025) would be about 1.5x away from any reference. Moving the reference to f32 makes it closer to the truth than the fused kernel under test, which is the right direction for a reference, and keeps the 2% bound meaningful.

## 4. Validation

From the PR body, on gfx1151 with every GPU run under `scripts/rocm_gpu_guard.sh --hold`:

- `mlxcel-core` `compiled_dtype_cast_tests`: 7/7 (2 failed before). The broader `compiled` filter: 42/42 in `mlxcel-core`, 8 pass and 1 ignored in `mlxcel`, re-run after the review commit.
- `mlxcel` `models::switch_layers` 57 pass (1 ignored, needs a checkpoint), `models::phimoe` 14/14, `loading::vlm::gemma_unified` 9/9, `models::mistral4` 2/2.
- `cargo clippy --profile test-fast --features rocm -p mlxcel -p mlxcel-core --lib --tests -- -D warnings` and `cargo fmt --all -- --check`: clean.
- The `verify-rocm-overlay` steps: `rocm_overlay.py records` (59 entries), `rocm_overlay_test.sh`, `allocator_ownership_test.sh`, `rocm_runtime_dir_test.sh`, `rocm_overlay.py drift`: pass.
- The full `make verify-rocm-held` gate is run centrally before merge. Its result is not part of this report.

## 5. Known Gaps

- **Fused `abs` of complex64 probably still fails to build.** MLX's `abs` builds an `Abs` node whose output dtype is complex64 and then casts it to float32 (`ops.cpp`). The codegen declares each tape value with the node's dtype, so it emits `complex64_t tmp = Abs{}(tmp_z);`. The new `Abs` overload returns `float`, and copy-initialization cannot use the explicit `complex64_t(float)` constructor, so this line should not compile. MLX CUDA returns `T` (complex64 with zero imaginary part) from `Abs`. This is not a regression (the graph failed to build before too) and it fails loudly, but the `Abs` overload as written does not deliver what `LOCAL_FIXES.md` item 59 describes. Returning `complex64_t(hypotf(z.x, z.y), 0.0f)` would match CUDA. Not verified by running.
- **Only `real` and `imag` are tested.** The arithmetic, `Square`, `Conjugate`, `bool` cast and `NaNEqual` overloads have no compiled-vs-eager test. The `bool` and `NaNEqual` behaviour in particular was reasoned from the MLX CPU and CUDA sources, not exercised.
- **Complex comparisons and min/max are not defined.** `<`, `>`, `Minimum` and `Maximum` on complex64 will fail to compile with a generic error rather than the `MLX_ROCM_REAL_ONLY` message. MLX's ordering of complex values (lexicographic) is not implemented.
- **`Divide` uses the textbook formula.** No scaling against overflow or underflow, unlike `hypotf` in `Abs`. Large-magnitude denominators can overflow `b.x * b.x + b.y * b.y`. Whether the eager ROCm path scales its division was not checked, so fused and eager results may differ at the extremes.
- **No ROCm coverage of NVFP4 numerics.** On ROCm the gated tests stop at the refusal. The global-scale arithmetic is checked only on Metal and CUDA, which the PR did not re-run after the change. The underlying gap (no NVFP4 `GatherQMM` with a per-expert global scale on ROCm) remains, tracked under #1806.
- **The PhiMoE test was only re-run on ROCm.** The f32 reference should only make the test easier to pass on Metal and CUDA, but that was not confirmed.
- **JIT cache churn.** The preamble change alters the JIT cache key, so cached ROCm kernel modules rebuild once after upgrade. Expected and harmless, but visible as a slower first run.

## 6. Learning Points

- A shared codegen and a backend preamble form a contract that nothing checks at build time: the codegen can name a type that the preamble never defines, and the gap shows up only when a graph with that dtype is first compiled at runtime.
- Adding an implicit-looking conversion to a JIT preamble widens which templates compile. A `static_assert` on the new type in templates that would otherwise silently accept it keeps "compiles" from turning into "computes the wrong thing".
- In C++, a non-template conversion operator wins over a template one with the same target, which is a clean way to special-case one target type (`bool`) without touching the general rule.
- A test skip for a backend gap is stronger when it asserts the expected refusal instead of just returning early: it still fails if the backend's behaviour changes.
- When a parity test sits on its bound, measure each side against a higher-precision truth before widening anything. Here the reference, not the kernel under test, was the less accurate side.
