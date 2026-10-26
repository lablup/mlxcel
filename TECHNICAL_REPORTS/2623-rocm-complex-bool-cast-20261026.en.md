# PR #2623: Match MLX in eager ROCm complex64 and bool casts

**Date**: 2026-10-26
**Status**: Completed
**Languages**: C++ (HIP), Rust
**Risk Level**: Low

## Executive Summary

The eager ROCm cast kernels converted complex64 to bool with `&&` and bool to complex64 as `1+1i`. MLX CUDA and CPU use `real != 0 || imag != 0` and `1+0i`, and the ROCm fused path follows them since LOCAL_FIXES item 59 (#2613). The two `CastOp` specializations in the ROCm overlay now match, so eager and compiled results agree (issue #2616, LOCAL_FIXES item 60).

## 1. Problem Statement

`astype(z, bool)` returned false for `1+0i` and `0+1i` on the eager ROCm path, and `astype(true, complex64)` returned `1+1i`. After item 59 made the fused kernels' `complex64_t` follow CUDA, the same graph produced different answers eagerly and under `mx::compile` on ROCm.

## 3. Technical Decisions

The reference is MLX CUDA `mlx/backend/cuda/device/cast_op.cuh:22-39` and the CPU `complex64_t::operator bool` (`mlx/types/complex.h:50`). Both define truthiness as either part nonzero and the bool to complex conversion as a real 1, so the ROCm overlay adopts those semantics rather than keeping its own.

The tests feed the cast source as an eagerly built input (probe graphs 12 and 13 in `mlx_cxx_compile_probe.cpp`), so eager mode runs the standalone cast kernel in `copy.hpp` instead of a fused graph. Graph 13 packs both complex parts into one float as `real + 10 * imag`, which makes the old `1+1i` show as 11 and the correct `1+0i` as 1.

## 7. Change Summary

- `overlays/rocm/mlx/backend/rocm/copy/copy.hpp`: `||` for complex64 to bool; `make_hipFloatComplex(1.0f, 0.0f)` for true.
- `overlays/rocm/LOCAL_FIXES.md`: item 60.
- `mlxcel-core/cpp/mlx_cxx_compile_probe.cpp`: probe graphs 12 and 13 with eager inputs.
- `mlxcel-core/src/compiled_dtype_cast_tests.rs`: `eager_and_compiled_complex64_to_bool_follow_mlx` (1+0i, 0+1i, 0+0i give true, true, false) and `eager_and_compiled_bool_to_complex64_follow_mlx` (true gives 1+0i), each also requiring compiled to equal eager.

## Validation

On gfx1151 before the fix, both new tests failed: eager complex to bool gave `[0, 0, 0]` against `[1, 1, 0]`, and eager bool to complex gave 11 where compiled gave 1. After the fix and an overlay rebuild, `compiled_dtype_cast_tests` passed 10 of 10. The contract checks (`verify-llama-compat`, `verify-env-var-docs`, `verify-kernel-port-dispatch`, `verify-kernel-dtype-keys`, `verify-rocm-overlay`) and clippy on `mlxcel-core` lib and tests are clean. Only the ROCm overlay changed; CUDA, Metal and CPU behavior is untouched.
