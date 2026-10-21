# PR #2325: Stop casting comparison inputs to bool in Metal compiled kernels

**Date**: 2026-10-16
**Status**: Implemented; Metal revert check and fix both verified on the self-hosted macOS runner, real checkpoint verified on CUDA (GB10) before merge; whether the overlay can be dropped is open (see "Not verified and follow-up")
**Risk**: Low (one bool-output branch in a Metal-only overlay; non-bool ops emit unchanged source; the bridge entry is test-only)

## Summary

Issue #1792 fixes wrong results from compiled (fused) comparisons on Metal.

1. In the mlxcel overlay of MLX `backend/metal/compiled.cpp`, a bool-output op (a comparison, `isnan`, `isinf`) now casts mismatched inputs to the promoted input type instead of to `bool`. The overlay header records upstream 243bdf09 as its base.
2. A test-only bridge entry, `compile_dtype_probe_for_test` (`src/lib/mlxcel-core/cpp/mlx_cxx_compile_probe.cpp`), runs one of seven fixed graphs eagerly, compiled, or compiled shapeless.
3. `src/lib/mlxcel-core/src/compiled_dtype_cast_tests.rs` (5 tests) asserts that both compiled modes equal eager.

## Problem

The overlay casts every op input whose type differs from the op's output type to that type, added for mixed bf16/f32 arithmetic (1923da32). For a comparison the output type is `bool`, so float operands were cast to `bool` before comparing. A compiled `less(0.2, 0.7)` became `Less(true, true)`, which is false, and `isnan` passed NaN through. Compiled `where(a < b, a, b)` returned `[0.7, 0.3, 0.2, 0, 3]` instead of `[0.2, -0.5, 0.2, 0, 3]`, and the min-p filter (#1379) returned its input unfiltered. CUDA has no such overlay and is unaffected.

## Technical decisions

- Fold `promote_types` over the op's inputs, starting from `bool_`, only when the output is `bool`. Non-bool ops keep the old cast, so the source emitted for the bf16/f32 arithmetic the cast was added for does not change.
- Do not change the non-bool path. Casting to the promoted input type for every op would also fix `Real`/`Imag`, but it alters the path the original cast protects, and this host cannot test Metal.
- Test through a C++ bridge probe with fixed graphs rather than Rust-side graph building, so eager, compiled and shapeless-compiled run the same graph and differences are attributable to compilation.
- Anchor the eager same-dtype f32 results to host arithmetic, so a wrong eager result cannot make the comparison pass.
- Correct the stale "Synced to 81ba1c6a" header. Diffed against upstream 243bdf09, the overlay differs only in the header and the cast block.

## Changes

- `src/lib/mlxcelverse/overlays/metal/mlx/backend/metal/compiled.cpp`: the bool-output branch of the cast (about ten lines) and the header; `overlays/metal/README.md` describes the rule.
- `src/lib/mlxcel-core/cpp/mlx_cxx_compile_probe.cpp` (new), plus `mlx_cxx_bridge.cpp`, `mlx_cxx_bridge.h`, `build.rs` and `lib.rs` wiring for `compile_dtype_probe_for_test`.
- `src/lib/mlxcel-core/src/compiled_dtype_cast_tests.rs` (new): the six comparisons on f32; mixed f16/f32, f32/f16 and bf16/f32 comparisons; `where(isnan(x), 7, x)` and `where(isinf(x), 9, x)` in f32/f16/bf16; the min-p graph; the mixed bf16/f32 arithmetic chain.

## Behavior change

Compiled comparisons, `isnan`/`isinf` and graphs that use them (min-p) now give the same results as eager on Metal. Nothing changes on CUDA, and no API is added outside the test bridge.

## Verification

- Revert check on Metal via Nightly verify, on a test-only branch without the fix: run 38780888405 failed exactly 4 tests (same-dtype f32 comparisons, mixed f16/f32 comparisons, isnan/isinf, min-p) with 2093 passed. The mixed bf16/f32 arithmetic test passed, as expected.
- With the fix: run 38780921019 passed all 5 tests and the full suite.
- On CUDA the 5 tests pass with or without the fix, because CUDA does not use the overlay. The revert check therefore has to run on Metal.
- Real checkpoint `qwen3-0.6b-4bit`, `generate` on CUDA with this branch: coherent output ending in "The capital of France is Paris."
- fmt, clippy and the contract checks pass.

## Not verified and follow-up

- Open question: whether the overlay can be dropped entirely at 243bdf09. The cast exists because macOS 26.4 failed template deduction on `Divide(bfloat16_t, float)` in a fused kernel. If upstream now promotes binary inputs before building the primitive, a fused op never sees mixed dtypes. Settling it needs a Metal build without the overlay and a model from 1923da32 (gemma3-4b, nemotron-h-30b).
- Follow-up candidate: `Real` and `Imag` (complex64 in, f32 out) still get inputs cast to float on Metal, so a compiled `imag()` would return 0. No compile site uses it.
- The Metal fix was verified on the macOS runner only through the test suite, not with a real checkpoint on Metal.
