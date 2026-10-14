# Technical Report: PR #2132 - Guard the TF32 test pin and record its decode impact (Issue #1065)

**Date**: 2026-10-06

**Status**: Implemented and validated on GB10, awaiting merge.

**Language**: Rust (tests and doc comments), Markdown

**Risk**: Low (no runtime code change)

## Summary

Issue #1065 reported four f32 numeric-agreement tests failing on M5 Max at fp16-epsilon scale and framed it as an Apple generation 17 defect. Later comments established the cause as MLX's default `MLX_ENABLE_TF32=1` and showed it reproduces on CUDA. #1260 pinned the variable to 0 in both lib test binaries. This PR adds a sentinel test for that pin, corrects the doc comments that called the problem Apple-only, makes the last two chunked-SDPA assertions print their divergence, and documents the variable with a measured decode A/B.

## Mechanism

MLX selects reduced-precision f32 GEMM on two backends: cuBLAS `CUBLAS_COMPUTE_32F_FAST_TF32` on CUDA (`gemms/cublas_gemm.cpp`) and the NAX kernel on Apple GPU generation 17 (`is_nax_available() && (enable_tf32() || dtype != float32)`). On CUDA, `can_use_gemv` sends `M == 1` against a transposed weight (and `N == 1`) to a gemv kernel that ignores the setting, and SDPA has its own kernels. That is why the four `mlxcel-core` tests #1065 named pass on CUDA even under `MLX_ENABLE_TF32=1`, while three root-lib tests (bailing GLA, deepseek_v2 MLA, florence2) fail there.

## Sentinel

`tf32_pin_tests::f32_gemm_runs_at_full_precision_in_the_test_process` multiplies 16x512 by 512x16 f32 matrices (a real GEMM on both backends) and compares with an f64 host sum at a 1e-4 absolute bound. GB10: passes by default and with `=0`; fails 3/3 with `=1` at 5.84e-3.

## Decode A/B (GB10, greedy, 256 tokens)

| checkpoint | weights | result |
|---|---|---|
| plamo-2-1b | f32 | 1 of 5 prompts diverges after about 40 words; each arm deterministic over 5 runs; 38 tok/s both |
| llama-3.1-8b-bf16 | f16 | identical, 3 prompts x 2 runs |
| qwen2.5-0.5b-bf16 | bf16 | run-to-run variation in both arms, not attributable |
| qwen3-0.6b | 4-bit | run-to-run variation in both arms, not attributable |

Decision: the production default stays on MLX's value. TF32 only reaches dense f32 GEMMs, which in practice means f32 checkpoints (and on CUDA mostly prefill-sized ones, since unbiased single-row matmuls go to gemv; read from MLX source, not measured); the docs give `MLX_ENABLE_TF32=0` as the switch for f32-exact output.

## Not verified

No Metal hardware on this host: M5 and M1 Ultra reruns, the sentinel's red arm on M5, and M5-vs-M1 decode quality.
