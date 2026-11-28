# Technical Report: PR #3136 - Absorb DeepSeek V2 decode by default where the fused kernel runs

**Date**: 2026-11-16
**Status**: Completed (CUDA and ROCm keep the decompressed default until #3133 measures them)
**Risk Level**: Medium (changes the default cache layout for every DeepSeek V2 user on Apple Silicon)

## Executive Summary

DeepSeek V2 now caches the MLA latent `(ckv, kpe)` and runs absorbed decode by default wherever the fused Metal absorbed-decode kernel from #3129 can run. Elsewhere it keeps the decompressed per-head K/V cache. `MLXCEL_MLA_ABSORBED` becomes a tri-state: unset follows the fused-kernel availability, `=1` forces absorbed and `=0` forces decompressed on any backend. Kimi-VL, which always absorbs, was validated end to end on the fused kernel: 0 decided disagreements, with decode 6.7 to 7.5% faster than the composed path.

## 1. Problem Statement

#3102 kept the decompressed default because absorbed decode was about 10% slower at short context. #3129 removed most of that cost with quantized folds and a fused Metal kernel, and re-measured both switch conditions as met. The switch itself was left for this change. The fused kernel is Metal-only, so the measured evidence covers only builds where it runs.

## 3. Technical Decisions

- **The default follows the kernel, not the platform.** `AbsorbedMode::resolve(fused_available)` maps Default to absorbed only when `mla::fused_decode_enabled()` holds: the Metal kernel port exists and `MLXCEL_MLA_FUSED_DECODE` is not off. Without the kernel, absorbed decode takes the composed path, which still carries the short-context cost #3102 measured. So CUDA and ROCm, and Metal with the kernel disabled, stay decompressed until #3133 measures them. A side effect is that `MLXCEL_MLA_FUSED_DECODE=0` alone now also gives decompressed V2 on Metal.
- **An unrecognized value means Default, not Off.** This follows the policy of `MLXCEL_MLA_FUSED_DECODE`: only a deliberate off spelling leaves the default layout.
- **Teacher-force the greedy parity test.** `tests/deepseek_v2_greedy_parity.rs` compared 20 free-running greedy ids with the mlx-lm reference. Under the new default it failed at step 1. The decompressed layout scores `.` at 29.109 and `,` at 29.094, one f16 ULP apart, while the fused kernel scores both at 29.094, so argmax picks the lower id and every later token differs. The composed absorbed path matched the reference, so this is a near-tie in rounding, not a kernel defect. The test now feeds the reference id at every step. It accepts a different argmax only when the reference id is within 0.0625 logits of the top, at most twice in 20 steps. A bidirectional prefill (#991) misses by whole logits and still fails it.

## 7. Change Summary

Decode throughput, `mlxcel-bench-decode` tg128 median tok/s, M5 Max. The host was shared with another unit's compiles, so absolute rates are 10 to 18% below #3129's, but the ratios held across two sweeps.

| prompt | V2 default (absorbed) | V2 `=0` | change | Kimi-VL fused | Kimi-VL fused off | change |
|---|---|---|---|---|---|---|
| 512 | 168.5 | 171.3 | -1.7% | 159.5 | 149.4 | +6.7% |
| 2048 | 153.5 | 150.9 | +1.8% | 149.8 | 139.2 | +7.5% |
| 16384 | 105.8 | 71.4 | +48% | | | |

Teacher-forced trace (`--decided 2`, width 1):

| comparison | context | decided disagreements |
|---|---|---|
| V2-Lite, default vs `=0` | 1024 | 0 / 209 |
| V2-Lite, default vs `=0` | 4096 | 0 / 106 |
| Kimi-VL, fused vs off | 1024 | 0 / 192 |
| Kimi-VL, fused vs off | 4096 | 0 / 50 |

On deepseek-vl2-small-4bit, which uses the same decoder, both layouts give identical greedy output for a text prompt and for an image prompt.

Tests: `the_default_absorbs_only_where_the_fused_kernel_runs` covers all six mode and kernel-availability cases. The `mla::` and `deepseek` lib suites pass, and `deepseek_v2_greedy_parity` and `deepseek_vl_v2_parity` pass under the default, `=0`, `=1` and `MLXCEL_MLA_FUSED_DECODE=0`.

Results: `docs/benchmark_results/mla-absorbed-default-m5max-2026-11-16.md`.

## 8. Follow-up Actions

- #3133: measure absorbed against decompressed on CUDA and ROCm before changing the default there.
- `src/execution/kv_arch.rs` still sizes `deepseek_v2` KV memory as decompressed, which overcounts under the new Metal default. That is safe for budgeting, but the estimate is not exact.
