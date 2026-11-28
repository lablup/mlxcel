# Technical Report: PR #3152 - Absorb DeepSeek V2 decode by default on ROCm

**Date**: 2026-11-18
**Status**: Completed
**Risk Level**: Medium (changes the default cache layout for every DeepSeek V2 user on ROCm)

## Executive Summary

DeepSeek V2 now caches the MLA latent `(ckv, kpe)` and runs absorbed decode by default on ROCm, extending the rule #3131 set on Metal and #3133 set on CUDA. ROCm has no fused absorbed-decode kernel, so this default runs the composed path, which measured 3.4% behind the decompressed path at 512 prompt tokens (median of five runs per arm), 17% ahead at 2048 and 65% ahead at 16384, with teacher-forced traces showing 0 decided disagreements at 1024 and 4096.

## 1. Problem Statement

#3133 measured the 2026-11 MLA and prefill-memory work on CUDA (#3149) and left ROCm unmeasured because no host was available. It closed with the ROCm default still decompressed, so a `gfx1151` machine ran 8.9x larger KV caches than the layout was already known to fit in, and `kv_arch`, which reads `mla::absorbed_enabled()`, sized them the same way. The gap was not a code hole but a missing evidence row.

## 2. What Changed

- `mla::default_absorbs_on(fused_available, backend)` returns true where the fused kernel runs and on `GpuBackendKind::Cuda` and `GpuBackendKind::Rocm`. The CPU backend stays decompressed.
- `docs/mla-absorbed-decode.md`, `docs/environment-variables.md` and `CHANGELOG.md` describe the new default and link the results record.
- `docs/benchmark_results/mla-and-prefill-memory-rocm-gfx1151-2026-11-18.md` records sections 3, 4, 5 and 7 on this host, with per-run CSV files carrying the measured commit.
- `src/execution/kv_arch.rs` gains the ROCm reference in its module comment only; the sizing follows the runtime layout with no code change.

## 3. Technical Decisions

**Switch on the evidence rule, not on a single cell.** The rule set in #3102 is that the latent layout becomes the default when decode throughput stays within a few percent of decompressed and the teacher-forced trace shows no material disagreement. ROCm meets both: -3.4% at 512, which is inside the band Metal was switched under (-1.7%), and 0 of 208 and 0 of 105 decided disagreements.

**The 512-token cell was measured twice.** An initial pair of runs read +5.6%, but the decompressed arm spread 14% across its two runs, so the cell was repeated five times per arm. That repeat read -3.4% with an 11.7% spread in the composed arm; dropping its slowest run moves the composed median to 41.45, still about 3% behind. The report states the repeat rather than the more favourable first pair.

**Keep gpt-oss prefill as measured, not as hoped.** The sink prefill from #3119 takes MLX's causal mode only where `maskless_route_is_fused` reports that MLX fuses it, and that answer is Metal-only. CUDA and ROCm keep the band, so gpt-oss peaks at 26.21 to 26.43 GB on both trees and a 19,773-token single pass asks for a 100.35 GB allocation in both. Recording this as an unchanged backend property avoids implying a regression or an unearned gain.

**Accept a host-specific gate invocation.** The first `make verify-rocm-held` at the gate's default of 6 jobs was killed while building test binaries on this 30 GiB machine. The repeat with `ROCM_JOBS="-j 3"` passed every prerequisite and all workspace tests, so the record states the cap and the reason instead of downgrading the gate.

## 4. Measurements

Decode, `mlxcel-bench-decode`, deepseek-v2-lite-chat-4bit-mlx, `--ignore-eos`, tg128, ABBA order with 60 s idle before each run:

| prompt | decompressed | composed | change |
|---|---|---|---|
| 512, five runs per arm (median, range) | 42.72 (41.62 to 43.57) | 41.28 (36.94 to 41.78) | -3.4% |
| 2048 | 35.28 | 41.41 | +17% |
| 16384 | 22.04 | 36.49 | +65% |

Phase peak behind 16K falls from 16.06 to 12.39 GB. KV sizing at 32K goes from 8.44 GiB to 972.0 MiB.

Prefill: GLM-4.7-Flash 2048-token chunks 25.26 to 20.53 GB with `generate` prefill 919.2 s to 306.9 s (3.0x), and a 28K single pass that fails to allocate 62.72 GB on the pre-#3114 tree now fits in 21.13 GB.

Gate: `make verify-rocm-held ROCM_JOBS="-j 3"`, 192 test binaries, 15,954 passed, 0 failed, 524 ignored.

## 5. Learning Points

**A missing evidence row looks like a code gap.** #3133 closed with "ROCm and the CPU backend are unmeasured" in the code comment, and the comment was accurate: the backend had no kernel and no measurement, so the conservative default was the only defensible one. What unblocked the change was one host and one afternoon of a measurement protocol that already existed in the sibling CUDA record.

**Noise in one arm changes the sign of a verdict.** The same cell read +5.6% at two runs and -3.4% at five. Repeating per-arm rather than adding cross-arm runs is what separated the two, because the variance was inside the decompressed arm's run-to-run spread, not between the layouts.

## 6. Follow-up Actions

- #3153: the DeepSeek V2 KV estimate models only an int8 decline while the runtime declines every non-FP16 cache mode, so a turbo-quant cache can undercount the estimate. The default change makes that path reachable by default.
- Llama 4 Scout (#3117, #3126) does not fit this host and stays covered by the CUDA and Metal records.
