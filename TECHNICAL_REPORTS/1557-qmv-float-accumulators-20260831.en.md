# Technical Report: PR #1557 - float accumulators for qmv below Ampere

**Date**: 2026-08-31

**Author**: mlxcel maintainers

**Status**: Completed with one acceptance criterion deliberately unticked, and one pre-existing defect surfaced. sm_80-and-later runtime validation deferred to a GB10 host.

---

## Executive Summary

PR #1557 (issue #1539) makes `qmv` accumulate in float on pre-Ampere hardware for bf16 checkpoints, instead of accumulating in the element type as upstream does below 8 bits. On a V100 this is worth **1.87x end-to-end decode** on `qwen3.8-27B-4bit` (219.96 to 117.83 ms/token) and 1.73x on `gemma-4-12B-it-4bit`, with the 8-bit sibling unchanged at 1.00x as the control predicts.

The change is small, the reasoning behind where it applies is not. It is confined to bf16 and to `__CUDA_ARCH__ < 800`, and both boundaries are load-bearing.

Two findings came out of the work that are worth more than the speedup. The issue's roofline acceptance criterion was set wrong and is left unticked rather than fudged. And `tests/cuda_qmm_determinism.rs` fails on `main` today, independent of this change.

## 1. Problem Statement

`qmv.cu` selected the accumulator from the weight bit width alone: `float` at `bits >= 8`, otherwise the element type `T`. That rule assumes `T`'s own ALU is fast. From Ampere on, it is. Before Ampere it is not, because sm_70 and sm_75 have no bf16 arithmetic unit at all, so a `cutlass::bfloat16_t` accumulator turns every FMA in the k-loop into convert-to-float, fma, convert-back.

`qmv` is the decode kernel for essentially every quantized matmul, so this sat on the hot path of every single-stream decode step on Volta.

The controlled evidence, from #1538's record: on the `gemma-4-12B-it-{4bit,8bit}` pair, which differs in exactly one config key, `qmv` spent 12.84 s at 4 bits against 5.99 s at 8 bits over **identical** 39,151 launches. That 2.14x gap accounted for 97.6% of the end-to-end decode difference, while `qmm_naive`, which accumulates in float at every bit width, moved the other way in the same profile. The accumulator was isolated before a line was written.

## 2. Change Summary

5 files, +309 / -7. The functional change is 57 lines in one file.

A `qmv_accumulator<bits, T>` trait in `qmv.cu` selects `float` under `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800` when `bits >= 8` **or** `T` is `cutlass::bfloat16_t`, and otherwise keeps upstream's rule. Both `qmv_kernel_impl` and `qmv_multirow_kernel_impl` use it; `gather_qmv` inherits it. The float specializations of `dequant_fma` and `fma_tile` already existed and were already exercised by the `bits >= 8` path, so this instantiates nothing new.

Also: 80 lines of tests in `ffi_tests.rs`, a benchmark record at `docs/benchmark_results/qmv-float-accum-v100-2026-08-31.md`, the post-program row in #1538's baseline doc, and a changelog entry.

## 3. Technical Decisions

### 3.1 `__CUDA_ARCH__` guard rather than host-side dispatch

`qmv` is AOT-compiled. The host only takes `&qmv_kernel<...>` as a function pointer and launches it through `add_kernel_node_raw`, and **the accumulator appears in neither the template signature nor the mangled name**. That is what makes the device-side guard safe: nvcc emits one device pass per entry in `MLX_CUDA_ARCHITECTURES`, so the guard produces exactly per-architecture behavior inside a fat binary, and the single host pass has nothing to disagree with.

The alternative, a `bool FloatAccum` template parameter fed from #1537's capability probe, would have worked but adds a host dispatch and changes the symbol. #1537's probe turned out not to be needed here at all.

### 3.2 Narrowed to bf16, not to every type below Ampere

The issue said "regardless of bit width", and that half is implemented as written. But the promotion is conditioned on `T` being `bfloat16_t` rather than applied to every element type below Ampere, and that narrowing is deliberate.

sm_70 and sm_75 **do** have a native fp16 FMA, at twice the fp32 rate. Promoting f16 accumulation to float there would spend throughput to buy precision, which is the opposite of the trade being made for bf16, where there is no native path at all. Widening the guard would have quietly slowed down a case nobody asked about. f16 policy below Ampere is #1542's subject.

Every checkpoint on this host is bf16, so the narrowing costs nothing measured, and it keeps the two issues from colliding.

### 3.3 The multirow kernel had to move with it

#725 guarantees that the multirow kernel produces bit-identical per-row results to the per-row kernel, and `qmv_multirow_matches_per_row_qmv_bitwise` enforces it. Changing the accumulator in one and not the other would have broken that invariant silently. Both moved.

## 4. Validation

Host: Tesla V100-PCIE-32GB (sm_70), CUDA 12.9.41, warm PTX cache, following the six methodology rules committed in #1538. The before arm reproduces the committed baseline within 1.2%.

| Checkpoint | Before | After | Speedup |
|---|---|---|---|
| `qwen3.8-27B-4bit` | 219.96 ms/tok | **117.83** | **1.87x** |
| `gemma-4-12B-it-4bit` | 124.26 | **71.66** | **1.73x** |
| `gemma-4-12b-it-8bit` (control) | 66.75 | 66.73 | 1.00x |
| `gemma-4-26b-a4b-it-4bit` (MoE) | 35.45 | 29.56 | 1.20x, ordinal only |

`qmv` achieved bandwidth on qwen: **66.3 to 133.2 GB/s, 7.37% to 14.80%** of the 900 GB/s roofline, a 2.01x improvement over 11,431 identical launches, with nsys reconciling at 101.3% before and 101.9% after. On the dense pair: 57.7 to 112.8 GB/s over 39,151 identical launches, attribution 99.4%. The `qmm_naive` control moves by -0.12% and +0.55%, confirming nothing else shifted.

The 8-bit arm being unchanged at 1.00x is the strongest single check: it already had float accumulators, so the change must be a no-op there, and it is.

**Occupancy.** Checked with `-Xptxas -v` rather than assumed. `qmv_kernel` at `<8,16,64>` goes 55 to 61 registers with blocks-per-SM unchanged at 4; the `has_residue_k=true` variant, which is the one this workload actually runs, goes 59 to 66 and 4 to 3 blocks. **Nothing spills**: `STACK` and `LOCAL` are zero across all 378 instantiations in both arms. The measured 1.96x already includes the residue-k occupancy cost. `elems_per_thread` was left alone.

## 5. Validation Limits and Follow-up

### 5.1 The roofline criterion was set wrong, and is left unticked

Issue #1539 asked for `qmv` decode to reach **>= 25%** of the bandwidth roofline. It reached 14.80%, and that box is unticked.

The criterion was not achievable by this change and should not have been written that way. The controlled measurement set the ceiling at 2.14x, so from 7.37% the arithmetic maximum was about 15.8%; 14.80% is 94% of it. The 25% figure came from the epic-level target and conflated two different things: the 8-bit arm itself only reaches 26.7% of roofline *while reading 1.9x the bytes*, so the residual gap is the sub-byte unpack cost, not the accumulator. Closing that is #1543's territory, not this issue's.

Recording this as an unmet criterion rather than quietly relaxing it is the point. The number to carry forward is 14.80%.

One consequence worth noting: the 4-bit against 8-bit inversion that #1538 documented falls from 1.886x to **1.074x**. The workaround this change replaces is now nearly moot on dense checkpoints.

### 5.2 Quantized prefill on sm_70 is not bitwise reproducible, and was not before this change

`tests/cuda_qmm_determinism.rs` fails at the prefill step. This was verified not to be caused by this PR: reverting `qmv.cu`, rebuilding `origin/main`, and rerunning reproduces the same divergence at step 0. Prefill runs at `M = 64`, which routes to `qmm_naive`, a kernel `qmv` never serves, and iteration 0's prefill logits hash identically on both builds.

The test has never actually run in CI, because its default checkpoint `models/llama-3.2-1b-4bit` is absent and it skips rather than fails.

Substance was verified another way instead: three greedy 32-token decodes give one distinct output on each build, and both builds emit the same tokens. `qmv_multirow_matches_per_row_qmv_bitwise` passes, and a new `qmv_matches_qmm_across_bits_and_group_sizes` covers bits 4/8 against group sizes 32/64/128.

This is the same class as #910, which fixed `qmm_sm80` on sm_121. It is out of scope here and is tracked separately.

### 5.3 Deferred to GB10, and one criterion recovered

**Recovered.** The sm_121 SASS-diff criterion was the one flagged as most at risk of being assumed rather than checked, and it did not need GB10 hardware: nvcc does not need the target part present. Compiling `qmv.cu` from unpatched and patched sources with production flags gives **byte-identical `cuobjdump --dump-sass` output at sm_80 (389,998 lines) and sm_121 (437,519 lines)**, with identical `-res-usage`, while sm_70 differs as intended. The guard provably does not reach Ampere or later.

**Still deferred**, in the PR body and in #1536's continuation section: byte-identical greedy output on a GB10 device, GB10 throughput unmoved, and `cargo test --features cuda` on an sm_121 host.

**Unquantified risk.** The multirow kernel halves its occupancy, 2 blocks per SM to 1, at 110 to 168 registers. Batched decode was not measured here. The row window is autotuner-tunable through #906 if it proves to matter.

## References

- Issue #1539, epic #1536 (including its GB10 continuation section)
- The controlled measurement this change was derived from: #1538 and `docs/benchmark_results/volta-sm70-baseline-2026-08-31.md`
- This change's record: `docs/benchmark_results/qmv-float-accum-v100-2026-08-31.md`
- Multirow bitwise-equality invariant: #725
- Prior determinism defect of the same class: #910
- Autotuner for launch configurations: #906
