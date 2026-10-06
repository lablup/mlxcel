# Technical Report: PR #2137 - perf(moe): share gather indices across SwitchGLU projections

**Date**: 2026-10-06
**Author**: mlxcel maintainers
**Reviewer**: implementation review cycle
**Status**: Completed (graph and greedy-output gates validated on CUDA; Metal throughput not measured)
**Languages**: Rust, C++ (one bridge function)
**Risk Level**: Low (graph-shape change in a shared helper; the kernels receive identical index values)

---

## Executive Summary

Issue #1713 found that decode pays a host-side graph build proportional to the number of MLX ops, and it named the shared MoE helper `SwitchGLU` as the first place to cut ops. The cost turned out to sit inside MLX rather than in the helper's own `expand_dims` calls. Each `gather_qmm` / `gather_mm` call without `lhs_indices` synthesizes `arange -> reshape -> broadcast`, and each call casts non-`uint32` expert ids. `SwitchGLU` made three such calls per layer. PR #2137 builds the index arrays once per block and passes them in. It also merges each `expand_dims` pair into one node. The decode graph loses 4 nodes and 7 edges per MoE layer, and greedy output is byte-identical on every model tested.

---

## 1. Problem Statement

### 1.1 Background

The issue measured `forward` (host graph build) as flat with model size while device time scales, so small models and SSM/MoE hybrids lose the most. granite-4.0-h-tiny-4bit built 231 nodes per layer against 89 for qwen3-8b, with the excess in Broadcast, AsType, ExpandDims, Arange, Full, Reshape and Slice.

### 1.2 Attribution

The MLX source (`ops.cpp`) for `gather_qmm` and `gather_mm` calls `indices_or_default` for both index arguments. That function returns `astype(indices, uint32)` when indices are given, and `reshape(arange(total, uint32), batch_shape)` when they are not. The two results then go through `broadcast_arrays`. On GB10 the exported decode graph confirmed it: granite-tiny's 120 Arange nodes are exactly 3 gather calls x 40 layers, and 80 of its 120 ExpandDims come from `SwitchGLU`'s double `expand_dims` on the input.

---

## 2. Technical Decisions

### 2.1 Pass the indices MLX would have built

`prepare_gather_indices` produces `rhs = astype(ids, uint32)` and `lhs = broadcast_to(reshape(arange_u32(total), batch), broadcast_shape(batch, ids))`. MLX's `astype`, `reshape` and `broadcast_to` all return their input unchanged when dtype or shape already match, so passing these arrays adds nothing inside MLX. Gate and up share the lhs. In the unsorted path, down keeps MLX's default lhs because its input batch is `[n, k]` rather than `[n, 1]`. In the sorted prefill path all three share one lhs. When the shapes do not broadcast, `lhs` is `None` and MLX validates and raises exactly as before.

### 2.2 A uint32 arange in the bridge

The first version built the lhs with `arange_i32` plus a cast, which added an AsType node for every one it removed elsewhere (measured: AsType 160 -> 200 on granite-tiny). The new `arange_u32(stop)` bridge function calls `mlx::core::arange(stop, uint32)`, the same call `indices_or_default` makes, which recovered the fourth node per layer.

### 2.3 Scope kept to the shared helper

`SwitchLinear::forward` keeps its signature and delegates with `lhs = None`. Fifteen families (DeepSeek, GptOss, Phixtral, Qwen3Next and others) drive `SwitchLinear` directly, and they could adopt `forward_indexed` in a follow-up. Granite's remaining Full, Slice and AsType excess sits in its SSM and attention paths, which the issue scoped out.

---

## 3. Validation (GB10, CUDA)

| Model | nodes | edges |
|---|--:|--:|
| granite-4.0-h-tiny-4bit | 3419 -> 3259 | 9057 -> 8777 |
| qwen3-30b-a3b-4bit, fused MoE off | 2895 -> 2703 | 7575 -> 7239 |
| qwen3-30b-a3b-4bit, default (fused decode kernel) | unchanged | unchanged |
| granite-4.0-h-350m-4bit (no experts) | unchanged | unchanged |

- Greedy `--temp 0` output was byte-identical between the base and new binaries on granite-4.0-h-350m, granite-4.0-h-tiny, qwen3-30b-a3b and qwen3.5-35b-a3b, with fused MoE on and off. A repeated base run matched the first one, and the prompt covers both the sorted (prefill) and unsorted (decode) paths.
- New unit tests compare the prepared indices with MLX's defaults, and compare the block output bit for bit with the per-call form in decode, small-prefill and sorted-prefill shapes, for int32 and uint32 ids and for the Inkling expert-scale path.
- Throughput: n=3 interleaved rounds with a null arm. The null arm deviated up to 34% from base within one round, and the base/new medians fell inside that spread. The expected gain (about 0.05 ms/token on granite-tiny) is below this host's noise, so the PR makes no throughput claim.

## 4. Not Verified

The M5 Max tok/s ratios against `benchmarks/pylm_m5max_2026-09-06.csv` and the Metal `forward` ms/token need Apple Silicon.
