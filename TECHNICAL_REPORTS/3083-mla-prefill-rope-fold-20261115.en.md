# Technical Report: PR #3083 - Fold the MLA rope score into SDPA on multi-token steps

**Date**: 2026-11-15
**Status**: Completed (DeepSeek V2 default-path peak needs a separate investigation)
**Risk Level**: Medium

## Executive Summary

DeepSeek-V3-family MLA prefill computed `q_pe @ k_peᵀ` as a dense `[B, H, L, L + offset]` tensor per layer, added the causal mask in f32, and passed the sum to SDPA as an additive mask. The new `mla::rope_folded_attention` concatenates `[q_nope | q_pe]` and `[k | k_pe]` along the head dimension, so SDPA computes the full score itself and the mask is causal mode, the caller's mask, or V3.2's per-row top-k mask. On an M5 Max, kimi-vl-a3b dropped from 76.31 to 32.03 GB in a single pass and from 20.0 to 14.43 GB with 2048-token chunks, and its prefill got 8 to 24% faster.

## 1. Problem Statement

Each layer of the multi-token path built its own score tensor of the full query length times the cache length, and it was the dominant prefill term for Kimi-VL, Kimi-K2 and Youtu (#3071).

## 3. Technical Decisions

- **Concatenate rather than add a mask.** This matches mlx-lm's `deepseek_v3.py`. MLX does not fuse this shape, because the query width (192, or 576 absorbed) differs from the value width (128, or 512 absorbed). The call therefore takes MLX's fallback, which the #3049 gate already splits along the query axis within the Metal 256 MiB budget. The additive tensor is gone, and the fallback's working set stays bounded.
- **Decode is unchanged.** Single-token decode keeps the old rope-mask form.
- **Where it applies:** `deepseek_v3`, `deepseek_v32` (V3.2's top-k mask becomes `[B, 1, L, S]`, shared by every head), and the shared absorbed path used by DeepSeek V2 with `MLXCEL_MLA_ABSORBED=1`.
- **Arithmetic order changes.** Kimi-VL's single-pass greedy text differs from main at a near-tie wording choice. The teacher-forced logit trace puts decided-position disagreement at 1 of 2627 positions (0.038%) from a cold start, and 0 of 2731 behind 2048 tokens of context. Perplexity is 0.04 to 0.06% lower on the branch. Every 2048-chunk cell is byte-identical.

## 7. Change Summary

| model | prefill | peak, main | peak, branch | prefill, main / branch |
|---|---|---|---|---|
| kimi-vl-a3b-thinking-4bit | 2048 chunks | 19.33-19.99 GB | 14.43 GB | 28.07 / 25.78 s |
| kimi-vl-a3b-thinking-4bit | single pass | 76.31 GB | 32.03 GB | 35.37 / 30.88 s |
| deepseek-v2-lite-chat-4bit-mlx | 2048 chunks | 27.98 GB | 27.98 GB | 42.84 / 42.81 s |
| deepseek-v2-lite, `MLXCEL_MLA_ABSORBED=1` | 2048 chunks | 15.13 GB | 15.13 GB | 42.73 / 42.64 s |

The tests in `mla::` (53, including 5 new tests that check the folded call against an f64 host reference) and in the touched model modules pass. Details are in `docs/benchmark_results/mla-prefill-rope-fold-m5max-2026-11-15.md`.

## 8. Follow-up Actions

- DeepSeek V2-Lite's default (non-absorbed) path peaks 12.85 GB above the absorbed path at 2048 chunks. Its default path already concatenated, so the score tensor was not the cause; this needs its own investigation.
- Kimi-VL's remaining single-pass peak is mostly full-position logits (#3070).
- DeepSeek V3.2 has unit coverage only, because no checkpoint fits this host.
