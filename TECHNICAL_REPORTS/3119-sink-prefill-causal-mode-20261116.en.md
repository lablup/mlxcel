# Technical Report: PR #3119 - Run attention-sink prefill in MLX causal and window mode

**Date**: 2026-11-16
**Status**: Completed (MiMo v2 Flash and DeepSeek-V4 are covered by unit tests only, with no local checkpoint)
**Risk Level**: Medium

## Executive Summary

#3056 and #3097 moved plain causal and sliding-window prefills off dense mask arrays and onto MLX's causal mode. Layers with attention sinks were left out, because the bridge had no causal-mode SDPA entry that accepts sinks, so they still built a full (L, L) band per layer (#3098). The bridge now has `fast_scaled_dot_product_attention_with_sinks_causal`, and the new `prefill_sdpa::causal_prefill_attention` routes to it when MLX's probe says the sink variant fuses. Otherwise, or on an error from the new entry, it falls back to the band. GPT-OSS, Falcon-OCR, MiMo v2 Flash, all-local DeepSeek-V4 stacks and Laguna use the new route.

On an M5 Max, single-pass peaks drop:

| model | tokens | peak | prefill time |
|---|---|---|---|
| gpt-oss-20b | 19,773 | 22.54 → 17.86 GB | 6.63 → 3.62 s |
| laguna-xs.2 | 28,747 | 35.84 → 29.25 GB | not reported |
| falcon-ocr | 9,067 | 5.11 → 4.70 GB | not reported |

Output is identical in every cell.

## 3. Technical Decisions

- **Probe with sinks.** `sdpa_uses_fused_kernel` takes `has_sinks`, and its memo key includes it, so a shape that fuses without sinks but not with them still gets the band.
- **Out of scope.**
  - The OpenAI privacy filter uses a symmetric bidirectional band, which causal mode cannot express.
  - DeepSeek-V4 compressed and sparse layers extend the local mask with pooled rows, so any stack containing them keeps the array.
- **The roster sentence now counts 13 `create_causal_mask` files.**

## 7. Change Summary

| model | path | peak, main → branch | prefill, main → branch |
|---|---|---|---|
| gpt-oss-20b-mxfp4-q4 | single | 22.54 → 17.86 GB | 6.63 → 3.62 s |
| gpt-oss-20b-mxfp4-q4 | 2048 chunks | 13.42 → 13.16 GB | 3.96 → 3.84 s |
| laguna-xs.2-4bit | single | 35.84 → 29.25 GB | 39.03 → 7.32 s (main is one run on a busy host) |
| laguna-xs.2-4bit | 2048 chunks | 22.32 → 21.99 GB | 9.71 → 7.49 s |
| falcon-ocr | single | 5.11 → 4.70 GB | 0.79/0.94 → 0.59/0.28 s |

Tests: the new `prefill_sdpa_tests` compare the sink entry and the routed helper against the explicit-mask call and an f64 reference, in f32 and f16, across windows, head dims and GQA. Malformed inputs return `Err`. The `mlxcel-core` lib suite (2,324) and the full `mlxcel` lib suite (11,976) pass, and workspace clippy is clean.
