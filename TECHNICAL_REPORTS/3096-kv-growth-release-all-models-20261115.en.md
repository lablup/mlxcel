# Technical Report: PR #3096 - Release superseded FP16 KV buffers in every model's prefill

**Date**: 2026-11-15
**Status**: Completed
**Risk Level**: Medium (shared cache path for every dense-cache model)

## Executive Summary

When a chunked prefill piece grows a layer's dense FP16 `KVCache`, the old buffer lands in MLX's allocator cache. No later, larger request in that piece can reuse it, and the engine flushes only after each piece. #3091 fixed this for DeepSeek V2 only. The release now runs inside the shared growth in `KVCache::update_fp16`, so every model that uses the dense FP16 cache gets it (#3092). On an M5 Max with a 28K-token prompt at 2048-token pieces, the peak drops from 10.94 to 7.26 GB on qwen3-4b, from 10.39 to 8.09 GB on llama-3.1-8b, and from 5.25 to 4.86 GB on gemma-3-4b. Greedy output is byte-identical and prefill time stays within 3%. With the branch, 2048-token pieces now peak below a single pass on all three.

## 3. Technical Decisions

- **Shared path, narrow trigger.** The release runs only on a multi-token append that outgrows an existing buffer. In that case it empties the allocator cache, builds the grown buffer with the same rule as before, and queues the copy with `async_eval`. It never touches first fills, decode rows, appends that fit, or the quantized, rotating and pool-backed layouts.
- **DeepSeek V2's explicit calls are gone.** Both of its caches reach `update_fp16`, so `reserve_rows` is deleted and its peak is unchanged against main.
- **The memory guard is comparative.** The K/V inputs a piece consumes also land in the allocator cache, so the integration test (`tests/kv_growth_release.rs`) compares the release against a lazy-growth control (151 MB against 403 MB) instead of a fixed threshold. It also fails when the release is disabled on purpose.

## 7. Change Summary

| model | peak, main | peak, branch | prefill | output |
|---|---|---|---|---|
| qwen3-4b-4bit | 10.94 GB | 7.26 GB | -1.1% | identical |
| meta-llama-3.1-8b-instruct-4bit | 10.39 GB | 8.09 GB | +2.9% | identical |
| gemma-3-4b-it-4bit (5 of 34 layers use `KVCache`) | 5.25 GB | 4.86 GB | +2.7% | identical |
| deepseek-v2-lite-chat-4bit-mlx | 20.20 GB | 20.20 GB | +0.2% | identical |

Tests: the `mlxcel-core` lib suite (2308), the two retention binaries, and the full `mlxcel` lib suite serially (11,950) pass. Details: `docs/benchmark_results/kv-growth-release-all-models-m5max-2026-11-15.md`.
