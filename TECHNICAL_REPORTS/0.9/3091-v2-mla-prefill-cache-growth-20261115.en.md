# Technical Report: PR #3091 - Release superseded KV buffers during DeepSeek V2 prefill

**Date**: 2026-11-15
**Status**: Completed (the remaining gap is V2's decompressed cache layout)
**Risk Level**: Medium

## Executive Summary

DeepSeek V2's default MLA prefill peaked 12.85 GB above its absorbed path at 2048-token pieces (#3084). Per-layer attribution found two causes. The first, about 7.0 GB, is that the default path stores decompressed per-head K and V, which is by design. The second, about 6.2 GB, is dead buffers: when a piece grows a layer's cache, the old buffer goes to MLX's allocator cache, and no later request in that piece can reuse it, because each one asks for a larger size. `KVCache::reserve_rows` now grows a layer's buffer before that layer's graph is built, queues the copy with `async_eval`, and empties the allocator cache at the next layer's reservation. On an M5 Max with a 28,705-token prompt, the 2048-piece peak drops from 27.98 to 21.04 GB, and greedy output stays byte-identical.

## 1. Problem Statement

The issue's hypothesis was that the latent cache is re-expanded on every piece. Measurement did not support it. The allocator cache climbed from 3.05 to 10.47 GB across the 27 layers of the last 2048-token piece, and the engine flushes it only after each piece (#672).

## 3. Technical Decisions

- **Reserve per layer, not all at once.** Reserving every layer at the start of the forward holds every old and new buffer together. That variant measured 27.97 GB and was rejected.
- **Use the same growth rule.** `update_fp16` now calls the same growth helper, so the grown buffer is byte-identical to what the in-graph growth built.
- **Scope.** The reservation does nothing for decode rows, first fills, appends that fit, and quantized or paged layouts. DeepSeek V2 calls it before both its decompressed and its latent cache updates.
- **Cost.** About 2% on prefill, measured with one binary in both modes (off: 41.97 and 42.15 s; on: 43.13 and 42.87 s).

## 7. Change Summary

| cell | peak, main | peak, branch | output |
|---|---|---|---|
| default, 2048 pieces | 27.98 GB | 21.04 GB | identical |
| `MLXCEL_MLA_ABSORBED=1`, 2048 pieces | 15.12 GB | 14.35 GB | identical |
| default, single pass | 54.44 GB | 54.43 GB | identical |

Tests:
- 3 new `cache::fp16_growth_tests`.
- `cache::kv_trim_tests`, `mla::` and `models::deepseek_v2` pass, plus the last-row tests after the rebase onto #3090.

Details: `docs/benchmark_results/mla-v2-prefill-cache-growth-m5max-2026-11-15.md`.

## 8. Follow-up Actions

- Every model whose dense FP16 cache grows during chunked prefill should leave the same dead buffers, so other models can adopt `reserve_rows`.
- A first fill does not grow the cache, so the single-pass peak does not move.
