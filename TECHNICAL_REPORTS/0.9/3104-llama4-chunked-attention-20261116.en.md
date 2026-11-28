# Technical Report: PR #3104 - Compute Llama 4 chunked attention without a dense mask

**Date**: 2026-11-16
**Status**: Completed (a decode-side chunk-boundary bug was found and filed as #3105)
**Risk Level**: Medium

## Executive Summary

Llama 4's MoE (local) layers attend causally within `attention_chunk_size` blocks (8192 for llama-4-scout). The prefill built one dense `create_chunked_attention_mask` and shared it across every MoE layer, and after #3097 that mask set Llama 4's peak (#3100). In an unpadded prefill with no caller mask, these layers now go through a new `chunked_causal_attention`. It splits the queries at chunk boundaries and runs MLX causal-mode SDPA for each piece over keys from `max(chunk_start, start_position)` up to its last query, so a piece that starts mid-chunk still sees the cached keys of its own chunk. On an M5 Max with 19,717 tokens, the single-pass peak drops from 72.88 to 68.41 GB, and greedy output is byte-identical in both repetitions.

## 3. Technical Decisions

- **Where the dense mask stays.** It is kept wherever `prefill_uses_causal_mode` declines: batched rows, speculative verify blocks, and `MLXCEL_PREFILL_CAUSAL_MODE=0`. `forward_igqa` and `forward_igqa_rows` both choose through `Llama4CxxModel::chunked_prefill_mask`.
- **Decode and caller-mask steps are unchanged.** The legacy plain-`KVCache` forward never used the chunk mask.
- **Numerics.** The teacher-forced logit trace compared the dense mask with the new default. It used three 2048-token forwards behind 7,168 tokens of context, each crossing the 8192 boundary partway through. Result: 0 of 6,144 top-1 disagreements, 0 of 756 at decided positions, max logit delta 0.0000, and equal perplexity.

## 7. Change Summary

| path | peak, main → branch | output |
|---|---|---|
| single pass | 72.88 → 68.41 GB | identical |
| 2048 chunks | 65.86 → 65.70 GB | identical |

Prefill times were within host noise, so no timing claim is made. The new tests compare the new path with the dense mask at:
- offset 0, inside one chunk and across chunks;
- mid-chunk, including a mid-chunk piece that crosses a boundary;
- behind a trimmed front, and on an exact boundary;
- f16.

A split-prefill model test also checks the result against a single pass.

## 8. Follow-up Actions

#3105: at decode, the chunked layers attend without a mask over the last 8192 keys, so past position 8192 they include keys from the previous chunk. mlx-lm rebuilds the `block_pos == 0` chunk mask on every step. This predates this PR.
