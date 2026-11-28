# Technical Report: PR #3117 - Keep Llama 4 chunked decode inside the query's chunk

**Date**: 2026-11-16
**Status**: Completed (a separate Llama 4 RoPE mismatch with mlx-lm was found and is tracked separately)
**Risk Level**: High (changes decode output past 8192 tokens, toward the reference)

## Executive Summary

Llama 4's chunked (local) layers attended at decode without a mask, over the last 8192 keys the cache keeps. Past position 8192 they therefore read keys from the previous chunk. mlx-lm rebuilds the `block_pos == 0` chunk mask on every step, so each query attends only within its own chunk (#3105). `ChunkedKVCache::attend` now slices a decode step's window to the query's chunk start, `floor(p / chunk) * chunk`. A multi-token step keeps #3100's per-chunk causal split, which moved into the same function. Every route goes through that one function: single-row, batched rows, and batched decode.

On llama-4-scout, the teacher-forced trace past the boundary goes from 20 of 75 decided-position disagreements to 0, and perplexity over positions 8192 to 8448 goes from 27.87 to the reference's 5.86, for both scalar and B=2 batched decode. Long-context decode gets 4% faster, because past the boundary the chunked layers attend to at most 286 keys instead of 8192.

## 3. Technical Decisions

- **Slice instead of building a mask.** At decode the chunk restriction is a prefix cut of the window, so the existing unmasked SDPA runs on fewer keys and no mask array is built.
- **The reference is mlx-lm's rule.** The trace's reference is mlxcel's one-pass prefill, which agrees with mlx-lm (with mlx-lm's RoPE set to mlxcel's) on all 941 decided positions.

## 7. Change Summary

| Cell | decode tok/s, main → branch | output |
|---|---|---|
| 554-token prompt, 128 tokens | 45.55 / 44.94 → 45.56 / 44.77 | identical |
| 8077-token prompt, 400 tokens | 27.04 / 27.12 → 28.09 / 28.37 | diverges after the context crosses 8192 (the fix) |

New tests fail on main and pass on the branch:
- a chunked decode across three boundaries against a host-built mlx-lm chunk mask;
- a tiny-model decode across chunk boundaries against the one-pass prefill;
- the chunked attend against the mlx-lm mask reference.

The `mlxcel-core` lib suite (2,319) and the `llama4` tests pass.

## 8. Follow-up Actions

Stock mlx-lm disagrees with mlxcel's Llama 4 on 46% of decided positions at every context length (perplexity 1.18 against 14.05). Swapping mlxcel's RoPE into mlx-lm closes the gap: mlx-lm uses `traditional=True` and the checkpoint's `llama3` rope scaling, which mlxcel's Llama 4 attention does not apply.
