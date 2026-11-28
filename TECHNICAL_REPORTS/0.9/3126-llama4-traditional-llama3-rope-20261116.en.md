# Technical Report: PR #3126 - Apply Llama 4's traditional llama3-scaled RoPE

**Date**: 2026-11-16
**Status**: Completed (VLM image prompts and multi-node runs share the fixed attention but were not run)
**Risk Level**: High (changes every Llama 4 output, toward the reference)

## Executive Summary

mlxcel's Llama 4 rotated q and k with half-split RoPE (`traditional = false`) and `rope_scale = 1.0`. It ignored the checkpoint's `rope_scaling` (llama3, factor 16, original 8192). mlx-lm uses `initialize_rope(..., traditional=True, scaling_config=rope_scaling)`. Every Llama 4 position was therefore rotated wrongly (#3120, found while tracing #3105). A new `Llama4Rope` pairs adjacent dimensions and takes its frequency table and base from the existing `RopeScalingKind::resolve` and `rope_overrides::resolve_base`. All 8 `fast_rope` and `fast_rope_batched` call sites in `CxxAttention` go through it. The VLM loader and the pipeline stage build the same block, so they are covered too.

On llama-4-scout against stock mlx-lm 0.31.3, decided-position disagreement falls from 83-92% to 0.00-0.05% across prefill, decode and B=2 batched decode. Perplexity goes from 1,800-3,000 to within 0.1% of mlx-lm's 10-16. Greedy output goes from a repetition loop to coherent text, and decode speed is unchanged.

## 3. Technical Decisions

- **Reuse the shared llama3 scaling.** The frequency table is the one llama3 and helium already use, so no new table was written. As a side effect, the `--rope-*` overrides now reach Llama 4, and the family lists in `rope_overrides.rs`, `rope_utils.rs` and `docs/llama-server-compat.md` say so.

## 7. Change Summary

| shape | decided positions | main | branch | ppl mlx-lm / main / branch |
|---|---|---|---|---|
| prefill width 1280 | 1965 | 83.82% | 0.05% | 14.213 / 1868.2 / 14.217 |
| prefill width 4608, context to 9216 | 3643 | 87.90% | 0.00% | 11.803 / 2392.9 / 11.800 |
| decode width 1, context 512-2560 | 750 | 83.33% | 0.00% | 16.351 / 1819.6 / 16.349 |
| batched decode B=2 | 750 | 83.47% | 0.00% | 16.351 / 1815.7 / 16.367 |
| decode width 1, context 4608-9216 | 1886 | 92.42% | 0.00% | 10.268 / 3059.5 / 10.260 |

Tests:
- Two new tests compare single-row and batched rotation against an f64 transcription of mlx-lm's traditional `Llama3RoPE` at offsets up to 20000, and assert that the old rotation fails that reference.
- The full `mlxcel` lib suite (11,986) passes, and workspace clippy is clean.

## 8. Follow-up Actions

On main, a single 9216-wide Llama 4 forward aborted with a Metal out-of-memory error; the unit used two 4608-wide forwards instead. Not investigated.
