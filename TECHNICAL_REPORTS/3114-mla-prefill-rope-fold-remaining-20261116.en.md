# Technical Report: PR #3114 - Fold the remaining hand-masked MLA prefills into SDPA

**Date**: 2026-11-16
**Status**: Completed (KimiLinear, LongcatFlashNgram and the GLM4MoeLite pipeline stage are covered by unit tests only)
**Risk Level**: Medium

## Executive Summary

#3083 added `mla::rope_folded_attention`, which concatenates the rope part into SDPA so prefill needs neither a per-head additive score tensor nor a dense mask. glm4_moe_lite, kimi_linear's MLA layers, kimi_k3 and longcat_flash_ngram still added the causal mask by hand. Their multi-token steps now go through the folded call with MLX causal mode, on both the absorbed and the materializing branch, and the GLM4MoeLite pipeline stage drops its mask the same way (#3099).

On an M5 Max, glm-4.7-flash-4bit (22,754 tokens) at 2048-token chunks drops from 28.82 to 20.51 GB, and prefill goes from 35 to 50 s down to 12.3 to 12.6 s. A single pass previously failed with a Metal GPU timeout at 82.8 GB; it now completes at 24.11 GB.

## 3. Technical Decisions

- **Decode and caller masks.** Single-token decode is unchanged. A caller-supplied mask is still applied as given, so the GLM4MoeLite MTP drafter keeps its own mask and stays on the `create_causal_mask` roster. The roster sentence now reads 17 files.
- **Arithmetic order changes.** Folding reorders the score sum. The teacher-forced trace on glm-4.7-flash puts decided-position disagreement at 2 of 3,335 (0.060%) cold at width 2048, and 3 of 3,360 (0.089%) at width 4096 behind 2048 tokens of context. Perplexity moves by +0.23% and +0.05%. In 81 to 83% of the disagreements, the branch's second choice is main's top token. Greedy output at 2048 chunks is byte-identical.

## 7. Change Summary

| model | prefill | peak, main → branch | prefill, main → branch |
|---|---|---|---|
| glm-4.7-flash-4bit | 2048 chunks | 28.82 → 20.51 GB | 35-50 → 12.3-12.6 s |
| glm-4.7-flash-4bit | single pass | GPU timeout at 82.8 GB → 24.11 GB | fails → 11-16 s |
| kimi-k3-2l-mla-synth (2 layers, 2,831 tokens) | single / 2048 | 41.51 → 39.69 GB | 0.50 → 0.45 s |

Tests:
- New tiny-weight tests for each family compare the folded prefill against the single-token path, and the pipeline stage against the single-process model.
- The `mlxcel-core` lib suite passes (2,318).
- The full `mlxcel` lib suite passes except `server::chat_template_json::tests::non_string_object_keys_are_coerced_like_python`, which fails identically on main after #2819 and is fixed in #3115.
