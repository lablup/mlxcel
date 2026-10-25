# Technical Report: PR #2591 — MLA batched decode

**Date:** 2026-10-25

**Related issues:** #2560, #2566, #2587

## Executive Summary

DeepSeek V2/V3/V3.2, MiniCPM3, Mistral4, GLM-4 MoE Lite and Youtu gain native multirow decode; GLM MoE DSA and DeepSeek-VL2 delegate to their converted text backbones. Projections and feed-forward/expert operations run across the batch while each sequence retains its own ragged cache. The final DeepSeek V2 fix also batches both folded projections across compatible absorbed rows, including when absorbed and expanded cache layouts are interleaved. Metal performance acceptance remains in #2587.

## Problem Statement

MLA cache layouts cannot be represented by the dense split-attention helper's single head dimension. Some paths cache expanded keys and values, while others retain a compressed latent plus rotary key components. DSA adds sequence-dependent sparse selection, and a cache-storage policy can put absorbed and expanded rows in the same batch.

## Change Summary

Expanded paths use shared batched cache attention. The new latent MLA helper performs each row's cache append, optional selection, positional score calculation and latent attention, with `embed_q`, `unembed_out` and output projection remaining batched. Single-row and empty-batch behavior use the shared routing contract.

DeepSeek V2 stably partitions rows by their established absorbed or expanded cache layout, including mixed FP16 latent and Int8 expanded storage. Absorbed rows run the folded query projection once as a group, retain row-local cache update and Stage 2 split-KV selection or fallback, then run the folded output projection once over the concatenated latent results. Expanded rows similarly share their `kv_b_proj` and batched attention. Results are restored to the original request order, so an absorbed/expanded/absorbed layout cannot reorder outputs or change cache representation. DeepSeek V3.2 evaluates top-k selection against each row's live post-trim length and gathers aligned latent/rotary cache entries. MiniCPM3 and Mistral4 preserve their rotary and residual/output scaling. GLM's plain decode does not mutate its separate MTP drafting state. Both wrappers are exercised through actual forward calls.

## Validation

Fifteen focused CPU checks covered classification, expanded/absorbed ragged attention, mixed cache policies, DSA dense/top-k after front trim, seven full tiny-model B>=3/B1/B0 harnesses, two actual wrapper delegation paths and invalid-input diagnostics. Core CPU-safe checks additionally cover the single-row path, latent-cache behavior, causal masking and absorbed/decompressed parity. A preprojected one-chunk split test pins the new boundary without requiring the backend merge kernel, and an interleaved absorbed/expanded/absorbed regression verifies stable grouping, batched folded projections and original-order assembly.

The full `split_kv` filter produced five CPU-safe passes. Four custom-GPU merge cases cannot run on this CPU host because `paged_attention_merge_states` has no CPU port; that backend boundary, together with real-checkpoint outputs, Metal logit traces and serving ladders, is explicitly deferred to #2587. Formatting, source and license checks passed, and final full-workspace, all-target Clippy with warnings denied passed in 32.90 seconds.

Review strengthened cache mismatch diagnostics to identify the family, request row and first missing or unexpected layer; the regression asserts the layer index explicitly. The final re-review found no remaining issues: stable grouping and order, batched folded projections, unchanged Stage 2 row planning and fallback, and the original single-row mathematics were all preserved. Local CI and review are the approved merge gate; native classification does not claim measured performance.

## Technical Decisions and Learning Points

Do not coerce different MLA formulations into the dense attention interface. Share the cache/attention operation at the latent boundary and keep surrounding weight-heavy operations batched. For ragged mixed-layout batches, group compatible rows only around projections, keep cache mutation and score/softmax row-local, then restore original order. Logical sequence position and live cache length serve different roles after trimming: rotary offsets follow the former, DSA fallback/selection follows the latter. Preserve cache layout per request instead of allowing another row to change it.
