# PR #2611: Native batched decode for Gemma 4

**Date**: 2026-10-26
**Status**: Complete on Metal; ROCm and CUDA performance remain unmeasured
**Languages**: Rust, Markdown
**Risk Level**: Medium

## Executive Summary

PR #2611 (issue #2546) replaces Gemma 4's per-row serving decode with one native batched transformer graph for identified single-token rows. Embeddings, projections, normalization, dense and MoE feed-forward work, per-layer inputs, and the LM head now execute over `[B, 1, hidden]`; attention and mixed full/sliding cache updates remain row-local. On an Apple M3 Ultra, aggregate output at concurrency four rose from 51.4 to 102.7 tok/s for Gemma 4 12B and from 154.2 to 189.8 tok/s for Gemma 4 26B A4B, while concurrency-one throughput remained within 1.5% of baseline.

The implementation also hardens the shared sequence-state boundary: duplicate `SequenceId` values are rejected before mutable cache slices are reconstructed, and invalid model-owned batches no longer remove live state. Review also added an exact-byte regression proving that the existing compiled proportional-RoPE path handles uniform-offset BF16 decode batches correctly.

## 1. Problem Statement

Gemma 4 advertised batching support, but its VLM wrapper sliced each decode batch and invoked the complete text model once per row. Every row therefore reread the weights and repeated roughly 1,400 decode kernels, leaving aggregate throughput nearly flat as concurrency increased. Gemma 4 also combines sliding-window and full attention, proportional RoPE, KV-shared layers, sparse experts, and token-derived per-layer inputs, so replacing the fallback required preserving independent offsets and heterogeneous cache state rather than merely stacking input tokens.

The original ROCm measurement for the 26B A4B checkpoint stayed near 74–79 aggregate tok/s from concurrency one through eight. Without a native path, adding clients mainly divided the single-request decode rate and increased latency.

## 2. Change Summary

| Area | Change |
|---|---|
| Native text decode | `Gemma4Wrapper` identifies eligible `[B, 1]` calls with sequence IDs and runs a new batched decoder path. B=1 delegates directly to the existing single-sequence graph; empty batches return empty logits. |
| Transformer work | Token embedding, Q/K/V and output projections, norms, dense or SwitchGeGLU/MoE blocks, router work, per-layer projections, and the tied LM head remain batched. Row logits are joined with one `concatenate_many` operation. |
| Attention and cache | Each row retains its own full or rotating cache and logical position. KV-shared layers reuse the source layer's row-specific K/V and offset. Standard and proportional RoPE accept independent row offsets. |
| VLM and Unified wrappers | Text-only identified decode delegates to the native text path. Prefill, masks, missing IDs, and pending image/audio per-layer inputs retain the exact wrapper fallback. |
| Sequence-state safety | `CachePool` rejects duplicate IDs before creating mutable slices. `ModelOwnedSequenceState` prevalidates all IDs before removing entries, so duplicate or missing IDs cannot alias or partially discard live state. |
| RoPE correctness | Mixed offsets continue to use row-specific proportional RoPE, while uniform offsets retain the compiled full-batch shortcut. A realistic BF16 regression proves that shortcut exactly matches independent rows for B=2 and B=4. |
| Classification and docs | Gemma 4 text, VLM, and Unified models are classified and documented as native batched-decode families. |

## 3. Technical Decisions

### 3.1 Batch the weight-heavy graph, keep attention row-local

A paged-attention backend was not introduced. The chosen path amortizes weight reads and projection launches immediately while preserving the existing `Cache` variants and attention semantics. Each layer projects the full batch once, then slices Q/K/V by row for cache append and attention before concatenating the row results. This leaves some per-row launch overhead, but avoids a larger storage migration and directly addresses the dominant repeated-weight cost.

### 3.2 Preserve the exact B=1 and conditional wrapper paths

The one-row branch calls the pre-existing forward method instead of the new batched implementation. The VLM and Unified wrappers delegate only identified, unmasked, single-token text decode with no pending multimodal per-layer input. This keeps prefill, image/audio state consumption, anonymous model-owned state, and malformed or ineligible shapes on their established paths.

### 3.3 Retain the uniform-offset proportional-RoPE shortcut

A review hypothesis suggested that a transposed `[B, H, 1, D]` tensor with equal batch and sequence strides might make the compiled proportional-RoPE path rotate row `b` at `offset + b`. A realistic test with Gemma 4 geometry (`head_dim=512`, partial factor 0.25), BF16, non-contiguous decode layout, and B=2/B=4 disproved the hypothesis: the uniform full-batch result was byte-for-byte identical to independent row calls. The production shortcut remains unchanged, and the positive regression now pins that behavior.

### 3.4 Validate IDs before unsafe borrowing or destructive extraction

`CachePool::get_batch_caches_mut` reconstructs mutable slices from raw pointers, so a repeated ID would create overlapping mutable references and violate Rust's aliasing contract. The pool now rejects duplicates before collecting pointers. The model-owned state container likewise checks the entire ID set before removing any entry; an error can no longer strand an earlier row outside the map.

## 4. Validation and Measured Results

### 4.1 Functional coverage

The checkpoint-free regression uses three rows with different prompt lengths and covers sliding attention beyond its window, proportional full attention, a KV-shared layer, MoE experts, token-derived per-layer inputs, B=1 exact dispatch, and the empty-batch contract. Separate tests cover anonymous fallback, pending multimodal input, native classification, duplicate pool IDs, duplicate and missing model-owned IDs without state loss, and uniform-offset proportional RoPE in the transposed decode layout.

The full workspace run reported 13,811 passed, 10 failed, and 441 ignored tests. A separate clean-baseline run at `01047d75` reproduced all 10 failures with the same values and source lines; they are existing cross-backend BF16 oracle failures rather than regressions from this PR. The focused duplicate-ID `CachePool` regression passed, as did all seven model-owned state tests, including the new invalid-ID state-retention cases. The native mixed-cache/MoE/per-layer-input/B=1/empty regression also passed after the sequence-state fixes. The realistic B=2/B=4 uniform-offset proportional-RoPE regression passed with exact byte equality. Formatting and clippy passed at implementation revision `a9fcef02`; the final release build remains with the parent finalization workflow.

### 4.2 Metal serving throughput

Measurements used an Apple M3 Ultra with 512 GiB unified memory, release builds with `metal,accelerate`, and 445-token prompts followed by 128 requested output tokens. The baseline was `01047d75`; the measured implementation was `a9fcef02`, before the later sequence-state hardening changes.

| Checkpoint | Revision | C=1 | C=2 | C=4 | C=8 |
|---|---|---:|---:|---:|---:|
| Gemma 4 12B dense / Unified | Baseline | 50.5 | 51.3 | 51.4 | 49.5 |
| Gemma 4 12B dense / Unified | Native batch | 49.9 | 78.3 | 102.7 | 110.8 |
| Gemma 4 26B A4B MoE | Baseline | 95.8 | 127.8 | 154.2 | 127.1 |
| Gemma 4 26B A4B MoE | Native batch | 94.4 | 145.0 | 189.8 | 220.7 |

At C=4 the native path reaches 2.06 times its own C=1 aggregate rate for 12B and 2.01 times for 26B. Relative to the old C=4 path, this is a 2.00× and 1.23× improvement respectively. Greedy `Hello` generation was byte-identical on both checkpoints and stopped at EOS after nine generated tokens.

### 4.3 Numerical behavior

Teacher-forced B=4 versus B=1 traces at decode width one showed 19/256 top-1 disagreements and 2/152 decided-position disagreements for 12B, and 25/256 plus 3/157 respectively for 26B. These results must not be described as bit-exact batched inference. With `MLXCEL_QMV_WIDE=0`, the 12B trace became identical at all 256 positions, attributing its default differences to Metal's width-dependent quantized projection kernels rather than cache offsets. The 26B control retained 18/256 top-1 and 2/157 decided-position differences, consistent with different single-token fused-expert and batched `gather_qmm` reductions.

The old 12B Unified trace is not a valid identified-state baseline because its trait-default batch method discarded sequence IDs after identified prefill. B=1 preservation is instead established by direct single-request generation and the exact one-row dispatch regression.

## 5. Learning Points

- A model can report batching support while still running one full graph per row. Decode classification needs to describe the graph actually executed, not merely whether batched input is accepted.
- Native batching does not require batching every operation. Keeping cache mutation and attention row-local can preserve complex cache semantics while still amortizing the weight-heavy majority of the decoder.
- Backend hypotheses need production-realistic geometry and dtype before they justify removing a fast path. Here, an exact-byte BF16 comparison disproved the suspected proportional-RoPE stride problem and preserved the compiled uniform-offset shortcut.
- APIs that reconstruct multiple mutable references from raw pointers must reject duplicate logical keys before borrowing. Validation after pointer collection is already too late.

## 6. Follow-up and Limitations

- Re-run the issue's MI300X ROCm ladder and collect CUDA GB10 results when those hosts are available. The Metal results do not establish cross-backend performance or numerical behavior.
- Tensor-parallel Gemma 4 and a paged decode backend remain out of scope.
- The per-layer attention result still performs row-local attention and then concatenates once; further batching should be driven by profiling rather than assumed to win.
- Keep monitoring decided-position divergence when Metal quantized kernels or MoE routing kernels change.

## References

- Issue #2546: Gemma 4 decode runs one forward per row
- PR #2611: batch Gemma 4 decode across active rows
- `docs/benchmark_results/gemma4-native-batched-decode-2026-10-26.md`: full protocol, trace interpretation, and raw comparison context
- `docs/CONTINUOUS_BATCHING.md`: serving decode classification and operational behavior
