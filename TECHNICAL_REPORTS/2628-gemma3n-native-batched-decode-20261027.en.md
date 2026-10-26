# PR #2628: Native batched decode for Gemma 3n

**Date**: 2026-10-27
**Status**: Partial; final checkpoint and serving validation pending
**Languages**: Rust, Markdown
**Risk Level**: Medium

## Executive Summary

PR #2628 (issue #2561) replaces Gemma 3n's trait-default row-by-row server decode with a native batched path for active single-token rows. Embeddings, AltUp, LAuReL, token-derived per-layer inputs, projections, normalization, MLPs, and the tied LM head execute over the full `[B, 1, hidden]` tensor, while attention preserves each row's concrete or KV-shared cache state and sliding-window behavior.

The text model now reports `BatchedDecodeKind::Native`, and the Gemma 3n VLM delegates text-only batched decode to that path. Checkpoint-free E2B- and E4B-shaped regressions and decode-classification coverage are present. Final full-workspace checks, checkpoint greedy parity, teacher-forced logit traces, server-path confirmation, and serving throughput measurements are intentionally left pending until the final validation owner records them.

## 1. Problem Statement

Gemma 3n advertised batching support but did not override `LanguageModel::forward_batched`. The server therefore accepted multiple active rows and then invoked the complete single-sequence `forward` graph once per row, repeating weight reads and projection work instead of amortizing them across the batch.

The model cannot safely use a simple stacked forward. It combines AltUp planes, a LAuReL residual branch, token-derived per-layer inputs, alternating sliding and full attention, and logical layers that read K/V from earlier owner caches without updating them. E2B and E4B also use different numbers of KV-shared layers. A native path must retain row-specific offsets and cache ownership while preserving the established single-row graph exactly.

## 2. Change Summary

| Area | Change |
|---|---|
| Routing contract | `Gemma3nModel` reports native decode and uses `route_batched_decode`, preserving the shared B=0 empty result and direct B=1 single-sequence dispatch. |
| Batched transformer work | Embedding and scaling, AltUp predict/correct, LAuReL, per-layer input gather/projection, norms, attention projections, MLPs, final norm, softcap, and LM head remain on the full batch graph. |
| Concrete attention | `Gemma3nAttention` implements `SplitAttention` and calls the shared `forward_split_attention` helper. Sliding layers retain their row-local windowed attention behavior. |
| KV-shared attention | Each logical shared layer maps every row to its concrete owner cache, rotates queries at that owner's offset, reads the live K/V range without appending, and concatenates row results once. |
| Cache diagnostics | Shape, owner-cache count, missing-cache, and offset inconsistencies report the row and layer involved. |
| VLM integration | `Gemma3nVLModel` delegates batched text decode and its decode-kind classification to the text model; multimodal prefill remains on the existing embedding path. |
| Tests and documentation | E2B- and E4B-shaped unequal-length regressions cover the sliding boundary and shared-layer layouts; coverage lists and `docs/CONTINUOUS_BATCHING.md` classify Gemma 3n text/VLM correctly. |

## 3. Technical Decisions

### 3.1 Batch the weight-heavy graph while keeping cache-sensitive attention row-local

The implementation reuses the shared split-attention boundary rather than introducing a new cache representation or paged backend. Q/K/V and output projections are computed for the batch, but cache append, live-range selection, and sliding attention preserve each sequence's independent history. This removes repeated transformer work without changing the cache contract used by single-sequence inference.

The trade-off is that attention still contains a per-row portion. This is deliberate: a broader storage redesign is outside issue #2561, and any additional batching should be justified by the pending serving measurements and later profiling.

### 3.2 Treat KV sharing as read-only ownership mapping

Later Gemma 3n layers do not own caches. The batched path uses `layer_idx_to_cache_idx` for every row, reads the owner's live K/V tensors, and applies the owner's logical offset to RoPE without calling cache update. Centralizing this in one helper prevents each shared layer from reimplementing the row loop and, critically, avoids appending the same token more than once to an owner cache.

### 3.3 Preserve B=1 through routing, not approximate parity

`route_batched_decode` sends one-row input directly to the pre-existing `forward` method. Exact single-row behavior therefore does not depend on numerical tolerance between differently shaped kernels. The new implementation is used only for two or more rows, while an empty batch follows the shared empty-logit contract.

### 3.4 Keep the VLM boundary narrow

The VLM wrapper delegates `forward_batched` to the text model because decode rows without input embeddings are exactly text-model calls. Image processing and multimodal per-layer input construction remain unchanged in prefill, avoiding an unrelated expansion of the native decode path.

## 4. Validation Status

### 4.1 Completed before final reporting

- Focused Gemma 3n tests passed: 3 tests.
- Batched-decode coverage tests passed: 12 tests.
- `cargo check --lib` passed.
- `cargo fmt --all -- --check` passed.
- The final release build passed in 12 minutes 37 seconds.
- The full workspace suite completed with 13,826 passed, 10 failed, and 441 ignored tests. The 10 vision-test failures are under baseline comparison, so this report does not yet classify them as pre-existing or as regressions.
- Implementation and security reviews reported no required fixes.
- `git diff --check origin/main...HEAD` passed during finalization.

The native regression runs three rows with distinct prompt lengths below, at, and beyond a four-token sliding boundary. It exercises one KV-shared layer for an E2B-shaped configuration and two for an E4B-shaped configuration, plus full attention, sliding attention, token-derived per-layer input, multi-row sequential parity, exact B=1 dispatch, and B=0 output. Static coverage also pins the text model as native and the VLM as delegating.

### 4.2 Pending final validation

- Clippy with warnings denied and baseline classification of the 10 full-suite vision-test failures.
- Byte-identical greedy B=1 generation on the Gemma 3n E2B and E4B checkpoints.
- Teacher-forced B=4 versus B=1 logit traces, including decided-position disagreement counts.
- Server startup confirmation that both checkpoints select `decode path: native batched forward`.
- Metal serving concurrency measurements against the main-branch baseline, including the B=1 regression check and B=4 aggregate throughput.

No performance or checkpoint-parity result is claimed until those measurements are attached to the PR by the final validation owner.

## 5. Learning Points

- A batching capability flag does not prove that one batched model graph runs; decode-kind coverage must classify the actual execution path.
- Native decode can batch most transformer work without merging independent sequence caches. A split boundary around attention is a practical intermediate architecture for complex cache semantics.
- Logical KV-sharing layers need both the owner's tensors and the owner's position. Reusing K/V with a separately derived logical-layer offset would silently rotate queries at the wrong location.
- Exact B=1 preservation is strongest when enforced by control flow. It avoids treating a tolerance-based comparison between different kernel shapes as proof of graph identity.

## 6. Follow-up and Limitations

- Record the pending checkpoint, trace, server-path, and concurrency evidence in PR #2628 before merge.
- The path remains row-local inside attention and does not add paged-backend support; both are outside this PR.
- The VLM change covers text-only decode delegation, not batching of the audio or vision towers.
- Backend-dependent reduction order may make multi-row logits numerically close rather than bit-identical; the pending trace results must describe any observed disagreement precisely.

## References

- Issue #2561: batched decode forward for Gemma 3n
- PR #2628: batch Gemma 3n decode across active rows
- Issue #2546 / PR #2611: Gemma 4 native batched decode reference design
- `docs/CONTINUOUS_BATCHING.md`: serving decode classification and behavior
