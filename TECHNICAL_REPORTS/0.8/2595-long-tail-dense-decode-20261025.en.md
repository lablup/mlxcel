# Technical Report: PR #2595 — Long-tail dense batched decode

**Date:** 2026-10-25

**Related issues:** #2557, #2566, #2587

## Executive Summary

Thirteen previously per-row decoder families now execute a native batched decode graph: BitNet, GPT-2, GPT-BigCode, GPT-NeoX, MiniCPM, MiniMax M3, NanoChat, Nemotron, OLMo, OpenELM, Phi, Phi-3-small and StableLM. MiniMax M3 VL delegates text-only decode to the same native text-tower path. The conversion preserves exact B=1 routing and family-specific position, normalization, residual, scaling and attention behavior while moving weight-heavy projections over the full active-row batch.

The “long-tail dense” grouping describes the issue wave, not every architecture literally. MiniMax M3 remains its current hybrid dense/MoE decoder with block-sparse attention; its MoE selection and experts are unchanged, while its attention and sparse-index projections gain the native batch path. No throughput improvement is claimed without the deferred Metal acceptance work in #2587.

## Problem Statement

These models reported batching support but inherited the trait fallback that invoked one single-sequence forward per active row. A server decode tick therefore reread the same layer weights and rebuilt projection graphs B times. A shared implementation still had to preserve several incompatible details: learned absolute positions in GPT-2 and GPT-BigCode, partial RoPE in GPT-NeoX, Phi and StableLM, depth and muP scaling, OpenELM's per-layer head widths, and Phi-3-small's mixture of dense and block-sparse attention layers.

MiniMax M3 added a second constraint. Its sparse index keys and queries must be projected once for the batch but consumed with the correct request-owned cache row. Its existing single-row sparse path also exposed a historical shape error: after appending an index-key head, `KVCache` allocated the value storage with the augmented key-head count, so attention could receive one real K/V head but two value heads.

## Technical Decisions

### Preserve the shared split-attention contract

Each family implements `SplitAttention`, builds per-layer cache metadata with family and layer diagnostics, and uses `forward_split_attention` instead of introducing a second row loop. `route_batched_decode` keeps B=0 on the defined empty result and B=1 on the family's original single-sequence graph. The native B>=2 closure batches embeddings, normalization, QKV and output projections, MLPs and final logits.

GPT-2 and GPT-BigCode gather row-specific learned positions from distinct cache offsets, including offsets near the configured position limit. GPT-NeoX, Phi and StableLM rotate only their configured leading RoPE dimensions, leaving passthrough channels bit-exact. OpenELM derives the split shape from each layer so varying query/KV head widths are not flattened into a model-wide assumption. Phi-3-small dispatches each row through the same dense or block-sparse attention implementation selected by that layer while keeping the surrounding projections batched.

### Add a backward-compatible indexed-row hook

`SplitAttention::attend_row_indexed(row, ...)` defaults to the existing `attend_row(...)`, so every prior implementation retains its behavior. MiniMax M3 overrides only this hook. It computes index-key and index-query projections once on the normalized batch, slices them by the row index supplied by the shared helper, and then applies the existing request-local cache and sparse-mask logic.

The MiniMax cache continues to retain the extra index-key head needed for scoring. Only the value tensor view passed to attention is sliced back to `num_kv_heads`. This fixes the observed single-row K/V head-shape failure as well as the batched path without deleting sparse index state. The corrected single-row path is therefore a deliberate exception to the otherwise unchanged B=1 computation, and the actual MiniMax M3 VL wrapper now delegates to the corrected native text model.

## Validation

The serial native-decode filter passed 53 tests. The new family regressions use nontrivial deterministic weights and three unequal prompt histories, then verify cached B=3 decode against sequential execution, exact B=1 routing and B=0. Separate focused checks cover learned positions, partial-RoPE passthrough, OpenELM per-layer shapes, Phi-3-small dense and sparse layers, MiniMax mixed cache lengths, the actual MiniMax M3 VL wrapper, and the shared indexed hook's custom and default behavior. The coverage ledger passed all 7 tests, and the core indexed-hook filter passed both tests.

The first Phi-3-small fixture produced non-finite values in both sequential and batched references because its tiny block-window configuration was degenerate for the selected short histories. The enhanced parity diagnostic reported zero finite max-absolute difference, ruling out a batch-only divergence. The fixture was replaced with finite prefill histories and a decode position that actually enters block-sparse attention; the sensitive parity test then passed without relaxing tolerance.

Before the MiniMax value-view fix, the full-model witness failed with a key/value head mismatch after the sparse index head entered the cache. After retaining the index-key cache head and trimming only the attention value view, cached decode with three mixed prompt lengths passed against the corrected sequential reference. The sparse additive-mask path is exercised; the optional fused paged path correctly reports that the tiny test budget is not serviceable and falls back.

The final repository source gates passed: workspace crate versions, documentation coverage for 320 `MLXCEL_*` names, 15 environment-documentation tests, and the Llama compatibility manifest plus its validator cases. Full-workspace, all-target Clippy with warnings denied passed in 55.33 seconds, and formatting and diff checks passed. The final architecture, correctness, security and performance review reported no findings.

BitNet's CPU-safe test covers native classification, B=3 routing, exact B=1 and B=0 through embedding, normalization and logits, but the CPU build has no BitLinear ternary kernel or graph fallback. A full ternary-layer parity test is feature-gated for a supported Metal, CUDA or ROCm backend. That runtime evidence, real-checkpoint logit traces, greedy generation and serving throughput remain pending in #2587.

## Learning Points

A batching marker is only trustworthy when the weight-heavy graph is batched and request-owned mutable state remains row-aligned. Auxiliary projections such as sparse indexers belong before the row split; cache mutation and row-specific masks remain after it. A default trait hook can extend the shared mechanism without forcing unrelated families to reimplement existing behavior.

Parity diagnostics must distinguish arithmetic divergence from invalid fixtures. Reporting finite maximum absolute and relative differences made it clear that the initial Phi-3-small failure was shared non-finite output, not a looser-tolerance problem. Likewise, exact B=1 compatibility does not mean preserving a demonstrably shape-invalid historical path; the MiniMax correction is recorded explicitly and covered at runtime.

## Change Summary

| Item | Value |
|------|-------|
| Files changed | 28 |
| Lines added | 2,387 |
| Lines deleted | 35 |
| Native families added | 13 |
| Delegating wrappers added | 1 |
| Commit | `3c58bcee` |

The implementation updates the shared batching helper, thirteen model runtimes, MiniMax's sparse index/cache adapter, the MiniMax M3 VL wrapper, focused runtime tests, the classification ledger and continuous-batching documentation. Metal and real-checkpoint acceptance remain the only required follow-up and are tracked by #2587.
