# Technical Report: PR #2592 — Native TP batched decode

**Date:** 2026-10-25

**Related issues:** #2565, #2566, #2546, #2587

## Executive Summary

The tensor-parallel runtimes for Llama, Qwen 3, Qwen 3.5, Gemma 3, ERNIE 4.5 and Hunyuan v1 Dense now execute decode as one native batch per rank and layer. The implementation retains exact single-row routing, the established rank-reduction order and precision, and sequence-isolated caches and recurrent state. Metal checkpoint parity and throughput remain explicitly deferred to #2587.

## Problem Statement

These TP runtimes advertised batching but decoded rows through repeated single-row forwards. Each server step therefore reread rank-local weights and repeated collectives for every request. Qwen 3.5 added a harder boundary: its full-attention KV caches and gated-delta convolution/recurrent state must remain associated with each external sequence cache through creation, mixed established/fresh batches, release and reuse.

## Change Summary

Each converted runtime transposes request-owned caches into per-rank, per-layer columns. Norm, attention, MLP and output projections operate on the full row batch for each rank. Rank outputs are reduced in the same rank order and with the same normal or f32 reduction policy as the original single-row forward, preserving the existing arithmetic contract. The shared batch router keeps B=1 on the original forward and returns the defined empty result for B=0.

Qwen 3.5 batches gated-delta projections while applying convolution and recurrent updates per row to separately owned state; full-attention layers use batched split attention. State lookup remains keyed by the external cache identity, and the server sequence-ID lifecycle releases it when a row is removed. The single-device Qwen 3.5 runtime remains classified as a per-row fallback. TP Gemma 4 is also excluded because its prerequisite #2546 is still open.

`MLXCEL_TRACE_TP_SIZE` extends `examples/logit_trace` through the production tensor-parallel loader. Using the same TP size with `MLXCEL_TRACE_BATCH_ROWS=1` and `=4` enables directly comparable teacher-forced traces for future hardware acceptance; unset preserves the previous single-device loader.

## Validation

Twelve synthetic runtime tests cover TP sizes 1 and 2 for all six families. Every case exercises three unequal prompt histories, exact B=1 routing, B=0 and the server sequence-ID entry point. A separate Qwen 3.5 test covers mixed new and established rows plus released-state reuse. Two Qwen3Next tests, three GatedDelta cache tests, seven logit-trace example tests and six full coverage-ledger tests passed on the warm debug CPU build.

Formatting and source gates passed, including repository versions, environment-variable documentation and Llama compatibility. Full-workspace, all-target Clippy with warnings denied passed in 1 minute 2 seconds. The final architecture, mathematical-correctness, security and performance review reported no findings.

An initial cold `test-fast` attempt stopped before test execution because the host could not resolve the MLX dependency endpoint. It is not counted as test evidence. No Metal checkpoint trace, greedy-text comparison or serving ladder was run; those hardware checks remain in #2587, and native classification does not claim a measured speedup.

## Technical Decisions and Learning Points

Tensor parallelism changes where weights live, not the sequence-ownership contract. Batch the weight-heavy graph within each rank, keep per-sequence cache and recurrent mutation isolated, and preserve collective ordering and precision. A production-loader trace selector is necessary because changing only the batch width on a single-device model does not exercise TP code or its collectives.
