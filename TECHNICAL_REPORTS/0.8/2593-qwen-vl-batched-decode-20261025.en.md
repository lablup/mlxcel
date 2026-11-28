# Technical Report: PR #2593 — Qwen VL batched text decode

**Date:** 2026-10-25
**Related issues:** #2562, #2566, #2567, #2587

## Executive Summary

Qwen2-VL, Qwen3-VL and Qwen3-VL-MoE text towers batch decode operations by positional-encoding mode. Five Qwen VL/Omni wrappers reach these paths through the real sequence-ID entry point. Metal performance and checkpoint acceptance remain deferred to #2587.

## Problem Statement

The prior wrapper dispatch preserved request-specific MRoPE state by calling each row separately, repeating weight projections and feed-forward work. Combining every row under one positional encoding would change the computation for text-only requests, so batching must preserve the original per-request choice.

## Change Summary

The towers partition decode rows into plain-RoPE and MRoPE groups, batch embedding, projections, feed-forward or expert operations, final normalization and logits within each group, and scatter outputs back to request order. Attention and cache updates remain request-specific. MRoPE tables are computed once per group and reused across layers. Singleton groups keep the original single-request call.

The shared MRoPE helper constructs positions from each request's cache offset and delta. Prefill, absent sequence IDs, stored-position coverage and model-global DeepStack state retain per-row dispatch. This preserves existing behavior but does not solve the global DeepStack isolation defect tracked by #2567.

The logit trace tool now prepares fresh sequence IDs. Its image-batch arm performs actual image prefill for row zero and separate text prefill for companions, then emits row-zero teacher-forced decode logits. Existing TP selection remains restricted to the text arm.

## Validation

CPU checks passed for all three tower parity fixtures, actual execution through five constructed wrappers, two MRoPE position tests, two exact DeepStack fallback tests, eight source-coverage tests and seven trace-example tests. Fixtures exercise unequal prompt lengths, distinct deltas, mixed groups, stored-position fallback and B0/B1 contracts. Compilation, formatting, source contracts and full workspace/all-target Clippy passed; the final Clippy run took 2m38s.

Final review results are recorded in the PR. Real image checkpoints, Metal traces, server concurrency and throughput remain unmeasured in #2587. Local CI and review are the approved merge gate, without awaiting hosted CI.

## Technical Decisions and Learning Points

Partition by the same predicate used by single-request execution, even where two positional encodings are mathematically equivalent. Preserve stored-position and global-state fallback explicitly. Runtime wrapper fixtures must call the real public entry point; marker declarations alone do not establish server integration.
