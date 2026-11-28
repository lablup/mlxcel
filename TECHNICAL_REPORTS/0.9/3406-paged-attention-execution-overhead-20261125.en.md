# PR #3406: paged attention execution overhead

**Date**: 2026-11-25
**Status**: Completed implementation; platform validation limits below

## Executive Summary

The full #3385 change reduces separate GPU helper kernels and excessive CUDA graph splitting. Qwen3 8B improved 1.16% on GB10 at 8192 input tokens and B=4. The owner accepts the measured Qwen3 0.6B regression of 1.16–1.36%; this is not a universal throughput or FLOPs reduction claim.

## Problem Statement

Registering whole paged KV slabs counts their reserved size against the CUDA graph byte budget even when attention reads only the selected pages. On GB10 this can commit graphs after individual attention launches. A separate Q-to-f32 cast and GPU scale fill also add per-layer work.

## Change Summary

The CUDA custom-kernel overlay excludes paged-attention inputs from byte-budget accounting while preserving dependency tracking and output accounting. The shared v2 kernel widens Q while staging it and accepts a scale array initialized from host data. A byte-equality regression test covers fp16 Q versus its explicit f32 cast at B=1 and B=2.

Q conversion still occurs, potentially once per partition; its arithmetic has not been eliminated. The benefit is removing a standalone cast, an intermediate f32 array, a fill kernel and excessive graph submissions. Graph grouping does not change attention mathematics.

## Technical Decisions

Choose the complete change rather than the budget-only variant: 8B full-change per-round gains were +0.86%, +1.24%, +1.12%; budget-only pooled +0.12% overlapped controls and its per-round changes were -1.23%, +0.06%, +0.19%. The original no-regression acceptance expectation is explicitly superseded by the owner's acceptance of the small-model tradeoff.

The earlier profiling counts came from the 0.6B experiment and the separate sliding-window branch, not from 8B. Benchmark binaries precede latest-main integration; current integration was checked with focused tests, not another performance run.

## Validation

After merging main at 7a66c862: 256 focused CUDA tests passed, 1 ignored; the new fp16 equality test ran. Core clippy, formatting, kernel dtype-key checks and port-dispatch checks passed. All 144 measured 8B requests completed 3072 output tokens and matched corresponding control text hashes. Remote CI is cancelled under the local-validation policy; cancellation is not a passing CI result. Metal and HIP execution remains unverified.

Raw data and binary hashes: [benchmark record](../../docs/benchmark_results/issue-3385-qwen3-8b-gb10-2026-11-24/README.md).

## Follow-up

This PR closes #3385 only. #3334 remains a stopped draft; #3256, #3257, #3259 and #3250 remain open. #3387 tracks other pool-slab kernels. The sliding-window B=1 regression, server acceptance tests, Gemma 4 migration and replay acceptance are not resolved by this isolated optimization.
