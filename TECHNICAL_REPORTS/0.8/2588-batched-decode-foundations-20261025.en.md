# Technical Report: PR #2588 — Batched decode foundations

**Date:** 2026-10-25 · **Status:** Implementation reviewed; integrated Metal acceptance tracked in #2587 · **Related issues:** #2555, #2566, #2587

## Executive Summary

The change centralizes the split-attention operations shared by Llama3, Qwen3 and Qwen3-MoE, adds explicit decode-path reporting, and supplies reusable regression tools for the remaining families in epic #2566. It does not establish new family throughput measurements.

## Problem Statement

A `forward_batched` method does not prove that a model shares its weight reads across decode rows. Several implementations split requests and execute complete single-sequence forwards, while the server previously did not distinguish those paths. Duplicated attention implementations and family-specific test scaffolding also made subsequent ports harder to verify consistently.

## Change Summary

- `mlxcel_core::batched_decode::SplitAttention` centralizes projection reshaping, optional Q/K normalization, RoPE, shared cache attention and family-specific row attention hooks.
- Routing helpers preserve empty-batch output and the exact single-sequence path while dispatching multirow work to the supplied native implementation.
- Trait markers propagate through loaded-model and vision wrappers into startup logging. Qwen3.5, Qwen4Exp, Muse Glimmer and pipeline serving remain fallback because their actual decode paths iterate full forwards per row.
- A source coverage ledger pins native, delegating and fallback implementations and rejects native or delegating markers without a batched override. Paths are normalized for Windows.
- Shared harnesses compare ragged rows with independent decoding and check cache advancement, single-row equality and empty batches.
- `MLXCEL_TRACE_BATCH_ROWS` traces teacher-forced width-one decode. Row zero has identical input and context for B=1 and B=4; companion rows use distinct stream positions and prefix lengths. Each step is evaluated and row zero is emitted immediately with its original position index, avoiding retained full-batch logit buffers.

## Technical Decisions

Path classification follows the serving path with sequence IDs, not the existence of a method or a prefill-only optimization. Review corrected initially optimistic classifications for Muse Glimmer and Qwen4Exp. Pipeline scheduling likewise remains fallback. Performance benefits must be demonstrated separately on Metal; no local CPU test is presented as that evidence.

## Validation

Fourteen focused CPU tests passed: coverage (4), Llama3/Qwen3 harnesses (2), Qwen3-MoE harnesses (3), pipeline classification (1), and trace planning and streamed output (4). Temporarily removing the Qwen3-MoE native override and Qwen3.5-VL delegating override produced the expected named failures; restoring them returned the guard to green. Formatting, workspace-wide all-target clippy, source contracts, 139 Python tooling tests, license headers, dependency policy, binary asset and Windows path checks passed.

The owner selected local CI plus review as the merge gate and deferred real-checkpoint generation, startup capture, B=1/B=4 traces and Metal concurrency ladders to #2587. These measurements remain pending after implementation merge.

## Learning Points

Batching capability, batch-shaped inputs and native batched decode are different properties. A source guard detects marker/override drift, but reviewers must still inspect the scheduler's real path. Trace tooling also needs bounded graph construction and linear accumulation work to avoid measuring its own overhead.
