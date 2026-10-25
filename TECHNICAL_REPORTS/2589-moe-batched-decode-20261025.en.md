# Technical Report: PR #2589 — Mixtral-style MoE batched decode

**Date:** 2026-10-25

**Related issues:** #2558, #2566, #2587

## Executive Summary

Seven routed-expert model families now execute embeddings, projections, normalization, experts and the output head across decode rows together. DeepSeek-OCR and OCR-2 delegate text decode to the same implementation. Hardware throughput acceptance remains pending in #2587.

## Problem Statement

These models previously inherited the trait's per-row fallback, repeating the complete forward and weight reads for each concurrent request. A native path must preserve each family's mathematical differences and the existing single-row fused-expert path.

## Change Summary

The implementation reuses the shared split-attention and routing helpers from #2555. Mixtral, PhiMoE, Phixtral, Qwen2-MoE, OLMoE, DBRX and DeepSeek v1 run their dense operations and existing expert modules on a batched hidden tensor, while attention updates each sequence's own cache. B=1 calls the unchanged single-sequence forward; empty batches retain their defined shape.

Family-specific behavior remains explicit: PhiMoE chooses LongRoPE tables per sequence, Phixtral retains f32 attention and parallel residual order, DBRX clips fused QKV before splitting, OLMoE preserves its Q/K normalization layout, and Qwen2-MoE/DeepSeek retain shared experts. DeepSeek preserves dense-first-layer dispatch before routed MoE layers. Both OCR wrappers forward their text-only decode to the tower and propagate its decode kind.

## Validation

Fifteen focused CPU tests passed: six shared MoE tests, two concrete OCR wrapper tests, one DBRX contract, one Phixtral contract and five coverage checks. Fixtures exercise three unequal prompt lengths, exact B=1 equality, B=0, mixed LongRoPE regimes, clipping that is actually activated, shared experts, a dense-to-MoE stack and repeated/disjoint expert routing. Compilation, formatting, source/license checks and workspace-wide all-target Clippy with warnings denied passed. Implementation and security/performance reviews found no blocking issues.

Metal, real-checkpoint greedy/logit comparisons and serving performance were explicitly deferred by the owner to #2587. Native classification describes the implemented execution path and does not claim a measured speedup.

## Technical Decisions and Learning Points

Reuse the expert implementation that already handles multiple tokens instead of duplicating routing logic. Tests need nontrivial routed weights and actual boundary conditions: a configured clip that never clips or a LongRoPE fixture that never switches tables cannot validate those branches. Concrete wrapper tests establish delegation at the callable boundary rather than matching source text.
