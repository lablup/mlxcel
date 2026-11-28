# Technical Report: PR #2594 — Additional MoE batched decode

**Date:** 2026-10-25
**Related issues:** #2559, #2566, #2587

## Executive Summary

GLM-4 MoE, ERNIE 4.5 MoE, Hunyuan MoE, Bailing MoE, Dots1, MiniMax, Klear, Solar Open, Exaone MoE and Cohere 2 MoE gain native batched decode. Existing expert implementations process the flattened batch without new kernels. An existing Exaone trait-adapter cache-loss defect is also corrected.

## Problem Statement

Per-row decode repeated model projections and expert launches. Batching must retain family-specific Q/K normalization order, partial rotary channels, shared experts, sliding attention and Hunyuan cross-layer K/V sharing while keeping request caches independent.

## Change Summary

Shared split attention batches projections and delegates ragged cache attention by request. Sliding layers retain their windowed row operation; review corrected Exaone and Cohere global layers to use the shared batched cache-dispatch path. Hunyuan carries raw batched K/V from each owning layer to its sharing layers, applies each layer's rotary/normalization rules, and updates every layer's separate request cache. Diagnostic errors include the layer index.

Exaone previously allocated fresh temporary caches on each trait call and copied back only offsets. Its initial regression observed offset 1 where 4 was expected. The corrected trait path retains standard KV history and matches the existing persistent `AnyKVCache` path through seven decode steps beyond a window of four. This intentionally changes historical B=1 behavior; the exact single-row routing contract applies to the corrected path. Standard sliding caches retain O(context) physical storage rather than a rotating O(window) buffer, visible to existing KV accounting and limits.

## Validation

The native-batch filter passed 52 tests, including routed three-unequal-row/B1/B0 fixtures for all ten families; the filter passed again after review corrections. Eight coverage checks, the Hunyuan missing-owner diagnostic and the dedicated Exaone persistent-cache reference test passed. Hunyuan tests include two owner/sharer groups; GLM exercises partial RoPE/QK normalization and the sliding families cross their window boundary. Compilation, formatting and source gates passed. Full workspace/all-target Clippy and final review results are recorded in the PR.

Metal kernels, checkpoints, real server ladders and throughput remain unmeasured in #2587. That issue explicitly records the Exaone baseline exception. Local CI and review are the owner-approved merge gate, without waiting for hosted CI.

## Technical Decisions and Learning Points

A batch-versus-single test can reveal pre-existing single-request state loss. Validate that correction against an independent persistent-cache path rather than only comparing two newly changed methods. Keep custom row attention limited to layers that actually require it so global layers retain shared cache-storage dispatch.
