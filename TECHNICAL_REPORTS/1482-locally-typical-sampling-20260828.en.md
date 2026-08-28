# Technical Report: PR #1482 - Locally typical sampling (typical_p)

**Date**: 2026-08-28

**Status**: Completed

**Languages**: Rust

**Risk Level**: Medium

## Executive Summary

Adds locally typical sampling (`typical_p`, default `1.0` disabled) on the #1375 row-filter hook: tokens whose surprisal is closest to the row entropy are kept in typicality order until `typical_p` probability mass accumulates. Unlike the two earlier filters this one flips its two llama-compat manifest entries (`--typical`, `field:typical_p`) to `supported`, which pulled the server-wide flag on both binaries and the native `/completion` field into the same PR.

## 1. Problem Statement

mlxcel had no typical sampler, and the `--typical` / `field:typical_p` manifest entries are owned by issue #1377 itself: the epic's closure rule requires every entry owned by an issue to leave `deferred` before that issue closes, and the llama-compat gates tie a supported option claim to both server binaries and a supported field claim to `NativeCompletionRequest`. The scope therefore had to cover the engine semantics, both flag surfaces, the native field, and the manifest flip in one change.

## 2. Technical Decisions

### 2.1 Chain position: an idempotent top-k pre-mask

The review caught the load-bearing subtlety: b10621's chain is `top_k -> typ_p` and its typical sampler re-softmaxes the surviving candidates, so entropy comes from the renormalized top-k distribution. mlxcel's hook runs before the C++ chain's top_k, so the filter initially computed entropy over the full vocabulary, diverging for essentially every enabled request (server default top_k 40). The fix masks the row to top-k ahead of the typical filter whenever `1 < top_k < vocab`; masking to top-k is idempotent, so the C++ chain's own top_k re-selects the same set and no stage runs out of order. A regression test uses a row where the two orders provably disagree (full-vocab typicality keeps only token 1, top-k-renormalized typicality keeps only token 0).

### 2.2 Value domains split by surface

OpenAI-shaped endpoints validate `(0.0, 1.0]` and return 400 otherwise. The native `/completion` field mirrors b10621's limit-free schema: a present out-of-domain value resolves to the explicit disabled `1.0`, still overriding the server default the way an upstream request value replaces it. The server-wide `--typical` flag accepts what b10621 accepts but folds out-of-domain values to disabled at startup with a warning, and `/props` reports the resolved value so the operator can read it back. The CLI generate/chat flag rejects out-of-domain values at parse time.

### 2.3 Effective-value normalization and serde default

`FusedSampleParams::from_config` and the speculative-window `sampling_config_eq` compare `effective_typical_p()` (greedy and out-of-domain fold to `1.0`), so necessarily-identical rows stay on the fused batch, pipelined lookahead, and shared speculative windows. The disaggregated wire struct defaults the field to `1.0` through a serde default fn, because the f32 zero default would turn an older peer's frame into an invalid always-on cutoff.

## 3. Change Summary

| Item | Value |
|------|-------|
| Files changed | 35 (+8 in the review follow-up) |
| Lines | +884 / -33 across both commits |
| Tests added | 25 |

Validation: 91 core sampling tests green including a 40-case f64 host-reference test; compat integration test validates the flipped `--typical` claim against both freshly built binaries; manifest checker green with zero remaining #1377-owned deferred entries; real-checkpoint check on Qwen3-4B-4bit (fluent at `typical_p 0.5` with and without `top_k 40`, greedy byte-identical control).

## 4. Follow-up Actions

- #1373 (`p_less`) fills the remaining slot in `apply_row_filters`.
- #1436 completes the remaining b10621 sampling semantics; the `+inf` handling difference between `typical_p_filter` (masked) and `top_n_sigma_filter` (kept) is documented in the filter docs.
