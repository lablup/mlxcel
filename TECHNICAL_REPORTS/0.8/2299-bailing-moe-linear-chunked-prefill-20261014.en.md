# PR #2299: Correct the Bailing MoE Linear no-ID prefill contract

**Date**: 2026-10-14

**Status**: Complete

**Risk**: Medium (prefill policy and long-prompt transient memory)

## Summary

Bailing MoE Linear now declares `supports_chunked_prefill() == false`. A multi-token call without a sequence ID deliberately replaces its model-owned recurrent and KV state so independent prompts do not inherit history; advertising chunk continuation to no-ID callers could therefore discard earlier chunks.

## Problem

The model inherited the trait default of chunked-prefill support even though `forward_for_sequence` resets fallback state at the start of every multi-token no-ID forward. Current DirectEngine uses a stable sequence ID and avoids that reset, so this change does not claim that its previous state handling truncated prompts.

## Design

- Override the capability while preserving independent no-ID prompt reset and stable-ID continuation.
- Load the existing deterministic nonzero six-layer fixture through `BailingMoeLinearModel::from_weights`; it contains four recurrent linear-attention layers and two global-attention layers.
- Force no-ID two-token and three-token forwards, then verify layer 0 ends at offset 3 rather than 5.
- Force prepared stable-ID two-token and three-token forwards, then verify the same recurrent layer continues to offset 5.

## Verification

The root and surgery CUDA `test-fast` build graph passed in 14 minutes 20 seconds. The GPU-locked `bailing_moe_linear_loaded_model_chunking_contract_is_safe` regression passed 1 of 1 tests in 0.78 seconds. The source also passes `git diff --check` and independent implementation review. The acceptance fixture is synthetic and checkpoint-shaped; no external checkpoint run is claimed.

## Tradeoff

The capability cannot express support conditional on a sequence ID. Every honoring caller, including DirectEngine, now receives a whole-prompt prefill plan even though DirectEngine's stable-ID state handling is safe. This can increase peak transient memory for long prompts compared with chunk-bounded prefill.

## Change summary

The model adds one capability override, factors the existing deterministic nonzero fixture fill into a helper, and adds one loaded-model state-contract regression without new dependencies.

## Final integration validation

Integrated main `b159c89d`, including MLX `243bdf09`, mlxcelverse, and the preceding inference fixes, without source conflicts. The rebuilt CUDA `test-fast` binary passes all 36 `models::bailing_moe_linear` tests, including the loaded six-layer request-state regression. This freshly checks no-ID prompt reset and stable-ID continuation on the final pin. Formatting and diff checks pass. The model uses synthetic loaded weights; no external-checkpoint inference result is claimed.
