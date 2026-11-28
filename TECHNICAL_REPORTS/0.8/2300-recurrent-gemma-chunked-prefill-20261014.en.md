# PR #2300: Correct the RecurrentGemma no-ID prefill contract

**Date**: 2026-10-14

**Status**: Complete

**Risk**: Medium (prefill policy and long-prompt transient memory)

## Summary

RecurrentGemma now declares `supports_chunked_prefill() == false`. A multi-token call without a sequence ID deliberately replaces its model-owned RGLRU state so independent prompts do not inherit history; advertising chunk continuation to no-ID callers could therefore discard earlier chunks.

## Problem

The model inherited the trait default of chunked-prefill support even though `forward_for_sequence` resets fallback state at the start of every multi-token no-ID forward. Current DirectEngine uses a stable sequence ID and avoids that reset, so this change does not claim that its previous state handling truncated prompts.

## Design

- Override the capability while preserving independent no-ID prompt reset and stable-ID continuation.
- Load a deterministic nonzero one-layer recurrent Griffin model through `GriffinModel::from_weights`.
- Populate fallback convolution and RGLRU state with a single-token forward, then prove a new multi-token no-ID prompt matches the same prompt on a fresh model.
- Split a five-token prompt into two and three tokens under one stable sequence ID, then compare the continuation logits with the matching suffix of a one-pass prompt using absolute or relative tolerance below `1e-3`.

## Verification

The root and surgery CUDA `test-fast` build graph passed in 14 minutes 25 seconds. The GPU-locked `recurrent_gemma_` focused suite passed 5 of 5 tests in 0.84 seconds, including the loaded-model contract and existing model-owned state tests. The source also passes `git diff --check`. The acceptance fixture is synthetic and checkpoint-shaped; no external checkpoint run is claimed.

## Tradeoff

The capability cannot express support conditional on a sequence ID. Every honoring caller, including DirectEngine, now receives a whole-prompt prefill plan even though DirectEngine's stable-ID state handling is safe. This can increase peak transient memory for long prompts compared with chunk-bounded prefill.

## Change summary

One Rust model file adds the capability override, a deterministic loaded-model fixture, and one state-contract regression without new dependencies.

## Final integration validation

Integrated main `5f334629`, including MLX `243bdf09`, mlxcelverse, and preceding model fixes, without source conflicts. The rebuilt CUDA `test-fast` binary passes all seven `models::recurrent_gemma` tests, including the loaded nonzero-weight RGLRU fixture. This freshly checks convolution/recurrent state population, no-ID prompt reset, and stable-ID split logits against one-pass execution on the final pin. Formatting and diff checks pass. No external-checkpoint inference result is claimed.
