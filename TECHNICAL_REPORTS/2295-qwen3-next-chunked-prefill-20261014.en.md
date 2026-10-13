# Technical report: PR #2295 — Qwen3-Next chunked-prefill opt-out

**Date**: 2026-10-14

**Status**: Completed

**Risk level**: Medium

## Executive summary

Qwen3-Next now advertises `supports_chunked_prefill() == false`. This prevents cache-less `LanguageModel` callers from splitting a prompt across no-sequence-ID forwards that each reset the model-owned GatedDelta and KV state. A one-layer GatedDelta model loaded through the production weight constructor verifies the capability and both state paths.

## Problem statement

`Qwen3NextModel::forward_for_sequence` treats every multi-token forward without a sequence ID as a new prompt. That reset isolates independent cache-less prompts, but a no-ID chunking caller would discard the state built by earlier chunks. The inherited trait default advertised chunking support despite this contract.

Current DirectEngine prefill passes one stable sequence ID through every piece, so it retains the per-sequence state. This change defines the conservative trait capability for legacy and alternate cache-less callers without changing the current server state lifecycle.

## Technical decision

The model opts out instead of changing `ModelOwnedSequenceState`. An explicit no-ID continuation mechanism would need a cross-family lifecycle signal that still resets between independent prompts. The local opt-out preserves that isolation, but it can replace per-chunk transient memory bounds with one graph over a long prompt.

## Change summary

- Added the Qwen3-Next `supports_chunked_prefill` override with the no-ID state rationale and DirectEngine distinction.
- Added deterministic nonzero weights for a real one-layer GatedDelta fixture loaded by `Qwen3NextModel::from_weights`.
- Forced two no-ID prompt forwards and verified the second prompt resets its cache offset to 3; forced two forwards with one stable `SequenceId` and verified continuation reaches offset 5.

## Validation

The CUDA `mlxcel` and `mlxcel-surgery` library test graph builds successfully. The focused `qwen3_next_loaded_model_chunking_contract_is_safe` test passes 1/1 under the GPU lock in 0.99 seconds. Rust formatting and diff checks pass, and independent implementation review found no correctness issue.

All required repository contracts pass, including 123 Python tests and 49 WebUI contract tests. An initial timing-sensitive GPU guard sample missed a short foreign process during concurrent native compilation; the isolated guard retry and broad contract rerun passed without code changes.

## Follow-up actions

- Parent integration validates the combined issue set after all authorized follow-up units land.

## References

- Issue #2285
- Issue #1749 and PR #2293
- Issue #672 and PR #1741
