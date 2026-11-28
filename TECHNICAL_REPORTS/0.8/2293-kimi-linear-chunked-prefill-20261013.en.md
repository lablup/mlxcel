# Technical report: PR #2293 — Kimi Linear chunked-prefill opt-out

**Date**: 2026-10-13

**Status**: Verified with runtime follow-up

**Risk level**: Medium

## Executive summary

Kimi Linear now advertises `supports_chunked_prefill() == false`. This prevents cache-less `LanguageModel` callers from splitting a prompt across no-sequence-ID forwards that each reset the model-owned Delta and MLA state. A tiny model loaded through the production weight constructor verifies the capability through the trait interface.

## Problem statement

`KimiLinearModel::forward_for_sequence` treats a multi-token forward without a sequence ID as a new prompt. That reset keeps two independent cache-less calls isolated, but a no-ID chunking caller would discard the recurrent state built by every earlier chunk. The shared trait default advertised chunking support despite that contract.

Current DirectEngine prefill passes one stable sequence ID through every piece, so its path does not enter the no-ID reset. The change still defines the conservative trait capability and prevents legacy or alternate cache-less callers from selecting an unsafe split.

## Technical decision

The model opts out instead of changing `ModelOwnedSequenceState`. An explicit continuation mechanism would require a cross-family state-contract redesign and must preserve independent-prompt resets. The opt-out is local and reversible, but it can replace per-chunk transient memory bounds with one graph over a long prompt.

## Change summary

- Added the Kimi Linear `supports_chunked_prefill` override with the no-ID state rationale.
- Added a zero-layer, tied-embedding fixture that loads through `KimiLinearModel::from_weights` and asserts the policy through `&dyn LanguageModel`.
- Reconfirmed the five sibling reset/default-capability combinations and filed #2285 through #2289.

## Validation

The CUDA `mlxcel` and `mlxcel-surgery` library test graph builds successfully. The focused loaded-model test passes 1/1 under the GPU lock. Formatting and diff checks pass, and independent implementation and security/performance reviews reported no findings.

Real-checkpoint validation used `mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit` revision `919175d75427f1c10cfc17e7254b53295edddf72`. Baseline main `190b83c3` and the candidate built from main `448935b6` plus the frozen policy diff both emitted token ID 0 for all 64 outputs of a 3,411-token prompt under the default setting and `MLXCEL_PREFILL_CHUNK=0`; both also emitted ID 0 for a 21-token chunk-disabled arithmetic prompt. These invalid matching outputs are not an accuracy or numerical-parity pass. The candidate CLI SHA256 was `c0b0319f0b4f0302e9e5dccf02f4f403453ec3dc9aa7e5e1a4ce625f3f060eb7`; diagnosis and exact-answer acceptance are tracked in #2292.

## Follow-up actions

- #2292 must restore normal Kimi Linear CUDA inference and prove exact short and long answers.
- #2285 through #2289 cover the audited sibling no-ID contracts without widening this change.

## References

- Issue #1749
- Issues #2285, #2286, #2287, #2288, #2289, and #2292
- Issue #672 and PR #1741
