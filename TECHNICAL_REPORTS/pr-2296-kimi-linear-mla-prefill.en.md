# PR #2296: Restore finite Kimi Linear MLA prefill

**Date**: 2026-10-14

**Status**: Implemented and independently reviewed; focused and pinned-checkpoint validation passed

**Risk**: Medium (causal masking in Kimi Linear's full-attention layers)

## Summary

PR #2296 fixes Kimi Linear inference on CUDA. The model previously emitted token ID 0 repeatedly for both a 21-token arithmetic prompt and a 3,411-token retrieval prompt. Kimi's MLA path copied the upstream boolean-mask operation, but mlxcel's shared `create_causal_mask` returns an additive mask. Treating zero and negative infinity as conditions reversed visibility and made every final prefill query softmax an entirely blocked row.

The MLA path now casts the additive mask to the positional-score dtype and adds it. Two regressions pin the mask values and drive a real three-token Kimi MLA prefill through the affected operation. The pinned 48B-A3B 4-bit checkpoint then answers `4` and `AMBER-RAVEN-7429` exactly under the required chunk settings.

## Root cause

`KimiLinearModel::forward` creates a causal mask with `mlxcel_core::utils::create_causal_mask`. Its contract is zero for a visible key and negative infinity for a blocked key. `KimiMLAAttention::forward` passed that array to `where_cond(mask, pe_scores, negative_infinity)` as if it were boolean.

Zero converts to false while negative infinity converts to true. Visible keys were therefore replaced by negative infinity and future keys were retained. The final query row in any multi-token prefill has no future keys, so its whole mask row is zero and all positional scores became negative infinity. Adding the content scores did not restore finite values. Softmax of the all-negative-infinity row produced NaNs, which flowed through the first MLA layer, later residuals and recurrent state, and the final logits. Greedy argmax then returned token ID 0.

The defect was independent of chunking. The shipped chat template produced the expected token IDs, checkpoint packing matched the loader, and the checkpoint's grouped routing configuration is inert because both group counts are one.

## Change

`apply_mla_causal_mask` preserves the repository mask contract. It casts the mask only when its dtype differs from `pe_scores`, then adds it. This matches the existing Kimi K3 and DeepSeek MLA adaptations and works with the cached `[L, L + offset]` key axis through normal broadcasting.

The patch deliberately leaves KDA RMS normalization unchanged. A newer upstream epsilon formula is a separate porting difference, but it does not explain the deterministic all-token-0 failure. The mask-only candidate restores exact checkpoint answers, so including the unrelated numerical change would obscure the demonstrated cause.

## Regression tests

`kimi_mla_applies_the_repository_causal_mask_as_additive_values` applies a three-by-three causal mask to known scores. It requires rows `[score, -inf, -inf]`, `[score, score, -inf]`, and three visible scores. This catches both mask reversal and a missing mask.

`kimi_mla_multi_token_prefill_keeps_the_last_query_row_finite` loads a small zero-weight `KimiMLAAttention`, performs a three-token prefill, and requires every output to remain finite. A single-token fixture would not reach the causal-mask branch.

Both tests pass with the fix. As a controlled negative check, replacing only the helper with the former `where_cond` semantics made both tests fail with exit 101: the first visible score was negative infinity instead of zero, and the real MLA fixture produced a non-finite final row. The fixed source was then restored byte for byte before the final build.

## Pinned-checkpoint validation

The acceptance checkpoint is `mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit` at revision `919175d75427f1c10cfc17e7254b53295edddf72`, running on an NVIDIA GB10 with CUDA compute capability 12.1. Greedy generation used temperature zero and seed 1749.

The baseline at `190b83c3`, binary SHA-256 `592c33a141ccd7a497732e132ac6da302661e22400e27fbcdb2cb3ca858ae8ff`, emitted token ID 0 for all 64 generated tokens on the long prompt under both default chunking and `MLXCEL_PREFILL_CHUNK=0`. The decoded output was 64 exclamation marks. The 21-token arithmetic prompt also emitted token ID 0 and decoded to `!`. PR #2293's chunk-capability change produced the same invalid values, ruling out the no-ID cache reset as the numerical cause.

The first mask-only candidate used base `c9e062af`, source SHA-256 `55c28b20dfd54398e8f6603149e8affbeb6883904c6ee012b277e12d370b429e`, and binary SHA-256 `2b2402c0316e9c6d937fa32998c360dba5214dcd7169fc2d9153e6af19472817`. It produced:

- Short prompt, 21 input tokens, `MLXCEL_PREFILL_CHUNK=0`: `4`, token ID `[19]`.
- Long prompt, 3,411 input tokens, default chunking: `AMBER-RAVEN-7429`, token IDs `[2102, 12169, 16134, 8858, 1254, 12, 51413, 24]`.
- Long prompt, 3,411 input tokens, `MLXCEL_PREFILL_CHUNK=0`: the same exact answer and token IDs.

The final branch merged main commit `398f9ba8`, which contains PR #2293, at head `7c3f697706c55d79a54b099b0728892e10ad2115`. Its binary SHA-256 was `b2cb19e07aa833d90e25c039daf13ac43190bf67ce4cf0074c6e074238c40de0`. The three required runs repeated the exact initial answers and token IDs. Two additional runs set `MLXCEL_SDPA_DETERMINISTIC=0`; the short prompt still returned `4` and the default long prompt still returned `AMBER-RAVEN-7429`, proving the repair does not depend on the comparison harness's deterministic-SDPA setting.

## Verification

- `cargo test --locked -p mlxcel -p mlxcel-surgery --lib --test chat_template_kwargs --profile test-fast --features cuda -j4 --no-run`: passed in 5 minutes 58 seconds.
- Focused final regressions: 3 passed, 0 failed. They include both new MLA mask tests and PR #2293's loaded-model chunk-capability test.
- Shared issue contracts: 123 Python checks and 49 WebUI checks passed with exit 0.
- Formatting and patch whitespace checks passed after the main merge.
- Independent correctness and security review: clear, with no finding in scope.

## Scope

This PR repairs Kimi Linear MLA masking and closes #2292. PR #2293 separately disables no-ID chunked prefill for Kimi Linear because its model-owned recurrent state cannot safely restart between chunks. The final branch contains both changes, while the tests and before/after evidence keep their causes distinct.

## Final integration with mlxcelverse and the updated MLX pin

Integrated main `adf0e589`, including #2304's MLX `243bdf09` pin, mlxcelverse layout, and rustls 0.23.46, without source conflicts. The actual-inference validation used staged source tree `2e4f5211f42b21973d097a00ed60a7768d7f05ba` and CLI SHA-256 `45753012854036e8b44d1adc622a6eba8bc9d15b601e098d340769581cb1a064`. This is a fresh build of the final combined source, not a reused pre-integration model result.

On NVIDIA GB10, `mlx-community/Kimi-Linear-48B-A3B-Instruct-4bit` at revision `919175d75427f1c10cfc17e7254b53295edddf72` passes all five exact-answer controls: the short prompt returns `4`, and the 3,411-token long prompt returns `AMBER-RAVEN-7429` with default prefill, explicit single-chunk prefill, and ordinary SDPA. The short prompt also passes ordinary SDPA. A separate ordinary-SDPA natural-language smoke produces 58 tokens in two coherent sentences explaining daytime Rayleigh scattering and the longer atmospheric path at sunset. There is no repeated token-0 collapse. No performance claim is made from these correctness runs.

All ten `models::kimi_linear` CUDA tests pass on the rebuilt final source, including the additive-mask, finite-last-query-row, and loaded-model chunk-capability regressions. Formatting and the diff check pass.
