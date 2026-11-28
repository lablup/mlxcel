# PR #2297: Correct the RWKV7 no-ID chunked prefill contract

**Date**: 2026-10-14
**Status**: Implemented; CUDA build and focused GPU validation passed
**Risk**: Medium (prefill policy plus a recurrent output-shape correction)

## Summary

This change declares that RWKV7 does not support cache-level chunked prefill. A multi-token call without a sequence ID deliberately replaces the fallback recurrent state so separate prompts remain independent; advertising chunk continuation to legacy no-ID callers could therefore discard earlier chunks. Current DirectEngine calls carry a stable sequence ID and continue using their per-sequence model-owned state.

The loaded-model regression also exposed an existing shape defect in RWKV7's recurrent output. The implementation now removes the singleton column axis produced by the step matrix product before stacking timesteps.

## Problems

RWKV7 inherited the default `supports_chunked_prefill() == true` even though its no-ID path resets recurrent state at the start of every multi-token forward. A caller that split one prompt into multiple no-ID chunks could receive logits computed without the earlier chunks.

During focused GPU validation, the real one-layer fixture aborted with a native broadcast exception between shapes `(2,4)` and `(1,1,2,4,1)`. This was a production forward defect rather than invalid fixture geometry. `wkv7_step_ops` computed `state @ r[..., None]` as `[B, H, D, 1]`; stacking step outputs then produced `[B, T, H, D, 1]`. `LayerNormPerHead` expects activations shaped `[B, T, H, D]` so its `[H, D]` weight can broadcast across batch and time.

## Design

- Override `supports_chunked_prefill()` to return `false` while preserving the existing independent-prompt reset behavior.
- Squeeze only the final singleton axis of each recurrent step output. The output contract becomes `[B, H, D]` per step and `[B, T, H, D]` after stacking, while recurrent state stays `[B, H, D, D]`.
- Build a real one-layer model from deterministic nonzero synthetic weights.
- Contaminate the fallback state with a single token, then prove a new multi-token no-ID prompt matches the same prompt on a fresh model.
- Split a prompt across two calls with one stable sequence ID and compare the continuation logits with the matching suffix of a one-pass prompt.

## Verification

The initial root and surgery CUDA `test-fast` build graph passed in 14 minutes 26 seconds. The first focused GPU run then reached the real RWKV7 forward path and reproduced the shape defect with the native broadcast exception recorded above. After the fix, the CUDA graph rebuilt successfully in 2 minutes 22 seconds. The GPU-locked `rwkv7_` suite passed 5 of 5 tests, including the loaded one-layer contract regression in 0.64 seconds. The fixture loads deterministic nonzero synthetic weights through the production model loader; no external checkpoint run was performed or claimed. `git diff --check` passes.

The broader contract gate also passed across a controlled retry. The initial run passed every stage before Python tooling, then the timing-sensitive ROCm guard fixtures reported one missed 2.5-second monitor sample and one holder-metadata `FileNotFoundError` while concurrent native builds and external load saturated the host. No product or test source was changed. After lowering the competing build priority, both named guard tests passed; the full Python tooling suite passed 123 tests and the WebUI suite passed 49 tests.

## Dependency refresh

The validation above was recorded before the branch merged main commit `4b0c4519` from PR #2298. The merge commit `9b05507b` brings in the new MLX pin and CUDA/Metal overlay bases without changing `src/models/rwkv7.rs`; its SHA-256 remains `fa18e8e5e33f4af42fe7c7c8574e55db8578f90724d3a70d0cad4a7cf10adea7`. Fresh runtime validation after the final pin integration is recorded below.

## Tradeoff

Every caller that honors the model capability, including DirectEngine and cache-less callers, now receives a whole-prompt prefill plan. This can use more peak transient memory for long prompts than chunk-bounded prefill. DirectEngine still supplies a stable sequence ID, and this change does not claim that its previous state handling truncated prompts.

## Change summary

One Rust model file changes. Production changes include the capability override and one recurrent output-axis correction. The regression adds one loaded-model contract test without new dependencies.

## Final integration validation

Merged main `d3cf28ac`, including MLX `243bdf09`, the mlxcelverse migration, and the Kimi inference fix, without source conflicts. The rebuilt CUDA `test-fast` unit binary passes all five `models::rwkv7` tests, including the loaded nonzero-weight contract regression. This revalidates output shape, no-ID prompt reset, and stable-ID split continuation on the final pin. Formatting and diff checks pass. The fixture result is not an external checkpoint inference claim.

The refreshed Clippy gate also found a test-fixture style warning in the parity calculation. Replace `% 2 == 0` with `is_multiple_of(2)` without changing generated weights.
