# PR #2417: Cast SDPA masks to the attention dtype, not q's

**Date**: 2026-10-16
**Status**: Implemented and verified on CUDA (GB10) with a real checkpoint before merge; two related dtype gaps are left as follow-ups (see "Not verified and follow-up")
**Risk**: Low (a dtype cast in the C++ SDPA bridge that is the same cast as before whenever q, k and v share a dtype; two new tests)

## Summary

Issue #2336 (refs #1793, #2330): on CUDA, `mlxcel-server --kv-cache-mode fp16+turbo4` aborted on Gemma 4 with `[scaled_dot_product_attention] Mask type must promote to output type float32` for any prompt past the 1024-token sliding window. The bridge cast additive masks to the query's dtype, but SDPA attends in `result_type(q, k, v)`, and for a bf16 query over fp16 TurboQuant K/V that is float32. This PR adds a shared `sdpa_mask_for_inputs` helper that casts floating masks (and attention sinks) to `result_type(q, k, v)`, and uses it in all three SDPA bridges.

## Problem

The issue's hypothesis was a float32 q, k or v. Logging the SDPA call on the crashing layer showed otherwise: `q` bf16 `[1,16,1980,256]`, `k` and `v` fp16 `[1,8,1980,256]`, mask f32 `[1980,1980]`.

- TurboQuant caches always return fp16 K/V, so a bf16 model hands SDPA a bf16 query over fp16 K/V.
- SDPA attends in `result_type(q, k, v)`, which is float32 for bf16 + fp16, and rejects an array mask whose dtype does not promote into that type.
- The bridge cast the additive mask to q's dtype (bf16). The CUDA overlay's patched promotion table keeps bf16 + float32 at bf16, so the mask no longer promoted into the float32 output and SDPA threw. The exception escaped on the worker thread and aborted the process.
- Below the sliding window Gemma 4 attends maskless, so nothing checked a mask. Only prompts past 1024 tokens crashed.
- Metal's unpatched table promotes bf16 + float32 to float32, so the failure was CUDA-only.

It reproduced with and without `--draft-kind mtp`, and before and after #2330, so #2330 did not cause it. Default FP16 KV did not abort. It blocked the last open #1793 criterion.

## Technical decisions

- A shared `sdpa_mask_for_inputs(mask, q, k, v)` in `mlx_cxx_internal.h` replaces three copies of the "cast to q's dtype" block in `fast_scaled_dot_product_attention`, `fast_scaled_dot_product_attention_with_sinks` and `fused_metal4_attention`. Bool and integer masks pass through unchanged, since their select semantics must not become additive 1.0/0.0 offsets. A floating mask is cast to `result_type(q, k, v)`. SDPA casts an accepted mask to the attention dtype itself, so when q, k and v share a dtype this is the cast it always did, and otherwise it only drops an extra rounding through q's dtype.
- Why the bridge: it is the one place that already adapts mask dtypes to MLX's contract, and its stated purpose was to satisfy "must promote to output type". It picked the wrong target type.
- Sinks: SDPA applies the same promotion rule to attention sinks, so the sinks bridge passes them through the same helper (a review finding). GPT-OSS greedy output on `gpt-oss-20b-mxfp4-q4` is byte-identical before and after.
- `mask_in_score_dtype` on the query-chunked path now takes q, k and v and casts to q's dtype only when K and V share it. With mixed dtypes it leaves the mask alone and the bridge casts it.
- Rejected: casting K/V back to bf16 in Gemma 4. It fixes one model, adds a per-layer K/V copy, and changes turbo4 numerics everywhere.
- Rejected: making turbo caches return the input dtype. It changes the cache's fp16 contract that the fused turbo kernels rely on.

Trade-off: the fix lives in the C++ bridge, so a Rust-side caller that builds its own SDPA call outside the bridge would not benefit. None exists today.

## Changes

- `src/lib/mlxcel-core/cpp/mlx_cxx_internal.h`: `sdpa_mask_for_inputs`.
- `src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp`: both SDPA bridges use the helper; sinks go through it too.
- `src/lib/mlxcel-core/cpp/mlx_cxx_ext.cpp`, `mlx_cxx_bridge.h`: the Metal 4 bridge uses the helper; comment updates.
- `src/lib/mlxcel-core/src/layers.rs`: `mask_in_score_dtype(mask, q, k, v)` and its two callers; test `additive_mask_with_bf16_query_over_fp16_kv_is_accepted`.
- `src/models/gemma4.rs`: test `turbo4_past_window_sliding_mask_is_accepted_by_sdpa`.
- `CHANGELOG.md`: Fixed entry.

Total: 7 files, +229 / -39.

## Behavior change

When q, k and v share a dtype, nothing changes. For a bf16 model on an fp16 TurboQuant KV cache, masked attention now runs instead of aborting on CUDA. On Metal the mask now reaches SDPA as float32 instead of bf16, which SDPA would have promoted to float32 anyway.

## Verification

- New test `layers::tests::additive_mask_with_bf16_query_over_fp16_kv_is_accepted`: bf16 q over fp16 K/V with a sliding-window causal f32 mask, through direct dispatch and the query-chunked path, compared with float32 attention (tolerance 1e-3). It also asserts `mask_in_score_dtype` returns `None` for the mixed dtypes.
- New test `gemma4_unified_mask_tests::turbo4_past_window_sliding_mask_is_accepted_by_sdpa`: a Turbo4Asym `RotatingKVCache` fed bf16 K/V, the past-window `create_sliding_window_prefill_mask`, `trim_mask_to_keys` and `attention_from_ptr`, compared with float32 attention (tolerance 1e-2). It asserts the fixture really has bf16 q over fp16 K/V and runs past the window.
- Revert check: with the bridge change reverted, both tests abort with `Mask type must promote to output type float32`. With it, they pass.
- `layers::tests` (100) and `models::gemma4` (120) pass. Clippy (`-p mlxcel-core --lib --tests`, root `--lib --tests`), fmt and the `scripts/ci` checks are clean.
- Real checkpoint: GB10, `~/models/mlx/gemma-4-26b-a4b-it-4bit`, `mlxcel-server`, a 1987-token prompt, 300 tokens at temperature 0, `MLXCEL_SDPA_DETERMINISTIC=1`. On main, `--kv-cache-mode fp16+turbo4` aborts. With the fix it completes, with no drafter and with `--model-draft ~/models/mlx/gemma-4-26b-a4b-it-assistant-bf16 --draft-kind mtp` (`MLXCEL_MTP_ALLOW_INEXACT=1`). The turbo4 output is coherent and matches the FP16 KV arm for the first 287 characters, then diverges on wording, as expected from 4-bit V quantization.
- #1793 re-check (reverted-#2330 arm, same setup, 400 greedy tokens, about 143 verify rounds, with a temporary diagnostic that is in no PR). Fixed arm, turbo layers: 3124 verify-block calls, each a 1027-column mask over 1027 keys, no discard. Reverted arm, turbo layers: 3080 calls returning 1024 keys against the 1027-column mask, which `trim_mask_to_keys` crops silently, so past the window the pre-#2330 defect shows as a right-crop, not a warning. Both arms log exactly one `mask shorter than key length` warning, and every discard (393 fixed, 387 reverted) comes from the three sliding layers that keep FP16 storage and carry the MTP rollback buffer, the legitimate buffered-verify case. The same warning appears with default FP16 KV plus MTP. The criterion as worded ("no warning") is therefore met in neither arm, but the warning is not from the turbo4 append. Results are recorded on #1793.

## Not verified and follow-up

- The CUDA chunk gate (`cuda_sdpa_materializes_scores`, `materializing_sdpa_query_chunk`) judges flash eligibility and score bytes from q's dtype, which understates mixed bf16/fp16 attention that runs in float32.
- The softcap composite still casts its mask to q's dtype and returns v's dtype.
- The CUDA `dtype.cpp` overlay header says bf16 + fp16 gives bf16, while its table gives float32.
- Greedy parity with the FP16 KV arm holds only for the first 287 characters; further agreement is not expected with 4-bit V.
