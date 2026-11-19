# Technical Report: PR #3138 - Reject mid-buffer KV slices in the CUDA cuDNN decode test

**Date**: 2026-11-16
**Status**: Completed
**Risk Level**: Low (one extra comparison in the CUDA overlay's dispatch test; affected calls fall back to MLX's own SDPA kernels)

## Executive Summary

During the CUDA measurement for #3133, Llama 4 Scout on GB10 decoded garbage after any prompt longer than about 8448 tokens: `Item` followed by runs of U+FFFD, the same with and without prefill chunking. Metal was fine, and `MLXCEL_SDPA_DETERMINISTIC=1` fixed CUDA, which pointed at the cuDNN dispatch in the CUDA SDPA overlay. The overlay's `is_kv_cache_slice` accepted a KV view that starts in the middle of its buffer. `unslice_kv` then widened that view from the buffer's base, so cuDNN attended to the keys at the front of the buffer instead of the decode window. The fix adds a `kv.offset() != 0` rejection, the same check `kv_cache_slice_extent` already had, plus a regression test that fails without it.

## 1. Problem Statement

Llama 4 uses chunk-local attention (`attention_chunk_size` 8192): a query attends only to keys from the start of its own chunk. `ChunkedKVCache` serves the decode step past the first chunk with `keys[..., chunk_start:end, :]`, a slice whose strides are those of the whole buffer and whose `offset()` is non-zero.

On CUDA, a one-row decode over 256 or more keys (`kv_cache_step`) is sent to cuDNN when `use_cudnn_for_decoding` says the K/V are "KV cache slices". For those, the overlay calls `unslice_kv`, which rebuilds the view with `T_kv = strides(1) / strides(2)` rows and `offset = -kv.offset()`, i.e. from the start of the allocation. It then tells cuDNN the true key count through the sequence-length padding. The keys cuDNN reads are therefore the first `k_len` rows of the buffer, not the window.

`is_kv_cache_slice` (upstream's test, kept unchanged by the overlay) only checked that the widened view's element count equals the allocation's. A mid-buffer window passes that test because slicing along the sequence axis leaves every stride unchanged. `kv_cache_slice_extent`, the overlay's own extended version used by the prefill bucketing path, already rejected a non-zero offset, but its comment said no mlxcel producer passes such a slice, so the guard was framed as protection against a future case.

The trigger condition explains why earlier checks missed it:
- The window must reach 256 keys, so the prompt must extend at least 256 tokens past a chunk boundary (8448 tokens for Scout). An 8720-token prompt produced a 529-key window (`MLXCEL_SDPA_PLAN_DEBUG=1` showed `k_len=529 decode=1`).
- A decode that crosses 8192 after an 8001-token prefill showed 0/100 decided disagreements, because the trace ended at position 8448, just as the new chunk's window reached 256 keys. Its last three steps are the defect's onset: NLL 13.1 vs 4.3, 23.2 vs 10.2 and 12.4 vs 5.5 against the reference, at positions the decided filter did not count.
- A single-pass 9216-token forward against two 4608 chunks agreed (0/3642), because that comparison runs prefill only, where the decode dispatch is not involved.
- Metal does not use this overlay.

## 3. Technical Decisions

- **Reject in the dispatch test, not in the producer.** `ChunkedKVCache` handing SDPA a strided window is a valid MLX view; the bug is the consumer assuming every view that matches the allocation's size begins at its base. Fixing the test protects every current and future producer, and the rejected calls take MLX's own SDPA kernels, which honor the view's offset.
- **Mirror the existing guard.** `kv_cache_slice_extent` already carried exactly this check with a full explanation. The new comment points there instead of repeating it, and the old comment that called the case hypothetical is corrected to name the chunked cache as a live producer.
- **No attribution to a specific PR.** The garbage output also reproduced on builds from before #3114 and #3119 (which predate #3117's chunk-local decode rework), so the report and the code comment describe the mechanism without naming an introducing change. Exactly when a mid-buffer window first reached this path is not established.
- **Test at the cache level, on the real dispatch.** The regression test drives `ChunkedKVCache::attend` in f16 with a chunk of 512 and an 800-token prefill, so the first decode window is 289 keys of a buffer the front trim left mid-allocation. It compares against `chunked_reference` (the mlx-lm masked formulation). On a CUDA build this takes the cuDNN branch without the fix; on Metal or CPU it is an ordinary parity test.

## 7. Change Summary

| Area | Change |
|---|---|
| `overlays/cuda/.../scaled_dot_product_attention.cpp` | `is_kv_cache_slice` returns false when `kv.offset() != 0`; the comment in `kv_cache_slice_extent` is corrected |
| `mlxcel-core/src/cache/attend_tests.rs` | New `chunked_decode_over_a_long_mid_buffer_window_matches_the_chunk_mask` |

Validation on GB10 (sm_121, CUDA 13.0):

| Check | Before | After |
|---|---|---|
| Scout, 8720-token prompt, `--temp 0`, `MLXCEL_PREFILL_CHUNK=0` and default | garbage | correct answer (item 275 in bin 6206, weight 136), identical in both modes |
| Teacher-forced decode, 300 positions after the 8720 prefill, vs one 9216-wide prefill forward over the same text | first step NLL 21.5 vs 2.85, then non-finite | 0/149 decided disagreements, mean NLL 1.853 vs 1.854 |
| New regression test | fails, relative rms 1.12 | passes |

Scout throughput after the fix on the 8720-token prompt: prefill 633 tok/s single pass and 451 tok/s with default chunking, decode 10.9 and 11.5 tok/s. There is no meaningful "before" number because the old output was wrong. All `cache::attend` tests pass on CUDA; PR CI passed (20 checks, 12 skipped by path filter).

## 8. Follow-up Actions

- ROCm was not checked. If the ROCm backend carries a similar slice test feeding a rebasing helper, it needs the same guard.
- Under the plain debug profile, two `cache::paged_detach` tests trip the `debug_assert` at `paged_detach.rs:1095`, with or without this change. They pass under `test-fast`. This is unrelated to SDPA and is tracked separately.
- Scout's single-pass prefill at 8720 tokens (about 630 tok/s) was faster than the default chunking (about 450 tok/s) on GB10. The default chunk size for Llama 4 on CUDA may be worth revisiting with a proper ABBA measurement.
