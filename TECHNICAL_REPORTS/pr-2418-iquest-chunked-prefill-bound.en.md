# PR #2418: Bound iquestloop chunked prefill by measured bf16 noise

**Date**: 2026-10-16
**Status**: Implemented and verified on CUDA (GB10) with the real 40B checkpoint; the full `make verify-test-cuda` workspace gate and Metal were not run
**Risk**: Low (test code and doc comments only; no production change)

## Summary

Issue #2143 reported `iquestloopcoder_matches_the_float32_oracle` failing in `check_chunked_prefill` on GB10 at the hard-coded 0.25 logit bound (`chunk 97: logit for token 5123 is 29.5, single-pass says 29.125`). The cause is bf16 activation precision, not chunk-boundary logic: with every bf16 tensor promoted to f32 at load, chunked and single-pass prefill agree within 2.1e-05 over the full 76 800-logit row. This PR replaces the 0.25 bound with a measured bf16 bound, and moves the check that can actually see boundary errors into a new f32 synthetic test.

## Problem

Chunk 97 straddles the 64-token `loop_window_size`, where pass 2's `RotatingKVCache` (`kept + L` keys) and the windowed mask from `mlxcel_core::causal_attention` must agree. The delta (3 bf16 ulps at magnitude 29) could have been either an off-by-one at that seam or accumulated rounding. The issue also paired the failure with 1.7 to 1.8 deltas against the f32 oracle on the 218-token prompt and a swap of ranks 2 and 3.

## Root cause

The two routes use the same keys, offsets and masks and round differently in bf16. Measured on GB10 with `~/models/mlx/iquest-coder-v1-40b-loop-instruct-4bit` and `--test-threads=1`, deterministic across runs and identical with and without `MLXCEL_SDPA_DETERMINISTIC=1`, using a temporary uncommitted patch that promotes all 1445 bf16 tensors to f32 at load:

| run | chunked vs single-pass, full row | mlxcel vs f32 oracle, 218-token top-5 |
|---|---|---|
| bf16 (as shipped) | top-5 worst 0.000 / 0.500 / 0.375 / 0.625 for chunk 128 / 97 / 64 / 32, row max up to 1.23 | 1.65 to 1.87, ranks 2 and 3 swapped |
| f32, TF32 on | row max at most 0.018 | about 0.037 |
| f32, `MLX_ENABLE_TF32=0` | row max at most 2.1e-05 (chunk 1: 5e-04) | 4.7e-04, oracle rank order (38-token prompt: 1.3e-04) |

Chunks 63 and 65 behave the same. The 4-bit weights are not the cause, since the f32 run uses them too. The oracle deltas and the rank swap also come from bf16 activations. Neither bf16 route is more accurate: KL(f32 || bf16 single) is 0.0047 and the chunked rows sit at 0.0041 to 0.0071. Different chunk sizes change attention call shapes (a short masked tail can leave cuDNN for MLX's unfused fallback, and continuation keys start at a different block offset), and each shape rounds its bf16 intermediates differently.

## Technical decisions

- `check_chunked_prefill` measures all four chunk sizes before asserting and reports every failure. It keeps the greedy token (it may only move to a token within `TIE_GAP` of the single-pass top) and the `TIE_GAP`-separated order, bounds the worst top-5 delta at 1.0 (measured worst 0.625) and `KL(single || chunked)` over the full row at 0.015 (measured worst 0.0074). The doc comment records the measurement table.
- A new synthetic f32 test, `chunked_prefill_matches_single_pass_past_the_window` in `src/models/iquestloopcoder_tests.rs`: 218 tokens, window 64, chunk schedules 128/97/65/64/63/32/1, plus 100, then 30 single tokens, then 88 (a multi-token append after the ring has wrapped). It compares every position against a 1e-4 bound (clean max 5.4e-07). The existing 64+32 test checks only the last row and missed a one-key cache slip: the oldest key a continuation chunk keeps is visible to that chunk's first row alone.
- Both checks assert that every logit is finite first, because `f32::max` and the bound comparisons skip NaN, which is this seam's historical failure (#401/#408).
- The doc comments that blamed f16 and a quantized `lm_head` now name bf16.
- The proposed corrections in the issue's investigation comment (scoped cuDNN dispatch, alignment-preserving K/V padding) would narrow a precision gap, not fix a logic error, so they are left for a follow-up.

Trade-off: the bf16 bound is coarse by necessity. The gate gives pass 2's local branch about 12% of the mix, so pass-2 faults stay small at the logits.

## What each check can and cannot catch

Faults injected into `forward_pass2` on continuation chunks:

| fault | real checkpoint, bf16 | new f32 test |
|---|---|---|
| cache keeps one key too few | passes (KL at most 0.0031) | fails, 6.4e-03 at position 128 |
| pass-2 RoPE offset minus 1 | passes (KL at most 0.0015) | fails, 7.2e-02 |
| pass-2 rotated from position 0 | fails on KL (0.024 to 0.030) | not run |

The bf16 check catches only gross faults. The boundary seams are pinned by the f32 test.

## Changes

- `tests/iquestloopcoder_parity.rs` (+159, -35): `CHUNKED_LOGIT_TOLERANCE`, `CHUNKED_KL_TOLERANCE`, `kl_divergence`, `top_k_of_row`, the reworked `check_chunked_prefill`, and the f16 to bf16 doc corrections.
- `src/models/iquestloopcoder_tests.rs` (+95): the new f32 test.

## Behavior change

None in production. Test code and comments only.

## Verification

- The original test on main fails as the issue reports. The new test passes in 43 s: `gpu-lock run -- target/test-fast/deps/iquestloopcoder_parity-<hash> --test-threads=1`, run from `$HOME` so `models/mlx/...` resolves.
- `models::iquestloopcoder_tests` passes 18/18. Clippy (`--lib --tests -D warnings`), fmt and the five contract checks pass.
- Fault injection results are in the table above.

## Not verified and follow-up

- Metal was not run, and the full `make verify-test-cuda` workspace gate is left to the merge gate.
- Scoped cuDNN dispatch and alignment-preserving K/V padding could shrink the bf16 chunked-vs-single gap; they are optional precision work.
