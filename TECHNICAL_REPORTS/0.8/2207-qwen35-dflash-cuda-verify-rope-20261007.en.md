# PR #2207: Qwen 3.5 DFlash verify byte-identical to decode on CUDA

**Date**: 2026-10-07
**Status**: Implemented and verified on CUDA (GB10); Metal unchanged by construction
**Risk**: Medium (the CUDA DFlash verify path and the CUDA exactness probe for the Qwen 3.5 family; removes a CUDA-wide decline)

## Summary

Qwen 3.5 never ran DFlash on CUDA. PR #1944 made `probe_block_chain_exactness` return `NotRun` for head_dim 256/288 while `MLXCEL_SDPA_VECTOR_LARGE_D` was on, because the served width 2 and 4 greedy output on `qwen3.5-4b-4bit` + `qwen3.5-4b-dflash` parted from classic decode and the probe could not see it. Issue #2191 tested whether the cause was MLX's CUDA RoPE kernel choice, which PR #2185 had found for Gemma 4. It was.

MLX's CUDA RoPE runs `rope_single` on a row-contiguous `B=1, L=1` input and the general `rope` kernel on anything wider. The two compute the same expressions as written but occasionally round a bf16 element differently: 1 byte in 144 op-level cases. A verify block's block-rotated K went into the KV cache, so the difference persisted across every later token. On the real 200-token transcript, a sub-op capture named post-RoPE Q or K as the first differing sub-op on 69 of the 73 rows whose first difference it located. On the other 4 rows the attention output differed first, and only after a K difference was already in the cache. The attribution matrix (per-row RoPE off/on x `LARGE_D` 1/0, widths 2 and 4, n=3, logit bytes) shows 67 or 129 of 200 rows differing with block RoPE under either `LARGE_D` setting, and 0 with per-row RoPE under both. `LARGE_D=0` moves the divergence rather than removing it, which corrects #1944's attribution: that rested on a text sha256 and an argmax count.

## Design notes

- `Qwen3NextAttention::verify_rope_rows` is set once from `cuda_is_available()` at construction. On the `fast_rope` branch, a `B=1` verify block with `L>1` rotates Q and K one row at a time at `offset + row` (`fast_rope_rows_like_decode`), which is the exact call classic decode makes. `B>1` blocks and Metal keep the block call. Patching MLX's `rope.cu` was rejected because it would move classic decode numerics for every CUDA model.
- The CUDA probe adds a long draw: 128 tokens walked as full-accept verify blocks and as single-token steps from one prefill, compared row by row in logit bytes. The probe shipped before this change passed widths 2 and 4 at every prompt length from 8 to 512 while the served path diverged. The new draw diverges at walk position 19 with the fix reverted and returns `Equal` with it.
- Every CUDA probe draw now prefills 160 tokens. Each KV length the probe visits can capture new CUDA graph topologies, and those misses count toward MLX's fatal lifetime cache-miss limit (#818). Bisecting `MLX_CUDA_GRAPH_CACHE_SIZE`, a width-4 startup now fits a 200-entry cache. Walking from an 8-token prompt needed more than 400 entries.
- `greedy_parity_dflash_qwen35_4b` requires bursts at widths 2 and 4 on CUDA, and accepts a decline at 8 or 16 only when the probe measured a divergence.

## Verification on GB10 (CUDA, release, `--features cuda`)

Driver 580.178.04, kernel 7.0.0-1019-nvidia, MLX pin `81ba1c6a`, `MLX_CUDA_ARCHITECTURES=121`.

- Served `/v1/completions`, temperature 0, no env overrides, three prompts of 200 tokens: widths 2, 4 and unset (which resolves to 4) run bursts and are byte-identical to classic. Widths 8 and 16 decline with `Diverges` at position 0.
- Full `speculative_parity --ignored --test-threads=1`: 7 passed, 1 failed. The failure is `b1_batched_baseline_probe`, the known Gemma 4 failure #2190 addresses. `greedy_parity_dflash_qwen35_4b`: ran [2, 4], declined [8, 16].
- `long_probe_draw_sees_the_verify_rope_hazard_and_passes_with_the_fix`: passed 3 of 3 runs.
- Throughput at width 4, three interleaved rounds with a classic null arm: DFlash is +6.7% to +10.7% over classic, against a null spread of -0.6% to +3.7%. Every arm produced an identical completion. Acceptance was 0.455, with 2.34 tokens emitted per verify.
- Per-row RoPE verify cost: within noise at width 2, +1.2% to +1.6% at width 4.
- `cargo clippy -p mlxcel --lib --tests --features cuda -- -D warnings` and `cargo fmt --check` are clean.

## Not verified on this host

Metal. Nothing to list for #2158. The per-row RoPE flag, the long draw, the 160-token probe prompt and the parity test's burst requirement are all gated on `cuda_is_available()`, and the removed decline was CUDA-only.

## Follow-up

- Throughput is about 1.08x, against the 1.18x the 2026-09-21 record measured for an inexact width-4 burst. That run followed a different trajectory (acceptance 0.511, against 0.455 here). This change does not separate the trajectory difference from the per-row RoPE cost beyond the verify timing above.
- Widths 8 and 16 still decline (`qmm_sm80` at `M*B >= 8`), as the issue scoped.
- The batched (`B>1`) DFlash verify keeps the block RoPE. The batched gate governs its exactness, not this probe.
