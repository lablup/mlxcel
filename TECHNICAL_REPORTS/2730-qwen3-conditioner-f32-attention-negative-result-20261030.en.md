# PR #2730: Record the Qwen3 conditioner f32-attention negative result

**Date**: 2026-10-30
**Status**: Complete (negative result recorded; no behavior change). Measured on CUDA (GB10) only; Metal and ROCm not run
**Languages**: Rust (comments, one ignored test), Markdown
**Risk Level**: Low (`src/models/qwen3.rs` is byte-identical to main; the only code added is an ignored diagnostic test)

## Executive Summary

PR #2730 (closes #2666, epic #2348) tests one hypothesis and reports that it is false. #2666 proposed running the shared Qwen3 conditioner's attention in f32, as upstream mlx-vlm `73d366ec` does (`models/flux2/qwen/attention.py:148-162`), to close part of the FLUX.2, Z-Image, Bonsai and Ideogram 4 parity drift.

The change was implemented as an opt-in flag and measured on GB10 against the CUDA oracle. f32 attention does not move the conditioning toward upstream: the FLUX.2 prompt embeds stay 1.19e-2 rel-RMS from upstream in both arms. End to end it only reshuffles bf16-noise drift, and it pushes Bonsai (29.21 dB against a 30 dB gate) and Ideogram 4 (pooled latents 4.74e-2 against 4.5e-2) past their committed gates. The flag was therefore removed. The PR ships corrected comments and docs plus one ignored diagnostic test. The suspected real cause, bf16 GEMM reduction order and kernel selection against mlx 0.32.4, is tracked in #2732.

## 1. Problem Statement

Four image families share `Qwen3TextEncoder` (FLUX.2, Z-Image, Bonsai, Ideogram 4). Each carried a parity test comment attributing its conditioner drift against upstream to "bf16 attention". The only concrete lead was a real difference: upstream casts q, k and v to f32 for SDPA and casts the output back to the query dtype, while mlxcel's shared Qwen3 attention stays in the activation dtype.

#2666 asked for that cast. Its target was a conditioner distance of 1.9e-3, taken from upstream's own sensitivity to its attention dtype. The open question was whether the port's 1.19e-2 residual was that same effect.

## 2. Change Summary

The branch history has three steps, and the merged result is only the last.

| Commit | What it did |
|---|---|
| `670c34ce` | Added `attend_in_f32` to the shared `Attention` in `src/models/qwen3.rs` (false everywhere, graph unchanged when off) and `Qwen3Model::set_attention_f32`. The cast sat after the fused-decode and graph branches converge, before `cache.attend`, and the output returned to the query dtype before `o_proj`. Added a tiny-encoder test, an ignored FLUX.2 prompt-embeds test reporting both arms, and a looser f32 check in the real Qwen3 logits test. |
| `3ff261c7` | Merge of `origin/main`. |
| `63eafc08` | Dropped the flag. `src/models/qwen3.rs` is back to main; the measured result stays in comments and docs without referring to the flag. |

What the PR ships (6 files, +55 / -6):

| File | Change |
|---|---|
| `src/imagegen/qwen3_text_encoder.rs` | Module header and the `from_weights` comment state that attention runs in the weights' dtype where upstream uses f32, with the measured result. |
| `src/imagegen/bonsai/parity_tests.rs`, `z_image/mod_tests.rs`, `ideogram4/parity_tests.rs` | Comments no longer blame the drift on bf16 attention and record the f32-attention numbers. |
| `docs/image-generation.md` | Qwen3 conditioner, Z-Image and Ideogram 4 sections corrected the same way. |
| `src/imagegen/flux2/parity_tests.rs` | New ignored `klein_4b_prompt_embeds_match_upstream`: encodes the generate prompt through the pipeline's bucketed `encode` path and reports rel-RMS against upstream's `embeds_bf16_bits` (`[1, 32, 7680]`). Measured 1.190e-2; the 0.1 assert is a sanity bound, not a parity gate. |

No bound, noise check or PSNR gate changed.

## 3. Measurements

All arms ran on GB10 with `MLX_ENABLE_TF32=0` against the CUDA oracle (mlx 0.32.4, mlx-vlm `73d366ec`).

Conditioner distance to upstream (rel-RMS):

| Conditioner output | bf16 attention (main) | f32 attention |
|---|---|---|
| FLUX.2 prompt embeds `[1, 32, 7680]` | 1.190e-2 | 1.192e-2 |
| Z-Image `cap0` (turbo / base / edit) | 9.308e-3 / 9.308e-3 / 9.303e-3 | 9.349e-3 / 9.349e-3 / 9.314e-3 |
| Ideogram 4 text features | 6.4013e-2 | 6.4005e-2 |
| Bonsai prompt embeds `[1, 512, 7680]` | 2.306e-2 | 2.228e-2 |

End to end (final-latent rel-RMS, PNG PSNR). Initial noise was bit-identical in every arm.

| Family / case | main (= this PR) | f32 attention | Gate |
|---|---|---|---|
| FLUX.2 generate | 1.026e-1, 32.27 dB | 1.094e-1, 32.02 dB | 0.2, 30 dB |
| FLUX.2 edit | 4.109e-2, 40.54 dB | 4.244e-2, 39.07 dB | 0.2, 30 dB |
| Bonsai generate | 1.001e-1, 31.18 dB | 1.348e-1, **29.21 dB (fails)** | 0.2, 30 dB |
| Z-Image turbo | 8.607e-2, 32.86 dB | 6.402e-2, 34.53 dB | 1.2e-1, 30 dB |
| Z-Image base | 6.198e-2, 34.71 dB | 9.965e-2, 30.19 dB | 1.0e-1, 30 dB |
| Z-Image edit | 1.725e-2, 46.14 dB | 1.758e-2, 45.50 dB | 5e-2, 30 dB |
| Ideogram 4 turbo_512 (pooled latents, pooled image) | 3.16e-2, 43.72 dB (PNG 37.13 dB) | **4.74e-2 (fails)**, 38.37 dB | 4.5e-2, 38 dB |

The Ideogram 4 f32 run never reached its full PNG PSNR because the pooled-latent assert fired first.

Oracle sources: FLUX.2 and Bonsai use the committed fixtures (the Bonsai step-0 inputs came from `generate_bonsai_fixtures.py --step0` with the CUDA oracle venv); Z-Image uses the epic's oracle directory; Ideogram 4 uses the PR #2427 oracle directory.

## 4. Technical Decisions

### 4.1 Drop the flag

**Context:** f32 attention is the one documented difference between the port and upstream in this module, so it was the natural candidate.

**Evidence:**

- Upstream's own bf16-to-f32 attention step moves its FLUX.2 prompt embeds by 1.03e-3. The port's f32 attention moved ours by 9.4e-4, the same size, so the port does reproduce that effect.
- The port stays 1.19e-2 from upstream in both arms, an order of magnitude more than the attention dtype accounts for, and the two deltas are uncorrelated (cosine -0.12). The attention dtype is therefore not on the path that creates the gap.
- Z-Image and Ideogram 4 features do not move either (9.3e-3 and 6.4e-2 in both arms). Bonsai improves slightly (2.306e-2 to 2.228e-2) and still loses 1.97 dB end to end.
- End-to-end results move in both directions (Z-Image turbo improves, Z-Image base degrades by 3.7 dB) with no pattern tied to the conditioner distance, which is what amplified rounding noise looks like.

**Decision:** remove the flag. A dormant switch on the live text model's attention with no production caller adds surface without benefit, and turning it on fails two committed gates. While it existed it was off by default, and teacher-forced `logit_trace` runs were byte-identical to main on `qwen3-0.6b-4bit` and on the bf16 Qwen3-4B text encoder. With it removed the file is byte-identical to main.

**Rejected alternatives:**

- Keep the flag off by default for future use: rejected for the surface reason above; the result is recorded, and the flag is a few lines to reintroduce.
- Enable it in the conditioner only: rejected because it breaks the Bonsai and Ideogram 4 gates.
- Loosen the Bonsai or Ideogram 4 bounds to admit it: rejected because it would trade a gate for a change that does not reduce the distance to upstream.

### 4.2 Keep the diagnostic test

`klein_4b_prompt_embeds_match_upstream` is the only place the conditioner distance for FLUX.2 is measured directly, rather than inferred from final latents. It is `#[ignore]`d (needs the checkpoint and a GPU), so it does not affect CI, and its 0.1 bound only catches gross breakage. It gives #2732 a baseline number to compare against.

### 4.3 Correct the comments instead of leaving the old attribution

The old comments sent the next reader to the attention dtype, which is now ruled out. They now say what was measured and that the cause is unconfirmed.

## 5. Validation

All on GB10, `MLX_ENABLE_TF32=0`, `--test-threads=1`.

| Filter | Passed | Failed | Ignored |
|---|---|---|---|
| `imagegen::qwen3_text_encoder` | 17 | 0 | 1 |
| `imagegen::flux2` | 29 | 0 | 7 |
| `models::qwen3` | 98 | 0 | 21 |

The ignored `klein_4b_prompt_embeds_match_upstream` also passes (1.190e-2). `cargo clippy --profile test-fast --features cuda -p mlxcel --lib --bins --tests --no-deps -- -D warnings`, `cargo fmt --check`, the `make verify-*` targets, `check_binary_assets.py` and `check_env_var_docs.py` pass.

Mage-Flow, Qwen-Image, LLaDA-Image, ERNIE-Image and Ming do not reach `Qwen3TextEncoder` and are unaffected.

## 6. Learning Points

- A reproduced upstream sensitivity is not a reproduced gap. The port's f32 attention moved the output by the same 1e-3 as upstream's, which only shows the cast works; the 1.19e-2 residual is a different quantity. Compare the size of the effect to the size of the gap before expecting a fix.
- Look for a delta correlation, not only a magnitude. Cosine -0.12 between the attention-dtype delta and the port-to-upstream residual says they are independent.
- A conditioner change that is neutral on the conditioner can still move end-to-end PSNR by several dB in either direction. Judge such changes by the conditioner distance first and treat end-to-end movement as noise unless the conditioner moved.
- A target number taken from the reference's sensitivity to its own setting (1.9e-3) is not a number the port can reach.

## 7. Follow-up and Limitations

- The cause of the 1.19e-2 conditioner residual is not established. The working suspicion is bf16 GEMM reduction order and kernel selection differing between this build's MLX and mlx 0.32.4. This was not measured here and is tracked in #2732.
- Measured on CUDA (GB10) only. Metal and ROCm were not run.
- The f32 arm no longer exists in the tree. Reproducing it means reapplying `670c34ce`.
- Per-layer conditioner distances were not collected, so where the residual first appears is unknown.

## References

- Issues #2666 (closes), #2732 (follow-up), epic #2348, maintainer decision #932 (parity contract).
- Related: PR #2427 (Ideogram 4 oracle), PR #2717 (LLaDA-Image).
- Upstream: mlx-vlm `73d366ec`, `models/flux2/qwen/attention.py:148-162`.
- Docs: `docs/image-generation.md`.
