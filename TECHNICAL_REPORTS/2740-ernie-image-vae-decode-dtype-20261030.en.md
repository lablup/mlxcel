# PR #2740: Decode ERNIE-Image latents in the VAE dtype and pin the encoder to bf16

**Date**: 2026-10-30 (report written pre-merge; the PR was open and in draft at the time)
**Status**: Open PR, validated on CUDA (GB10) only. Metal, ROCm and pre-Ampere CUDA were not run
**Languages**: Rust, Markdown
**Risk Level**: Low to medium (the VAE cast is additive; the encoder loader change alters how the Ministral 3 weights are read on every backend and removes the quantized-checkpoint path)

## Executive Summary

PR #2740 (closes #2734, epic #2348) fixes an out-of-memory failure in ERNIE-Image at 1024x1024 when `MLXCEL_CUDA_F16_NORMALIZE=1` and `MLX_ENABLE_TF32=0`. Under that switch the VAE weights are f16 while the bf16 transformer returns bf16 latents. bf16 x f16 promotes the whole VAE decode to f32, and a non-TF32 f32 decode at 1024 fails in `cudaMallocAsync`. The fix adds `ernie_image::pipeline::decode_latents`, which casts the latents to a new `Flux2Vae::dtype()` accessor before decoding.

The parity run for that fix exposed a second problem. With the env var, the Ministral 3 text encoder also loaded as f16 through `load_text_weights`, which moved the conditioning to 3.509e-3 rel-RMS and Turbo from bit-exact to 2.93e-1 final-latent rel-RMS (26.04 dB). The encoder and prompt enhancer are now read with the raw `load_weights_from_dir`, so they stay bf16 under any load policy. Because the raw loader skips the NVFP4/ModelOpt repack, `load_ministral3` now refuses quantized configs with a clear error.

Both arms (with and without the env var) pass all 42 real-checkpoint parity tests with the original bounds. The one cost is that the f16 VAE decode lowers Turbo PNG PSNR from 57.00 to 53.56 dB under the f16 policy.

## 1. Problem Statement

`MLXCEL_CUDA_F16_NORMALIZE` converts bf16 weights to f16 at load. It is opt-in on sm_80 and later and on by default below sm_80. The ERNIE-Image transformer is bf16 on every backend (f16 weights give NaN latents), so `sample_latents` returns bf16. `Flux2Vae::default_dtype()` returns f16 under the switch. `decode_to_pixels` passed the bf16 latents straight to `decode_packed_latents`, so every conv and attention op ran in f32.

Issue #2734 measured it with a VAE-only probe on `ernie-image-turbo`: f16 VAE with bf16 latents at 1024 and TF32 off decodes in f32 and OOMs, even with a small attention chunk budget (so the mid-block attention is not the cause). The same VAE with f16 latents, or a bf16 VAE with bf16 latents, decodes in 3.84 GiB. A pure f32 VAE with f32 latents also OOMs at 1024 with TF32 off, which is a separate CUDA question the issue put out of scope. FLUX.2, Ideogram 4 and LLaDA-Image already cast to the VAE dtype first; ERNIE-Image did not.

The second problem was found only because the first fix made the parity tests runnable under the env var. Before it, the 1024 cases crashed in the VAE and nothing downstream of the encoder could be compared.

## 2. Change Summary

Four commits, six files (+103 / -6).

| File | Change |
|---|---|
| `src/imagegen/vae/model.rs` | `Flux2Vae` stores the `compute_dtype` it was built with and exposes `dtype()`. Purely additive, outside the ERNIE directory. |
| `src/imagegen/ernie_image/pipeline.rs` | New `decode_latents(vae, latents)` casts to `vae.dtype()` and calls `decode_packed_latents`. `decode_to_pixels` uses it. |
| `src/imagegen/ernie_image/text_encoder.rs` | `load_ministral3` reads weights with `mlxcel_core::weights::load_weights_from_dir` instead of `crate::models::load_text_weights`. New `reject_quantized_config` runs before any weight is read. |
| `src/imagegen/ernie_image/pipeline_tests.rs` | New `decode_latents_runs_in_the_vae_dtype`. |
| `src/imagegen/ernie_image/text_encoder_tests.rs` | New `quantized_configs_are_refused_before_any_weight_loads`. |
| `docs/image-generation.md` | ERNIE-Image precision prose and the text encoder and VAE rows of the Precision table. |

Commit sequence: `a07f2683` (VAE cast, with parity bounds temporarily relaxed for the f16 policy), `9d52ba0a` (encoder pinned to bf16, relaxed bounds removed), `b4804580` (docs and doc comment alignment, no behavior change), `3be12dd2` (quantized-config guard).

## 3. Technical Decisions

### 3.1 Cast in ERNIE, to `vae.dtype()`

**Context:** the cast could live in `Flux2Vae::decode_packed_latents` or in the ERNIE caller.

**Decision:** keep it in ERNIE. Z-Image deliberately feeds an f32 latent stream into the shared VAE, so the VAE must not own the dtype.

**Deviation from the issue text:** the issue proposed `Flux2Vae::default_dtype()` as the target. The PR uses `vae.dtype()`, the dtype the weights were actually loaded in. For ERNIE's real VAE the two are identical. Without the env var `default_dtype()` is bf16 on the test host, so an f16-VAE unit test following the literal wording would fail. Casting to the loaded dtype is also correct for any VAE built with an explicit dtype. The cost is one new public accessor in `vae/model.rs`.

### 3.2 Pin the encoder to bf16 by loading raw

**Evidence:** with the VAE fix alone, the real-checkpoint arm under the env var showed conditioning at 3.509e-3 rel-RMS and Turbo at 2.93e-1 (26.04 dB). The transformer is bf16 on every backend, and its first forward on upstream's own inputs gave 0.000e0 in both arms, so the encoder was the component that moved.

**Decision:** follow the project rule that a component stays bf16 when f16 fails a seed-fixed threshold. `load_ministral3` calls the raw loader, so no f16 round trip happens and neither `sanitize.rs` nor the shared loader changes. This also keeps the encoder and enhancer bf16 on Metal, where the load policy would otherwise convert to f16.

**Rejected alternatives:**

- Relax the parity bounds for the f16 policy (done in `a07f2683`, reverted in `9d52ba0a`): it would have hidden a real 26 dB regression behind a looser gate.
- Add a bf16 opt-out to the shared text-weight loader: a larger blast radius for one family.

### 3.3 Refuse quantized configs

The raw loader does not run the NVFP4 or ModelOpt repack that `load_text_weights` performs. A quantized checkpoint would fail later with a missing-weight error. `reject_quantized_config` checks `quantization` and `quantization_config` at the top level and under `text_config`, ignoring null values, and fails early. No quantized ERNIE encoder exists, and quantized conversions are already out of scope in the docs.

## 4. Validation

All on GB10, `MLX_ENABLE_TF32=0`, `--test-threads=1`.

- New `decode_latents_runs_in_the_vae_dtype`: tiny f16 VAE from `vae_tiny.safetensors`, bf16 packed latents, asserts f16 output and that uncast `decode_packed_latents` gives f32. Since `decode_latents` does not exist on main, the fails-on-main proof was an in-branch cast removal: the assertion failed with `left: 10` (f32), `right: 9` (f16). It passes with the fix, with and without the env var.
- Non-ignored `imagegen::ernie_image` on the final head: 39 passed, 4 ignored.
- Real-checkpoint `imagegen::ernie_image --include-ignored`: 42 passed in both arms with the original bounds, no case skipped. The two full arms ran at `9d52ba0a`. The later commits are docs, a doc comment and the quantized guard, and on the final head `the_real_encoder_matches_upstream` still prints conditioning rel-RMS 0.000e0 with the env var.
- The oracle directory was regenerated with `tests/fixtures/generate_ernie_image_fixtures.py --control` on mlx-vlm 0.7.8 git `73d366ec`, mlx 0.32.4 CUDA; its summary is byte-identical to the committed `ernie_image_parity.safetensors`.

End-to-end PNG PSNR and final-latent rel-RMS at 1024x1024:

| Case | no env var | `MLXCEL_CUDA_F16_NORMALIZE=1` |
|---|---|---|
| turbo_1024 | 57.00 dB, 0 | 53.56 dB, 0 |
| turbo_edit_1024 | 32.23 dB, 1.35e-1 | 32.02 dB, 1.45e-1 |
| base_1024 | 24.24 dB, 3.03e-1 | 24.24 dB, 3.03e-1 |
| base_8step_1024 | 31.43 dB, 1.64e-1 | 31.42 dB, 1.64e-1 |

Conditioning rel-RMS is 0 in every case. Under the f16 policy the VAE decode is the only f16 component, and it accounts for the Turbo drop from 57.00 to 53.56 dB; it barely moves the CFG cases. Base 50-step misses the epic's 30 dB criterion in both arms, as already documented on main (CFG batch-2 GEMM rounding, unrelated).

CLI smoke on a binary built at `9d52ba0a` (before the quantized guard): `MLXCEL_CUDA_F16_NORMALIZE=1 MLX_ENABLE_TF32=0 mlxcel generate-image -m models/mlx/ernie-image-turbo ... --size 1024x1024 --seed 42` finished in 73 s and produced a coherent image. It OOMed before the change. The prompt enhancer ran through the new raw-loaded path.

Other `decode_packed_latents` callers were audited: LLaDA-Image, FLUX.2 and Ideogram 4 already cast, Z-Image feeds f32 on purpose, the rest are tests. No follow-up is needed. Lint and verify gates (`cargo fmt`, clippy `-D warnings`, the `make verify-*` targets, `check_binary_assets.py`) pass.

## 5. Learning Points

- Mixed bf16 and f16 operands silently promote to f32 in MLX. An allocation failure far from the cause can be a dtype-promotion bug, so print dtypes before tuning chunk budgets.
- A fix that unblocks a test arm can reveal that the arm was hiding a second regression. Here the crash masked an encoder drift of 26 dB. Running the parity arm after the fix, instead of relaxing its bounds, is what found it.
- Loading through the shared text-weight loader couples a component to a global dtype policy. A component with a parity contract against a bf16 reference should load raw and say so.
- Use the dtype a module was actually built with (`vae.dtype()`), not a global default, as the cast target.

## 6. Limitations and Unverified Points

- Measured on GB10 CUDA only. On pre-Ampere CUDA and Apple Silicon the policy previously converted the encoder and enhancer to f16 and they now run bf16. This was not measured. The encoder is one forward per prompt, but the enhancer decodes token by token and could be slower on parts without native bf16.
- The text-generation `--surgery` weight hook lives inside `load_text_weights`, so it no longer reaches the ERNIE encoder or enhancer.
- Quantized ERNIE encoders are now rejected rather than loaded. No such checkpoint exists today.
- The CLI smoke ran before the quantized-config guard; the guard is covered by its unit test and by the real-encoder test on the final head.
- The f32 non-TF32 1024x1024 VAE OOM itself is untouched.
- The Turbo PSNR drop under the f16 policy (57.00 to 53.56 dB) is the accepted cost of an f16 decode.

## References

- Issue #2734 (closes), epic #2348, PR #2706 (where it was found), maintainer decision #932 (parity contract).
- Related: PR #2730 (Qwen3 conditioner negative result), PR #2717 (LLaDA-Image).
- Upstream: mlx-vlm `73d366ec`.
- Docs: `docs/image-generation.md` (ERNIE-Image precision, Precision table); `docs/environment-variables.md` (`MLXCEL_CUDA_F16_NORMALIZE`).
