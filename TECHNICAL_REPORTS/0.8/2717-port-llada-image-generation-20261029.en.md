# PR #2717: Port LLaDA-Image text, VQ and edit generation

**Date**: 2026-10-29
**Status**: Complete on CUDA (GB10) with an accepted parity exception; Metal and ROCm not run
**Languages**: Rust, Python (oracle generators), Markdown
**Risk Level**: Medium (new family; shared-code changes are opt-in, but the end-to-end own-feature parity target is not met)

## Executive Summary

PR #2717 (closes #2346, epic #2348) ports mlx-vlm `llada_image` (git 73d366ec) for `inclusionAI/LLaDA-Image-Turbo` and `inclusionAI/LLaDA-Image`. It adds three modes: text-to-image, VQ (the LLaDA-2 prior generates discrete image tokens that SigVQ embeds) and single-image editing. It is wired into `mlxcel generate-image`, `/v1/images/generations`, `/v1/images/edits`, the probe and the registries.

The 30 dB end-to-end PSNR criterion and the exact VQ token-id match are not met on mlxcel's own features, and the maintainer accepted that gap as a documented exception. Own-feature PSNR is 16.10 dB (Turbo text), 18.02 dB (base text), 26.71 dB (Turbo edit), 22.24 dB (base edit) and 14.34 dB (Turbo VQ); 254 of 1024 VQ ids match. The gates that apply instead are: end-to-end tests on upstream's own caption and semantic features at 40 to 50 dB bounds (measured 44.79 to 58.45 dB), per-layer text-encoder exactness, bit-identical noise draws, and wrong-port controls that must fail.

## 1. Problem Statement

LLaDA-Image conditions a 30-block DiT on features from a LLaDA-2 MoE text encoder (through a QueryFormer and a text projection), optionally on SigVQ tokens produced by block-diffusion sampling, and on VAE latents of a source image for edits. #2348 brought image generation into scope, and the two checkpoints were recognized by nothing in mlxcel.

Existing pieces that could be reused: the `llada2_moe` model and its sampler helpers, the shared `dit_blocks`, and the FLUX.2 VAE. Missing: the LLaDA-Image config, QueryFormer, SigVQ, VQ sampler, schedulers, transformer wiring and pipeline.

The hard part is numerical, not structural. The pipeline is chaotic: MoE routing flips and the sampler amplify sub-ulp differences, so a correct port can land far from upstream's PNG. That forced a parity design that separates port defects from amplified rounding drift.

## 2. Change Summary

| Area | Change |
|---|---|
| Family module | New `src/imagegen/llada_image/`: `config` (load-time shape and dtype checks), `conditioning` (QueryFormer, text projection), `sigvq` (ViT encoder, prior projector), `vq_sampling` (block-diffusion sampler reusing `llada2_moe::generate` helpers plus an argsort-fallback top-count helper), `transformer` (built from `dit_blocks`), `scheduler`, `weights`, `pipeline`, `mod`, each with a `*_tests.rs` sibling, plus `parity_tests.rs` and `transformer_stage_tests.rs`. |
| Schedulers | Turbo: uniform sigmas, shift 3, stochastic re-noise, 4 steps, guidance 1. Base: f64 nonlinear schedule, shift 1, 50 Euler steps, guidance 5. Upstream's global RNG draw order is kept. |
| Text numerics | `Llada2MoeModel` gains an opt-in `Llada2Numerics::LladaImage` path (`image_numerics.rs`, `forward_embeds`): bf16-rounded RMSNorm before the gain, RoPE tables rounded to bf16 with per-row positions, experts sorted by descending score, routing score applied before the down projection, f32 expert sum. The default text path is unchanged. |
| GELU | QueryFormer, text projection and SigVQ use mlxcel-core's compiled replicas of the `nn.gelu_approx` and `nn.gelu` graphs MLX compiles. Eager op chains rounded 0.8% of bf16 outputs differently. |
| `dit_blocks` (additive) | Weightless block norms and per-token modulation for edits; opt-in `BiasAdd::Fused` (one `addmm`, as upstream `nn.Linear`); opt-in `MRoPE::compute_freqs_on_device`. Z-Image, Mage-Flow and Ming-Image keep their current paths. |
| Request handling | Options `generation_mode`, `stochastic_sampling`, `negative_prompt`. Invalid requests raise `InvalidImageRequest` (HTTP 400). Edits are capped at `MAX_EDIT_IMAGE_TOKENS` = 65536 denoiser image tokens; VQ prompts at `MAX_VQ_PROMPT_TOKENS` = 2048. Both are checked before any weights load. |
| Registration and docs | Image registry, probe (`LLaDAImagePipeline`) and probe test, architecture registry, env-var allowlist for the parity test; LLaDA-Image section in `docs/image-generation.md`, row in `docs/supported-models.md`. |
| Oracle tooling | `tests/fixtures/generate_llada_image_oracle.py` and `generate_llada_image_denoiser_stages.py`. No fixtures are committed because the cases are full resolution. |

Size: 40 files, +6589 / -47.

## 3. Technical Decisions

### 3.1 Parity: accepted exception and replacement gates

**Context:** Against upstream on the same seed, mlxcel's whole pipeline does not reach the 30 dB own-feature PSNR the issue asked for, and the VQ ids do not match exactly. The maintainer accepted this as an exception (epic #2348). The conclusion below is the evidence behind it, not a claim that the numbers are good.

Oracle: upstream mlx-vlm `73d366ec` on mlx 0.32.4 CUDA (GB10), `MLX_ENABLE_TF32=0` on both sides, seed 42, prompt "A red panda serving tea in a moonlit bamboo forest". Edits use the checkpoint's `assets/edit_demo.jpg` at 1440x704 with "make it blue". "Own" is mlxcel's whole pipeline; "upstream features" is mlxcel's denoiser and VAE fed upstream's caption and semantic features.

| Case | First prediction rel-RMS | Own: latents / PSNR | Upstream features: latents / PSNR | Bound (upstream features) |
|---|---|---|---|---|
| Turbo text 1024 | 0 | 0.882 / 16.10 dB | 0 / 57.97 dB | 5e-3 / 50.0 dB |
| Turbo VQ 512 | 3.00e-3 | 1.087 / 14.34 dB | 0.0454 / 44.79 dB | 0.08 / 40.0 dB |
| Turbo edit 1440x704 | 0 | 0.154 / 26.71 dB | 0 / 56.21 dB | 5e-3 / 50.0 dB |
| Base text 1024 | 0 | 0.623 / 18.02 dB | 0 / 57.94 dB | 5e-3 / 50.0 dB |
| Base edit 1440x704 | 0 | 0.547 / 22.24 dB | 0 / 58.45 dB | 5e-3 / 50.0 dB |
| Base VQ 512 | no oracle | | | |

Base VQ has no oracle: upstream fails twice in mlx 0.32.4's CUDA graph code (`cudaGraphAddKernelNode ... invalid argument`). mlxcel's own base VQ run completes.

**Gates asserted in `parity_tests.rs`:**

| Gate | Requirement | Measured |
|---|---|---|
| Token ids, every prompt | exact | exact |
| Initial and stochastic re-noise draws | bit-identical | bit-identical |
| Text projection on upstream's hidden states | below 1e-4 | 0 |
| Text backbone layers 0 and 1, each on upstream's own input | below 2e-5 | 0 and 7.7e-6 |
| SigVQ embedding of upstream's VQ ids | below 1e-2 | 0 |
| Edit source latents | below 2e-2 | 8.8e-3 |
| Edit SigVQ codebook ids identical | at least 97% | 973/990 |
| First prediction | below 1e-3 (text, edit), 1e-2 (VQ) | 0 (text, edit), 3.0e-3 (VQ) |
| VQ prior window positions | equal to upstream's | equal |
| End-to-end on upstream features | bounds in the table above | met in all five oracle cases |

Reported, not gated: caption features at 1.0e-2 to 1.4e-2 rel-RMS, and first VQ-window hidden states at 0.171 rel-RMS.

**Why the own-feature gap is not a port defect:**

- Teacher-forced (each backbone layer fed upstream's own input), attention and the dense layer are bit-exact in all 20 layers, and every MoE layer routes every token to the same experts in the same order. 0.6% of MoE outputs differ by one rounding because mlxcel's expert-sorted `gather_mm` accumulates in a different order than the oracle's unsorted call.
- With the unsorted call all 20 layers are bit-exact, yet the prompt still drifts through a 28-row QueryFormer `addmm` whose GEMM kernel differs between this build's MLX and mlx 0.32.4 (40 of 57344 values). Measured with the unsorted call (before the RoPE fix, so with the old 6.0e-2 denoiser residual): Turbo text 18.5 dB, Turbo edit 36.0 dB, VQ ids equal for the first 48 tokens (377/1024), VQ prior 206 s instead of 101 s.
- Upstream amplifies its own perturbations the same way: a separate bias add in its x_embedder moves its first prediction by 8.6e-2 and its PNG to 26.5 dB against itself; a 2^-9 perturbation of its QueryFormer queries gives 17.9 dB against itself; one bf16 ulp in layer 2 of its text backbone moves the final hidden states by 0.167 rel-RMS because MoE routing flips (mlxcel against upstream is 0.172).
- VQ ids: 254 of 1024 equal at seed 42, 512x512, first 6 identical. The chaos enters through the first VQ window. On upstream's ids, SigVQ and the denoiser reproduce upstream within the gates (44.79 dB).

**Decision:** keep upstream's sorted `gather_mm` call (see 3.2) and accept the own-feature drift, with the gates above as the acceptance criteria.

### 3.2 Sorted `gather_mm` kept, even though it is wrong on upstream's MLX

Upstream's `TextMoE` calls `gather_mm(sorted_indices=True)`. On mlx 0.32.4 CUDA that call returns wrong rows (rel-RMS 0.80 against the unsorted call and a per-row matmul, also with random weights; CPU f32 is correct). So upstream's own text encoder is wrong from the first MoE layer on that MLX. mlxcel's MLX pin does not have the bug (sorted path matches f32 CPU to 6e-3). The oracle is generated with `--unsorted-gather` to get a correct reference; the port keeps the sorted call because that is what upstream issues. The alternative, the unsorted call, improves own-feature PSNR (see 3.1) but doubles VQ prior time and departs from upstream.

### 3.3 Second oracle correction: VAE `conv_in`

Upstream `_match_conv_layout` never transposes the VAE encoder's `conv_in` because its shape is [128,3,3,3] in both layouts. Upstream's VAE then round-trips the edit source at 17.98 dB, against 33.79 dB with the weight transposed. mlxcel loads it correctly. Oracle flag: `--fix-conv-in`. Consequence: the base edit checkpoint output is a blue water-caustics texture in both the corrected oracle and mlxcel, while uncorrected upstream renders the blue collage.

### 3.4 Denoiser exactness through three opt-in switches

Teacher-forced stage tests (`transformer_stage_tests.rs`) feed every denoiser stage upstream's input and compare outputs. Before the fix the only non-exact stage was the image RoPE table (2.8% of entries off by an f32 rounding, from the host `powf` path), which alone produced the 6.03e-2 first prediction. Computing the inverse frequencies on the device, as upstream does, made every stage and the whole Turbo text pass bit-identical, and moved upstream-feature PSNR from 29.58 dB to 57.97 dB. `BiasAdd::Fused` fixes a separate-add path that moved the first prediction to 0.18. All three are opt-in, so Z-Image and Mage-Flow keep their behavior (dit_blocks 18, z_image 33, mage_flow 42 unit tests pass).

The remaining residual is the VQ 512 final projection (`[1024, 3840] x [3840, 128]` bf16 `addmm`), where this MLX picks a different kernel than mlx 0.32.4 (3.0e-3 rel-RMS, 38.6% of outputs) with identical inputs. The same projection on 4096 rows is exact. No code change reaches it.

### 3.5 Compiled GELU replicas

Eager GELU chains rounded 0.8% of bf16 outputs differently from MLX's compiled graphs. Using mlxcel-core's compiled replicas made the text projection exact (eager measured 1.9e-3).

### 3.6 Resource bounds and panic-free loading

- Edits are limited to 65536 denoiser image tokens (source plus target grids, `2 * (H / 16) * (W / 16)`, the Mage-Flow value). Upstream has no bound. The check is in `validate_dimensions` before any component loads; HTTP 400 names the count and the limit. A 4096x4096 edit (131072 tokens) is rejected, 2880x2880 passes, 2912x2912 is rejected, generation at 4096x4096 is unaffected.
- VQ prompts over 2048 tokens (template included) are rejected before weights load, because the VQ prior recomputes the whole window at every step.
- Text and edit prompts are truncated at 2048 tokens, as upstream does.
- QueryFormer and text projection load before either is stored, so a failed load leaves no half-built state; the request path returns errors instead of calling `expect`.
- Errors split into `InvalidImageRequest` (HTTP 400) and evaluation failures (HTTP 500).

### 3.7 Smaller decisions

- `n > 1` reloads the transformer per image. This is upstream's order and was left as is.
- The edit source is resized with the `image` crate's Lanczos3 (shared `z_image::pipeline::reference_rgb`), not PIL LANCZOS.
- The wrong-port controls run on upstream's features so that the text-encoder drift does not mask them.

## 4. Validation

Hardware: NVIDIA GB10, CUDA, under `gpu-lock`, `--profile test-fast`.

Unit tests (`--test-threads=1`): `imagegen::llada_image` 35 passed (6 ignored: parity, stage and checkpoint-tokenizer tests); `imagegen::dit_blocks` 18; `imagegen::z_image` 33; `imagegen::probe` 26; `models::llada2_moe` 23; `imagegen::mage_flow` 42.

Real-checkpoint parity tests (`--ignored`, `MLX_ENABLE_TF32=0`):

- `turbo_matches_upstream`: pass, 255 s.
- `base_matches_upstream`: pass at `7f8d27d5`, 929 s. Base VQ skipped (no oracle).
- `turbo_controls_are_rejected`: pass, 140 s. Every control misses the bounds on every case:

| Control | Text | VQ | Edit |
|---|---|---|---|
| Seed 43 | 1.256 / 12.78 dB | 0.978 / 17.71 dB | 0.505 / 20.46 dB |
| Shift 1 instead of 3 | 0.791 / 18.02 dB | 0.669 / 22.20 dB | 0.230 / 25.37 dB |
| FP32 q and k after RoPE (first prediction in parentheses) | 0.250 / 28.62 dB (8.43e-2) | 0.117 / 35.28 dB (6.69e-2) | 0.0152 / 48.85 dB (1.28e-2) |

- `denoiser_stages_on_upstream_inputs`: pass, 40 s.
- `edits_past_the_image_token_bound_are_rejected`: passes; fails with the bound disabled.
- `an_overlong_vq_prompt_is_an_invalid_request`, `prompt_ids_carry_the_grid_and_count`: pass.

Real checkpoints, CLI (seed 42, rc 0 on all): Turbo text 1024 52 s (14 s repeated, byte-identical PNG), Turbo VQ 512 84 s, Turbo edit 1440x704 40 s, base text 1024 166 s, base VQ 512 131 s, base edit 327 s. Server (`--image-model`): Turbo and base text, VQ and edit return 200; size 1000x1000, `generation_mode: "editing"` on generations and a non-multiple-of-32 edit return 400. At head `7c3becea` the server text image is pixel-identical to the CLI PNG.

Regression on the default `llada2_moe` text path (`llada2.0-mini-preview-4bit`, `examples/logit_trace`, 256 tokens x 4 chunks, this branch against the same tree with the diff reverted): 0 of 1024 positions disagree, 0 of 674 decided positions disagree, TSVs byte-identical, perplexity 7.6092 on both.

Other checks: `make verify-*` targets, `check_binary_assets.py` and `cargo clippy ... -D warnings` pass. Two identical CLI Turbo runs write byte-identical PNGs (#2696 fused LayerNorm nondeterminism does not reach this pipeline).

## 5. Learning Points

- In a routing-chaotic pipeline, end-to-end PSNR on the port's own features measures sensitivity, not correctness. Teacher-forcing each stage on the reference's input, and running the downstream stages on the reference's features, separates the two.
- A parity gate must be shown to fail. The controls (seed, shift, FP32 RoPE) all miss the bounds, and FP32 RoPE now misses even the first-prediction bound.
- Host and device floating point differ: computing RoPE inverse frequencies with host `powf` changed 2.8% of table entries by one f32 rounding and that alone moved the first prediction by 6.0e-2.
- A reference implementation can be wrong on its own platform (sorted `gather_mm`, `conv_in` layout). Check the oracle against an independent computation before treating it as ground truth.
- Fused versus separate bias add (`addmm` vs matmul plus add) is a visible numeric difference in bf16; mirror the reference's op granularity.
- Identical GELU math in eager form is not identical to MLX's compiled graph in bf16.

## 6. Follow-up and Limitations

- The 30 dB own-feature PSNR and the exact VQ id match are not met (accepted exception). Closing the gap needs bit-compatible GEMM kernel selection for the 28-row QueryFormer `addmm` and the unsorted-vs-sorted MoE accumulation order, neither of which this PR can change.
- Base VQ has no oracle until upstream's CUDA graph failure on mlx 0.32.4 is resolved.
- VQ 512 final-projection GEMM residual (3.0e-3) has no code fix.
- EXIF orientation: upstream applies `ImageOps.exif_transpose` to the edit source; mlxcel's shared loader does not, for any family.
- Edit source resize uses Lanczos3 from the `image` crate instead of PIL LANCZOS (server edit 43.8 dB against the parity run on PIL pixels).
- The text backbone layer gates run only for `turbo_text_1024`, whose oracle was regenerated with the backbone layer tensors; the other cases skip them until regenerated.
- Not run: Metal, ROCm.

## References

- Issue #2346, epic #2348, #2696 (fused LayerNorm nondeterminism), maintainer decision #932 (parity contract).
- Related ports: PR #2703 (Ideogram 4), PR #2688 (Mage-Flow).
- Upstream: mlx-vlm `llada_image` at `73d366ecf9233b3fc6549531a7f2386726d09ff6`.
- Docs: `docs/image-generation.md` (LLaDA-Image section).
