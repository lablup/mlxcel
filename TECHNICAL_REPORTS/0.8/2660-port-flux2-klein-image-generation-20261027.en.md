# PR #2660: Port FLUX.2 Klein text-to-image and editing

**Date**: 2026-10-27
**Status**: Complete on CUDA (GB10); Metal and ROCm runs not recorded
**Languages**: Rust, Python (fixture generator), Markdown
**Risk Level**: Medium

## Executive Summary

PR #2660 (issue #2424, epic #2348) adds the first runnable image-generation family to mlxcel: FLUX.2 Klein (`flux2`), ported from mlx-vlm `models/flux2` at git 73d366ec. It covers text-to-image and reference-image editing for the Klein 4B and 9B checkpoints, both distilled (4 steps, guidance 1.0) and base (50 steps, guidance 4.0), the `9b-kv` reference K/V cache variant, dense bf16 Diffusers trees, and MLX affine 4-bit exports. The family is registered in `imagegen::FAMILIES` and in the model registry, so `mlxcel generate-image`, `mlxcel-server --image-model` (`/v1/images/generations` and `/v1/images/edits`), and `mlxcel arch` all see it.

Against the CUDA build of upstream mlx-vlm on the same GB10 host, the 4B bf16 acceptance case (512x512, 4 steps, seed 9909) has bit-identical initial noise and a decoded PSNR of 32.27 dB, and the 4B edit case reaches 40.54 dB. Two acceptance numbers in the issue were shown to be unreachable for any implementation and were replaced with stronger checks (Section 3.3). Review added the upstream 1 MP edit sizing and fixed an unbounded batched VAE decode that could exhaust host memory.

## 1. Problem Statement

#2341 landed the shared image-generation runtime (`src/imagegen/`: the `ImageGenerationModel` trait, checkpoint probe, `Flux2Vae`, latent packing, the shared Qwen3 hidden-state conditioner, CLI and server routes) with an empty family table. A FLUX.2 directory was recognized by the probe and then rejected with `image generation family flux2 is not supported by this build`. FLUX.2 Klein also differs from the Qwen-Image style pipeline that #2341 had scheduler support for: it uses an empirical `mu` time shift keyed on image sequence length, conditions on Qwen3 layers 9, 18 and 27 concatenated, runs a dual-stream plus single-stream DiT with 4-axis RoPE, needs tiled VAE decode for large images, and has an edit path that concatenates reference-image latents into the token sequence. Prism Bonsai (#2428) is a ternary FLUX.2 variant, so whatever lands here must be reusable rather than private.

## 2. Change Summary

| Area | Change |
|---|---|
| Family module | New `src/imagegen/flux2/` (config, embeddings, attention, blocks, transformer, weights, scheduler, prompt, tiling, kv_cache, edit, pipeline, mod) implementing `ImageGenerationModel` for Generate and Edit. |
| Variant resolution | `config.rs` mirrors upstream `_defaults_variant`: `9b-kv` from the directory name or its single-file weight, otherwise `model_index.json` `is_distilled` picks distilled or base defaults. Transformer shape comes from `transformer/config.json`, or from index metadata for quantized exports without it. |
| Transformer | Dual-stream and single-stream blocks with the per-token modulation blend, timestep and guidance embeddings, 4-axis RoPE, SwiGLU. Linears are built through a `LinearSource` trait (`embeddings.rs`) whose default is affine or MLX-quantized `UnifiedLinear`. |
| Scheduler and prompt | `flux2/scheduler.rs` ports the empirical-`mu` scheduler in f32 (kept out of the shared Qwen-Image scheduler file). `prompt.rs` applies the chat template, sequence-length buckets (32..512), stacked layers 9/18/27 from the shared conditioner, and `(0, 0, 0, i)` text ids. |
| Latent helpers (shared) | `latent::prepare_packed_latents` and `prepare_grid_ids`, matching upstream's RNG call and draw order so initial noise is bit-identical. |
| VAE (shared) | `Flux2Vae::bn_normalize` and `denormalize_packed_latents` added; `tiling.rs` ports upstream's tiled decode including the mlx-vlm#1381 blending fix, auto-enabled at `max(h, w) >= 256`. |
| Edit and 9b-kv | Reference images resized and VAE-encoded, reference latents placed at `t = 10 + 10 * i`; `9b-kv` stores reference K/V on step 0 and reuses it, with a zero-timestep modulation for reference tokens. |
| Trait hooks (shared) | `ImageGenerationModel::check_request` (default accepts) lets a family reject a request before work; the image worker maps a failure to HTTP 400. `default_edit_size` (default keeps the reference size) lets FLUX.2 apply upstream's 1 MP sizing on the CLI and `/v1/images/edits`. |
| Registration and docs | Rows in `imagegen::FAMILIES` and `src/models/registry.rs` (`image_generate` runtime, `image` output); `docs/image-generation.md` and `docs/supported-models.md` updated; `MLXCEL_TEST_FLUX2_DIR` allowlisted as a test-only variable. |
| Fixtures | `tests/fixtures/flux2/` (about 2.6 MB): generate and edit latents plus PNGs, an f32 transformer step-0 tensor, the reference image, and the generator script. |

## 3. Technical Decisions

### 3.1 A family-local scheduler and a linear-layer seam

The FLUX.2 scheduler lives in `flux2/scheduler.rs`, not in the shared `src/imagegen/scheduler.rs`, because its `mu` derivation (branch at 4300 tokens) is specific to FLUX.2 and Bonsai, and the shared file is edited in parallel by the Z-Image unit. The transformer reads every linear through `LinearSource`, so #2428 can supply a ternary source and reuse the blocks, embeddings, scheduler, prompt path and latent helpers without forking them. `dit_blocks` RoPE and timestep embedding were not reused because their layout and precision differ from FLUX.2's.

### 3.2 Precision per component

Upstream runs bf16 throughout. Measured under the f16 policy: the transformer does not hold in f16 (NaN latents, 10.2 dB), so it is bf16 on every backend; the VAE holds (54.76 dB tiled, 52.37 dB untiled) and follows `Flux2Vae::default_dtype`; the Qwen3 text encoder holds and follows the Qwen3 loader policy. Edits run the transformer in f32 by promotion from f32 reference latents, as upstream does.

### 3.3 Replacing two unreachable acceptance numbers

- **Final-latent relative RMS below 1e-2.** Measured 1.03e-1 (generate) and 4.11e-2 (edit). The first diverging tensor is the context embedder output, one 7680-wide bf16 matmul on identical inputs and weights, while the 128-wide image embedder is bit-identical. The cause is bf16 GEMM reduction order between this build's MLX pin and mlx 0.32.4, amplified by four Euler steps; upstream's own f32 run differs from its own bf16 run by 7.1e-2. The test instead asserts: the f32 transformer is bit-identical to upstream f32 (16384 of 16384 values), the VAE decodes upstream's latents at 57.59 / 58.45 dB, end-to-end PSNR is at least 30 dB, and a 0.2 latent sanity bound that a wrong port (about 1.4) cannot pass.
- **9b-kv edit within 35 dB of the 9B edit.** The 9b-kv checkpoint has its own weights, and upstream's own 9b-kv and 9B edits differ by 25.15 dB (this port: 25.09 dB). The meaningful check is 9b-kv against upstream 9b-kv: 42.94 dB.

### 3.4 Batched seeds and batched decode

Following mlx-vlm#2355, `n` generated images share conditioning and come from one noise tensor of batch `n`, so every image reports `seed` and image `i` is not reproducible as `seed + i` alone; the trait documentation and `docs/image-generation.md` record the exception. Upstream decodes the batch untiled, which at 2048x2048 is about 15 GiB per image, so the maximum `n = 10` would exceed the 121 GB host. The port decodes one image at a time with single-image tiling (0.55 GiB peak for `n = 2` at 2048x2048). A checkpoint-gated test asserts that this is byte-identical to decoding each image alone; against upstream's untiled batch decode it differs by the tiled vs untiled gap.

### 3.5 Upstream VAE encoder layout bug

Upstream's `_match_conv_layout` cannot distinguish the PyTorch and MLX layouts of `encoder.conv_in.weight` at shape `[128, 3, 3, 3]` and leaves it in PyTorch order. The port computes the correct layout, and the edit fixture was generated with that tensor transposed (`--fix-conv-in`). Parity against unpatched upstream would therefore compare against a known bug.

## 4. Validation and Measured Results

Oracle: mlx-vlm 0.7.8 at git 73d366ec on mlx 0.32.4, CUDA backend, same GB10 host, `MLX_ENABLE_TF32=0`.

| Check | Result |
|---|---|
| Initial noise, generate seed 9909 and edit seed 123 | bit-identical |
| 4B bf16 generate, 512x512, 4 steps | 32.27 dB; CLI and `/v1/images/generations` pixel-identical |
| 4B edit "Add black sunglasses", seed 123 | 40.54 dB; CLI and `/v1/images/edits` pixel-identical |
| 4B 4-bit generate | 35.47 dB vs upstream 4-bit |
| base-4B at 12 steps (resolves to 50 / 4.0) | 32.80 dB |
| 9B bf16 / 9B 4-bit generate | 25.78 / 20.70 dB (same composition, fine detail differs) |
| 9B / 9b-kv edit | 44.39 / 42.94 dB |
| Tiled vs untiled decode at 512 | 25.05 dB, equal to upstream's own 25.05 dB |

Unit tests cover the scheduler at 4 and 50 steps for sequence lengths 1024 and 5000 against oracle bit patterns, prompt buckets for lengths 1, 33 and 600, the text-id layout, tiled decode, kv extract vs cached equivalence, config and variant handling, `--gen-kwargs` parsing, edit sizing, and the 400 mapping of `check_request`. Removing the `FAMILIES` row makes four registration tests fail. Final lib filters: `imagegen::` 167 passed, `flux2` 35, `server::image` 13, `server::routes::images` 18, `models::registry` 16; bin `generate_image` 15; checkpoint-gated parity 6 of 6.

## 5. Learning Points

- A bf16 end-to-end tolerance can be unreachable even for a correct port when the two sides run different MLX pins; the decisive evidence is an f32 run of the same graph on identical inputs, which isolates arithmetic order from port errors.
- Acceptance criteria written before checkpoints are inspected can assume equivalences that do not exist (9b-kv vs 9B weights). Measuring upstream against itself is the quickest test of whether a threshold is meaningful.
- Upstream behavior that is fine on a single Apple device (untiled batch decode) can become a denial-of-service path behind an HTTP endpoint with `n` up to 10.

## 6. Follow-up and Limitations

- Metal and ROCm runs are not recorded; the `arch` row reports Metal as partial.
- The 4B generate case clears the 30 dB gate by about 2 dB, so a future MLX pin change can move it.
- Open MEDIUM findings from the security review: absolute checkpoint paths in 500 responses for load failures (shared behavior across routes), no per-request compute budget beyond the shared caps, and a text-encoder reload for each new prompt (matches upstream eviction).
- LOW: edits with `n > 1` re-encode the same references per image; `9b-kv` step 0 evaluates cached K/V arrays individually; `generate_flux2_fixtures.py` reproduces only the 4B fixtures.
- `250x512` reports the multiple-of-16 rule rather than the 256..=2048 bounds, because the shared validator runs before the family check.

## References

- Issue #2424, epic #2348, runtime issue #2341 (PR #2497), follow-up #2428 (Prism Bonsai)
- Upstream: Blaizzy/mlx-vlm `mlx_vlm/models/flux2` and PRs #1232, #1236, #1381, #1729, #1730, #2355, #2373
- `docs/image-generation.md`, `docs/supported-models.md`
