# PR #2662: Port Z-Image text-to-image and img2img

**Date**: 2026-10-27
**Status**: Complete on CUDA (GB10); Metal and ROCm runs not recorded
**Languages**: Rust, Python (fixture generator), Markdown
**Risk Level**: Medium

## Executive Summary

PR #2662 (issue #2425, epic #2348) ports mlx-vlm `models/z_image` (git 73d366ec) as an image-generation family: text-to-image and img2img for Tongyi-MAI Z-Image-Turbo (8 steps, guidance 0) and Z-Image base (50 steps, guidance 4). It loads the raw Diffusers tree with no conversion step, and also accepts an mlx-vlm conversion (affine, mxfp4, mxfp8 and nvfp4, dequantized at load). The family is registered in `imagegen::FAMILIES` next to FLUX.2 (#2660) and in the model registry, so `mlxcel generate-image`, `mlxcel-server --image-model` and `mlxcel arch --json` all see `z_image`.

Against upstream on the same GB10 host, initial noise is bit-identical in every case and PNG PSNR is 32.9 dB (Turbo), 34.7 dB (base) and 46.1 dB (img2img). The issue's original bound of final-latents relative RMS below 1e-2 is not met (8.6e-2, 6.2e-2, 1.7e-2) and was shown to be below upstream's own reproducibility; during epic #2348 the user replaced it with per-case latent bounds plus PSNR >= 30 dB plus bit-identical noise. Review added a guarded f16 weight cast, query-chunked attention that removes a server-aborting allocation, `try_eval` recovery, and Rust-side validation of quantized tensors.

## 1. Problem Statement

#2341 (PR #2497) ported the Z-Image DiT blocks (`src/imagegen/dit_blocks.rs`) as shared code for LLaDA-Image and Ming-Image, but Z-Image itself did not load. Its Diffusers directory has no top-level `model_type`; after #2341 the probe classified it as `z_image` and then failed with `image generation family z_image is not supported by this build`. Missing pieces were the top-level transformer (patchify, position ids, noise and context refiners, final layer), a VAE built from the FLUX.1 `AutoencoderKL` config (16 latent channels, no `quant_conv` in some exports, no batch-norm stats), conditioning on the penultimate Qwen3 hidden state, the static-shift scheduler with `steps - 1` scheduler steps, CFG with `cfg_truncation`, and the img2img start-index rule.

Upstream asks users to run `convert.py` first. Its load path already sanitizes raw Diffusers weights on the fly, so mlxcel loads the raw tree directly, per the project's direct-checkpoint rule.

## 2. Change Summary

| Area | Change |
|---|---|
| Family module | New `src/imagegen/z_image/` (`config`, `prompt`, `transformer`, `weights`, `pipeline`, `mod`) implementing `ImageGenerationModel` for Generate and Edit. |
| Config and variant | `config.rs` picks Turbo when scheduler `shift <= 3.0`, else base, and parses `cfg_truncation` (default 1.0), `strength` (img2img, default 0.6) and `negative_prompt` from `--gen-kwargs` or extra JSON fields. `img2img_start_index` ports upstream's `int(max(steps - min(steps * strength, steps), 0))`. |
| Transformer | `ZImageTransformer` composed from `dit_blocks` (blocks not copied), with upstream's key remap (`all_final_layer.2-1.`, `all_x_embedder.2-1.`, `t_embedder.mlp.0/2`). |
| Prompt | Chat template with `enable_thinking=True`, padded to 512, the shared Qwen3 conditioner's `hidden_states(...)[num_layers - 1]`, trimmed to the valid length. |
| Weights | `weights.rs`: index metadata, mlx-vlm format detection, quantization table, quantized-tensor validation, dequantize, the guarded f16 cast, VAE key renames. |
| Shared runtime | `Flux2Vae` `quant_conv` optional; `apply_rotary` broadcasts a batch-1 table; DiT and VAE mid-block attention route through `mlxcel_core::layers::attention` (query-chunked); new `InvalidImageRequest` error type that the image worker maps to 400. |
| Registration and docs | Rows in `imagegen::FAMILIES` and `STANDALONE_ARCHITECTURE_FAMILIES` (aliases `z-image`, `z-image-turbo`); `docs/image-generation.md`, `docs/supported-models.md`, `docs/vision-model-scope.md`; test-only variables allowlisted in `scripts/ci/check_env_var_docs.py`. |
| Fixtures | `tests/fixtures/imagegen/z_image_tiny_refs.safetensors` (1.6 KB) and `z_image_parity.safetensors` (198 KB, 4x4-pooled latents and images), generator `tests/fixtures/generate_z_image_fixtures.py`. |

Statistics: 35 files, +3933 / -51. Commits: `e540035b` (port), `7aceac10` (f16 cast guard), `ab732046` (attention memory bound and MLX failure recovery), `dde3116c` (quantized-tensor validation), `94f6553d` (merge of `origin/main` with FLUX.2).

## 3. Technical Decisions

### 3.1 Revising the 1e-2 final-latents bound

The issue asked for final-latents relative RMS below 1e-2 at 512x512, seed 42. The PR measured upstream against itself on the Turbo case: computing only the caption embedder linear in f32 instead of bf16 moves upstream's latents by 2.7e-2 (41.3 dB); running only its text-encoder attention in bf16 instead of f32 moves them by 1.0e-1 (32.1 dB); turning TF32 on moves them by 9.5e-2 (31.6 dB). Eight Euler steps through 30 joint layers amplify bf16-level differences in the caption stream, so 1e-2 is below the floor for any bf16 implementation.

This port's drift is attributed mostly to one known difference: the shared Qwen3 conditioner runs attention in bf16 where upstream casts q, k and v to f32. Caption features differ by 9.3e-3, and feeding upstream's caption features into this denoiser drops Turbo latents from 8.6e-2 to 3.4e-2. `Qwen3Model` was not changed here because it is the live text model, shared with FLUX.2 and Bonsai, and a change would need a logit-trace regression run.

During epic #2348 the user revised the criterion to per-case latent bounds that reject a wrong port, PSNR >= 30 dB, and bit-identical initial noise; measured rel-RMS is recorded as drift and does not gate the merge. The ignored real-checkpoint tests in `src/imagegen/z_image/mod_tests.rs` encode that: `TURBO_LATENTS = 1.2e-1`, `BASE_LATENTS = 1.0e-1`, `EDIT_LATENTS = 5.0e-2`, `MIN_PSNR_DB = 30.0`. The conditioner fix is follow-up #2666, which expects caption features near upstream's own 1.9e-3 and Turbo latents moving toward 3.4e-2, not below 1e-2.

### 3.2 f16 storage with a measured round-trip guard

The CUDA build of mlxcel's MLX promotes bf16 with f32 to bf16, so upstream's f32 latent stream would run in bf16 with bf16 weights (2.1e-2 after one forward, 1.1e-1 final). The transformer and VAE weights are therefore stored as f16, which promotes with f32 to f32. The DiT blocks cast to upstream's promotion at rotation, SDPA and residual adds (no-op on Metal). The caption embedder and the first context-refiner pre-attention path (`reads_caption_stream`) keep bf16, because upstream computes them in bf16 and the joint layers amplify that rounding about 25x.

The first version described the bf16 to f16 cast as exact. Review found it changes 355 of 521 transformer tensors and 76 of 244 VAE tensors on the published checkpoints: range-safe (maximum magnitude 47.75) but not bit-exact, because subnormal bf16 values lose low bits. Commit `7aceac10` measures each tensor's f16 round trip at load and keeps any tensor above `F16_MAX_REL_RMS = 1e-5`, or one that overflows, in f32; if any VAE tensor fails, the whole VAE goes f32. The published checkpoints measure 2.0e-7 worst case, so their numbers did not change. The guard costs two host syncs per tensor at load.

### 3.3 Query-chunked attention instead of a raw SDPA call

The DiT attention called the raw SDPA FFI with f32 q and k. MLX CUDA sends only f16 and bf16 to cuDNN, so the fallback built the full `[1, 30, L, L]` f32 score matrix: about 34 GB per layer at 2048x2048 and about 500 GB at the shared 4096x4096 request cap. The VAE mid-block attention (one head, head_dim 512) had the same shape problem. The allocation failure surfaced as a C++ exception from a non-try `eval`, which crosses the cxx boundary and terminates `mlxcel-server`; `run_guarded` catches only Rust panics.

Commit `ab732046` routes both through `mlxcel_core::layers::attention`, which splits scores into query chunks above `MLXCEL_ATTENTION_CHUNK_BUDGET_MB` (default 1024) and is the same single SDPA call below it. The denoise, reference encode, VAE decode and prompt encode now use `try_eval` and `try_array_to_raw_bytes`, so an MLX failure returns 500 instead of killing the process. Forced-chunk tests (run in a child process with a 1 MiB budget, since the budget is read once per process) match the unchunked output with max abs diff 0 for the DiT block and 3.6e-7 for the VAE attention, and Turbo parity PNGs are byte-identical with `MLXCEL_ATTENTION_CHUNK_BUDGET_MB=64`. Half-precision SDPA would avoid the unfused f32 path but conflicts with upstream parity, so it was not taken. The OOM was not reproduced on GB10 because of the driver-wedge risk; the case rests on the arithmetic.

### 3.4 Validating quantized tensors before MLX

`dequantize_weights` passed checkpoint metadata to the non-try MLX `dequantize`, which throws on unsupported bits or group sizes, shape mismatches, a non-uint32 packed weight, or missing affine biases. Commit `dde3116c` checks each triple in Rust first, matching upstream `_apply_weights`: affine bits {2, 3, 4, 5, 6, 8} and group size {32, 64, 128}; mxfp4 32, mxfp8 32, nvfp4 16; uint32 packed weight of at least two dimensions with a packed width whole at the bit width; scales shaped `[..., in / group_size]`; float scales with matching biases for affine, uint8 scales and no biases for the fp formats. Each error names the tensor key.

### 3.5 Model-raised request errors are 400

Some request errors are only detectable inside the family (an out-of-range `strength`, the wrong number of reference images, guidance on Turbo). Z-Image's `validate_request` checks them before any GPU work and wraps each failure in `InvalidImageRequest` (`src/imagegen/pipeline.rs`); the image worker finds it with `downcast_ref` through any added `anyhow` context and answers 400 instead of 500.

### 3.6 Merge with FLUX.2 (#2660)

FLUX.2 merged first, so `94f6553d` merges `origin/main` into this branch. Both PRs had added a pre-run request check to `src/server/image_worker.rs`. The merged worker runs FLUX.2's `validate_request` (which calls `ImageGenerationModel::check_request`) first and maps its failure to 400, then calls `run_request` with `classify_model_error`, which maps an `InvalidImageRequest` to 400 and anything else to 500. `FAMILIES` became a two-row table (`flux2`, `z_image`), and the registry test now asserts both rows and rejects duplicates. The Z-Image registry row had used `category: "image_generation"`; the merge changed it to `"image generation"`, the spelling FLUX.2 uses, and switched its runtimes to the shared `IMAGE_GENERATE` constant. The docs tables were combined into one family table.

## 4. Validation and Measured Results

Oracle: upstream mlx-vlm at git 73d366ec on mlx 0.32.4, CUDA backend, same GB10 host, `MLX_ENABLE_TF32=0` on both sides, 512x512, seed 42.

| Case | Final latents rel-RMS | PNG PSNR | Test bound |
|---|---|---|---|
| Turbo, 8 steps | 8.6e-2 | 32.9 dB | 1.2e-1 |
| Base, 50 steps, guidance 4, negative "blurry, low quality", `cfg_truncation` 1.0 | 6.2e-2 | 34.7 dB | 1.0e-1 |
| img2img, strength 0.6 | 1.7e-2 | 46.1 dB | 5e-2 |

Initial noise is bit-identical in all three cases. The table's full-resolution numbers come from the scratch oracle tensors; the committed fixture stores 4x4-pooled summaries (Turbo there: pooled latents 3.15e-2, pooled PSNR 36.25 dB).

- CLI, Turbo 1024x1024, seed 42, defaults: 36.9 s wall, clean PNG.
- Server (`mlxcel-server -m qwen3-0.6b --image-model z-image-turbo-bf16`): generation 200 (10.7 s, 512) and edit 200 with a data-URL image and strength 0.6 (5.9 s); strength 0, strength 1.5, two images, and Turbo with guidance 4 each return 400. With TF32 on (server default) the PNGs are 32.4 dB and 46.6 dB against the oracle.
- Unit tests fail on main (module absent): `static_shift` sigmas at shift 3 and 6 with 8 and 49 steps, the `_img2img_start_index` table, transformer key sanitize, VAE sanitize no-op on converted directories, the quant table and validation, the worker 400 mapping.
- Final run (CUDA, TF32 off): imagegen lib 173 passed with 4 ignored, `server::image` 12, `server::routes::images` 17, `models::registry` 16; clippy `-D warnings` and fmt clean.
- ERNIE VAE ignored test after making `quant_conv` optional: f32 numbers unchanged (3.3e-6 / 1.2e-6 / 3.0e-6). Its default-dtype assert fails on CUDA identically on main (bf16 default), so it is pre-existing.

Upstream's Turbo generation default is 9 steps (README); this port uses the 8 that upstream's code and the issue specify.

## 5. Learning Points

- Measure upstream against itself before accepting a numeric tolerance. Toggling one precision setting on the reference showed that 1e-2 was unreachable, and feeding upstream's intermediate (caption features) into the port localized the remaining drift to one component.
- On this MLX build, bf16 with f32 promotes to bf16, unlike upstream MLX. Storing weights as f16 restores upstream's f32 activation stream, but the cast has to be measured per tensor, not assumed exact.
- A C++ exception from a non-try `eval` terminates the server process regardless of Rust panic guards. Any request-driven MLX work behind an HTTP route needs `try_eval` and a bounded allocation shape.
- Two family PRs touching the same worker and registry should converge on one validation path and one spelling for shared enum-like strings during the merge, not in a later cleanup.

## 6. Follow-up and Limitations

- #2666: run the shared Qwen3 conditioner attention in f32 like upstream, via an opt-in flag on `Qwen3Model` attention that only `Qwen3TextEncoder` enables. It affects FLUX.2, Z-Image, Bonsai and Ideogram 4, and must leave the live text model byte-identical.
- Open review findings, not fixed: transformer tensor shapes are not checked against the config at load (a wrong-sized checkpoint aborts on the first request); one request within the shared limits (1000 steps, n=10, 16M pixels, CFG doubling) can hold the single image worker for a long time; timestep-independent work (position ids, RoPE tables, caption embed, context refiner) is recomputed every step (about 1 to 2%); img2img with n>1 re-encodes the reference per image; the text encoder stays resident (about 8 GB) instead of being evicted before the denoiser.
- On CUDA the DiT attention always takes the unfused f32 path (about 2.5 GB of score traffic per layer at 1024x1024).
- Metal and ROCm runs are not recorded in the PR, although the registry row marks Metal as supported.
- `server::image_worker::tests::a_full_queue_sheds_load` is a pre-existing timing flake on main (about 1 in 10 runs).

## References

- Issue #2425, epic #2348, runtime issue #2341 (PR #2497), FLUX.2 Klein PR #2660, follow-up #2666
- Upstream: Blaizzy/mlx-vlm `mlx_vlm/models/z_image` and PRs #1952, #2354, #2373
- `docs/image-generation.md`, `docs/supported-models.md`
