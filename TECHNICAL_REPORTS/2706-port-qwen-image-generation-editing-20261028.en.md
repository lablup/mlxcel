# PR #2706: Port Qwen-Image-2.1 generation and editing (qwen_image)

**Date**: 2026-10-28
**Status**: Complete on CUDA (GB10); Metal and ROCm not run
**Languages**: Rust, Python (fixture generator), Markdown
**Risk Level**: Low to medium (new family; one additive change on the live Qwen3-VL model, one numeric change in a scheduler mode with a single caller)

## Executive Summary

PR #2706 (issue #2342, epic #2348) ports mlx-vlm `models/qwen_image` (git 73d366ec) as the `qwen_image` image family. It loads the original `Qwen/Qwen-Image-2.1` Diffusers tree without conversion and serves text-to-image and instruction-based editing with one or more reference images, including RGBA references whose alpha goes through the VAE and comes back in the PNG. The family is wired into `mlxcel generate-image`, `/v1/images/generations`, `/v1/images/edits`, the probe, the model registry and `mlxcel arch --json`. Its VAE is built to be reused by Ming-Image-0.1-Design (#2347).

Against the CUDA oracle (mlx 0.32.4, TF32 off, 1024x1024, seed 7, 4 steps) every case clears the epic's 30 dB PSNR gate by a wide margin: generate 52.22 dB, RGBA edit 52.61 dB (alpha 53.58 dB), and the edit with the reference KV cache disabled is bit-identical to the cached run. Initial noise is bit-identical to upstream. Final-latent rel-RMS drift is 7.55e-3 (generate) and 6.38e-3 (edit), under even the issue's original 1e-2 bound.

## 1. Problem Statement

Qwen-Image-2.1 pairs a Qwen3-VL 8B text/vision conditioner (36 layers x 4096) with a 32-block single-stream transformer (32 heads x 128, 3-axis RoPE `[16, 56, 56]`, `causal_condition`) and a 3D-causal residual VAE run per frame as 2D convolutions (`base_dim` 96, `decoder_base_dim` 144, `z_dim` 64, spatial scale 16). Before this PR a Qwen-Image directory was recognized by nothing.

Only the conditioner had an in-tree counterpart, and not exactly: `Qwen3VLModel::forward_hidden` always applies the final RMSNorm, while Qwen-Image consumes the last decoder layer's output before that norm. The issue required reusing the Qwen3-VL decoder rather than copying it, keeping the live Qwen3-VL behavior byte-identical, and shaping the VAE so Ming-Image's 4-channel non-residual configuration can reuse it.

## 2. Change Summary

| Area | Change |
|---|---|
| Family module | New `src/imagegen/qwen_image/` (`config`, `text_encoder`, `transformer`, `blocks`, `kv_cache`, `vae`, `reference`, `pipeline`, `mod`, plus `*_tests.rs` and `parity_tests.rs`) implementing `ImageGenerationModel` with Generate and Edit. |
| Conditioner | `text_encoder.rs` loads Qwen3-VL from `<model>/text_encoder/` through the existing loader with the processor from `<model>/processor/`, computes `drop_idx` (14) from the tokenizer, and runs the vision tower for edit references. The text encoder is dropped after encoding. |
| Qwen3-VL (live model) | `src/models/qwen3_vl.rs` gains `forward_hidden_pre_norm` via a second const generic `FINAL_NORM` on the shared inner forward; every existing caller passes `true`. `// Used by:` now names MageFlow and QwenImage. |
| Transformer | Single-stream blocks with zero-centered RMSNorm, SwiGLU, 3-axis RoPE and causal-condition modulation (two modulation rows), block-causal attention through `mlxcel_core::layers::attention` (query-chunked at large sizes). |
| Edit KV cache | `kv_cache.rs`: prompt and reference K/V are extracted on step 0 and later steps run the target tokens alone; `use_kv_cache: false` disables it. |
| VAE | `QwenImageVae::new` / `load` with a config carrying every field the issue lists plus `decoder_base_dim`; handles residual and plain layouts and both Diffusers and MLX conv layouts (double-permute guarded). `// Used by: QwenImage, MingImage`. |
| References | `reference.rs`: Pillow-exact Lanczos resize and white compositing, so preprocessing matches upstream to the byte. |
| Scheduler | `FlowMatchEulerDiscrete`'s `AccumulateF32` step widens the velocity delta to f32 before the add. Only `FlowMatchEulerDiscrete::dynamic` builds that mode, and Qwen-Image is its only caller. |
| Registration and docs | `imagegen::FAMILIES`, `models/registry.rs` (image generation category plus a test), `docs/image-generation.md`, `docs/supported-models.md`, `docs/vision-model-scope.md`; `MLXCEL_QWEN_IMAGE_ORACLE_DIR` and `MLXCEL_QWEN_IMAGE_OUT` allowlisted; fixtures declared in `check_binary_assets.py`. |
| Fixtures | `tests/fixtures/imagegen/qwen_image_tiny_refs.safetensors` (304 KB), `qwen_image_parity.safetensors` (96 KB) and the generator `tests/fixtures/generate_qwen_image_fixtures.py`. |

Statistics: 29 files, +5684 / -14. Commits: `397e84ed` (port), `45839010` and `076f371d` (precision and CLI numbers in docs), `1c5519f6` (vision model scope doc), `f0fb5440` (edit memory bounds), `22a68ed3` (early request rejection), and merges of `origin/main` (`8887f10b`, `6fb5c1f0`, `b5b6ebe9`).

## 3. Technical Decisions

### 3.1 Parity gate (epic #2348 user decision)

The issue asked for final-latent rel-RMS below 1e-2 and decoded-image rel-RMS below 2e-2. Epic #2348 replaced that with per-case latent bounds that must reject a wrong port, PNG PSNR >= 30 dB against the CUDA oracle, and bit-identical initial noise, with rel-RMS recorded as drift.

| Case | Final latents rel-RMS (drift) | 8x8-pooled latents (bound 2e-2) | PNG PSNR |
|---|---|---|---|
| Generate | 7.55e-3 | 5.57e-3 | 52.22 dB |
| Edit, RGBA reference, KV cache on | 6.38e-3 | 4.45e-3 | 52.61 dB (alpha 53.58 dB) |
| Edit, KV cache off | 6.38e-3 | 4.45e-3 | 52.61 dB; inf against the cached run |

The pooled-latent bound is shown to reject a wrong port: swapping the RoPE height and width axes gives 0.770 and 13.22 dB, and putting every token on the real-timestep modulation row gives 0.234 and 18.34 dB. Component checks on identical inputs: first-step velocity 6.3e-3 / 6.9e-3, reference latents 1.66e-3, VAE decode of upstream's final latents 97.79 dB, real VAE decode 2.07e-4 and encode 5.7e-7, conditioner text positions 3.65e-2 / 3.71e-2 (the `<|image_pad|>` rows differ more, 0.59, but the transformer replaces them with VAE tokens).

Unlike Mage-Flow (#2688), the edit case did not need an exception. The conditioner drift here is about the same as Mage-Flow's generate case, and the transformer drops the image-pad rows where the bf16 vision tower diverges most.

### 3.2 Pre-norm hidden states through a const generic

The pre-norm path is a second const generic on the existing inner forward, not a runtime flag or a copy. Existing callers instantiate `<_, true>` and compile to the same ops, and a teacher-forced logit trace (fixed vs reverted `qwen3_vl.rs`) shows 0 disagreements on 512 text positions and 128 image positions, with byte-identical TSV output. The alternative, a runtime `bool`, would have added a branch to the live decode path; a copy would have violated the issue's reuse rule.

### 3.3 Widening the velocity delta in the f32 Euler step

Upstream adds a bf16 delta to an f32 sample and MLX promotes the sum to f32. In mlxcel's CUDA MLX build that add rounds to bf16, which cost the f32 accumulation its purpose. The fix casts the delta to f32 first. It touches shared code, but only the `AccumulateF32` mode, which only `dynamic` builds and only Qwen-Image calls, so Z-Image, Mage-Flow and the other families are unaffected.

### 3.4 Edit memory bounds (security review)

The first version copied Mage-Flow's 65536 image-token cap. Qwen-Image costs more per token: the `S x S` bool block-causal mask is about 4.4 GB at 65536 plus intermediates, and the edit KV cache keeps `S * 512 KiB` per CFG branch (about 34 GB, 68 GB with CFG) next to a 40 GB transformer. The cap is now 32768 (a 2048 edit with one 2048 reference, or a 1024 edit with up to seven). Step-0 extraction also evaluates each layer's compacted prefix as it is stored, so the full-sequence K/V of all layers is not held alive through the whole first-step graph; values are unchanged (the edit parity numbers did not move).

Request checks that need no tokenizer (vision markers in the prompt or CFG negative, references the processor would resize again, the token cap) moved into `resolve`, so a bad edit is a 400 before the 36-layer text encoder loads.

### 3.5 Precision

The transformer follows `bf16_to_f16_at_load` (f16 on Metal, bf16 on CUDA unless `MLXCEL_CUDA_F16_NORMALIZE=1`). With f16 forced on CUDA it stays finite at 48.32 dB (generate) and 47.92 dB (edit). The VAE stays f32 as stored, as upstream runs it.

## 4. Verification

- `imagegen::` 373 passed / 30 ignored; `server::routes::images` 18; `image_worker` 14; `models::registry` 18; `qwen3_vl` 51.
- `imagegen::qwen_image --ignored` against the real checkpoint and oracle outputs: 6 passed on the final head.
- Scoped clippy with `-D warnings`, `cargo fmt --check`, the contract `make verify-*` targets, `check_binary_assets.py` and `check_env_var_docs.py`: pass.
- Real checkpoint: CLI generate (30 steps, 119.5 s) produced a coherent red teapot; CLI edit recolored it blue; the RGBA edit kept the elliptical alpha mask (39% transparent vs 41% in the input); the server returned 200 on both routes and 400 on a 1000x1000 size, `use_kv_cache` on a generate request, and `<|image_pad|>` in an edit prompt.

## 5. Open Items and Risks

- An explicit edit size equal to the first reference's default is treated as "no size given" (same heuristic as Mage-Flow). Fixing it needs a "size given" flag on the shared `ImageRequest`.
- The shared `src/vision/qwen3_vl.rs` `get_input_embeddings` uses non-fallible evals, so an MLX failure in the vision tower aborts the server instead of returning 500. This is shared with Mage-Flow and Qwen3-VL chat.
- The block-causal mask is rebuilt on every uncached step; the token cap bounds it.
- With `MLXCEL_CUDA_F16_NORMALIZE=1`, ERNIE-Image's parity tests fail with a VAE decode error. That is independent of this PR but has no issue yet.
- Metal and ROCm were not run. The parity gate used TF32 off; CLI and server runs used the default.

## 6. Learning Points

- Check where the bf16 drift lands before judging a conditioner. The edit conditioner's image-pad rows are 0.59 off, which looks alarming, but the transformer overwrites those rows with VAE tokens, so the end-to-end edit is at 52 dB.
- Dtype promotion is a backend property. A mixed f32 + bf16 add that promotes in upstream MLX may round to bf16 in another build; widen explicitly where accumulation precision matters.
- Copying a request cap from a sibling family is not safe when the per-token memory differs. Derive it from the family's own mask and cache sizes.
