# PR #2703: Port Ideogram 4 text-to-image with per-row FP8 weights

**Date**: 2026-10-28
**Status**: Complete on CUDA (GB10); Metal marked Partial (not run), ROCm not run
**Languages**: Rust, Python (fixture generator), Markdown
**Risk Level**: Low (new family; shared-code changes are additive)

## Executive Summary

PR #2703 (issue #2427, epic #2348) ports mlx-vlm `models/ideogram4` (git 73d366ec) as the `ideogram4` image family for the license-gated `ideogram-ai/ideogram-4-fp8` snapshot. It covers the two FP8 transformers (conditional and unconditional), per-row FP8 dequantization, interleaved M-RoPE, the logit-normal scheduler with three sampler presets, JSON caption wrapping, a 13-layer stacked text-feature path through the existing Qwen3 text encoder, and the FLUX.2 VAE decode. It is wired into `mlxcel generate-image`, `/v1/images/generations`, the probe, the model registry and `mlxcel arch --json`. Generation only: `/v1/images/edits` is a 400.

Against upstream at 512x512, `V4_TURBO_12`, seed 0, tokens and initial noise are bit-identical, the pooled final latents are at 3.16e-2 rel-RMS and the PNG is at 37.13 dB. Most of the remaining drift comes from the bf16 text tower in the shared conditioner (issue #2666): with upstream's own text features the PNG reaches 45.95 dB. The real checkpoint runs on GB10 through both the CLI and the server and produces images that match upstream (47.88 dB for the issue's command).

## 1. Problem Statement

Ideogram 4 is a text-to-image model that conditions a DiT on a stack of 13 hidden-state layers of a Qwen3 text tower, using a structured JSON caption as the prompt. Its checkpoint stores both transformers and the text tower as FP8 (`F8_E4M3` bytes plus a per-row f32 `weight_scale`). #2348 brought image generation into scope, and after the #2341 runtime merged, an `ideogram-4-fp8` snapshot was recognized by nothing and failed at load.

The pieces with an in-tree counterpart were the Qwen3 text encoder (`Qwen3TextEncoder::from_weights`), the FLUX.2 VAE and the FP8 byte decoder `f8_e4m3_to_f32`. The DiT blocks, M-RoPE layout, scheduler, prompt normalization and per-row dequantization had none. The issue also required a reference-parity gate against the upstream implementation and a real-checkpoint CLI run.

## 2. Change Summary

| Area | Change |
|---|---|
| Family module | New `src/imagegen/ideogram4/`: `config`, `scheduler`, `prompting`, `latent_norm`, `weights`, `blocks`, `transformer`, `pipeline`, `mod`, each with a `*_tests.rs` sibling, plus `parity_tests.rs`. Implements `ImageGenerationModel` for generate only. |
| Scheduler | Logit-normal scheduler with the three presets (`V4_DEFAULT_20`, `V4_TURBO_12`, `V4_QUALITY_48`) and an inverse normal CDF following CPython's AS241 so the schedule matches upstream literals. Guidance 7 then 3 on the last two steps for the default preset. |
| Prompting | JSON caption passthrough, minimal-caption wrapping when `auto_json_caption` is on, escaping, and caption warnings. The wrapped caption is returned as `revised_prompt`. |
| Weights | `weights.rs` dequantizes per-row FP8 to bf16 in upstream's rounding order, for the transformers and the text tower. Shape and dtype checks run at load. |
| Transformer | `blocks.rs` and `transformer.rs`: DiT blocks with interleaved M-RoPE, conditional and unconditional variants. The unconditional one skips `llm_cond_*`. |
| Pipeline | Tokenizer via the snapshot chat template, 13-layer stacked text features, global-seed noise, guidance loop, FLUX.2 VAE decode. |
| Shared code (additive) | `pub(crate)` re-export of `f8_e4m3_to_f32`, a pointer comment in `fp8_block.rs`, a `Used by` update in `imagegen/vae/mod.rs`. The shared Qwen3 conditioner is not modified. |
| Registration and docs | Probe `IMAGE_FAMILIES`, image `FAMILIES`, `STANDALONE_ARCHITECTURE_FAMILIES` (aliases `ideogram-4`, `ideogram-4-fp8`); new section, precision rows and parity table in `docs/image-generation.md`; rows in `docs/supported-models.md` and `docs/vision-model-scope.md`. |
| CI and fixtures | `ideogram4_parity.safetensors` (46 KB) and `ideogram4_tiny_refs.safetensors` (61 KB) with `tests/fixtures/generate_ideogram4_fixtures.py`; asset budget in `scripts/ci/check_binary_assets.py`; `MLXCEL_IDEOGRAM4_ORACLE_DIR` and `MLXCEL_IDEOGRAM4_OUT` allowlisted in `scripts/ci/check_env_var_docs.py`. |

Size: 34 files, +4818 / -8.

## 3. Technical Decisions

### 3.1 Parity gate design

**Context:** The port drifts from upstream for reasons outside this PR (bf16 text tower, #2666), so a single end-to-end PSNR threshold cannot tell a correct port from a wrong one.

**Gates (CUDA, oracle mlx 0.32.4, mlx-vlm 0.7.8, `MLX_ENABLE_TF32=0` on both sides):**

| Measure | Value | Gate |
|---|---|---|
| Token ids (45) | equal | exact |
| Initial noise | bit-identical, FNV-1a `0xa671abe7` | exact |
| 4x4-pooled final latents rel-RMS | 3.16e-2 | at most 4.5e-2 |
| 8x8-pooled image PSNR | 43.72 dB | at least 38 dB |
| Full PNG PSNR | 37.13 dB | at least 30 dB |
| Full final latents rel-RMS | 6.18e-2 | recorded, not gated |
| Text features rel-RMS | 6.40e-2 | recorded, not gated |
| First-step velocities rel-RMS | 6.17e-3 cond, 3.17e-3 uncond | recorded, not gated |
| Upstream's text features fed to the port | latents 2.87e-2, PNG 45.95 dB | diagnostic |

This follows the maintainer's #932 contract: per-op same-input parity plus greedy exactness gate, accumulated intermediate drift is recorded.

**Controls:** Two perturbed upstream runs check that the gate can fail. Wrong noise (seed 1) gives pooled latents 4.2e-1, 12.5 dB and 12.3 dB, rejected by everything. Ignoring the preset schedule (guidance 7 instead of 3 on the last step) gives pooled latents 5.2e-2, 51.7 dB and 48.1 dB: it passes every image gate.

**Decision:** The pooled-latent bound was tightened from 6e-2 to 4.5e-2 in review. The schedule control passed 6e-2 and all image gates because the text-tower drift costs more PSNR than a one-step guidance error, so the latent bound is the only discriminator for that class of bug. After the change the control (5.2e-2) is rejected and the port (3.16e-2) passes. The margin is narrow on both sides, which is why the numbers are in the PR and the docs.

### 3.2 Request errors are `InvalidImageRequest`

`size` and aspect, empty prompt, prompt over 1 MiB, malformed JSON caption, unknown `sampler_preset`, `guidance_schedule` length, more than 2048 prompt tokens, `prompt_expansion_model`, non-empty `negative_prompt` and the edit task all map to `InvalidImageRequest`, so the server answers 400 and a bad request cannot surface as a 500.

A non-empty `negative_prompt` is rejected as a deliberate deviation: upstream's `generate_array` takes `**kwargs` and silently ignores it. `prompt_expansion_model` is rejected rather than ported.

### 3.3 Input hardening added in review

- A 1 MiB prompt byte cap, checked before JSON parsing and tokenization. It cannot reject a prompt that fits in 2048 tokens, since the longest vocab token is 128 bytes.
- Caption warnings are collected in linear time, capped at 64, with 8 logged and control characters escaped.
- Overflow-checked config arithmetic: `heads * head_dim`, `mrope_section` entries bounded by `head_dim / 2`, `num_layers` in 1..=4096, and `guidance_schedule` length capped before collection.
- Shape and dtype checks on every transformer tensor against the config at load. A mismatched or unscaled FP8 weight is a load error instead of a C++ exception inside a bridge op on the first request.

### 3.4 Memory lifecycle follows upstream

The tokenizer loads at `load()`. The text encoder (about 16 GB bf16) loads per request and is evicted with `clear_memory_cache` before the transformers run. The two transformers and the VAE load lazily and stay resident (about 38 GB). The unconditional transformer no longer reads or dequantizes `llm_cond_*` (about 0.5 GB, unused there); the conditional one still requires them.

**Tried and reverted:** Projecting only the text rows of the LLM features would save about 4 GB at 2048x2048. It changed the GEMM row count versus upstream and dropped the upstream-features diagnostic from 45.95 dB to 31.8 dB, so the projection stays over all rows. Left as a possible follow-up.

### 3.5 Shared conditioner left alone

The text tower goes through the existing `Qwen3TextEncoder::from_weights`, and attention precision in it is not touched. #2666 owns that, and changing it here would have mixed a family port with a change to every family that uses the conditioner.

### 3.6 Smaller decisions

- `num_images` follows the shared convention, `seed + i` per image. Upstream draws one batched noise tensor from one seed. Documented.
- The noise unit test allows one ulp at element 510 because CUDA and CPU `erfinv` differ there. Bit-identity of the whole tensor is gated only in the ignored CUDA parity test.
- The registry marks Metal as `Partial` without a Metal run.

## 4. Validation

All with `MLX_CUDA_ARCHITECTURES=121`, `--profile test-fast --features cuda`, NVIDIA GB10, final head at the time of the PR body.

- `imagegen:: models::registry server::image_worker --test-threads=1` under `gpu-lock`: 377 passed, 0 failed, 24 ignored (real checkpoints).
- New tests: 36 non-ignored plus 1 ignored real-checkpoint parity test. They cover scheduler literals for all three presets at two resolutions, prompt wrapping literals and caption warnings, per-row FP8 dequantization against an upstream 4x8 literal, noise literals for seeds 0 and 7, and a tiny two-layer transformer with random bf16 weights against upstream (conditional and unconditional outputs rel-RMS 0, M-RoPE tables equal).
- Regression check: removing `"ideogram4"` from `IMAGE_FAMILIES` makes the probe test fail with `left: Some("flux2") right: Some("ideogram4")` and the routing test fail as well.
- Clippy with `-D warnings` is clean; `make verify-*` targets and `check_binary_assets.py` pass.
- Parity test run with `MLX_ENABLE_TF32=0`: reproduces every table figure exactly, also after the review commits.

Real checkpoint (`models/mlx/ideogram-4-fp8`):

- CLI, the issue's command (1024x1024, seed 0, `V4_DEFAULT_20`): exit 0, 3 min 01 s including first-run kernel compile; upstream 173 s. The output is the checkpoint's own "Image blocked by safety filter" placeholder and matches upstream at 47.88 dB (max abs pixel difference 27). That is what the checkpoint renders for this prompt and seed, not an mlxcel filter.
- Server, 512x512, `V4_TURBO_12`, seed 1: a plain-text prompt returns 200 in 43.4 s with a `revised_prompt` byte-identical to upstream's wrapped caption (image is again the placeholder, 51.26 dB vs upstream). With `auto_json_caption: false` it returns 200 in 17.6 s, `revised_prompt: null`, a watercolor of two red foxes at 26.0 dB against upstream (detail-level difference; the text-tower attribution is inferred from the turbo_512 diagnostic, not rerun on this case). `/v1/images/edits` returns 400 naming the unsupported `edit` task; `sampler_preset: "V4_FAST"` returns 400 listing the valid presets.

## 5. Learning Points

- Per-row FP8 is a storage format, not a compute format: the checkpoint multiplies each row of the FP8 matrix by its own f32 scale, and the result is rounded to bf16 in a fixed order. Matching upstream's rounding order is what makes the tiny-transformer test bit-identical.
- A parity gate must be shown to fail. The schedule control proved a single-step guidance error survives every image-space metric when another component adds larger drift; only the latent bound caught it.
- Reproducing a sampler's RNG needs the same inverse CDF. Implementing AS241 as CPython does gave bit-identical noise from the same seed.
- A passing real-checkpoint run is not evidence of a useful demo: the issue's required prompt triggers the checkpoint's safety placeholder, which also confirms the port since upstream renders the same image.

## 6. Follow-up and Limitations

- #2666: bf16 attention precision in the shared Qwen3 conditioner is the main source of the 37 dB vs 46 dB gap.
- Possible follow-up: project only text rows of LLM features (about 4 GB at 2048x2048), if a way to keep the upstream GEMM shape is found.
- Not run: Metal, ROCm. `V4_QUALITY_48` was not run end to end; its schedule is checked against upstream literals only.
- Memory figures in the docs (about 38 GB resident, about 54 GB while encoding) are computed from tensor sizes, not measured peaks. The 2048x2048 peak was reviewed from the graph, not measured.
- The text encoder reloads on every request, matching upstream. Reusing it across requests would trade about 16 GB of residency for latency.
- The issue's example prompt is not a good documented example because it hits the safety placeholder.

## References

- Issue #2427, epic #2348, issue #2666 (conditioner attention precision), maintainer decision #932 (parity contract).
- Related ports: PR #2688 (Mage-Flow), PR #2687 (ERNIE-Image).
- Upstream: mlx-vlm `models/ideogram4` at `73d366ecf9233b3fc6549531a7f2386726d09ff6`.
- Docs: `docs/image-generation.md` (Ideogram 4 section and parity table).
