# PR #2667: Port Prism Bonsai ternary 2-bit FLUX.2

**Date**: 2026-10-27
**Status**: Complete on CUDA (GB10); Metal runs and the Metal megakernel comparison not verifiable on this host
**Languages**: Rust, Python (fixture generator), Markdown
**Risk Level**: Low

## Executive Summary

PR #2667 (issue #2428, epic #2348) adds the `bonsai` image-generation family for `prism-ml/bonsai-image-ternary-4B-mlx-2bit`: the FLUX.2 Klein 4B transformer with ternary linears packed as 2-bit MLX affine weights (group 128) and a 4-bit MLX Qwen3 text encoder. It is built on the FLUX.2 port merged in PR #2660 rather than a copy: the new module only supplies a packed-weight `LinearSource`, the quantized text encoder, and prompt padding, and reuses the FLUX.2 pipeline, scheduler, prompt path, latent helpers and VAE. The family runs through `mlxcel generate-image`, `mlxcel-server --image-model` (`/v1/images/generations`), and `mlxcel arch`.

Against the CUDA build of upstream mlx-vlm (with Bonsai's Metal-only fused kernels replaced by their arithmetic), the 512x512, 4-step, seed 9909 case has bit-identical initial noise and a decoded PSNR of 31.18 dB. The f32 transformer agrees with upstream f32 to 1.19e-6 relative RMS. FLUX.2's own parity numbers are unchanged.

## 1. Problem Statement

Bonsai's `model_index.json` names `Flux2KleinPipeline`, but its checkpoint layout differs from a FLUX.2 Diffusers tree: `transformer-packed-mflux/` with a `quantization_config.json` and packed `uint32` linears, `text_encoder-mlx-4bit/`, and a `manifest.json`. The probe rule that classifies it as `bonsai` before the class-name scan had already shipped with #2341, but no loader existed, so the family was reported as unsupported. Upstream's fast path (`klein_fast`) uses `mx.fast.metal_kernel` with no CUDA or CPU fallback, so neither the implementation nor the parity reference could simply mirror upstream's runtime code on this host.

## 2. Change Summary

| Area | Change |
|---|---|
| Family module | New `src/imagegen/bonsai/` (`config.rs`, `loader.rs`, `mod.rs`, tests, checkpoint-gated `parity_tests.rs`). Generate only; defaults 4 steps, guidance 1.0; prompts padded to 512 tokens (upstream `bucketed_seq_len=False`); `n` images run one at a time with seed `seed + i`, as upstream's server loops. |
| Packed loader | Reads `quantization_config.json`; rejects any `format` other than `mlx-packed-affine` and any `bits` or `group_size` outside mlxcel-core's affine sets (the 1-bit binary variant fails on `bits (1)`). Modules in `quantized_modules` load as affine `UnifiedLinear`; others load dense. A listed module without `.scales`, or an unlisted module with `.scales`, is an error. Each packed layer passes `validate_quantized_packing`. |
| FLUX.2 seam | `Flux2Pipeline::with_linear_source` (`new` delegates to it with the default source and bucketing on); a default-accepting `LinearSource::check_weights` hook called from `Flux2Transformer::load`; wider visibility on a few helpers, each with a `Used by:` comment. |
| Registration | Rows in `imagegen::FAMILIES` and `STANDALONE_ARCHITECTURE_FAMILIES`, appended after FLUX.2 and Z-Image; a probe test for the no-manifest case. |
| Docs and CI | `docs/image-generation.md` Bonsai section with precision rows; `docs/supported-models.md` row; two test-only env vars allowlisted. |
| Fixtures | `tests/fixtures/bonsai/` (about 1.45 MB) and `generate_bonsai_fixtures.py`. |

## 3. Technical Decisions

### 3.1 Treat `quantized_modules` as the source of truth

Upstream treats any linear with a sibling `.scales` tensor as packed and never consults `quantized_modules` or `skip_patterns`. The port uses the list and errors when the list and the file disagree in either direction. On the real checkpoint the 100 modules with `.scales` are exactly the 100 listed, so behavior is the same for valid checkpoints while a corrupted or mismatched export fails at load with the module name.

### 3.2 Validate packed shapes at load

The security review found that biases shaped differently from scales, a leading-axis mismatch between weight and scales, or 1-D scales all loaded without error and then failed MLX's own shape check on the first forward pass. That failure cannot be caught across the C++ bridge, so `mlxcel-server` would abort on its first request. Running the existing core check `validate_quantized_packing` on every packed layer turns these into load errors. A checkpoint whose tensors are mutually consistent but packed for a different `hidden_size` than `config.json` declares still fails late; fixing that requires passing the expected input width through `LinearSource::load`, which affects FLUX.2 as well and was left as a follow-up.

### 3.3 Use the checkpoint's `vae/` instead of upstream's downloaded decoder

Upstream `bonsai/weights.py::load_vae` ignores the checkpoint `vae/` and downloads `black-forest-labs/FLUX.2-small-decoder` (channels 96/192/384/384) at load time. The issue names `vae/`, and a network fetch at model load is not acceptable for a server, so the port decodes with `vae/`. The two decoders differ from each other by 34.96 dB; the port measures 31.18 dB against an oracle that also uses `vae/`, and 29.82 dB against upstream as shipped. An optional local small-decoder path is a possible follow-up.

### 3.4 Building a CUDA reference without the Metal kernels

The fixture generator runs upstream's own `BonsaiImage` pipeline with the fused norm-RoPE kernels replaced by their arithmetic (f32 RMSNorm times gain, interleaved-pair RoPE, cast to bf16). A `--flux2-transformer` mode loads Bonsai's weights into upstream's FLUX.2 transformer for the f32 check, with weights dequantized because mlx 0.32.4 CUDA fails f32 2-bit `quantized_matmul` with more than one row. The megakernel itself cannot be compared against `quantized_matmul` here; by code reading, `QuantizedLinearKernel` calls `mx.quantized_matmul`.

### 3.5 Latent tolerance under the revised epic criterion

Following the epic-wide maintainer decision, the tests gate on PSNR at least 30 dB, bit-identical initial noise, a 0.2 final-latent bound (a wrong port scores about 1.4), and 0.1 on prompt embeddings. The measured final-latent relative RMS of 1.0e-1 and prompt-embedding 2.3e-2 are recorded as drift. The evidence that this is bf16 rounding rather than a port error: upstream's bf16 4-bit conditioner is 1.95e-2 from its own f32 run and this port is 1.74e-2 from the same f32 run, and upstream disagrees with itself by 3.5e-2 in final latents when its FLUX.2 transformer replaces `klein_fast` on the same weights. The shared Qwen3 conditioner precision was deliberately left unchanged; a separate follow-up covers f32 q/k/v attention there with a live-model logit-trace gate.

## 4. Validation and Measured Results

Oracle: mlx-vlm 0.7.8 at git 73d366ec, mlx 0.32.4 CUDA on GB10, `MLX_ENABLE_TF32=0`.

| Check | Result |
|---|---|
| Initial noise | bit-identical |
| f32 transformer vs upstream f32 | rel RMS 1.19e-6 |
| One bf16 step on upstream's step-0 inputs | rel RMS 8.1e-3 |
| VAE decoding upstream's final latents | 57.61 dB |
| End-to-end PNG (CLI acceptance command) | 31.18 dB; server output pixel-identical |
| Same case with `MLXCEL_CUDA_F16_NORMALIZE=1` | 31.21 dB |
| CLI run | 512x512 in 11.5 s including load, 3.6 GB RSS |

Lib filters on the final tree: `imagegen::` 220 passed, `models::registry` 16; `bonsai::parity --ignored` 5 of 5; `flux2::parity --ignored` 6 of 6 with PR #2660's exact numbers. Removing the `FAMILIES` row fails two registration tests; the loader and config tests do not exist on main.

## 5. Learning Points

- A quantized variant of an existing family is cheapest to port as a weight-source seam: one trait method for building linears and one for validating the weight map kept the FLUX.2 graph untouched and its parity bit-stable.
- A shape error that MLX raises inside a forward pass is fatal across the FFI boundary, so checkpoint validation belongs at load time where it can return an error.
- Upstream reference code that depends on Metal-only kernels can still serve as a CUDA oracle when the kernels are replaced by their documented arithmetic, provided the PR states exactly what was substituted.

## 6. Follow-up and Limitations

- Metal runs and the megakernel vs `quantized_matmul` comparison are not verifiable on this host.
- Shared MEDIUM: a checkpoint packed for a different `hidden_size` than its config still fails on the first forward pass (FLUX.2 and Bonsai).
- Optional: a local path for upstream's small decoder, if matching upstream as shipped becomes a goal.
- Shared Qwen3 conditioner f32 attention follow-up (filed by the epic orchestrator).

## References

- Issue #2428, epic #2348, FLUX.2 port #2424 (PR #2660), runtime #2341 (PR #2497), Z-Image #2662
- Upstream: Blaizzy/mlx-vlm `mlx_vlm/models/bonsai` and PRs #1226, #1234, #2373
- `docs/image-generation.md`, `docs/supported-models.md`
