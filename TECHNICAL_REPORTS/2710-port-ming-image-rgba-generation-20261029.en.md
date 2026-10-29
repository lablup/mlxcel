# PR #2710: Port Ming-Image-0.1-Design RGBA text-to-image (ming_image)

**Date**: 2026-10-29
**Status**: Complete on CUDA (GB10), with a maintainer-accepted parity exception for the gen case; Metal and ROCm not run
**Languages**: Rust, Python (fixture generator), Markdown
**Risk Level**: Low to medium (new family; additive changes to the live Bailing MoE model and to shared DiT blocks)

## Executive Summary

PR #2710 (issue #2347, epic #2348) ports mlx-vlm `models/ming_image` (git 73d366ec) as the `ming_image` image family. Ming-Image-0.1-Design is inclusionAI's 6B text-to-image model for text-rich design with native RGBA output. The family loads the raw bf16 tree and the MLX 4-bit and 8-bit trees, and is wired into `mlxcel generate-image`, `/v1/images/generations` (RGBA kept), the probe, the model registry and `mlxcel arch --json`. Edits and guidance other than 1.0 are rejected with 400, as upstream supports neither.

The RGBA case clears the epic's 30 dB gate end to end (30.67 dB, alpha 72.16 dB). The gen poster case is 24.98 dB end to end. The maintainer accepted it as a conditioner-precision exception: on upstream's own conditioning the port reaches 32.45 dB, and upstream perturbed by the same amount of drift lands at 16.37 dB against itself. The issue's RGBA criterion ("alpha below 16 over half the pixels") was revised to alpha PSNR >= 30 dB against the oracle, because upstream itself produces fully opaque output at seed 0.

## 1. Problem Statement

Ming-Image conditions a Z-Image NextDiT on a BailingMoeV2 multimodal LLM: 20 layers x 2048, 256 experts top-8, and two routers per MoE layer (a text `gate` and an `image_gate`, selected per token by an image-patch mask). 256 learned query tokens replace the image-patch span. The last hidden state at the query span goes through `proj_in`, a 28-layer bidirectional Qwen2-style connector and `proj_out` (`cap_feats`). A concatenation of selected layers' hidden states at text positions goes through `directvlm_norm` and `directvlm_proj` (`cap_feats_2`). Decoding uses a 4-channel, non-residual variant of the Qwen-Image VAE, so alpha is decoded directly.

mlxcel's `src/models/bailing_moe.rs` had one router per layer, no per-token router selection, and no per-layer hidden states. The Qwen-Image VAE had just landed in #2706 with Ming's reuse in mind. The issue required reusing the Bailing MoE layers, the Qwen-Image VAE, the Z-Image blocks and the shared scheduler, while keeping the live Bailing MoE text model byte-identical.

## 2. Change Summary

| Area | Change |
|---|---|
| Family module | New `src/imagegen/ming_image/` (`config`, `weights`, `text_encoder`, `transformer`, `pipeline`, `mod`, tests and `parity_tests.rs`), Generate only. Image `i` of a batch uses seed `seed + i`. |
| Image router | New `src/models/bailing_moe_image_router.rs`: per-token selection between `gate` and `image_gate` by a boolean `[B, L]` mask, and an `inputs_embeds` entry point that returns every layer's hidden state plus the final normed state. `bailing_moe.rs` changes only by delegating `router_prefix` and `BailingMoeGate::from_weights` to new `_named` variants with `"gate"`. |
| Conditioner | Learned query tokens, the bidirectional connector, `cap_feats` and `cap_feats_2`; connector and bridge in f32 with bf16 states widened explicitly; prompt templating and tokenization happen before the 33 GB encoder loads. |
| Transformer | Z-Image NextDiT from `src/imagegen/z_image` and `dit_blocks.rs` with a second caption stream and masked zero padding. `dit_blocks::multiply_promoted` applies upstream's dtype promotion to the modulation products; it is a plain multiply when dtypes agree, as they do for Z-Image. |
| VAE | `qwen_image::vae::QwenImageVae` with Ming's 4-channel non-residual config, plus a sanitizer for the raw tree's 5-D kernels. |
| Weights | Raw, MLX 4-bit and MLX 8-bit trees; quantized packing (scale and bias shapes and dtypes) validated before dequantizing at load; expert stacking checks shapes. |
| Request limits | guidance != 1.0, edits, `n` outside 1..=10, steps outside 1..=1000, sizes outside 256..=2048 or not multiples of 16, prompts over 256 KiB or 8192 tokens, and prompts spelling out block tokens are all `InvalidImageRequest` (400). |
| Registration and docs | `imagegen::FAMILIES`, `models/registry.rs`, `docs/image-generation.md`, `docs/supported-models.md`, `docs/vision-model-scope.md`; `MLXCEL_MING_IMAGE_*` env vars allowlisted; fixtures declared in `check_binary_assets.py`. |
| Fixtures | `tests/fixtures/imagegen/ming_image_parity.safetensors` (103 KB), `ming_image_tiny_refs.safetensors` (2 KB), and `tests/fixtures/generate_ming_image_fixtures.py` (modes `tiny`, `real`, `layers`, `perturb`). |

Statistics: 29 files, +4698 / -19. Commits: `0531b6bb` (port), `259397d9` (upstream-conditioning and final-norm checks), `c006b576` (gen gated on its conditioner; CLI, server, docs), `e7245d1d` (format and `Used by:`), `8c00c416` (prompt bounds and router mask validation), plus merges of `origin/main`.

## 3. Technical Decisions

### 3.1 The gen exception and how it is gated (maintainer decision, 2026-10-29)

At 1024x1024, 12 steps, seed 0, TF32 off, against mlx-vlm 73d366ec on mlx 0.32.4 CUDA (GB10), initial noise bit-identical:

| Case | Pooled latents (bound) | Final-latent drift | PNG PSNR | Alpha PSNR | Gate |
|---|---|---|---|---|---|
| gen, bf16 | 3.37e-1 (0.6) | 3.87e-1 | 24.98 dB | 65.81 dB | recorded |
| rgba, bf16 | 3.03e-2 (6e-2) | 9.82e-2 | 30.67 dB | 72.16 dB | gated, >= 30 |
| gen on upstream's conditioning | 3.90e-2 (8e-2) | 1.82e-1 | 32.45 dB | 67.19 dB | gated, >= 30 |
| rgba on upstream's conditioning | 1.38e-3 (1e-2) | 7.08e-3 | 52.69 dB | 82.02 dB | gated, >= 30 |
| q4 vs upstream on the 4-bit tree | 4.44e-2 | 1.70e-1 | 41.91 dB | 75.97 dB | recorded |

The diagnosis localizes the gen gap to accumulated bf16 rounding in the conditioner:

- Each backbone layer run alone on upstream's input state differs by 1e-4 to 7.5e-4, which is rounding level. The final norm alone is bit-exact.
- Through 20 layers the query-span drift reaches 8.8e-2 before the final norm and 2.16e-1 after it. `cap_feats` lands at 2.05e-1.
- Downstream of the conditioner the port is sound: first-step velocity on upstream's inputs is 4.2e-3, and the VAE decodes upstream's final latents at 97.7 dB.
- Upstream with each backbone layer perturbed by 1e-3, which gives the same drift, lands at 16.37 dB (gen) and 29.98 dB (rgba) against its own unperturbed PNG. The poster prompt is chaotic at this drift level even for upstream.

The end-to-end pooled latents cannot catch a broken conditioner. A run that routes query tokens with the text router lands at 1.07e-1, closer to upstream than the correct run. So gen is gated at the conditioner (`cap_feats` <= 0.3, `cap_feats_2` <= 0.12, text states <= 3e-2 on upstream's ids), and the wrong-router arm is asserted to fail at 4.59e-1. The transformer, scheduler and VAE are gated by the upstream-conditioning arm, and dropping `cap_feats_2` there is asserted to fail (pooled latents 7.73e-1).

### 3.2 RGBA criterion revised (maintainer decision)

Upstream's own output at seed 0 is opaque on both RGBA prompts tried (apple and poster), on bf16 and 4-bit: 0.0% of pixels below alpha 16, minimum alpha 227 to 252. The criterion is now that the output carries a real alpha channel and its alpha PSNR against the oracle is >= 30 dB (measured 72.16 dB). That measures port fidelity rather than a seed-dependent model behavior.

### 3.3 Additive router and live-model safety

The image router lives in its own module, and `bailing_moe.rs` only gains `_named` variants that the original functions call with `"gate"`, so keys and error text are unchanged. The logit-trace regression used two tiny seeded random `bailing_moe` checkpoints (4 layers, hidden 64, 8 experts, f32, one with a sigmoid router and expert bias and one with softmax), built fixed and reverted arms of `examples/logit_trace`, and found byte-identical traces (0 of 512 positions disagree, max delta 0). No Ling or Bailing text checkpoint exists on this host, and a text-only view of Ming's own LLM would need its keys renamed, which means copying weights.

### 3.4 Explicit dtype promotion

mlxcel's CUDA MLX build rounds bf16 x f32 to bf16, where upstream promotes to f32. Two places needed explicit widening: the connector and bridge inputs, and the modulation products in the shared Z-Image blocks (`multiply_promoted`). The shared helper does nothing when the dtypes match, so Z-Image and LLaDA-Image compute the same ops as before; LLaDA's denoiser stage test on upstream inputs is unchanged.

### 3.5 Prompt bounds (security review)

The shared request check only rejected empty prompts, and the JSON body limit allows very large ones. Prompt length drives the 20-layer MoE prefill, 21 retained hidden states and the DiT caption stream on every step, so one request could exhaust unified memory. Prompts are now capped at 256 KiB before anything loads and at 8192 tokens after templating. The model keeps its own tokenizer so these checks run before the 33 GB encoder loads.

## 4. Verification

- Narrow suites on the final head: `imagegen::` 440 passed / 40 ignored, `models::bailing_moe` 85, `models::registry` 19, `server::routes::images` 18, `image_worker` 14.
- Ignored real-checkpoint parity with the oracle directory on the final head: 4 of 4 passed, numbers unchanged.
- Scoped clippy, `cargo fmt --check`, contract `make verify-*` targets, `check_binary_assets.py`, `check_env_var_docs.py`: pass.
- CLI: the 4-bit poster renders a legible "HELLO" on blue in 12 default steps; the RGBA prompt writes a 4-channel PNG; `--guidance 3` errors with the guidance 1.0 message; `-n 2` writes seeds 0 and 1, pixel-identical to single runs; bf16 and 8-bit run.
- Server: `/v1/images/generations` 200 with RGBA kept (pixel-identical to the CLI run), `/v1/images/edits` 400 naming the task, bad guidance 400.

## 5. Open Items and Risks

- The gen poster case is below 30 dB end to end by maintainer decision. A future f32 conditioner path would be the way to close it, at memory cost.
- The text encoder reloads from disk per request after the first (same design as Qwen-Image), so peak memory includes all components.
- Unknown `--gen-kwargs` keys are ignored rather than rejected.
- The shared image worker checks the reply deadline only while a request is queued.
- `ming_image::weights::dequantize` duplicates Z-Image's dequantize loop. Quantized transformers are dequantized at load.
- The ignored parity assertions skip without `MLXCEL_MING_IMAGE_ORACLE_DIR`.
- `FinalLayer` in the shared Z-Image blocks still uses fused `fast_layer_norm`, which is exposed to the MLX CUDA LayerNorm race (#2696). Repeated runs here were stable.

## 6. Learning Points

- A loose end-to-end metric can rank a broken component as better than the correct one. Gate at the stage where the defect shows (here, the conditioner), and assert that the broken arm fails.
- Measure the upstream noise floor before calling a miss a bug. Perturbing upstream by the measured drift and comparing it with itself tells you whether 30 dB is attainable at all.
- Acceptance criteria that describe model behavior at one seed (here, transparency) should be checked against upstream first. A fidelity metric (alpha PSNR) is the portable form.
