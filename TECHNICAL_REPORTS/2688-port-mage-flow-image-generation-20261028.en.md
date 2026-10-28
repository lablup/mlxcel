# PR #2688: Port Mage-Flow image generation and editing

**Date**: 2026-10-28
**Status**: Complete on CUDA (GB10); Metal marked Partial (not run), ROCm not run
**Languages**: Rust, Python (fixture generator), Markdown
**Risk Level**: Low (new family; shared-code changes are additive)

## Executive Summary

PR #2688 (issue #2426, epic #2348) ports mlx-vlm `models/mage_flow` (git 73d366ec) as the `mage_flow` image family. It covers all six variants (Base, aligned, Turbo for text-to-image; Edit-Base, Edit, Edit-Turbo for multi-reference editing), the Qwen3-VL conditioner through the existing in-tree VLM loader, the 12-block dual-stream transformer with affine, mxfp4, mxfp8 and nvfp4 quantized projections, and the new Mage-VAE encoder and decoder. The family is wired into `mlxcel generate-image`, `/v1/images/generations`, `/v1/images/edits` (all references in one conditioning pass), the probe, the model registry (all six variant names as aliases) and `mlxcel arch --json`.

Against upstream at 512x512, 4 steps, seed 42, Turbo reaches 38.4 dB PNG PSNR. Edit-Turbo with two references reaches 21.8 dB on our own conditioning and 39.9 dB on upstream's conditioning. The user accepted Edit-Turbo as an exception to the epic's 30 dB criterion and replaced the issue's conditioner gate. The port also found a race in MLX's CUDA LayerNorm kernel, now filed as issue #2696. Mage-Flow avoids it by composing LayerNorm from f32 ops.

## 1. Problem Statement

Mage-Flow (Microsoft, 4B, MIT checkpoints under `mage-flow-community/`) is a native-resolution text-to-image and multi-reference editing family: a Qwen3-VL conditioner, an NR-MMDiT transformer (12 double-stream blocks, hidden 3072, 24 heads, multi-scale RoPE with `axes_dim [16, 56, 56]`), its own Mage-VAE (`DConvEncoder`, `CoDDecoder`-based denoiser with 32-pixel local attention, 128 latent channels, downsample 16) and a static-shift flow-match sampler (default shift 6.0). #524 had recorded it as out of scope. #2348 brought image generation in scope, and after the #2341 runtime merged, a Mage-Flow checkpoint was recognized by nothing and failed at load.

Only the conditioner had an in-tree counterpart: `Qwen3VLModel::forward_hidden` returns final-normed hidden states, which is what Mage-Flow consumes. The issue required reusing it instead of copying the decoder, dropping the conditioner after encoding (the 8B conditioner plus the 4B transformer exceed 24 GB in bf16), and porting the transformer and VAE, which have no counterpart in mlxcel.

## 2. Change Summary

| Area | Change |
|---|---|
| Family module | New `src/imagegen/mage_flow/` (`config`, `conditioner`, `transformer`, `layers`, `pipeline`, `weights`, `mod`, `vae/{blocks,encoder,decoder,mod}`) implementing `ImageGenerationModel`. Generate variants support only Generate and edit variants only Edit. |
| Conditioner | `conditioner.rs` loads Qwen3-VL from `<model>/text_encoder/` through the existing loader, applies upstream's `GENERATION_TEMPLATE` / `EDIT_TEMPLATE`, and drops the first 34 (generate) or 64 (edit) tokens. Edit inserts `Image N: <|vision_start|><|image_pad|><|vision_end|>` per reference and runs the existing Qwen3-VL processor and vision tower. `qwen3_vl.rs` is untouched. |
| Transformer | `transformer.rs`: dual-stream blocks, msrope position ids, `qwen_proj` time embedding; projections through the quantized-linear path when `.scales` are present. |
| VAE | `vae/`: encoder and decoder ports; reuses the FLUX.2 VAE `Conv2d` (now with a `groups` argument for depthwise convs), `GroupNorm` (new `forward_native` in the activation dtype) and the conv-layout probe. |
| Sampling | The shared `FlowMatchEulerDiscrete::static_shift`, as the issue's scheduler contract specified; CFG only when guidance > 1 and a negative prompt is set; optional `renormalization`. |
| Shared code (additive) | `ImageGenerationModel::variant` (default `None`) so task errors name the variant; quantized-tensor validation moved from Z-Image into `src/imagegen/quant.rs` with its error text kept and `QuantSpec` re-exported from `z_image::weights`; visibility changes in `imagegen/vae`; the shared VAE weight error now says `VAE weight not found` instead of `FLUX.2 VAE weight not found`. |
| Registration and docs | `imagegen::FAMILIES`, `probe.rs` (`MageFlowPipeline`), `models/registry.rs`, server image model and worker, `docs/image-generation.md`, `docs/supported-models.md`, `docs/vision-model-scope.md`; `MLXCEL_MAGE_FLOW_ORACLE_DIR` allowlisted in `scripts/ci/check_env_var_docs.py`; fixtures declared in `scripts/ci/check_binary_assets.py`. |
| Fixtures | `tests/fixtures/imagegen/mage_flow_parity.safetensors`, `mage_flow_tiny_refs.safetensors` and the generator `tests/fixtures/generate_mage_flow_fixtures.py`. |

Statistics at `c4468af0`: 40 files, +6107 / -125. Commits: `721e2a06` (port), `6453e542` (docs and fixture declarations), `c19bad02` (deterministic f32 conditioner reference), `2fe3da6a` (conditioner input bounds), `98653b52` (vision markers), `91e2a2a6` (DiT length cap and explicit size), `dd5b2cb9` and `e1efc517` (merges of `origin/main`), `c4468af0` (vision model scope doc).

## 3. Technical Decisions

### 3.1 Parity gate and the Edit-Turbo exception (user decision, epic #2348)

The issue first asked for final-latent relative RMS below 1e-2. During epic #2348 the user replaced that with an epic-wide criterion: per-case latent bounds that reject a wrong port, PNG PSNR >= 30 dB against the CUDA oracle, and bit-identical initial noise. Measured rel-RMS is recorded as drift and does not gate the merge. Issue #2426's acceptance revision of 2026-10-28 records the outcome.

Oracle: mlx-vlm 0.7.8 (git 73d366ec) on mlx 0.32.4, CUDA backend, NVIDIA GB10, `MLX_ENABLE_TF32=0`, 512x512, 4 steps, seed 42. Initial noise is bit-identical in both cases.

| Case | Final latents rel-RMS (drift) | PNG PSNR | On upstream's conditioning |
|---|---|---|---|
| Turbo | 6.6e-2 | 38.4 dB | 3.4e-2, 42.3 dB |
| Edit-Turbo, two references | 2.5e-1 | 21.8 dB | 5.1e-2, 39.9 dB |

Committed bounds:

| Case | Pooled latents bound (measured) | PNG bound |
|---|---|---|
| Turbo | 8e-2 (4.75e-2) | 30 dB |
| Edit-Turbo, own conditioning | 3.5e-1 (2.13e-1) | 18 dB |
| Edit-Turbo, upstream's conditioning | | 30 dB (`ORACLE_COND_PNG_PSNR`) |

Turbo passes the epic criterion. Edit-Turbo on our own conditioning does not, and the user accepted it as an exception with an 18 dB bound on our conditioning and 30 dB on upstream's. The reasoning:

- With upstream's conditioning fed into our transformer and VAE, the edit reaches 39.9 dB. The transformer, VAE and sampler are therefore sound, and the gap is in the bf16 Qwen3-VL conditioner.
- Upstream moves as far against itself. When only its conditioner runs in f32, upstream lands at 22.9 dB against its own bf16 PNG. 21.8 dB is the bf16 precision floor of this conditioner on a two-reference edit, not a port error.

The 18 dB bound only rejects a grossly broken pipeline. It is not a wrong-port detector for the conditioner. That job falls to the conditioner gate in 3.2 and to the 30 dB gate on upstream's conditioning, which catches any error downstream of the conditioner.

### 3.2 Conditioner gate (user decision, epic #2348)

The issue asked for conditioner hidden states within 1e-3 relative RMS of upstream. That bound sits below the bf16 kernel floor. One Qwen3-VL layer on identical input already differs from mlx 0.32.4 by 5.7e-3 (5.6e-3 with upstream's RoPE emulated). Upstream's own bf16 hidden states sit 1.8e-2 (generate) and 2.6e-1 (edit) from the same model widened to f32.

The user replaced the gate rather than loosening it. The new gate requires:

- exact token ids, image grid and M-RoPE positions against upstream;
- a distance of our bf16 hidden states from the f32 reference of at most 1.2x the distance of upstream's bf16 states from that same reference (`F32_DISTANCE_RATIO` in `conditioner_tests.rs`).

Measured ratios: 0.96 (generate, ours 1.7e-2 vs upstream 1.8e-2) and 1.04 (edit, 2.7e-1 vs 2.6e-1). Vision rows from identical pixels differ by 6.9e-2 against upstream's 6.4e-2. Commit `c19bad02` made the f32 reference deterministic (see 3.3).

Caveat from the implementation review, rated HIGH and left to the maintainer: the gate compares 64-channel pooled means (`pool64`), so an error confined within one channel group would not show. The 18 dB end-to-end edit bound would not catch it either. Our pooled edit states sit 2.7e-1 from upstream's bf16 run. That matches precision chaos, since upstream moves 2.8e-1 itself when its conditioner runs in f32, but a structural error of the same size would look identical. Settling it needs an f32-vs-f32 comparison, which needs an f32 Qwen3-VL load path that mlxcel does not have. The user accepted the gate with this gap recorded.

### 3.3 MLX CUDA LayerNorm race (issue #2696)

Identical Edit-Turbo runs gave PNGs 29.8 to 35.5 dB apart, in upstream mlx-vlm and in mlxcel alike. The cause is in MLX's `backend/cuda/layer_norm.cu`. The kernel declares one shared `temp` buffer and calls `BlockBroadcastReduce::Sum` twice, once for the mean and once for the variance. `Reduce` has a `block.sync()` between the `temp` write and the read, and none after the read. A fast warp can write its variance partial into `temp[warp]` before a slower warp has read the mean partial from that slot. On 384-wide bf16 rows (the Mage-VAE encoder's `norm_out`, input `[2, 32, 32, 384]`) a few rows per call come out wrong; on mlx 0.32.4, 30 of 30 calls differed. The edit path's reference latents moved by about 1e-1 between identical calls.

The PR works around it in the family only:

- `layers::LayerNorm::forward` composes LayerNorm from separate ops: cast to f32, mean, centered variance, rsqrt, affine in f32, one cast back. It is deterministic and matches the fused kernel's good rows to one bf16 ulp. The fused kernel rounds the normalized value to the activation dtype before `w * norm + b`, which is where the ulp comes from; the doc comment was corrected during review to say so.
- `vae_tests.rs::layer_norm_is_deterministic_on_384_wide_bf16_rows` is the regression test.
- The fixture generator applies the same composition to upstream's transformer and VAE so that the oracle is reproducible. It was regenerated twice byte-identically.

What stays exposed: GroupNorm in the CoD decoder (`GroupNorm::forward_native`) still calls the fused `fast_layer_norm`, as upstream does, and the generator leaves upstream's GroupNorm fused too. Whether that path is deterministic is not tested. Every other CUDA `fast_layer_norm` user in mlxcel is also exposed (`mlxcel_core::layers::LayerNorm`, Cohere, StableLM, Nemotron, the audio encoder, `dit_blocks`, FLUX.2 embeddings, the FLUX.2 VAE GroupNorm, dense vision towers), and `layer_norm_vjp` has the same hazard. #2696 proposes a CUDA overlay of `layer_norm.cu` with a second `block.sync()` after the `temp` read, which fixes forward and VJP. The file is byte-identical to `ml-explore/mlx` main, so a pin bump would not help.

### 3.4 Request limits added during review

These limits are mlxcel's own; upstream has none. The motivating fact is that the shared `Qwen3VLModel::get_input_embeddings` runs the vision tower with the non-fallible `mlxcel_core::eval`, so an MLX failure there aborts the server instead of returning 500. Before the fixes, `vl_cond_long_edge` accepted `null`, values <= 0 or up to 8192, which with the checkpoint's 4096x4096 processor ceiling and 16 references allowed about 262k Qwen3-VL tokens.

- `2fe3da6a`: `vl_cond_long_edge` at most 2048 (`MAX_VL_COND_LONG_EDGE`); each reference's processor pixel ceiling at 2048x2048 (`MAX_VL_COND_PIXELS`, at most 4096 visual tokens per reference); an edit instruction over 2048 tokens is a 400; a reference past the 200:1 aspect limit is a 400 instead of a 500. Defaults are unchanged.
- `98653b52`: an edit prompt containing a literal Qwen3-VL vision marker (`<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>`, `<|video_pad|>`) is a 400. Before, a stray `<|image_pad|>` failed placeholder expansion as a 500, and a stray `<|video_pad|>` passed it and pushed image features out of alignment with their positions in the shared merge. That gave wrong output with no error, because `masked_scatter` wraps its indices.
- `91e2a2a6`: total DiT image tokens `(1 + references) * (H/16) * (W/16)` are capped at 65536 (`MAX_DIT_IMAGE_TOKENS`, enough for a 2048x2048 edit with three references); above that is a 400 naming the count and the limit. Before, 16 references at 2048x2048 meant about 278k joint tokens per forward, doubled with CFG. Memory stayed bounded through chunked attention and per-step `try_eval`, but one request could hold the image worker for hours. The same commit lets an explicit edit size win over `max_size`, which then applies only when the size is the default derived from the first reference.

Other request rules: sizes 512..=2048 per side, multiples of 16, aspect up to 4:1; a generate variant asked to edit, or the reverse, is a 400 naming the variant. Known deviation: a reference over the shared 16 MP limit is rejected by the shared `ImageRequest::validate` before `max_size` could shrink the output. This fails closed, because the resolved size always passes through `validate_dimensions`, and is documented.

### 3.5 Smaller decisions

- **Variant detection.** The variant comes from the nearest path component naming `mage`, not upstream's whole-path match, so a model under `/data/database/` does not select `-base`.
- **`sample_posterior`** defaults to true, as upstream's pipeline does (the issue text said false; the code follows the pipeline).
- **VAE gates.** Each stage is gated at 5e-3 on upstream's inputs, except the last denoiser block at 2e-2 (measured 9.8e-3). The oracle's CUDA `mx.mean` over bf16 accumulates in bf16, the precedent from #932. End to end the VAE is bounded at 5e-2 (measured 1.0e-2 decode, 2.4e-2 encode).
- **Precision.** Weights stay bf16, as upstream runs them; the steady-state footprint is about 16.5 GB of bf16 weights.
- **Memory.** On the first request the conditioner is dropped after encoding, as the issue required. On later requests it reloads from disk while the transformer and VAE stay resident, so the steady-state peak is all three. The module docs were corrected during review to say so.

### 3.6 Merge with ERNIE-Image (#2687)

ERNIE-Image merged to main as `90d6409a` while this PR was open, and `e1efc517` merged it in. Nine files changed on both sides: `docs/image-generation.md`, `docs/supported-models.md`, both CI scripts, `src/imagegen/mod.rs`, `src/imagegen/registry.rs`, `src/imagegen/vae/model.rs`, `src/imagegen/z_image/weights.rs` and `src/models/registry.rs`. The registry tables, `FAMILIES` and `IMAGE_FAMILIES` now carry both `ernie_image` and `mage_flow`.

The one cross-dependency: `ernie_image/weights.rs` imports `z_image::weights::index_metadata`, and this PR had moved quantized-tensor validation out of `z_image::weights` into `imagegen::quant`. `index_metadata` stayed in `z_image::weights` and `QuantSpec` is re-exported there, so ERNIE's import kept compiling with no change. ERNIE's `// Used by:` comment additions on `index_metadata` and `Flux2Vae::bn_normalize` carried over.

The merge left `docs/vision-model-scope.md` saying `mage_flow` was not in `IMAGE_FAMILIES`, because ERNIE had rewritten the same section. `c4468af0` names `mage_flow` in the `FAMILIES` and `IMAGE_FAMILIES` lists and marks its row registered. The validation numbers in section 4 were measured at `dd5b2cb9`; the two later commits are the ERNIE merge and that doc fix. CI on `c4468af0` was still running when this report was written.

## 4. Validation

GB10, `--profile test-fast --features cuda`, `MLX_ENABLE_TF32=0`, head `dd5b2cb9`:

- `imagegen::mage_flow --include-ignored` 46/46, including the 4 ignored real-checkpoint tests (Turbo and Edit-Turbo parity, both conditioner gates). `imagegen::` 263 passed, 19 ignored; `server::routes::images` 18; `server::image` 14; `models::registry` 17; `arch` 55. Clippy `-D warnings` on lib, bins and tests, and the make verify targets pass.
- The implementation review re-ran the suite on `c19bad02` with a rebuilt binary and reproduced every number: Turbo 38.39 dB, edit 21.80 dB, edit on upstream conditioning 39.87 dB, conditioner ratios 0.960 and 1.042.
- Fixtures come from mlx 0.32.4 CUDA on GB10 and mlx-vlm 0.7.8 at 73d366ec, TF32 off, LayerNorm composed for the transformer and VAE only, and were regenerated twice byte-identically (Turbo 17.4 s / 16.6 GB, edit 19.1 s / 17.8 GB). Processor pixels differ from PIL bicubic by rel-RMS 1.95e-3, at most two 8-bit levels.
- CLI, 1024x1024 Turbo, seed 42, defaults (4 steps, guidance 1): a blue glazed teapot on a red table. Edit-Turbo with two references: the headphones of image 1 on image 2's marble slab; two runs byte-identical.
- Server: Turbo generation 200 (38.4 dB vs the oracle); Edit-Turbo with two references 200, byte-identical to the CLI PNG, and 1248x832 when no size is given. Wrong task, a size not a multiple of 16, a side outside 512..2048 and a bad `static_shift` return 400.
- The unit tests do not compile on main, where the module is absent. They cover variant resolution for all six directory names, msrope position ids against upstream, the VAE stages, templates and processor resize rules, the request bounds and the LayerNorm determinism test.

## 5. Learning Points

- Swap one stage's input for upstream's to localize a PSNR miss. Feeding upstream's conditioning into our transformer and VAE moved Edit-Turbo from 21.8 to 39.9 dB and cleared everything downstream of the conditioner in one run.
- Measure upstream against itself before calling a gap a defect. Upstream at 22.9 dB against itself with an f32 conditioner sets the floor for the edit case, and upstream's own f32 distance sets the scale for the conditioner gate.
- A ratio gate against a shared f32 reference replaces an absolute bound that sits below the precision floor. It inherits the resolution of whatever the comparison pools over, so record that resolution as a limit.
- When two identical runs disagree, check determinism before parity. The 29.8 to 35.5 dB spread between identical edit runs pointed to a kernel race, not to a port error, and it reproduced in upstream.
- A shared path with a non-fallible eval turns large inputs into a crash, not an error. Bound the inputs at the family's request layer until the shared path is fallible.

## 6. Follow-up and Limitations

Fixed during review: conditioner input bounds, vision-marker rejection, the DiT token cap, explicit edit size over `max_size`, a probe unit test for `MageFlowPipeline`, all six variant aliases in the registry (it listed three), the LayerNorm doc comment, and the module docs on steady-state memory.

Open MEDIUM findings:

- The shared `get_input_embeddings` vision path uses non-fallible evals, also on the chat path. The request limits in 3.4 bound the input but do not make the path fallible.
- With CFG, edits run the vision tower twice per request; `get_input_embeddings_with_cache` could reuse the result.
- After the first request the conditioner reloads from disk on every request.
- `GroupNorm::forward_native` (CoD decoder) still uses the fused kernel with the race, and its determinism is not tested.
- `layers.rs` re-implements `scalar_like`, `split_last`, `modulate` and `rms_norm`, which exist in `dit_blocks.rs` and `flux2/embeddings.rs`. Other families keep private copies too.

Open LOW findings: edits with n > 1 re-encode the references once per image; reference pixels are staged in one f32 host buffer (about 805 MB at 16 references of 2048x2048); unquantized transformer and VAE tensors are not shape-checked against the config at load (the checkpoint is operator-trusted).

External follow-ups:

- #2696: the CUDA `layer_norm.cu` overlay. Once it lands, Mage-Flow may switch back to the fused kernel, and the GroupNorm question above resolves.
- An f32 Qwen3-VL load path would allow the f32-vs-f32 conditioner comparison that closes the pooled-mean caveat in 3.2. The PR asks the maintainer whether to add it.

Metal is marked Partial in the registry and was not run; ROCm was not run.

## References

- Issue #2426, epic #2348, runtime issue #2341 (PR #2497), ERNIE-Image PR #2687, LayerNorm race #2696, precision precedent #932, earlier scope note #524
- Upstream: Blaizzy/mlx-vlm `mlx_vlm/models/mage_flow` and PRs #1708, #1979, #2373
- `docs/image-generation.md`, `docs/vision-model-scope.md`
