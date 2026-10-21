# PR #2459: LFM2-VL resampling, Qwen3-VL Conv3d layout guard and quantized pos_embed

**Date**: 2026-10-19
**Status**: Implemented and verified on CUDA (GB10) with LFM2-VL, LFM2.5-VL, Qwen3.8-27B-FP8 and Qwen3-VL checkpoints; PR open

## Summary

Closes #2388. Ports three upstream vision fixes: LFM2-VL position-embedding resampling with bilinear antialias plus the processor's configured resample filter, a Conv3d patch-embed layout guard at every Qwen3-VL tower load site, and quantized `pos_embed` support for Qwen3-VL. It also fixes a load failure on `lfm2-vl-1.6b-4bit` that blocked validating the first item.

## Problem

- **LFM2-VL tower.** The learnable 2D position table was resized with bicubic interpolation. The reference uses bilinear with antialias, so every image whose patch grid differs from the table size got shifted position embeddings.
- **LFM2-VL processor.** Images were always resized with one fixed filter regardless of the checkpoint's `resample` value.
- **Qwen3-VL Conv3d.** Four Qwen3-VL family sites and Cohere Compass built the vision tower without `normalize_patch_embed_layout`, so a hub-layout patch weight (`[O, C, T, H, W]`) was read as MLX layout. On Qwen3.8-27B-FP8 the model described "vertical stripes alternating red and green" for a picture of three shapes.
- **Qwen3-VL pos_embed.** `PositionEmbedding` assumed a dense table and could not load a quantized one.
- **LFM2-VL-1.6B load (outside the issue).** The config declares 27 layers and `vision_feature_layer: -2`, but the checkpoint ships 26, so loading failed with `Weight not found: vision_tower.encoder.layers.26...`.

## Technical decisions

- **Shared resample matrix.** SAM's `interp_matrix` moved to `src/vision/encoders/resample_matrix.rs`, now parameterized by a cubic or triangle kernel and doing the separable contraction. `Learnable2DInterpPosEmb` gains `PosEmbResample::BilinearAntialias`, which resamples in float32 and casts back to the table dtype; the LFM2 tower selects it through `POSITION_RESAMPLE`. SAM produces the same values as before, and MoonViT stays `Bicubic` with its own edge-replicated matrix, pinned bit for bit by a test.
- **Processor filter from the checkpoint.** `processor_config.json` (nested or top level) wins over `preprocessor_config.json`; absent means bicubic. PIL ids 0/1/2/3 map to Nearest, Lanczos3, Triangle and CatmullRom, and any other id is a load error instead of upstream's silent fallback. The filter reaches all three resize sites.
- **One load helper for the layout guard.** `load_qwen3_vl_vision_encoder` in `src/loading/vlm_qwen3_vision.rs` runs `normalize_patch_embed_layout` and builds the tower. Qwen3-VL, Qwen3-VL-MoE, Qwen3.5/3.8 and Qwen3-Omni in `vlm_qwen.rs`, and Cohere Compass, all use it; a structural test fails if any file calls the encoder constructor directly again.
- **Quantized pos_embed.** `PositionEmbedding` holds a `UnifiedEmbedding`, and `hidden_size` comes from the config rather than the table shape.
- **Vision feature layer.** The LFM2 tower keeps only the layers up to `vision_feature_layer` and still applies `post_layernorm`, matching HF `modeling_lfm2_vl.py` and mlx-vlm.
- **Image traces.** `examples/logit_trace` accepts `MLXCEL_TRACE_IMAGE` and `MLXCEL_TRACE_IMAGE_PROMPT`, and `scripts/mlx_vlm_image_logit_trace.py` produces the matching mlx-vlm reference (`--float32` dequantizes for the CPU-only mlx install).

## Changes

- `src/vision/encoders/resample_matrix.rs` (new), `kimi_vl_pos_emb.rs`, `lfm2_vl.rs`, `sam.rs`, `qwen3_vl.rs`: resample modes, feature-layer prefix, quantized pos_embed.
- `src/vision/processors/lfm2_vl.rs`, `lfm2_vl_tests.rs`: configured resample filter.
- `src/loading/vlm_qwen3_vision.rs` (new), `vlm_qwen.rs`, `vlm_cohere_compass.rs`: layout guard helper and its five callers.
- `tests/fixtures/lfm2_vl/reference.json` and `generate_reference.py`: torch 2.14.1+cpu and mlx-vlm 0.7.8 (git `73d366ec`) reference, agreeing to 3e-8.
- `examples/logit_trace/image.rs`, `scripts/mlx_vlm_image_logit_trace.py`, `docs/supported-models.md`, `docs/benchmarks.md`.

## Verification

GB10, `--profile test-fast --features cuda`, test binaries under `gpu-lock` with `--test-threads=1`. `vision:: loading:: multimodal::` with `MLX_ENABLE_TF32=0`: 1241 passed, 0 failed, 49 ignored; the 25 new tests also pass with TF32 at its default. Clippy (`--lib --tests` and `--example logit_trace`), fmt and the Python tooling tests (124) pass.

Eight new tests fail with their fix reverted: the hub/MLX patch-embed equivalence, quantized pos_embed against the dequantized table, the vision feature-layer prefix, the LFM2 grid against mlx-vlm (0.0399 against 0.1085 at 12x20 entry 0), and four processor resample tests. The site-coverage test fails on main by construction.

Image-prompted teacher-forced traces, 74 positions, 212-token prompt (mlx-vlm renders the same 212 tokens). Reference: mlx-vlm `73d366ec`, CPU, float32 after dequantization. The reverted arm sets the tower back to bicubic; on LFM2.5 its trace is byte-identical to a main build.

| Checkpoint | Arm | Top-1 disagreement | Disagreement where reference gap >= 2 | Logit delta p50 / max | ppl vs reference |
|---|---|---|---|---|---|
| LFM2.5-VL-1.6B-4bit | reverted | 4/74 | 0/36 | 0.193 / 0.673 | +2.36% |
| LFM2.5-VL-1.6B-4bit | fixed | 0/74 | 0/36 | 0.034 / 0.133 | +0.14% |
| LFM2-VL-1.6B-4bit | reverted | 0/74 | 0/41 | 0.092 / 0.396 | +1.25% |
| LFM2-VL-1.6B-4bit | fixed | 0/74 | 0/41 | 0.046 / 0.185 | +1.12% |

CLI and `mlxcel-server` at temperature 0: both LFM2 1.6B checkpoints describe the red square, blue circle and green triangle, with server output matching the CLI at 212 prompt tokens; LFM2-VL-450M is fluent on the CLI. Qwen3.8-27B-FP8 (hub patch weight `[1152, 3, 2, 16, 16]`) describes stripes on main and the correct shapes after the fix, on both CLI and server. `qwen3-vl-2b-4bit` (MLX layout, dense pos_embed) is byte-identical between main and the fix.

## Follow-ups

- All three local LFM2 checkpoints declare `resample: 2`, so the processor change has no numeric effect on this host.
- No local Qwen3-VL export quantizes `pos_embed`; that item is covered by the synthetic test only.
- SAM's antialiased bicubic uses `a = -0.75`, while ATen and mlx-vlm use `-0.5` for the antialiased kernel. Left bit-identical here.

## Statistics

| Item | Value |
|------|-------|
| Files changed | 24 |
| Lines added | 4799 |
| Lines deleted | 301 |

| Commit | Message |
|--------|---------|
| `dda17a4b` | feat(examples): trace image-prompted continuations in logit_trace |
| `5686d0a4` | fix(vision): port LFM2-VL resampling, Qwen3-VL Conv3d guard and pos_embed |
| `3bf82f87` | test(vision): pin the LFM2-VL tower's resample mode to the mlx-vlm grid |
| `377664c9` | chore(scripts): mark mlx_vlm_image_logit_trace.py executable |

## Related

- `normalize_patch_embed_layout` already existed for other Qwen VL loaders; this PR routes every Qwen3-VL tower site through it.
