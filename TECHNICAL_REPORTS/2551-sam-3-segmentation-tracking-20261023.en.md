# PR #2551: Add SAM 3 promptable segmentation and tracking

**Date**: 2026-10-23
**Reviewer**: -
**Status**: Partial (runtime parity complete; central integration and CI pending)
**Languages**: Rust, C++, Python, Markdown
**Risk Level**: High

## Executive Summary

PR #2551 adds native SAM 3 image detection, prompted instance segmentation, and memory-based video tracking to `mlxcel predict`. It extends the shared dense-prediction contract with repeatable text and pixel-box prompts, source-pixel instances, u8 source-resolution masks, stable PNG names, and video object identities, then ports the ViT-L, CLIP text, DETR, geometry, segmentation, SAM mask-decoder, and tracker components needed by `sam3` and `sam3_video` checkpoints.

Runtime validation on the pinned gfx1151 ROCm stack is complete for BF16 reference parity and real affine 4-bit inference. The final BF16 run matches the exact-native reference with zero error through all 24 CLIP text layers, image and box-prompt mask IoU of at least 0.9989319, and the pinned three-frame tracker behavior. End-to-end CLI artifact validation, architecture/redirect checks after central synchronization, and the fresh PR CI run remain pending.

## 1. Problem Statement

mlxcel's dense-prediction foundation could serialize maps and keypoints, but it had no model-neutral representation for a variable number of scored objects with optional masks. It also had no prompt flags or family capability declaration, so a promptable open-vocabulary segmenter could not expose text prompts, pixel boxes, per-instance PNGs, or stable object IDs through the existing CLI.

SAM 3 combines an image detector and segmenter with a stateful memory tracker. A faithful port therefore requires more than a model loader: it must preserve source-pixel geometry, task-specific thresholds, prompt semantics, source-resolution mask reconstruction, and video frame/object identity while loading both BF16 and quantized MLX checkpoints.

## 2. Change Summary

| Area | Change |
|---|---|
| Shared instance contract | Added `Instance { bbox, score, label, mask_index, object_id, frame }` and `DenseOutput.instances`. Bounding boxes use source-image pixels; mask rows are optional per instance. |
| Output artifacts | Added `instances.json`, u8 `masks` tensors, image names such as `masks/000.png`, video names such as `masks/f000012_o003.png`, and `boxes`/`scores` safetensors. The writer synthesizes absent box/score tensors and validates supplied tensors against the instance values. |
| Prompt surface | Added repeatable `--prompt` and `--boxes` parsing, finite ordered pixel-box validation, family-level prompt capability checks, and structured `prompts`/`boxes` predictor options. |
| Task selection | Added input-aware predictor defaults. SAM 3 images default to `segment`; video requires explicit `track`. |
| Native model | Ported the ViT-L vision encoder and FPN, 24-layer CLIP text encoder, DETR encoder/decoder, geometry prompts, segmentation heads, SAM prompt/mask decoder, and memory-based tracker. |
| Checkpoint loading | Added idempotent convolution-layout sanitization and `UnifiedLinear` support for dense and affine-quantized weights. |
| Registry and docs | Registered `sam3` and `sam3_video`, added the `masks` output kind, and documented prompt, output, and tracking behavior. |
| Acceptance fixtures | Pinned the two-cat image, box prompt, three-frame DAVIS clip, exact-native text features, sampled processor pixels, JSON boxes/scores, binary mask PNGs, and hashes/provenance. |

## 3. Technical Decisions

### 3.1 A model-neutral instance and mask surface

Instances and mask rows are separate counts. An instance may be maskless, while every emitted mask row must be referenced exactly once through `mask_index`. This supports detection-only output and later segmenters without inventing a SAM-specific file format.

`result.safetensors` stores source-pixel `boxes` and `scores` plus u8 `masks` with shape `(M, H, W)` and values 0 or 1. `instances.json` adds labels and optional video identity fields, while PNGs convert mask values to 0 or 255. Stable names let downstream consumers join JSON, tensor rows, and PNGs without relying on directory enumeration order.

Prompt support is explicit registry metadata. The CLI rejects `--prompt` and `--boxes` before model loading for families that do not declare that capability.

### 3.2 Native image and video execution

Image `detect` and `segment` share the cached vision backbone across prompts, combine query and presence logits, clip boxes in source coordinates, and apply cross-label NMS. Video `track` initializes one object from each passing text detection or pixel box, then propagates tracker memory with stable `object_id` values.

The image threshold defaults to 0.3 with NMS IoU 0.5. Video uses 0.15 only when initializing text-prompt objects. Later tracker scores remain raw upstream values and are not clamped or reinterpreted as probabilities.

### 3.3 Scoped BF16 arithmetic for linear layers and GELU

Upstream MLX evaluates biased dense projections through a fused `addmm` path. A separate BF16 matrix multiplication followed by bias addition rounds differently. SAM 3 therefore wraps `UnifiedLinear` locally: ordinary biased dense projections use the fused operation, while quantized, biasless, and active runtime-LoRA paths continue through the shared implementation. This preserves the shared layer contract and avoids changing unrelated model families.

The pinned mlx-vlm runtime evaluates GELU as `x * (1 + erf(x / sqrt(2))) / 2` with a specific BF16 operation order. SAM 3 now uses a local activation helper for its text, vision, encoder, mask-decoder, and tracker paths. This scoped implementation preserves other model activation behavior and makes the SAM 3 text pipeline bit-exact against the same native MLX backend.

The authentic layer-zero GELU oracle passes, and the final text diagnostic reports maximum and mean error 0 at embeddings, every inspected layer-zero substage, residual outputs for layers 0 through 23, final normalization, detector projection, and all 32 token rows.

### 3.4 Reproducible exact-native fixtures

The reference set records source assets, conversion rules, runtime, and SHA-256 values. Large numeric references were reduced to JSON metadata, one-bit PNG masks, sampled processor pixels, and the text safetensor needed for stage diagnostics. The exact-native reference uses the same pinned MLX ROCm backend as the Rust acceptance tests, which removes backend drift from the parity result.

## 4. Provenance

| Component | Pinned value |
|---|---|
| Upstream implementation | `Blaizzy/mlx-vlm` commit `abd3b2d7b728cb5372d4e092260da325e5ded209` |
| BF16 checkpoint | `mlx-community/sam3-bf16` revision `dfe573c3171dbcfda8399c650d9135afa7e94592` |
| Affine 4-bit checkpoint | `mlx-community/sam3-4bit` revision `53c009b4ef4cfa5b1d9fa4549e7b153e881b5636` |
| Native MLX pin | `bff10220c742907f0c68e55ff174aaea56f74e3d` |
| Reference runtime | MLX 0.32.5 from the exact mlxcel native archive; BF16 weights left uncast |
| Validation device | ROCm gfx1151, AMD Radeon 8060S Graphics, 20 compute units, warp size 32 |

The COCO two-cat image is losslessly recompressed. The three-frame video retains the pinned MP4 decode path. `tests/fixtures/sam3/provenance.json` records the upstream asset hashes, committed fixture hashes, preprocessing SHA-256, and reference conversion rules.

The shared dtype policy retained BF16 on this ROCm device both with and without `MLXCEL_KEEP_BF16`; this report does not claim a forced-F16 parity run.

## 5. Runtime Validation

### 5.1 BF16 text, image, and box parity

| Case | Result |
|---|---|
| 24-layer CLIP text path | Maximum error 0 and mean error 0 at every recorded stage after the scoped GELU fix |
| Processor | Full resized RGB8 SHA-256 matches Pillow; sampled normalized pixels pass maximum error `1e-6` and mean error `1e-7` bounds |
| Text-only cat segmentation | Two detections; maximum box-coordinate error below 0.0021 px; maximum score error below `6.7e-6`; mask IoU 1.0 and 0.99998 |
| Text plus box prompt | Two detections; maximum box-coordinate error below 0.0019 px; maximum score error below `1.2e-5`; mask IoU 0.9999825 and 0.9989319 |

The final combined BF16 run completed four ignored real-checkpoint tests in 27.74 seconds with 4 passed and 0 failed.

### 5.2 Video parity and upstream limitation

The pinned three-frame reference propagates stable `object_id = 1`. Its raw tracker scores are `0.9603320`, `-3.0101275`, and `-3.0060225`. Frame 0 passes the source-mask IoU threshold of 0.95. The pinned upstream model produces empty native and source-resolution masks on frames 1 and 2 while retaining the object; the Rust test requires those masks to be exactly empty and preserves the negative raw scores.

This fixture proves tracker memory execution, identity stability, source-resolution output, and exact empty propagation. It does not provide meaningful non-empty-mask IoU evidence after frame 0.

A second authentic upstream run used three repeated copies of the cat image with the box prompt. It also retained object ID 1 while producing empty masks on frames 1 and 2, so the limitation was reproduced with both moving-video and stationary repeated-image inputs.

### 5.3 Affine 4-bit inference

The real `mlx-community/sam3-4bit` checkpoint loads through `UnifiedLinear` and completes prompted image segmentation. The smoke test verifies non-empty instances, finite scores and boxes, one mask row per returned instance, 480x640 source-resolution masks, and non-empty foreground. It passed on gfx1151 in 6.50 seconds.

### 5.4 Validation logs

| Log | Evidence |
|---|---|
| `issue-2429-gelu-build.log` | Final test-fast unit-test executable built successfully |
| `issue-2429-gelu-oracle.log` | Authentic BF16 GELU oracle passed |
| `issue-2429-gelu-text-bf16.log` | Zero-error 24-layer text and projection parity |
| `issue-2429-gelu-full-bf16.log` | Final text, image, box-prompt, and video parity; 4 passed |
| `issue-2429-4bit-inference-smoke.log` | Real affine 4-bit segmentation inference passed |
| `issue-2429-focused-tests.log` | SAM 3 options, NMS, clipping, tokenizer, sanitizer, threshold, and preprocessing tests passed |

Earlier diagnostic logs record the narrowing process and are not acceptance evidence; the final logs above supersede them.

### 5.5 Pending integration evidence

- **Pending after central synchronization:** end-to-end `mlxcel predict` CLI artifact run for image and video.
- **Pending after central synchronization:** `mlxcel arch --json` lists SAM 3 under the `predict` runtime with `masks` output.
- **Pending after central synchronization:** `mlxcel generate` redirects both SAM 3 model types to `mlxcel predict`.
- **Pending after central synchronization:** fresh PR #2551 CI results, including format, clippy, repository asset/provenance gates, and required platform checks.

## 6. Risk and Compatibility

The port adds no server endpoint and does not change the token-generation or closed-vocabulary detection output contracts. Its new CLI flags are rejected for dense families without prompt support. Point-prompt CLI input, webcam streaming, overlay rendering, and background replacement remain outside this issue.

The main numerical risk is backend-sensitive BF16 operation order across a large model. Exact-native fixtures and stage diagnostics constrain that risk on gfx1151. Full-checkpoint Metal runtime parity was not executed on this Linux host; platform CI can establish build and test compatibility, but it is not full model parity evidence unless it runs the checkpoint acceptance tests. The video fixture also exposes an upstream tracker limitation: identity may persist with a negative raw score and an empty mask.

## 7. Follow-up Actions

- Complete the pending CLI, architecture catalog, and generate-redirect checks after central synchronization.
- Run the fresh PR CI and update this report with the final integration evidence.
- Reuse the shared prompt, instance, output, and model components for SAM 3.1 in issue #2430.
- Use a tracking acceptance clip with non-empty later masks if future work needs measured multi-frame mask quality rather than exact reproduction of the pinned empty propagation.

## References

- PR #2551: `feat: add SAM 3 promptable segmentation and tracking`
- Issue #2429: SAM 3 promptable segmentation and video tracking
- Epic #2382: dense-prediction model delivery
- `docs/dense-prediction.md`: shared CLI and output contract
- `tests/fixtures/sam3/provenance.json`: source and fixture provenance
