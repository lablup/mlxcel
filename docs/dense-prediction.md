# Dense prediction

`mlxcel predict` is the offline file interface for checkpoints that return depth, normal, mask, segmentation, point, or keypoint results. These outputs have shapes and lifetimes that do not fit the token stream from `generate` or the box rows from `detect`.

The same interface also exposes standalone image-encoder features. DINOv2 is the first active family in the registry; later dense-model ports reuse its encoder and the shared file contract.

The dense registry currently includes Sapiens2. The task is selected from the checkpoint: pose, semantic segmentation, surface normals, metric pointmaps, image matting, or backbone features.

```bash
mlxcel predict \
  --model models/dinov2-small \
  --image input.png \
  --output outputs/features \
  --task features

mlxcel predict \
  --model models/sapiens2 \
  --image person.png \
  --output outputs/pose \
  --opts '{"boxes":[[140,90,920,700]],"flip_test":true}'
```

| Family | `model_type` | Tasks | Input | Tensor outputs |
|---|---|---|---|---|
| DINOv2 | `dinov2`, `dinov2_with_registers` | `features` | image | `last_hidden_state`, `pooler_output`, `hidden_patch_tokens` |

DINOv2 accepts both standard Hugging Face checkpoints and checkpoints with register tokens. The predictor follows the checkpoint's `preprocessor_config.json`: it converts to RGB, resizes the shortest edge with Pillow-compatible bicubic uint8 rounding, center crops, rescales, and normalizes with the configured channel statistics. When that file is absent, it uses the public checkpoint defaults of a 256-pixel shortest edge, a 224-by-224 center crop, division by 255, and ImageNet normalization. `last_hidden_state` retains the class token and any register tokens, `pooler_output` is the normalized class token, and `hidden_patch_tokens` contains only spatial patches. Video input is rejected.

Exactly one of `--image` and `--video` is required. `--task` may be omitted when the selected family has one task. `--opts` must be a JSON object; the command adds the selected task under the reserved `task` key before calling the predictor. Video sampling keeps at most 32 frames by default; `--max-frames` changes that cap, and the command rejects a sampled set whose decoded RGBA worst-case size exceeds 512 MiB before ffmpeg decodes it. The output directory is created when absent. Existing files with the standard names are replaced, while unrelated files are retained.

Sapiens2 accepts still images. A task checkpoint exposes exactly its own task, so `--task` is optional. Pose accepts `boxes` as an array of pixel-space `xyxy` boxes and `flip_test` as a boolean. When `boxes` is omitted, pose runs on the full image. An empty array produces valid empty pose tensors. mlxcel clips boxes to the image and rejects malformed, non-finite, or degenerate boxes. The pose path preserves the upstream aspect-ratio crop, affine warp, flip-pair remapping, DARK refinement, and unbiased data processing (UDP) coordinate transform. It does not run a person detector.

Published Sapiens2 checkpoints use a fixed 1024x768 input (height by width). Segmentation and matting stretch the image to that canvas. Normal and pointmap estimation preserve the aspect ratio with symmetric padding, then remove that padding before resizing the result to source resolution. The loader accepts both published config layouts: the earlier nested backbone/deconvolution schema and the current flat Transformers schema. The backbone dimensions are config-driven for the standard 0.4B, 0.8B, 1B, and 5B architectures.

Sapiens2 retains the published BF16 weight dtype on every backend. Converting the public pose checkpoint to F16 changes some decoded keypoints by more than the 1.5-pixel upstream parity bound, so the family is an explicit exception to the general Apple Silicon load-time BF16-to-F16 optimization.

Every successful run writes:

| Path | Contents |
|---|---|
| `result.safetensors` | Every output tensor converted to contiguous f32, preserving the predictor's upstream dictionary key and shape. Video maps are stacked on axis 0. |
| `meta.json` | Family, selected task, source size, inference time after MLX evaluation, finite min/max for every tensor, model metadata, and flattened intrinsics when an `intrinsics` tensor is present. |
| `keypoints.json` | Present when people were detected. Each entry contains `bbox`, `(x, y, score)` points in source pixels, and optional point names. |
| `depth.png` | 16-bit grayscale. Finite map min/max become 0 and 65535; NaN and infinity become 0. |
| `normal.png` | 8-bit RGB after `(normal + 1) / 2`, clamped to `[0, 1]`. |
| `mask.png` | 8-bit grayscale, using 0 below or equal to 0.5 and 255 above 0.5. |
| `segmentation.png` | 8-bit class IDs. |

Sapiens2 uses these tensor keys and shapes:

| Checkpoint task | Tensor keys | Shape |
|---|---|---|
| Pose | `keypoints`, `scores`, `boxes` | `(N, K, 2)`, `(N, K)`, `(N, 4)` in source-image pixels |
| Segmentation | `segmentation` | `(H, W)` integer class IDs |
| Normal | `normals` | `(H, W, 3)` normal vectors |
| Pointmap | `pointmaps`, `scales` | `(H, W, 3)`, `(1)` |
| Matting | `alphas`, `foregrounds` | `(H, W)`, `(H, W, 3)` |
| Backbone | `last_hidden_state`, `pooler_output` | token and pooled feature tensors |

Pose also writes one `keypoints.json` entry per box. Point names come from the checkpoint's `id2label` map when present.

For video, each available preview is written as `<tensor-key>/000000.png`, `<tensor-key>/000001.png`, and so on. Preview normalization uses the same whole-tensor range recorded in `meta.json`.

## Model integration

A model implements the object-safe interface in `src/vision/dense/predictor.rs`:

```rust,ignore
pub trait DensePredictor {
    fn family(&self) -> &'static str;
    fn tasks(&self) -> &[&'static str];
    fn predict(
        &self,
        input: &DenseInput,
        opts: &serde_json::Map<String, serde_json::Value>,
    ) -> anyhow::Result<DenseOutput>;
}
```

`DenseOutput::tensors` is a `BTreeMap<String, UniquePtr<MlxArray>>` because `MlxArray` is an opaque C++ bridge type. Keep upstream output names such as `depth`, `points`, `normal`, `mask`, `intrinsics`, `segmentation`, `normals`, `pointmaps`, and `alphas`. `DenseOutput::meta` carries family-specific JSON values that the standard writer preserves.

Family predictors must attach stage context to internal preprocessing failures with `.context("preprocess")`. The command supplies the outer `predict <task>` context and separately names model loading and output writing, so the resulting error chain identifies the failing stage.

Expose a constructor with this signature and add its row to `src/vision/dense/registry.rs`:

```rust,ignore
fn load(path: &std::path::Path) -> anyhow::Result<Box<dyn DensePredictor>>;

DenseFamily {
    model_type: "video_depth_anything",
    load: crate::vision::video_depth_anything::load,
}
```

The registry key must equal the lowercase top-level `model_type` in `config.json`. Add the same model type to `KNOWN_MODEL_TYPES` only when the family is a dense predictor; that list directs operators who accidentally use `mlxcel generate` to `mlxcel predict`, even in builds where the family has not landed.

Shared preprocessing APIs live in `src/vision/dense`:

```rust,ignore
resize_bilinear_nhwc(x, (height, width), align_corners, antialias)
resize_bicubic_nhwc(x, Some((height, width)), None, antialias)
resize_bicubic_nhwc(x, None, Some((height_scale, width_scale)), antialias)
resize_nearest_nhwc(x, (height, width))
image_to_nhwc_f32(image)
normalize_imagenet(x)
resize_to_patch_multiple(x, DINOV2_PATCH_SIZE)
conv_weight_to_channels_last(weight, expected_in_channels)
```

The interpolation functions accept rank-4 NHWC arrays and match `torch.nn.functional.interpolate` with `align_corners=false` by default at call sites. Bilinear and bicubic interpolation math returns f32; their upstream-compatible same-size fast path returns a copy in the input dtype, and nearest preserves dtype. For bicubic `scale_factor`, the non-antialiased path retains the supplied factor while the antialiased path derives its scale from the floored output size, matching upstream. The convolution helper recognizes an already-channel-last trailing input-channel axis before applying OIHW to OHWI permutation, preventing a second conversion. A model loader should also use checkpoint-key layout detection for shapes whose channel and kernel axes are ambiguous.

Mesh and Gaussian export belong to the SAM 3D Objects implementation. Streaming, webcam input, and a server route remain outside this file contract until a client-facing dense-output wire format is defined.
