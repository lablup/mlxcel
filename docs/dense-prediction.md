# Dense prediction

`mlxcel predict` is the offline file interface for checkpoints that return depth, normal, segmentation, point, keypoint, per-instance mask, mesh, or Gaussian results. These outputs have shapes and lifetimes that do not fit the token stream from `generate` or the box rows from `detect`.

The same interface also exposes standalone image-encoder features. DINOv2 is the first active family in the registry; later dense-model ports reuse its encoder and the shared file contract.

The dense registry includes DINOv2, Sapiens2, Video Depth Anything, MoGe3,
SAM 3, SAM 3.1, SAM 3D Objects, and SAM 3D Body. The task is selected from the checkpoint or input kind.

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

mlxcel predict \
  --model models/video-depth-anything-small \
  --video clip.mp4 \
  --max-frames 64 \
  --output outputs/depth

mlxcel predict \
  --model models/sam3-bf16 \
  --image cats.jpg \
  --prompt "a cat" \
  --boxes "0,50,350,480" \
  --output outputs/cats
```

| Family | `model_type` | Tasks | Input | Tensor outputs |
|---|---|---|---|---|
| DINOv2 | `dinov2`, `dinov2_with_registers` | `features` | image | `last_hidden_state`, `pooler_output`, `hidden_patch_tokens` |
| Sapiens2 | `sapiens2` | `pose`, `segmentation`, `normal`, `pointmap`, `matting`, `features` | image | Task-specific keys listed below |
| Video Depth Anything | `video_depth_anything` | `depth` | image or video | `depth` `(T, H, W)` |
| SAM 3D Objects | `sam3d_objects` | `mesh` | image + mask or alpha | pose, sparse latent, OBJ mesh, Gaussian PLY |
| SAM 3D Body | `sam3d_body` | `body_mesh` | image and optional person box | `vertices`, `keypoints_3d`, `joint_coords`, `camera`; `mesh.obj`, `result.safetensors`, `meta.json`, empty `instances.json` |
| MoGe3 | `moge3` | `geometry` | image | `points`, `depth`, `mask`, `normal`, `intrinsics` |
| SAM 3 | `sam3`, `sam3_video` | `detect`, `segment`, `track` | image or video with text and box prompts | `boxes`, `scores`, source-resolution u8 `masks` |
| SAM 3.1 | `sam3.1`, `sam3.1_video` | `detect`, `segment`, `track` | image or video with text prompts and optional box filters | `boxes`, `scores`, source-resolution u8 `masks` |

DINOv2 accepts both standard Hugging Face checkpoints and checkpoints with register tokens. The predictor follows the checkpoint's `preprocessor_config.json`: it converts to RGB, resizes the shortest edge with Pillow-compatible bicubic uint8 rounding, center crops, rescales, and normalizes with the configured channel statistics. When that file is absent, it uses the public checkpoint defaults of a 256-pixel shortest edge, a 224-by-224 center crop, division by 255, and ImageNet normalization. `last_hidden_state` retains the class token and any register tokens, `pooler_output` is the normalized class token, and `hidden_patch_tokens` contains only spatial patches. Video input is rejected.

Exactly one of `--image` and `--video` is required. `--task` may be omitted when the selected family declares a default for that input. `--prompt` is repeatable. `--boxes` is also repeatable and accepts pixel `x1,y1,x2,y2` boxes separated by semicolons. Families without prompt support reject both flags. `--opts` must be a JSON object; the command adds the selected task under `task`, prompt strings under `prompts`, and parsed numeric boxes under `boxes` before calling the predictor. Video sampling keeps at most 32 frames by default; `--max-frames` changes that cap. Without `--target-fps`, Video Depth Anything and SAM tracking keep a consecutive prefix up to that cap; supplying `--target-fps` uses time-based uniform sampling. The command rejects a sampled set whose decoded RGBA worst-case size exceeds 512 MiB before ffmpeg decodes it. The output directory is created when absent. Existing files with the standard names are replaced, while unrelated files are retained.

Sapiens2 accepts still images. A task checkpoint exposes exactly its own task, so `--task` is optional. Pose accepts up to 16 `boxes` as an array of pixel-space `xyxy` boxes and `flip_test` as a boolean. It processes boxes one at a time to bound peak activation memory. When `boxes` is omitted, pose runs on the full image. An empty array produces valid empty pose tensors. mlxcel clips boxes to the image and rejects malformed, non-finite, or degenerate boxes. The pose path preserves the upstream aspect-ratio crop, affine warp, flip-pair remapping, DARK refinement, and unbiased data processing (UDP) coordinate transform. It does not run a person detector.

Published Sapiens2 checkpoints use a fixed 1024x768 input (height by width). Segmentation and matting stretch the image to that canvas. Normal and pointmap estimation preserve the aspect ratio with symmetric padding, then remove that padding before resizing the result to source resolution. The loader accepts both published config layouts: the earlier nested backbone/deconvolution schema and the current flat Transformers schema. The backbone dimensions are config-driven for the standard 0.4B, 0.8B, 1B, and 5B architectures.

Video Depth Anything supports the published Small, Base, and Large relative and metric checkpoints. It preserves the source aspect ratio, resizes the lower bound to 518 with OpenCV-compatible cubic interpolation and 14-pixel multiples, then applies ImageNet normalization. Inference uses the upstream 32-frame window, 10-frame overlap, keyframe conditioning, relative scale/shift alignment, and eight-frame linear blend. Metric checkpoints keep their absolute scale and report `units: "metres"`; relative checkpoints report `units: "relative_disparity"`. Depth returns at source resolution. A still image follows the same temporal path by repeating its frame. `--max-frames` retains consecutive decoded frames when no `--target-fps` is supplied.

The DINOv2 backbone runs frame chunks that respect its token budget, while each temporal head still sees the complete 32-frame window. The published checkpoints store F32 tensors. BF16 checkpoints retain BF16 on ROCm and other backends whose load policy supports it.

Sapiens2 retains the published BF16 weight dtype on every backend. Converting the public pose checkpoint to F16 changes some decoded keypoints by more than the 1.5-pixel upstream parity bound, so the family is an explicit exception to the general Apple Silicon load-time BF16-to-F16 optimization.

Every successful run writes:

| Path | Contents |
|---|---|
| `result.safetensors` | Every output tensor converted to contiguous f32, except `masks`, which remains u8. Instance output includes `boxes` `(N, 4)` and `scores` `(N,)`; `masks` is `(M, H, W)` at source resolution because instances may be maskless. Video maps are stacked on axis 0. |
| `meta.json` | Family, selected task, source size, inference time after MLX evaluation, finite min/max for every tensor, model metadata, and flattened intrinsics when an `intrinsics` tensor is present. |
| `keypoints.json` | Present when people were detected. Each entry contains `bbox`, `(x, y, score)` points in source pixels, and optional point names. |
| `instances.json` | Scored instances with source-pixel `bbox`, `label`, `mask_index`, optional video `object_id` and `frame`, and a relative `mask` PNG path when present. |
| `masks/000.png` | Image instance mask, 8-bit single-channel with values 0 and 255. The number is the `masks` tensor row. |
| `masks/f000012_o003.png` | Video mask for frame 12 and stable object 3. |
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

## MoGe3 geometry

MoGe3 estimates a camera-space point map, depth, validity mask, surface normals, and normalized camera intrinsics from one RGB image. The released `moge-3-vitl-mlx-fp32` and `moge-3-vitg-mlx-fp32` checkpoints use DINOv2-L/14 and DINOv2-g/14 respectively. Their weights remain F32. Four configured intermediate DINOv2 grids are projected and summed, decoded by the convolutional pyramid, and refined by a sparse 3D U-Net. The sparse path stores only occupied `(batch, row, column, depth-bin)` voxels; it never allocates a dense 3D volume.

```bash
MLX_ENABLE_TF32=0 mlxcel predict \
  --model models/moge-3-vitl-mlx-fp32 \
  --image room.png --task geometry \
  --opts '{"resolution_level":9,"refine_steps":3}' \
  --output outputs/room
```

`resolution_level` is an integer from 0 through 9 and maps linearly from the checkpoint's 1200-token minimum to its 3600-token maximum. `num_tokens` may select an exact value in that range and takes precedence when both fields are supplied. `refine_steps` defaults to 3 and accepts 0 through 16. `force_projection` defaults to true and reconstructs points from the recovered depth and intrinsics. `apply_mask` defaults to true: invalid depth and points become positive infinity, while invalid normals become zero. `fov_x` fixes the horizontal field of view in degrees and must be strictly between 0 and 180; when omitted, the model recovers focal length and depth shift. `return_per_step` adds `points_stepN`, `depth_stepN`, and `intrinsics_stepN` tensors to CLI serialization. The Rust library keeps those values as per-step vectors in `Moge3Output`.

MoGe3 derives checked live-work estimates from the input grid and validated configuration before constructing large decoder, sparse-refiner, and retained-output graphs. Its refiner releases optional neighbor maps and recomputes them when needed; supported mandatory phases have an explicit 8 GiB ceiling, while optional cached maps target 512 MiB. Published resolution level 9 inputs remain supported. Oversized inputs, unsupported configurations, and arithmetic overflow return a resource-budget error. These allocation guards preserve the complete computation and do not reduce the requested resolution or refinement steps.

The standard tensor shapes for a still image are:

| Key | Shape | Meaning |
|---|---|---|
| `points` | `(H, W, 3)` | Camera-space XYZ, with positive infinity outside the valid mask when masking is enabled |
| `depth` | `(H, W)` | Metric-scaled camera depth with the same invalid-value convention |
| `mask` | `(H, W)` | Boolean predicted validity combined with positive depth |
| `normal` | `(H, W, 3)` | Unit surface normal on valid pixels and zero elsewhere when masking is enabled |
| `intrinsics` | `(3, 3)` | OpenCV pinhole matrix normalized to image width and height |

Input integers are converted using the full range of their source dtype. A direct floating-point library input must contain finite RGB values in `[0, 1]`; other ranges are rejected. Batch CLI inference, video, streaming, and server routes are outside this family. Set `MLX_ENABLE_TF32=0` before the process starts when full F32 matmul accumulation is required for numerical comparisons on supported GPU backends.

For video, each available preview is written as `<tensor-key>/000000.png`, `<tensor-key>/000001.png`, and so on. Preview normalization uses the same whole-tensor range recorded in `meta.json`.

The instance writer requires every mask row to be referenced exactly once by an instance's `mask_index`. When a predictor supplies `boxes` or `scores` tensors, their values must match the instance records; when it omits them, the writer synthesizes the tensors from those records.

## Promptable segmentation and tracking

SAM 3 accepts text and box prompts. Image input defaults to `segment`; use `--task detect` to write boxes and scores without mask PNGs. `segment` needs at least one text prompt, while boxes may refine or add prompted objects. Repeat `--prompt` to run several labels against one cached image backbone:

```bash
mlxcel predict -m models/sam3-bf16 --image street.jpg \
  --prompt "a car" --prompt "a bicycle" \
  --boxes "20,40,300,420;340,80,620,430" -o outputs/street
```

For video, `--task track` initializes objects from frame-zero text detections and box prompts, then propagates them with stable `object_id` values:

```bash
mlxcel predict -m models/sam3-bf16 --video clip.mp4 --task track \
  --prompt "a car" -o outputs/tracked-car
```

`--opts '{"threshold":0.4,"nms":0.6}'` overrides the image score threshold and NMS IoU. SAM 3 defaults to threshold 0.3 and NMS IoU 0.5 for images. If no image instance passes the threshold, `instances.json` is an empty array and no mask PNG is written. Video defaults to threshold 0.15 and applies it when initializing each text-prompt object from the highest-scoring frame-zero query. A prompt below the threshold creates no track; established tracks remain present across later low tracker scores.

Tracking scores after frame zero are raw upstream tracker values rather than probabilities, so they can be negative and are written without clamping. The pinned upstream model keeps the object ID even when a later mask is empty. On the three-frame DAVIS car acceptance clip, the pinned reference retains ID 1 on every frame but produces empty masks on frames 1 and 2; the fixture records this limitation explicitly rather than treating two empty masks as a meaningful IoU result.

SAM 3.1 uses the same image detector and output files, with the model types
`sam3.1` and `sam3.1_video`. Its Object Multiplex tracker initializes every
frame-zero text detection in one session and calls the tracker once for each
later frame, including when more than 16 objects require several internal
buckets. Video boxes are region filters over text detections and therefore
still require at least one `--prompt`:

```bash
mlxcel predict -m models/sam3.1-bf16 --video traffic.mp4 --task track \
  --prompt "a car" --boxes "0,0,640,360" -o outputs/sam31-cars
```

Frame-zero scores use the detector confidence. Later SAM 3.1 scores are the
f32 sigmoid of the multiplex tracker's object logits; masks are resized with
nearest sampling and thresholded above zero before source-pixel boxes are
derived from their extents. The pinned public MLX checkpoint is named
`sam3.1-bf16`, but its stored and loaded tensors are all f32, so mlxcel keeps
them in f32 under the shared load-time dtype policy.

## Model integration

A model implements the object-safe interface in `src/vision/dense/predictor.rs`:

```rust,ignore
pub trait DensePredictor {
    fn family(&self) -> &'static str;
    fn tasks(&self) -> &[&'static str];
    fn default_task(&self, input: &DenseInput) -> Option<&'static str>;
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
    supports_prompts: false,
    load: crate::vision::dense::video_depth_anything::load,
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

Streaming, webcam input, and a server route remain outside this file contract until a client-facing dense-output wire format is defined.

## SAM 3D Objects geometry

Reconstruct an object from an RGB image plus a same-size mask, or an RGBA image whose alpha supplies the mask:

```bash
mlxcel predict -m models/sam-3d-objects-bf16 --image chair.png --mask chair_mask.png -o out/
mlxcel predict -m models/sam-3d-objects-bf16 --image chair-rgba.png --opts '{"estimate_depth":false}' -o out-no-depth/
```

The default uses seed 42, 25 structure steps, 25 latent steps, automatic bundled MoGe-3 depth, and `formats: ["gaussian", "mesh"]`. Set `seed`, `ss_steps`, `slat_steps`, `formats`, or `estimate_depth` through `--opts`; `gaussian_4` selects the four-Gaussian head. `--pointmap points.safetensors` supplies an HWC floating `points` tensor instead of estimated depth. Point maps use +X left, +Y up, +Z forward; NaN marks invalid points. Disabling depth without a point map uses the trained point-token dropout path.

`mesh.obj` contains RGB vertex colors and 1-based triangular faces. `gaussian.ply` and optional `gaussian_4.ply` contain binary little-endian Gaussian records with logarithmic scales and logit opacities. Geometry remains in object-local coordinates. `result.safetensors` records `pose.rotation` (WXYZ), `pose.rotation_matrix`, `pose.translation`, `pose.scale`, integer `coords`, `latents`, and boolean `pointmap_conditioned`; `meta.json` records the seed and actual step counts. Apply the returned pose separately when placing the geometry in a scene.

Masks must span at least two rows and two columns. Preprocessing checks source conversion, crop padding, square views, interpolation intermediates, and point-map normalization against a conservative 1 GiB live-buffer guard before large allocations. Extreme aspect ratios can exceed this guard even when their source pixel count is within the input limit; rejected inputs are not resized or truncated implicitly. Other prediction families reject `--mask` and `--pointmap`. The released main checkpoint stays BF16 and its bundled depth model stays F32. Shared DINO trunks are reused only after exact parameter comparison. Video, batch JSONL, streaming, scene placement, and serving are not supported for this family.

## SAM 3D Body human mesh

```bash
mlxcel predict -m models/sam-3d-body-dinov3-bf16 --image person.png \
  --opts '{"bbox":[100,50,400,500]}' -o outputs/body
```

The body-only predictor runs DINOv3, six camera-conditioned decoder layers, and the MHR body model. The released configuration explicitly specifies a 512×512 affine crop, which this runtime requires. The upstream dataclass default of 512×384 is distinct from the published checkpoint and is not selected when crop dimensions are missing. Dense checkpoint configuration reads are limited to 8 MiB before JSON parsing. The stored backbone stays BF16, decoder and MHR tensors stay F32, and integer topology remains integer. F32 input activations follow the upstream graph's natural promotion.

`bbox` is a source-pixel `[x1,y1,x2,y2]` box and defaults to the whole image. A partly outside box receives black padding without clipping; nonfinite, degenerate, and fully outside boxes fail. `cam_int` accepts nine row-major numbers or three rows of three numbers, with finite values, positive focal lengths and an invertible matrix. Defaults use the image diagonal as focal length and its center as principal point. Grayscale and RGBA inputs convert to RGB. Callers supply the person box; person detection and the separate hand-refinement pass are outside this runtime.

The colorless `mesh.obj` contains 18,439 six-decimal vertices and 36,874 one-based triangle faces. `result.safetensors` contains F32 `vertices [18439,3]`, `keypoints_3d [70,3]`, `joint_coords [127,3]`, and `camera [3]`. Vertices and keypoints are body-local meters with MHR Y/Z axes negated; camera remains the raw weak-perspective `[scale,tx,ty]` output. `meta.json` records the unchanged box, intrinsics and mesh bounds. The shared writer also emits an empty `instances.json`; no 2D `keypoints.json` is produced. Colored SAM 3D Objects OBJ formatting is unchanged.
