# Vision model scope

This page is the scope record for the task-specific vision and generative-media families that `lablup/mlxcel#524` collected from the VLM reference: object detection, instance segmentation, 3D body estimation, grounding, and diffusion image or video generation. Each family has one disposition with the evidence it rests on, so a coverage question has a recorded answer. Families that are generative chat or completion VLMs are tracked separately in [model-triage.md](model-triage.md) (`lablup/mlxcel#548`) and the individual port issues `lablup/mlxcel#533` through `lablup/mlxcel#547`.

Last checked against `main` at `68d7cc36`.

## Why these families were out of scope

mlxcel's generation runtime targets decoder LLMs and VLMs that emit a token stream. The families below emit boxes, masks, meshes, or pixels instead, or need an encoder-decoder loop the original runtime did not have. A checkpoint for a family without a detection entry fails in `src/models/detection.rs::get_model_type` in one of two ways:

- A `config.json` whose `model_type` has no match arm fails with `Unsupported model type: <type>`.
- A multi-component diffusion directory has no top-level `model_type`. When the image probe does not recognize it, it fails with `model_type not found`, or with the plain file-read error when there is no top-level `config.json` at all (Diffusers checkpoints key on `model_index.json`). A recognized image checkpoint is redirected instead, as described under image generation below.

mlxcel already has non-chat paths, and they are the precedent for any re-scoping:

- `mlxcel detect` (`src/commands/detect.rs`, `src/vision/detection/`): object detection with bounding-box output.
- Whisper (`src/models/whisper/`): encoder-decoder speech-to-text through `/v1/audio/transcriptions`.
- Kokoro (`src/models/kokoro/`): text-to-speech with audio output through `/v1/audio/speech`.
- Image generation (`src/imagegen/`, `src/commands/generate_image.rs`): the shared runtime behind `mlxcel generate-image` and `/v1/images/generations` / `/v1/images/edits`, added by `lablup/mlxcel#2341` (PR `lablup/mlxcel#2497`). See [image-generation.md](image-generation.md).

## Already supported

| Family | Checkpoint / Issues | In-tree evidence |
|---|---|---|
| `rt_detr_v2` | RT-DETRv2 checkpoints, including the Docling layout checkpoints (`docling-layout-heron`) (PR `lablup/mlxcel#80`) | `src/vision/detection/rt_detr_v2/` (ResNet-vd backbone, AIFI plus FPN/PAN hybrid encoder, 6-layer deformable decoder with iterative box refinement, labels through `id2label`), `src/vision/detection/rt_detr_v2/sanitize.rs` (layout-guarded conv permutation), `src/commands/detect.rs` (`RtDetrV2Predictor::from_pretrained`). It is a standalone runtime family outside the generate loop, listed by `mlxcel arch --json`. See [supported-models.md](supported-models.md). |
| `pp_doclayout_v3` | PaddlePaddle/PP-DocLayoutV3 and converted `PP-DocLayoutV3-mlx` checkpoints (`lablup/mlxcel#2412`) | `src/vision/detection/pp_doclayout_v3/` implements HGNetV2-L, the hybrid encoder, mask-enhanced query initialization, six-layer deformable decoder, and GlobalPointer reading-order head. The loader accepts the provenance-distinct Paddle and converted layouts, while `mlxcel detect` emits dense one-based model reading order for direct Falcon-OCR layout input. |
| `florence2` | `mlx-community/Florence-2-base-ft-bf16` and its 4-bit and 8-bit conversions (epic `lablup/mlxcel#850`, PRs `lablup/mlxcel#1060`, `lablup/mlxcel#1063`, `lablup/mlxcel#1064`, `lablup/mlxcel#1069`, `lablup/mlxcel#1071`, `lablup/mlxcel#1082`, `lablup/mlxcel#1083`) | `src/models/detection.rs` (`"florence2" => Florence2VLM`), `src/models/florence2/` (DaViT tower, BART encoder-decoder, task prompts, location-token post-processing), `src/loading/vlm_florence2.rs`, and the seq2seq server worker from PR `lablup/mlxcel#1083`. The seq2seq loop this record named as the blocker now exists. The `-large-ft` MLX conversions load but do not generate usable text (upstream `Blaizzy/mlx-vlm#1840`); [supported-models.md](supported-models.md) recommends `-base-ft`. |
| `locateanything` | `mlx-community/LocateAnything-3B-4bit` (`lablup/mlxcel#847`, PR `lablup/mlxcel#1070`) | `src/models/detection.rs` (`"locateanything" => LocateAnythingVLM`), `src/vision/locateanything.rs`, `src/loading/vlm_locateanything.rs`. The standard autoregressive decode path is supported. The parallel box-decoding head (`pbd`, `n_future_tokens = 6`) and coordinate-token-to-box post-processing are not implemented; [supported-models.md](supported-models.md) records them as a follow-up, and no open issue tracks them. |
| `nemotron_parse` | `nvidia/NVIDIA-Nemotron-Parse-2.0` and its `mlx-community/Nemotron-Parse-2.0-{4bit,8bit}` conversions (`lablup/mlxcel#1369`, PR `lablup/mlxcel#2034`) | `src/models/detection.rs` (`"nemotron_parse" => NemotronParseVLM`), `src/models/nemotron_parse/` (C-RADIO ViT-H/16 tower, compression neck, 10-layer pre-norm mBART decoder), `src/loading/vlm_nemotron_parse.rs`, `src/commands/generate_nemotron_parse.rs` (CLI driver), and `src/server/nemotron_parse_worker.rs` (single-stream server worker). It is a document OCR and layout model that emits markdown with box and class tokens, so it is not one of the families `lablup/mlxcel#524` collected, but it shares the encoder-decoder plumbing the `florence2` row describes. Greedy-token parity against the reference is checked by `tests/nemotron_parse_real_model.rs`, which is `#[ignore]`-gated on a local checkpoint. See [supported-models.md](supported-models.md). |

## Re-scoped to an epic and a port issue

The maintainer asked for image generation, depth, pose and 3D support, so these families moved from this record to dedicated work (`lablup/mlxcel#524` comments of 2026-10-16). Each now has a port issue under an epic.

### Image generation, epic `lablup/mlxcel#2348`

The shared runtime from `lablup/mlxcel#2341` (PR `lablup/mlxcel#2497`) is merged. `src/imagegen/registry.rs::FAMILIES` registers `z_image` (`lablup/mlxcel#2425`). `src/imagegen/probe.rs::IMAGE_FAMILIES` recognizes `bonsai`, `flux2`, `z_image`, `ming_image`, `ernie_image` and `qwen_image`. A recognized checkpoint of an unregistered family therefore fails in one of two ways depending on the command:

- `mlxcel generate`, `mlxcel serve` and the other text-model loaders stop in `src/models/detection.rs` with `image generation checkpoint (<name>): use mlxcel generate-image` (`src/imagegen/probe.rs::image_checkpoint_redirect_error`).
- `mlxcel generate-image` fails with `image generation family <name> is not supported by this build` (`src/imagegen/registry.rs::resolve_family`) until the port issue registers the family.

A checkpoint the probe does not recognize keeps the `model_type not found` or file-read failure described above.

| Family | Port issue | What it is | Probe today |
|---|---|---|---|
| `flux2` | `lablup/mlxcel#2424` | FLUX.2 Klein (4B, 9B): dual-stream plus single-stream MMDiT flow-matching text-to-image and editing, bundled text encoder, VAE. | Recognized, not registered. |
| `z_image` | `lablup/mlxcel#2425` | Z-Image / Z-Image-Turbo: Qwen3-style text encoder, diffusion transformer, VAE, flow-matching scheduler. | Registered: `mlxcel generate-image` and `mlxcel-server --image-model` run text-to-image and img2img. |
| `mage_flow` | `lablup/mlxcel#2426` | Mage-Flow and Mage-Flow-Edit (4B, six variants): Qwen3-VL conditioner, NR-MMDiT transformer, Mage-VAE, flow-matching sampler. | Not in `IMAGE_FAMILIES`, so not redirected. |
| `ideogram4` | `lablup/mlxcel#2427` | Ideogram 4: 34-layer DiT over VAE latents with an external LLM text encoder; FP8, license-gated checkpoint keyed by `model_index.json`. | Not in `IMAGE_FAMILIES`, so not redirected. |
| `bonsai` | `lablup/mlxcel#2428` | Prism Bonsai: ternary 2-bit FLUX.2 text-to-image. Also recorded in [model-triage.md](model-triage.md). | Recognized, not registered. |

### Dense prediction, segmentation and 3D, epic `lablup/mlxcel#2382`

The tree includes the `mlxcel predict` output surface (`lablup/mlxcel#2349`),
its dense-family registry, and the shared `mlxcel detect` surface. Every epic
family below is now implemented through one of those native runtimes.

| Family | `model_type` | Port issue | Output path |
|---|---|---|---|
| `dinov2` | `dinov2` or `dinov2_with_registers` | `lablup/mlxcel#2350` | Native `mlxcel predict` feature tensors and metadata. |
| `sapiens2` | `sapiens2` | `lablup/mlxcel#2371` | Native pose, segmentation, normal, pointmap, matting, and backbone prediction. |
| `video_depth_anything` | `video_depth_anything` | `lablup/mlxcel#2352` | Native still-image and temporal depth prediction. |
| `moge3` | `moge3` | `lablup/mlxcel#2355` | Native monocular geometry, depth, normal, validity, and intrinsics prediction. |
| `sam3` | `sam3_video` or `sam3` | `lablup/mlxcel#2429` | Native `mlxcel predict` detector, instance masks and memory tracker with text or pixel-box prompts. |
| `sam3_1` | `sam3.1_video` or `sam3.1` | `lablup/mlxcel#2430` | Same as `sam3`, plus Object Multiplex tracking of up to 16 objects per pass. The video checkpoint contains a `sam3.1` detector sub-config and shares most components with `sam3`. |
| `sam3d_objects` | `sam3d_objects` | `lablup/mlxcel#2381` | Native image-and-mask object reconstruction to RGB OBJ or Gaussian PLY. |
| `sam3d_body` | `sam3d_body` with a `dinov3` vision sub-config | `lablup/mlxcel#2432` | Native human mesh, 3D keypoint, joint-coordinate, and camera prediction. |
| `rf-detr` | `rf-detr` (alias `rfdetr`) | `lablup/mlxcel#2431` | Native `mlxcel detect` implementation with the released windowed DINOv2 backbone, C2f projector, deformable decoder, COCO boxes, and optional source-resolution segmentation masks through `--masks-dir`. |

## Not re-scoped

Neither epic names these families, so the 2026-10-16 re-scoping does not affect them.

| Family | Disposition | Evidence |
|---|---|---|
| `minimax_h3` | Out of scope. | MiniMax-H3 is joint video and audio generation (T2VA, FL2VA, Ref2VA): a 50-block rectified-flow transformer, a causal 3D-conv video VAE, a DAC/BigVGAN-style audio VAE and a Qwen3-VL-32B conditioner, about 130 GB bf16 per workflow. The image runtime excludes video generation ([image-generation.md](image-generation.md)), no port issue exists, and no detection entry matches it. |
| `nemotron_labs_diffusion` | In scope, awaiting a port issue. | Not a vision model despite the name: an 8B text-only diffusion language model (GQA, SwiGLU, YaRN RoPE, untied diffusion head, mask token) with autoregressive, block-diffusion (block size 32) and self-speculative decode modes. DiffusionGemma (`src/models/diffusion_gemma/`, `"diffusion_gemma" \| "diffusion_gemma_text"` in `src/models/detection.rs`, [block-diffusion.md](block-diffusion.md)) is the in-tree precedent for its runtime shape. `src/models/detection.rs` has no arm for it, so loading fails with `Unsupported model type`. No port issue exists: a repository issue search for the family name matches only this record, the f32 attention-guard fix `lablup/mlxcel#1710` (which lists it among the families to check) and the audit `lablup/mlxcel#1829` (`docs/f32-attention-widen-audit.md`), which records it as not ported. |

## Re-scoping a family

A family listed here moves to a dedicated port issue when all of the following hold:

1. A concrete user or product need names the family, with a checkpoint someone wants to serve.
2. The output path exists or is being built. Boxes ride `mlxcel detect`; images ride `src/imagegen/`; masks, depth maps, keypoints and meshes ride `mlxcel predict` once `lablup/mlxcel#2349` lands. Any other output path is its own infrastructure issue first.
3. A specific public checkpoint is named for validation, with its `config.json` or component layout confirmed.
4. Weight layout is accounted for. These families carry convolution stacks (CNN backbones, ViT patch embeddings, VAEs) serialized as `(out_channels, in_channels, kernel_h, kernel_w)`, while MLX runs channels-last convolutions expecting `(out_channels, kernel_h, kernel_w, in_channels)`. A port permutes these at load and must not double-transpose checkpoints already stored in MLX layout; `src/vision/detection/rt_detr_v2/sanitize.rs` is the precedent.

Write the port issue in the format of `lablup/mlxcel#533` through `lablup/mlxcel#547` (Summary with the `model_type` string, What it is, In-tree reuse, Scope, Effort), link it to `lablup/mlxcel#524` or to its epic, and update this page.

## Non-goals

- Implementing any of these families from this record; it is a decision record only.
- Building a general image-generation, segmentation-mask, or 3D-geometry output pipeline here; each is scoped under its epic.
- Extending the OpenAI-compatible API with image, mask or mesh response types outside those epics.
- Tracking generative VLM port candidates; those live in [model-triage.md](model-triage.md) and the port issues `lablup/mlxcel#533` through `lablup/mlxcel#547`.
