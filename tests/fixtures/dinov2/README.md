# DINOv2 reference fixtures

These complete f32 outputs come from public Hugging Face checkpoints evaluated with Transformers 4.57.1, PyTorch 2.14.1+cpu, eager attention and CPU inference. They are reference outputs, not seeded or reduced-size model weights. Safetensors 0.8.0 serialized the tensors. Each file embeds the model ID, pinned checkpoint revision, library versions and input convention as metadata.

| File | Checkpoint | Pinned revision | Last hidden state |
|---|---|---|---|
| `small_reference.safetensors` | `facebook/dinov2-small` | `ed25f3a31f01632728cabb09d1542f84ab7b0056` | `[1, 257, 384]` |
| `register_small_reference.safetensors` | `facebook/dinov2-with-registers-small` | `0d9846e56b43a21fa46d7f3f5070f0506a5795a9` | `[1, 261, 384]` |

Each fixture contains `pixel_values_nhwc`, `last_hidden_state`, `pooler_output` and `hidden_patch_tokens`. The source image is uint8 RGB `arange(224*224*3).reshape(224,224,3) % 256`. Each checkpoint's `AutoImageProcessor(use_fast=False)` applies its pinned `preprocessor_config.json`: shortest-edge bicubic resize to 256, center crop to 224, division by 255, and ImageNet normalization. `pixel_values_nhwc` stores this **already normalized** model input transposed to NHWC; callers must not normalize it again. Passing the original deterministic RGB image to the standalone predictor must reproduce the same output tensors. Patch outputs exclude the class token and any register tokens. Full tensors permit relative RMS error checks across every output element at the issue's `1e-3` tolerance.

Regenerate with the recorded dependencies installed:

```bash
python tests/fixtures/dinov2/generate_reference.py --cache-dir /tmp/dinov2-checkpoints
```

The script downloads only configuration and safetensors checkpoint files at the pinned revisions. Model weights remain in the external Hugging Face cache and are not committed. The published DINOv2 checkpoints use Apache-2.0 licensing; see the [upstream model card](https://huggingface.co/facebook/dinov2-small) and [register model card](https://huggingface.co/facebook/dinov2-with-registers-small).

SHA-256 of the committed fixtures:

```text
b8ceb38c4320c4601d08c859e1ebb5eabefc15adbb15044b97075832dcac8c9d  small_reference.safetensors
7b935e7fa433c4a04670d5f344e8f65df96d0324c23d1b1acde2bf1e8b8e8af0  register_small_reference.safetensors
```
