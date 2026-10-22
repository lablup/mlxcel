# DINOv2 reference fixtures

These complete f32 outputs come from public Hugging Face checkpoints evaluated with Transformers 4.57.1, PyTorch 2.14.1+cpu, eager attention and CPU inference. They are reference outputs, not seeded or reduced-size model weights. Safetensors 0.8.0 serialized the tensors. Each file embeds the model ID, pinned checkpoint revision, library versions and input convention as metadata.

| File | Checkpoint | Pinned revision | Last hidden state |
|---|---|---|---|
| `small_reference.safetensors` | `facebook/dinov2-small` | `ed25f3a31f01632728cabb09d1542f84ab7b0056` | `[1, 257, 384]` |
| `register_small_reference.safetensors` | `facebook/dinov2-with-registers-small` | `0d9846e56b43a21fa46d7f3f5070f0506a5795a9` | `[1, 261, 384]` |

Each fixture contains `pixel_values_nhwc`, `last_hidden_state`, `pooler_output` and `hidden_patch_tokens`. Input RGB values are `arange(224*224*3).reshape(1,224,224,3) % 256 / 255` in float32. The reference transposes this input to NCHW and applies ImageNet mean `[0.485,0.456,0.406]` and standard deviation `[0.229,0.224,0.225]`; it does not resize. Patch outputs exclude the class token and any register tokens. Full tensors permit relative RMS error checks across every output element at the issue's `1e-3` tolerance.

Regenerate with the recorded dependencies installed:

```bash
python tests/fixtures/dinov2/generate_reference.py --cache-dir /tmp/dinov2-checkpoints
```

The script downloads only configuration and safetensors checkpoint files at the pinned revisions. Model weights remain in the external Hugging Face cache and are not committed. The published DINOv2 checkpoints use Apache-2.0 licensing; see the [upstream model card](https://huggingface.co/facebook/dinov2-small) and [register model card](https://huggingface.co/facebook/dinov2-with-registers-small).

SHA-256 of the committed fixtures:

```text
4b7063f757f234e84561ee1cf2c4a993c81f0230bda65f77069b89dca08af091  small_reference.safetensors
3c33b5210dfbb60634781f14073d8ca5371c80c8005c104b83113a39add4ff82  register_small_reference.safetensors
```
