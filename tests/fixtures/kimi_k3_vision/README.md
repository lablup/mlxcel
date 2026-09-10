# Kimi K3 MoonViT3D reference fixture

`reference.json` is the out-of-band oracle for the Kimi K3 vision path (issue #1342): the projected features of `tests/fixtures/test_image.png` through the MoonViT3D tower, the `sd2_tpool` merger and the `patchmergerv2` projector, computed by `generate_reference.py` in numpy float32 from the bf16 vision shards of a local Kimi K3 checkpoint. The script is an independent transcription of the specification in the issue and of the checkpoint's own `media_utils.py` / `kimi_k3_vision_processing.py`; it does not import the Rust port, `torch`, `transformers` or `mlx`.

The Rust harness `kimi_k3_tower_real_weights` (`src/vision/encoders/moonvit3d_tests.rs`, `#[ignore]`) loads the same shards, runs the same image and compares the per-token statistics and a strided sample of the projected rows against this file.

## Contents

- `original_size`, `grid_thw`, `num_tokens`: the navit geometry of the image (224x224 keeps its size, so no resampling kernel is involved: grid `(1, 16, 16)`, 64 tokens).
- `pixel_values`: shape, mean absolute value and the first 16 normalized values of the `[256, 3, 14, 14]` patch tensor.
- `stages`: mean, mean absolute value and standard deviation after the patch embedding, blocks 0, 13 and 26, the final norm and the merge, for locating a divergence.
- `projected`: shape and statistics of the `[64, 7168]` projector output, the per-token mean absolute value, and every 64th column of every row (`sample_col_stride`).

## Regenerate

```bash
python3 tests/fixtures/kimi_k3_vision/generate_reference.py --checkpoint models/kimi-k3-8l-mxfp4 --image tests/fixtures/test_image.png
```

Any local copy of `moonshotai/Kimi-K3` that keeps the vision shards works; only `vision_tower.*` and `mm_projector.*` are read. Needs `numpy` and `pillow`.
