# PR #2163: Youtu-VL patch grid matches AutoProcessor

**Date**: 2026-10-06
**Status**: Implemented and verified on CUDA (GB10). Metal memory at the cap is unverified.
**Risk**: Medium. Large images now produce up to 9x more vision tokens than the old 4096-patch cap allowed.

## Summary

The checkpoint's documented entry point, `YoutuVLProcessor.__call__` in `processing_youtu_vl.py`, calls its image processor with `max_num_patches=36864`, and it sizes images with `get_image_size_for_patches`: each edge is rounded up to a multiple of 32 (at least one block), and `scale` drops by 0.02 until the grid fits. mlxcel had three differences: it capped the grid at `vision_config.num_patches` (4096), it rounded edges to the nearest multiple instead of up, and it applied Qwen2-VL `min_pixels`/`max_pixels` bounds. Because of these, 330-pixel edges, tiny images and anything above about 1024 pixels per side got a different grid from transformers. Closes #1611.

## Changes

- `DEFAULT_MAX_PATCHES_PER_IMAGE = 36864`, citing `processing_youtu_vl.py:53`. It is a constant because the value exists only as a Python default argument.
- `smart_resize` ports `get_image_size_for_patches`. It decrements `scale` in place in f64, as Python does (`1.0 - 0.02*k` rounds differently), and clamps the f64 edge before the cast. A bounded loop stops the walk at one block per edge when the cap is below 4 patches, a case the reference would loop on forever. `TooManyPatches` then rejects it.
- `build_processor` ignores `vision_config.num_patches`, the `max_num_patches: 256` default of the bare image processor, the dead `num_patches` key and the pixel bounds. `min_pixels`, `max_pixels`, `with_pixel_bounds` and `effective_max_pixels` are removed.

## Verification on GB10

- AutoProcessor (transformers 4.56.0, remote code) returns these `spatial_shapes` for 11 sizes: 224 14x14, 330 22x22, 336 22x22, 448 28x28, 512 32x32, 2048 128x128, 1080x1920 68x120, 3000x4000 166x220, 330x500 22x32, 100x3000 8x188, 32x32 2x2. A table test pins all 11. A synthetic loader test checks the 36864 cap. An ignored test runs `build_processor` on the real checkpoint for 330, 2048 and 3000x4000.
- `tests/youtu_vl_parity.rs --ignored`: 3/3 pass. The 224, 336 and 448 fixture grids are unchanged.
- In both CLI and `mlxcel-server`, the image token counts are 121, 4096 and 9130 (merged grid = patches / 4), and all three images are described correctly.
- Peak host-memory delta for one 48-token generation: 12.6 GB at 484 patches, 18.6 GB at 16384, 30.1 GB at 36520. No OOM, and `NV_ERR_NO_MEMORY` stayed at its baseline. The vision head_dim of 72 is eligible for cuDNN flash SDPA, and otherwise the 1024 MiB query-chunked fallback bounds the full-attention layers.

## Follow-ups

- Metal: if MLX's fused SDPA rejects head_dim 72, the four full-attention layers would materialize about 43 GB of scores at the cap. This needs a Metal measurement.
- The windowed attention concatenates its per-window outputs with a sequential fold, which copies quadratically in the number of windows (144 at the cap). The 19 s wall time at the cap is still acceptable, but a single concatenate would remove the cost.
