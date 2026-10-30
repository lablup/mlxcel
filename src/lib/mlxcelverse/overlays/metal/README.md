# overlays/metal: Metal backend overlay

This directory holds mlxcelverse's source overlays for MLX's Metal backend (`mlx/backend/metal/`). `mlx_apply_source_overlays` in `../../CMakeLists.txt` copies these files over the fetched MLX source tree before `libmlx` is compiled.

## Files

- `mlx/backend/metal/device.cpp`: Runtime override of the per-command-buffer input budget (`mlxcel_set_mb_per_buffer_override`), allowing decode steps to raise the budget while prefill keeps the device default.
- `mlx/backend/metal/kernels/utils.h`: Metal kernel utility additions (SIMD group matrix helpers).
- `mlx/backend/metal/kernels/quantized.metal`: adds `mlxcel_affine_qmm_t_rows`, a steel MMA quantized matmul tiled for at most 8 activation rows (BM = 8), so a batched decode step on GPU generations 13 and 14 dequantizes each weight tile once instead of running `qmv` once per row.
- `mlx/backend/metal/quantized.cpp`: `MLXCEL_QMV_WIDE` runtime off-switch on `use_qmv_wide` (issue #1187), and the dispatch that sends 5- to 8-row affine 4- and 8-bit products on generations 13 and 14 to `mlxcel_affine_qmm_t_rows` behind the same switch.

The `mlx/backend/metal/compiled.cpp` overlay (an input cast for mixed-dtype fused kernels) was retired in #2333: MLX ops promote their own inputs, so upstream emission is correct, and the cast broke comparisons (#1792) and `Imag` of complex64. `mlx_apply_source_overlays` restores a retired file to upstream in a cached `_deps/mlx-src`.

## Provenance and pin bumps

Every file in this directory is modified from upstream `ml-explore/mlx`, retains Apple's copyright header, and declares its upstream base commit on the line `// Modified from upstream MLX <40-char-sha> <upstream/relative/path>`. On an MLX pin bump, 3-way merge each file from its declared base commit to the new pin (`git merge-file`) and update the header SHA as documented in [CONTRIBUTING.md](../../../../../CONTRIBUTING.md).
