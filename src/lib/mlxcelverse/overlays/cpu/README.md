# overlays/cpu: CPU backend overlay

This directory holds mlxcelverse's source overlay for MLX's CPU backend (`mlx/backend/cpu/`). `mlx_apply_source_overlays` in `../../CMakeLists.txt` copies these files over the fetched MLX source tree on every build (Metal, CUDA, ROCm, and CPU-only), because every build compiles MLX's CPU backend and can execute ops on the CPU stream (`MLXCEL_DEVICE=cpu` or explicit CPU stream dispatch).

## Files

- `mlx/backend/cpu/quantized.cpp`: Modified from upstream `ml-explore/mlx` (`mlx/backend/cpu/quantized.cpp`). Carries the `mlxcel` float-accumulation fix (issue #2248) in the scalar `_qmm`, `_qmm_t`, `fp_qmm`, and `fp_qmm_t` loops used by builds without Apple Accelerate's SIMD header (all Linux builds and x86 macOS), accumulating in `float` via local row buffers instead of accumulating `bfloat16` in `bfloat16`.

## Provenance and pin bumps

Every file in this directory keeps upstream MLX's Apple copyright header followed by `// Modified from upstream MLX <40-char-sha> <upstream/relative/path>`. On an MLX pin bump, 3-way merge against the declared base commit and update the header SHA as documented in [CONTRIBUTING.md](../../../../../CONTRIBUTING.md).
