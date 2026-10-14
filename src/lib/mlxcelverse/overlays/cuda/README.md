# overlays/cuda: CUDA backend and core overlay

This directory holds mlxcelverse's source overlays for MLX's CUDA backend (`mlx/backend/cuda/**`, 22 files) and CUDA-only MLX core files (`mlx/*.cpp`, 3 files).

## How `CMakeLists.txt` applies this directory

`mlx_apply_source_overlays` in `../../CMakeLists.txt` applies the two subtrees with distinct gating:

1. `mlx/backend/cuda/**` (22 files: custom grouped GEMM and CUTLASS arch selection, quantized QMV/QMM/FP-QMV kernels and tile selection, reduce ops, binary/unary ops, compiled JIT module and CUTLASS include resolution, device/allocator adjustments, and `mlxcel` helper headers `grouped_gemm_arch.h` and `qmm_naive_tile.h`): copied into `${mlx_source_dir}/mlx/backend/cuda/` on every build. MLX only compiles `mlx/backend/cuda/` when `MLX_BUILD_CUDA` is `ON`, so copying them unconditionally is a no-op on Metal, ROCm, and CPU-only builds.
2. `mlx/{dtype,fast,ops}.cpp` (3 MLX core files): copied into `${mlx_source_dir}/mlx/` **only when `MLX_BUILD_CUDA` is `ON`**. These files adjust `bfloat16` + `float32` type promotion and norm precision to avoid `copy_v` kernel overhead on CUDA, which would alter Metal precision on `bfloat16` vision weights (such as SigLIP position embeddings) and conflict with the ROCm core overlay if applied outside CUDA builds.

## Provenance and pin bumps

Every file in this directory is modified from upstream `ml-explore/mlx` (except the two mlxcel-authored helper headers `grouped_gemm_arch.h` and `qmm_naive_tile.h`, which are marked `// New file for mlxcel (no upstream MLX counterpart)`), retains Apple's copyright header, and declares its upstream base commit on the line `// Modified from upstream MLX <40-char-sha> <upstream/relative/path>`. On an MLX pin bump, 3-way merge each file from its declared base commit to the new pin (`git merge-file`) and update the header SHA as documented in [CONTRIBUTING.md](../../../../../CONTRIBUTING.md).
