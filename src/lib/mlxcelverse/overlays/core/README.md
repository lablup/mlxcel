# overlays/core: MLX core overlay

This directory holds mlxcelverse's source overlay for MLX core files outside any backend. `mlx_apply_source_overlays` in `../../CMakeLists.txt` copies them over the fetched MLX source tree on every build (Metal, CUDA, ROCm, and CPU-only), before the backend overlays. The ROCm overlay replaces `mlx/device.cpp` with its own copy, which carries the same marked block.

## Files

- `mlx/device.cpp`: Modified from upstream `ml-explore/mlx` (`mlx/device.cpp`). Makes the default device per thread, as upstream already does for default streams, and adds `mlxcel_set_process_default_device` for threads that never set their own (issue #3066). Without it, a thread that pinned the CPU moved every other thread's unstreamed ops onto the CPU.

## Provenance and pin bumps

Every file in this directory keeps upstream MLX's Apple copyright header followed by `// Modified from upstream MLX <40-char-sha> <upstream/relative/path>`. On an MLX pin bump, 3-way merge against the declared base commit and update the header SHA as documented in [CONTRIBUTING.md](../../../../../CONTRIBUTING.md). Re-apply the same block to `overlays/rocm/mlx/device.cpp`.
