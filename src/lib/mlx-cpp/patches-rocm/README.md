# patches-rocm: the ROCm part of mlxcelverse

mlxcelverse is the name for everything mlxcel builds on top of upstream MLX: per-backend source overlays and mlxcel's own kernels. This directory is its ROCm overlay. It gives MLX an AMD GPU (ROCm/HIP) backend so that `cargo build --features rocm` produces a GPU build on Linux AMD hosts.

## What is in here

The tree mirrors the MLX source tree. Every file is a whole-file overlay, copied over the fetched MLX checkout by `mlx_apply_source_overlays` in `../CMakeLists.txt`, and only when `MLX_BUILD_ROCM` is on. Metal and CUDA builds never see these files.

- `mlx/backend/rocm/**`: the ROCm backend (106 files). Upstream MLX has no such directory, so these files never conflict with a pin bump; they only need to keep compiling against the API of the pinned commit.
- 15 MLX core files (`CMakeLists.txt`, `mlx/CMakeLists.txt`, `mlx/backend/common/{buffer_cache.h,compiled.cpp,compiled.h}`, `mlx/backend/gpu/primitives.cpp`, `mlx/{compile,device,fast,ops,primitives,stream}.cpp`, `mlx/{fast,fast_primitives}.h`, `mlx/io/safetensors.cpp`): the upstream file at the pinned commit merged with the fork's ROCm hooks, which are mostly `#ifdef MLX_USE_ROCM` blocks.

Files in this directory other than `CMakeLists.txt` and `mlx/` (this README, `UPSTREAM`, `LOCAL_FIXES.md`) are records and are not copied.

## Where it comes from

The backend is vendored from the `rocm-support` branch of [NripeshN/mlx](https://github.com/NripeshN/mlx/tree/rocm-support) (MIT), the head of the upstream draft [ml-explore/mlx#2300](https://github.com/ml-explore/mlx/pull/2300). `UPSTREAM` records the exact commit and the MLX pin it was retargeted to; `LOCAL_FIXES.md` lists every change relative to that commit. Vendored files keep their original headers; never add a Lablup header to them.

## On an MLX pin bump

1. For each of the 15 core files, 3-way merge the upstream change into the overlay (`git merge-file overlay old-upstream new-upstream`) and compare the result line by line with the new upstream file.
2. Build with `--features rocm`. Upstream API changes show up as compile errors in `mlx/backend/rocm/`, and new primitives without a ROCm kernel show up as undefined `eval_gpu` symbols at link time; give those a `NO_GPU` stub in `mlx/backend/rocm/primitives.cpp` or an implementation.
3. Record every fix in `LOCAL_FIXES.md` and update `retargeted_to_mlx_pin` in `UPSTREAM`.

Removing a file from `mlx/backend/rocm/` here also removes it from a cached MLX checkout on the next configure. Dropping one of the 15 core overlays does not restore the upstream file in a cached checkout, because the cache is keyed only by the pin; clear the `mlxcel-core` build directory (`cargo clean -p mlxcel-core`) after doing that.
