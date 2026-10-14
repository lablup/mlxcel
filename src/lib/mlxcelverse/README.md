# mlxcelverse

`mlxcelverse` (`src/lib/mlxcelverse/`) is mlxcel's MLX-side C++ layer: the CMake project that fetches upstream `ml-explore/mlx` at the pinned commit, lays mlxcel's per-backend source overlays over the fetched tree before compiling `libmlx`, and holds mlxcel's own cross-backend GPU kernels and launchers.

## Layout

```text
src/lib/mlxcelverse/
  README.md               # This file
  CMakeLists.txt          # Fetches pinned MLX, applies overlays, compiles libmlx
  overlays/               # Whole-file overlays copied into _deps/mlx-src before build
    cpu/                  # CPU backend overlays (mlx/backend/cpu/**)
    metal/                # Metal backend overlays (mlx/backend/metal/**)
    cuda/                 # CUDA backend and core overlays (mlx/backend/cuda/**, mlx/*.cpp)
    rocm/                 # ROCm backend and core overlays (CMakeLists.txt, mlx/**) + records
  kernels/                # mlxcel's own fused kernels, launchers, and port dispatch
```

## Two kinds of code, two maintenance rules

### 1. `overlays/`: modified upstream MLX files (`cpu/`, `metal/`, `cuda/`, `rocm/`)

Files under `overlays/<backend>/` mirror the upstream MLX directory tree (`mlx/...`, plus `CMakeLists.txt` in `rocm/`) and replace or extend upstream MLX files inside `${CMAKE_BINARY_DIR}/_deps/mlx-src` via `mlx_apply_source_overlays` in `CMakeLists.txt` before `add_subdirectory` compiles `libmlx`.

- **Provenance and headers**: Every file in `overlays/` originates from upstream `ml-explore/mlx` (or `NripeshN/mlx` for `overlays/rocm/mlx/backend/rocm/**`) and retains its original copyright header (`Copyright © 2023-... Apple Inc.`). Never add a Lablup Apache-2.0 header to an overlay file; `scripts/insert_apache_header.py` skips files carrying an Apple copyright notice.
- **`overlays/cpu/`, `overlays/metal/`, `overlays/cuda/`**: Every source and header file carries a `// Modified from upstream MLX <40-char-sha> <upstream/relative/path>` declaration immediately below its copyright header (plus `// New file for mlxcel (no upstream MLX counterpart)` on mlxcel-authored helper headers in `overlays/cuda/`). On an MLX pin bump, 3-way merge each file from its declared base commit to the new pin (`git merge-file <overlay> <old-pin-base> <new-pin>`) and update the header SHA; see [CONTRIBUTING.md](../../../CONTRIBUTING.md) ("Bumping the MLX upstream pin").
- **`overlays/rocm/`**: Vendored from `NripeshN/mlx@rocm-support` and retargeted to the MLX pin. Tracked by `UPSTREAM`, `LOCAL_FIXES.md`, and `CORE_RESIDUAL.diff`, and maintained with `scripts/mlxcelverse/rocm_overlay.py` (`records`, `drift`, `retarget`, `sync`); see [overlays/rocm/README.md](overlays/rocm/README.md) and [docs/mlxcelverse/rocm-overlay.md](../../../docs/mlxcelverse/rocm-overlay.md).

### 2. `kernels/`: mlxcel's own fused kernels and port dispatch

Files under `kernels/` (`paged_attention*`, `sparse_v_sdpa*`, `turbo4_delegated_sdpa*`, `sampling*`, `fused_norm*`, `fused_rope_append*`, `kv_inplace_write*`, `gpu_backend*`, `kernel_port*`) are authored by Lablup for mlxcel. They are compiled into the `mlx_cxx_bridge` library by `src/lib/mlxcel-core/build.rs` and call into `mlx::core::fast::{metal_kernel,cuda_kernel,hip_kernel}` at runtime through `mlxcel::select_kernel_port`.

- **Provenance and headers**: Every `.cpp` and `.h` file under `kernels/` is first-party mlxcel code and carries the Lablup Apache-2.0 header (enforced by `scripts/insert_apache_header.py --check`).
- **Port dispatch and JIT cache keys**: Every kernel launcher routes backend selection through `mlxcel::select_kernel_port` (`kernels/kernel_port.h`, enforced by `scripts/ci/check_kernel_port_dispatch.py`) and includes input dtypes in its CUDA/HIP `TemplateArg` list (`scripts/ci/check_kernel_dtype_keys.py`).
