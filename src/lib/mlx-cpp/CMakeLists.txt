cmake_minimum_required(VERSION 3.16)

project(mlx-cpp LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)

# ----------------------------- Fetch MLX C++ -----------------------------

include(FetchContent)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

set(MLX_BUILD_TESTS OFF)
set(MLX_BUILD_EXAMPLES OFF)
set(MLX_BUILD_BENCHMARKS OFF)
set(MLX_BUILD_PYTHON_BINDINGS OFF)

function(mlx_apply_source_overlays mlx_source_dir)
  # Metal patches: JIT type promotion fix + kernel utils (SIMD group matrix).
  #
  # The SDPA / steel-attention overlays (scaled_dot_product_attention.cpp,
  # scaled_dot_product_attention.metal, steel_attention.metal,
  # steel_attention_nax.metal) were dropped when the MLX pin moved to
  # v0.31.2 — that release already carries the NAX fixes we had been
  # carrying locally (`#3361 int16 overflow in SDPA NAX mask indexing`,
  # `#3422 Speed up NAX split-K`, `#3419 Segmented mm nax kernel`). Removing
  # the patches keeps our build binary-equivalent to the upstream Python
  # wheel and simplifies future upstream syncs.
  # The steel/gemm/mma.h overlay (issue #217: upstream safe-load fix
  # cherry-pick, PRs #3560/#3565) was retired when the pin moved to
  # 2026-06-11 upstream main, which carries the fix natively (issue #222).
  set(_metal_patch_files
      "mlx/backend/metal/compiled.cpp"
      "mlx/backend/metal/kernels/utils.h")
  foreach(_patch_file ${_metal_patch_files})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/patches/${_patch_file}")
      configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/patches/${_patch_file}"
        "${mlx_source_dir}/${_patch_file}"
        COPYONLY)
    endif()
  endforeach()
  message(STATUS "Applied Metal kernel patches")

  # CUDA patches: custom grouped GEMM, quantized kernels, reduce ops, binary ops.
  # These are in patches/mlx/backend/cuda/ and only compiled for CUDA targets,
  # so they are safe to apply unconditionally (harmless on Metal-only builds).
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx/backend/cuda")
    file(GLOB_RECURSE _cuda_patch_files
         RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/patches"
         "${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx/backend/cuda/*")
    foreach(_patch_file ${_cuda_patch_files})
      configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/patches/${_patch_file}"
        "${mlx_source_dir}/${_patch_file}"
        COPYONLY)
    endforeach()
    message(STATUS "Applied CUDA backend patches (${_cuda_patch_files})")
  endif()

  # CUDA-only patches: bf16 type promotion and norm computation changes.
  # These change dtype promotion rules (bf16+fp32→bf16) to avoid copy_v kernel
  # overhead on CUDA, but break Metal precision for bf16 vision weights
  # (e.g., SigLIP position embeddings). Apply ONLY for CUDA builds.
  if(MLX_BUILD_CUDA AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/patches-cuda")
    file(GLOB _cuda_only_patches
         "${CMAKE_CURRENT_SOURCE_DIR}/patches-cuda/*.cpp")
    foreach(_patch_file ${_cuda_only_patches})
      get_filename_component(_fname "${_patch_file}" NAME)
      configure_file(
        "${_patch_file}"
        "${mlx_source_dir}/mlx/${_fname}"
        COPYONLY)
    endforeach()
    message(STATUS "Applied CUDA-only dtype/precision patches")
  endif()
endfunction()

set(_mlx_cached_source_dir "${CMAKE_BINARY_DIR}/_deps/mlx-src")
set(_mlx_cached_binary_dir "${CMAKE_BINARY_DIR}/_deps/mlx-build")

# Reuse an already-populated source tree when present so offline rebuilds can
# proceed without re-running FetchContent. This is especially useful when Cargo
# changes the build hash but the developer already has a cached MLX checkout.
# NOTE: Stale-cache validation is handled by build.rs (purge_stale_mlx_cache)
# which runs *before* CMake, ensuring _deps/ is either valid or absent.
if(EXISTS "${_mlx_cached_source_dir}/CMakeLists.txt")
  set(mlx_SOURCE_DIR "${_mlx_cached_source_dir}")
  set(mlx_BINARY_DIR "${_mlx_cached_binary_dir}")
  message(STATUS "Using cached MLX source from ${mlx_SOURCE_DIR}")
else()
  FetchContent_Declare(
    mlx
    GIT_REPOSITORY "https://github.com/ml-explore/mlx.git"
    GIT_TAG e9463bbfc1a7cd9e0e6b96aaa3068a316e234a63)

  # Use FetchContent_Populate + add_subdirectory so we can apply source
  # overlays before the MLX build system processes the files.
  FetchContent_GetProperties(mlx)
  if(NOT mlx_POPULATED)
    FetchContent_Populate(mlx)
  endif()
endif()

mlx_apply_source_overlays("${mlx_SOURCE_DIR}")

# Sanity-check: detect deprecated MLX Device API usage in patch files that
# would cause build failures after an upstream MLX version bump.
set(_deprecated_patterns
    "d\\.get_command_encoder("
    "d\\.new_command_buffer(")
foreach(_patch_file ${_metal_patch_files_global})
  # _metal_patch_files is function-local; re-glob here
endforeach()
file(GLOB_RECURSE _all_patches
     "${CMAKE_CURRENT_SOURCE_DIR}/patches/mlx/backend/metal/*.cpp")
foreach(_pf ${_all_patches})
  file(READ "${_pf}" _pf_content)
  foreach(_pat ${_deprecated_patterns})
    string(FIND "${_pf_content}" "${_pat}" _found)
    if(NOT _found EQUAL -1)
      message(WARNING "Patch file ${_pf} uses deprecated API '${_pat}' — "
                      "update to match upstream MLX at GIT_TAG commit.")
    endif()
  endforeach()
endforeach()

add_subdirectory(${mlx_SOURCE_DIR} ${mlx_BINARY_DIR})

# ----------------------------- CUDA CCCL ---------------------------------
# Ensure MLX_CCCL_DIR is defined for CUDA JIT kernel compilation at runtime.
# MLX only sets this for test builds, but statically linked binaries need it too.
# Since upstream e9463bb the macro is consumed by cccl_dir() in dirs.cpp, which
# is compiled in the dedicated mlx_dirs OBJECT target (upstream isolates the
# dynamic defines there to avoid invalidating the compile cache), so define it
# on mlx_dirs when that target exists and fall back to mlx otherwise.
if(MLX_BUILD_CUDA)
  set(_cccl_include "${CMAKE_BINARY_DIR}/_deps/cccl-src/include")
  if(EXISTS "${_cccl_include}")
    if(TARGET mlx_dirs)
      target_compile_definitions(mlx_dirs
                                 PRIVATE MLX_CCCL_DIR="${_cccl_include}")
    else()
      target_compile_definitions(mlx
                                 PRIVATE MLX_CCCL_DIR="${_cccl_include}")
    endif()
  endif()

  # CUTLASS/CuTe headers for the runtime-JIT'd quantized kernels (MLX #3706/
  # #3576 moved qmm_sm80/sm90/naive and gather_gemm to NVRTC JIT compilation,
  # and those kernels include <cute/...> / <cutlass/...> at runtime). Upstream
  # only installs the headers next to installed binaries; statically linked
  # mlxcel binaries need a compiled-in fallback for dev builds. Consumed by
  # our jit_module.cpp overlay, which compiles into the mlx target.
  set(_cutlass_include "${CMAKE_BINARY_DIR}/_deps/cutlass-src/include")
  if(EXISTS "${_cutlass_include}")
    target_compile_definitions(mlx
                               PRIVATE MLX_CUTLASS_DIR="${_cutlass_include}")
  endif()
endif()
