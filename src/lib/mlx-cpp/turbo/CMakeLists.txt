# Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# mlxcel TurboQuant fused kernel launchers.
#
# Currently only Sparse-V SDPA on Metal. The kernels themselves are runtime-
# JIT-compiled via `mlx::core::fast::metal_kernel(...)`; this CMake target
# exists to keep the C++ launcher sources as a discrete library that the
# top-level mlxcel-core cxx-bridge build (build.rs) can list as an extra
# source via `cxx_build::bridge(..).file(...)`.
#
# IMPORTANT: This subdirectory is INFORMATIONAL only at the moment — the
# top-level `src/lib/mlx-cpp/CMakeLists.txt` does not yet add_subdirectory()
# this dir, because the cxx-build path in `mlxcel-core/build.rs` compiles the
# launcher .cpp files directly. The CMakeLists.txt below documents the build
# contract so MLX upstream-commit upgrades treat this directory as in-scope.
# See `CLAUDE.md` MLX upgrade checklist.

cmake_minimum_required(VERSION 3.16)

project(mlxcel-turbo-kernels LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

# Header search paths — these are exported to consumers via target_include
# directories so a parent CMakeLists.txt can pick this up via
# add_subdirectory(turbo).
set(MLXCEL_TURBO_KERNEL_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/sparse_v_sdpa.h
    ${CMAKE_CURRENT_SOURCE_DIR}/turbo4_delegated_sdpa.h)

set(MLXCEL_TURBO_KERNEL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/sparse_v_sdpa.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/turbo4_delegated_sdpa.cpp)

set(MLXCEL_TURBO_KERNEL_METAL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/sparse_v_sdpa.metal)

add_library(mlxcel-turbo-kernels STATIC
    ${MLXCEL_TURBO_KERNEL_SOURCES})

target_include_directories(mlxcel-turbo-kernels
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${mlx_SOURCE_DIR})

# Metal sources are compiled at runtime via mlx::core::fast::metal_kernel;
# the .metal file is kept as a reference and is not built by this CMake target.
# Link mlxcel-turbo-kernels against MLX so the metal_kernel JIT API is reachable.
if(TARGET mlx)
    target_link_libraries(mlxcel-turbo-kernels PUBLIC mlx)
endif()
