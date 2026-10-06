// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gpu_backend.h"

#include <mlx/backend/cuda/cuda.h>
#include <mlx/backend/metal/metal.h>

// `mlx/backend/rocm/rocm.h` ships in the ROCm overlay
// (`src/lib/mlx-cpp/patches-rocm/`), which CMake copies into the MLX source
// tree only for a ROCm build, so the include has to be gated the same way the
// Metal-backend-only symbols are. build.rs defines this alongside
// MLXCEL_BRIDGE_METAL_BACKEND.
#ifdef MLXCEL_BRIDGE_ROCM_BACKEND
#include <mlx/backend/rocm/rocm.h>
#endif

#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace mlxcel {

namespace {

GpuKernelBackend resolve_gpu_kernel_backend() {
  if (mlx::core::metal::is_available()) {
    return GpuKernelBackend::Metal;
  }
#ifdef MLXCEL_BRIDGE_ROCM_BACKEND
  // Checked before CUDA: on a ROCm build `cu::is_available()` is the no-CUDA
  // stub and returns false, but ordering it this way keeps the answer right if
  // a future build ever links both.
  if (mlx::core::rocm::is_available()) {
    return GpuKernelBackend::Rocm;
  }
#endif
  if (mlx::core::cu::is_available()) {
    return GpuKernelBackend::Cuda;
  }
  return GpuKernelBackend::None;
}

const char* backend_name(GpuKernelBackend backend) {
  switch (backend) {
    case GpuKernelBackend::Metal:
      return "metal";
    case GpuKernelBackend::Cuda:
      return "cuda";
    case GpuKernelBackend::Rocm:
      return "rocm";
    case GpuKernelBackend::None:
      return "none";
  }
  return "unknown";
}

} // namespace

GpuKernelBackend gpu_kernel_backend() {
  static const GpuKernelBackend backend = [] {
    GpuKernelBackend resolved = resolve_gpu_kernel_backend();
    if (std::getenv("MLXCEL_DEBUG_KERNEL_BACKEND") != nullptr) {
      std::fprintf(
            stderr,
            "[mlxcel] custom kernel backend: %s%s\n",
          backend_name(resolved),
          custom_kernels_available_for(resolved)
              ? ""
              : " (not every kernel family is ported; the rest use MLX graph fallbacks)");
    }
    return resolved;
  }();
  return backend;
}

bool custom_kernels_available() {
  return custom_kernels_available_for(gpu_kernel_backend());
}

namespace {

// `0` is "no override". Relaxed is enough: the value is a single int that a
// test sets before the reads it wants to affect, with no other data riding on
// it.
std::atomic<int> rocm_port_warp_size_override{0};

int rocm_hardware_warp_size() {
  static const int warp_size = [] {
#ifdef MLXCEL_BRIDGE_ROCM_BACKEND
    if (!mlx::core::rocm::is_available()) {
      return 0;
    }
    int w = mlx::core::rocm::device_warp_size();
    if (w != 32) {
      // Once per process, so an operator on CDNA can tell why decode runs
      // without the fused kernels rather than reading it as a slowdown.
      std::fprintf(
          stderr,
          "[mlxcel] ROCm device wavefront is %d lanes%s; fused kernels "
          "validated only on 32 lanes use their MLX graph fallbacks, or are "
          "refused at load where none exists (BitNet)\n",
          w,
          w == 0 ? " (query failed)" : "");
    }
    return w;
#else
    return 0;
#endif
  }();
  return warp_size;
}

} // namespace

int rocm_port_warp_size() {
  int forced = rocm_port_warp_size_override.load(std::memory_order_relaxed);
  return forced > 0 ? forced : rocm_hardware_warp_size();
}

void set_rocm_port_warp_size_for_tests(int warp_size) {
  rocm_port_warp_size_override.store(
      warp_size > 0 ? warp_size : 0, std::memory_order_relaxed);
}

} // namespace mlxcel
