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
              : " (no custom kernel ports; using MLX graph fallbacks)");
    }
    return resolved;
  }();
  return backend;
}

bool custom_kernels_available() {
  return custom_kernels_available_for(gpu_kernel_backend());
}

} // namespace mlxcel
