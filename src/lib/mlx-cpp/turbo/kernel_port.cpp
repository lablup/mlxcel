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

#include "kernel_port.h"

#include <stdexcept>
#include <string>

namespace mlxcel {

namespace {

// The single place that maps a backend to a table entry.
//
// The switch is exhaustive with no `default:` on purpose. `-Wswitch` flags a new
// enumerator here, and the `static_assert` below turns that warning into a build
// failure, because this repository compiles with warnings enabled but not
// warnings-as-errors and a new one would be lost among the MLX ones.
KernelPortGetter port_for(const KernelPorts& ports, GpuKernelBackend backend) {
  switch (backend) {
    case GpuKernelBackend::Metal:
      return ports.metal;
    case GpuKernelBackend::Cuda:
      return ports.cuda;
    case GpuKernelBackend::Rocm:
      // The shuffle-based HIP ports have run on 32-lane wavefronts only, and
      // their `#error` guards are inert on current compilers, so a wider
      // device gets "no port" here rather than an unvalidated kernel
      // (issue #2147). Checked only when a port exists, so a table with no
      // ROCm entry never queries the device.
      if (ports.rocm == nullptr ||
          !rocm_port_allowed(ports.rocm_any_wave_size, rocm_port_warp_size())) {
        return nullptr;
      }
      return ports.rocm;
    case GpuKernelBackend::None:
      return nullptr;
  }
  return nullptr;
}

static_assert(
    kGpuKernelBackendCount == 4,
    "GpuKernelBackend gained a value. Add a field for it to KernelPorts, a case "
    "to port_for, and an entry to every kernel's port table; then bump "
    "kGpuKernelBackendCount. Leaving this assert to pass on its own would let "
    "the new backend read as 'no port' at every kernel silently.");

} // namespace

bool has_kernel_port(const KernelPorts& ports) {
  return port_for(ports, gpu_kernel_backend()) != nullptr;
}

mlx::core::fast::CustomKernelFunction& select_kernel_port(
    const char* entry_point,
    const char* fallback,
    const KernelPorts& ports) {
  GpuKernelBackend backend = gpu_kernel_backend();
  if (KernelPortGetter getter = port_for(ports, backend)) {
    return getter();
  }
  if (backend == GpuKernelBackend::Rocm && ports.rocm != nullptr) {
    // The port exists and was held back by the wavefront width. Saying so
    // keeps a CDNA user from reading this as a missing port.
    throw std::runtime_error(
        std::string("[") + entry_point +
        "] the ROCm port is validated only on a 32-lane wavefront and this "
        "device reports " +
        std::to_string(rocm_port_warp_size()) +
        " lanes; mlxcel's callers take the " + fallback + " instead");
  }
  // Named from the table rather than from whichever port was tried, which is
  // what made the old per-site messages misleading: `[metal_kernel] No Metal
  // back-end.` on an AMD host reads as a missing Metal install when the real
  // fact is a healthy ROCm GPU with no port for this kernel.
  throw std::runtime_error(
      std::string("[") + entry_point +
      "] no custom kernel port for this GPU backend; mlxcel's callers take the " +
      fallback + " instead");
}

} // namespace mlxcel
