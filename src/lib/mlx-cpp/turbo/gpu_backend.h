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

#pragma once

// Which GPU backend mlxcel's custom kernels should target (issue #1803).
//
// Every fused-kernel launcher used to decide with `!metal::is_available()`,
// reading "not Metal" as "CUDA". That was true while Metal and CUDA were the
// only GPU backends. It stopped being true with the ROCm backend (issue #1802):
// there Metal is unavailable too, so every site took the CUDA branch and called
// `fast::cuda_kernel`, whose no-CUDA stub throws `[cuda_kernel] No CUDA
// back-end.` Most callers catch that and fall back to an MLX graph, but the
// throw crosses the cxx bridge into a `noexcept` extern wherever they do not,
// which ends the process in `std::terminate`.
//
// The replacement names the backend instead of negating one. On Metal and CUDA
// builds it resolves to exactly what the old boolean did, which is the point:
// `Metal` where `metal::is_available()` was true, `Cuda` otherwise on a build
// that has CUDA. Only a ROCm build sees a new value.

namespace mlxcel {

enum class GpuKernelBackend {
  // No GPU backend, or one with no custom-kernel port. Callers take their MLX
  // graph fallback.
  None = 0,
  Metal = 1,
  Cuda = 2,
  // ROCm is a real GPU backend with no `fast::hip_kernel` ports yet (issue
  // #1814). Callers treat it like `None` and take the graph fallback; they must
  // not call `fast::cuda_kernel`.
  Rocm = 3,
};

// True when this backend has custom kernel ports, that is Metal or CUDA. This
// is the condition the launchers gate on; it is deliberately not "a GPU is
// present", which is what the old idiom conflated it with.
constexpr bool custom_kernels_available_for(GpuKernelBackend backend) {
  return backend == GpuKernelBackend::Metal ||
      backend == GpuKernelBackend::Cuda;
}

// Resolved once from the compiled-in backends and the runtime device. Set
// MLXCEL_DEBUG_KERNEL_BACKEND to have the resolved value printed once.
GpuKernelBackend gpu_kernel_backend();

// custom_kernels_available_for(gpu_kernel_backend()).
bool custom_kernels_available();

} // namespace mlxcel
