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
  // ROCm is a real GPU backend whose `fast::hip_kernel` ports arrive one
  // kernel at a time (issue #1814). A family-wide gate
  // (`custom_kernels_available()`) treats it like `None` and takes the graph
  // fallback; a kernel with a HIP port is gated on its own predicate, which
  // reads its port table (`has_kernel_port`). Nothing may call
  // `fast::cuda_kernel` here.
  Rocm = 3,
};

// True when this backend has every custom kernel family ported, that is Metal
// or CUDA. Family-wide gates use it; a kernel with per-backend ports answers
// through `has_kernel_port` on its own table instead. It is deliberately not
// "a GPU is present", which is what the old idiom conflated it with.
constexpr bool custom_kernels_available_for(GpuKernelBackend backend) {
  return backend == GpuKernelBackend::Metal ||
      backend == GpuKernelBackend::Cuda;
}

// How many values `GpuKernelBackend` has.
//
// Pinned by a `static_assert` in `kernel_port.cpp`, so adding a backend fails
// the build there with a message naming what to extend rather than leaving every
// per-kernel port table silently short one field. Bumping this number is the
// deliberate acknowledgement that those places were reviewed.
constexpr int kGpuKernelBackendCount = 4;

// Resolved once from the compiled-in backends and the runtime device. Set
// MLXCEL_DEBUG_KERNEL_BACKEND to have the resolved value printed once.
GpuKernelBackend gpu_kernel_backend();

// custom_kernels_available_for(gpu_kernel_backend()).
bool custom_kernels_available();

} // namespace mlxcel
