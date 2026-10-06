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

// Whether a ROCm port may run on a device whose wavefront is `warp_size`
// lanes wide (issue #2147).
//
// Every shuffle-based HIP port was written and validated on a 32-lane
// wavefront (RDNA, gfx1151) only. Its `#error` guard on
// `__AMDGCN_WAVEFRONT_SIZE` cannot catch a 64-lane device, because AMD clang 23
// (HIP 7.15) defines neither spelling of that macro, so this host-side test is
// what keeps such a port off CDNA (gfx90a, gfx942) until someone runs it
// there. A port whose body has no lane-level operation says so with
// `KernelPorts::rocm_any_wave_size` and is allowed at any width. `0` means the
// width could not be read and counts as "not 32".
//
// Pure so it can be tested without a wave64 device; `port_for` in
// `kernel_port.cpp` is its only caller outside the test bridge.
constexpr bool rocm_port_allowed(bool any_wave_size, int warp_size) {
  return any_wave_size || warp_size == 32;
}

// The wavefront width `port_for` holds a ROCm port against: the hardware value
// of the current HIP device (`mlx::core::rocm::device_warp_size()`, which does
// not follow `MLX_ROCM_FORCE_WARP_SIZE`), read once per process, or `0` on a
// build without the ROCm backend or when the query fails.
//
// Read once, for the device that is current at the first call. A process that
// later moves to a GPU of another wavefront width (a host mixing RDNA and CDNA
// cards) keeps the first answer; that case is not handled.
//
// A test can replace the answer with `set_rocm_port_warp_size_for_tests`, so
// the wave64 refusal can be exercised on a wave32 host.
int rocm_port_warp_size();

// Test seam for `rocm_port_warp_size` (issue #2147): a positive value replaces
// the hardware answer for the rest of the process, `0` restores it. Only test
// code may call it; `make verify-kernel-port-dispatch` fails on a call from
// anywhere else. The Rust predicates cache a `true` port answer, so a test
// that wants them to see the override sets it before the first predicate
// call, in a process of its own.
void set_rocm_port_warp_size_for_tests(int warp_size);

} // namespace mlxcel
