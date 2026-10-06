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

// One way to choose a custom kernel's per-backend port.
//
// Every fused kernel used to do this by hand, and the hand-written form had a
// recurring defect: `use_cuda ? cuda_port : metal_port` treats "not CUDA" as
// "Metal", so each new backend silently took the Metal arm and
// `fast::metal_kernel` threw. Issue #1803 replaced the underlying
// `!metal::is_available()` test with a named backend, but left nine sites
// shaped that way; #1885 and #2018 then added a refusal to each by hand, one
// launcher at a time, and the hand-written guards drifted in wording and twice
// cited support predicates that did not exist.
//
// The point of this header is that adding a backend, or adding a kernel, is no
// longer per-site work:
//
//   - A kernel declares which ports it has, once, as a `KernelPorts` table. A
//     null entry means "no port for that backend", which is a first-class state
//     rather than something a dispatch falls into.
//   - `select_kernel_port` owns the refusal, so its message always names the
//     entry point, the missing port and the fallback the caller should take,
//     and cannot drift per site.
//   - `has_kernel_port` answers the support predicate from the same table the
//     dispatch reads. Before this, `ssm_kernel_available()` answered with
//     `cu::is_available()` while its launcher guarded on
//     `custom_kernels_available()`: two predicates for one question, free to
//     disagree.
//   - Adding a `GpuKernelBackend` enumerator fails to compile in
//     `kernel_port.cpp` with a message naming what to extend, rather than
//     leaving every table silently short one field.

#include <mlx/fast.h>

#include "gpu_backend.h"

namespace mlxcel {

// A holder's accessor. Captureless lambdas convert to this, so a port entry is
// written inline as `+[]() -> mlx::core::fast::CustomKernelFunction& { return
// get_x_kernel().get(); }`.
using KernelPortGetter = mlx::core::fast::CustomKernelFunction& (*)();

// Which backends a kernel has a port for. A null member is "no port here".
//
// Deliberately not defaulted to anything but null: a kernel that gains a
// backend has to say so, and one that has never had a given port reads as
// absent rather than as an accident.
struct KernelPorts {
  KernelPortGetter metal = nullptr;
  KernelPortGetter cuda = nullptr;
  KernelPortGetter rocm = nullptr;
  // True when the ROCm port is correct at any wavefront width, because its HIP
  // body has no lane-level operation (no shuffle, ballot or per-warp slot).
  // False, the default, holds the port to 32-lane devices: on a wave64 GPU
  // (CDNA) `port_for` answers "no port" and callers take their graph fallback
  // (issue #2147). A new port is therefore refused on wave64 unless its author
  // says otherwise, and `make verify-kernel-port-dispatch` pins which tables
  // say so and checks that their HIP source has no lane intrinsic.
  bool rocm_any_wave_size = false;
};

// True when the resolved backend has a port in this table, and on ROCm the
// device's wavefront width allows it (`rocm_port_allowed`).
//
// This is what a kernel's `*_available()` predicate should return, so that the
// predicate and the dispatch cannot answer differently. A caller that gates on
// it may then `expect` the launch, because an error would mean the two
// disagree.
bool has_kernel_port(const KernelPorts& ports);

// The port for the resolved backend.
//
// Throws when there is none, naming `entry_point` and `fallback` so the message
// says what is missing (or, for a wave32-only ROCm port on a wider device, the
// wavefront width) and what the caller should do instead, rather than
// naming whichever port happened to be tried. The bridge function that reaches
// here must be declared `-> Result<...>` in the cxx bridge, or the throw
// crosses a `noexcept` extern and ends the process instead of failing the call.
//
// Changing that declaration changes every Rust call site, and the two shapes
// are not caught alike. A launcher returning a value becomes `Result<T>`, so
// the old `let x = ffi::f(...)` stops compiling and `cargo check` finds it. A
// void launcher becomes `Result<()>`, and an ignored `Result<()>` is only the
// `unused_must_use` lint: `cargo check` stays green and the refusal is
// swallowed at runtime. Those sites surface solely under `-D warnings` clippy,
// and PR-time CI's `clippy` job lints `-p mlxcel` only, so a missed call in
// mlxcel-core reaches main. `make verify` and `make verify-rocm` are the gates
// that catch it, because both lint `--workspace --all-targets`; run one of them,
// not a hand-written `-p mlxcel` clippy, after touching a launcher's signature.
mlx::core::fast::CustomKernelFunction& select_kernel_port(
    const char* entry_point,
    const char* fallback,
    const KernelPorts& ports);

} // namespace mlxcel
