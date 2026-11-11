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

// Fused tanh-GELU custom kernel launcher, f32 arithmetic: a Metal port (#2857)
// and a CUDA port (#2780) of one body.
//
// This header exposes three C++ entry points that together implement the
// fused path:
//
//   - `fused_gelu_tanh_f32_available()`: whether the fused kernel runs on
//     this backend (Metal, CUDA), read the same way the launcher decides.
//   - `fused_gelu_tanh_f32_matches_eager_bits()`: whether the kernel that
//     runs here returns the eager chain's bits (CUDA).
//   - `fused_gelu_tanh_f32(const mlx::core::array& x)`: issue the
//     runtime-JIT kernel via `mlx::core::fast::metal_kernel` or
//     `mlx::core::fast::cuda_kernel` and return the activated tensor.
//
// The kernel is 1-D elementwise: one thread per element, upcast to f32,
// compute the tanh-GELU in f32, cast back to the output dtype:
//
//     x2 = x * x
//     x3 = x2 * x
//     inner = 0.7978845608028654 * (x + 0.044715 * x3)
//     cdf = 0.5 * (1 + tanh(inner))
//     out = x * cdf
//
// Same formula and op order as the eager `gelu_tanh_f32` chain
// (cpp/mlx_cxx_internal.h `gelu_tanh_approx`). The Metal body may contract
// `x + 0.044715 * x3` into a fused multiply-add, so its lanes may differ from
// the eager chain by f32 rounding; the CUDA body rounds every operation on its
// own and returns the eager chain's bits. The upcast and the downcast happen
// inside the kernel.
//
// Used by: `cpp/mlx_cxx_bridge.cpp` (`fused_gelu_tanh_f32`), which is the
// `GeluActivation::TanhF32Compiled` dispatch target on Metal and CUDA and the
// `GeluActivation::TanhF32` one on CUDA.

#include <mlx/array.h>

namespace mlxcel::turbo {
// Whether the fused kernel runs on this backend rather than refusing.
//
// Custom kernels run only on the GPU stream, so this is false with
// MLXCEL_DEVICE=cpu: callers take `compiled_gelu_tanh_f32` instead of
// the launch throwing (the same check as fused_xielu_kernel_available).
bool fused_gelu_tanh_f32_available();

// Whether the fused kernel on this backend returns the bits the eager
// `gelu_tanh_f32` chain returns, so a caller pinned to that chain may launch
// the kernel instead.
//
// True on CUDA with a GPU default device: the CUDA body rounds each multiply
// and add on its own (`__fmul_rn` / `__fadd_rn`), as the eager chain's
// one-kernel-per-op evaluation does. False on Metal, whose body may contract
// a multiply-add, on a backend without a port, and with MLXCEL_DEVICE=cpu.
// Implies `fused_gelu_tanh_f32_available()`.
bool fused_gelu_tanh_f32_matches_eager_bits();

// Evaluate tanh-GELU on `x` with one fused custom kernel (Metal or CUDA).
//
// Inputs:
// - `x`: any shape; flattened to 1-D for the launch and reshaped back on the
//   way out. f32, bfloat16 and float16 are supported (the kernel upcasts its
//   element type to f32 and rounds back at the store).
//
// Output:
// - activated array with `x`'s shape and dtype. On Metal it may differ from
//   the eager `gelu_tanh_f32` chain by f32 rounding (FMA contraction inside
//   the fused kernel); on CUDA it is that chain's bits.
//
// Panics (throws) when there is no port on this backend (no GPU default
// device, or no port for the running backend in the table, which is ROCm
// today). Rust-side callers must gate on `fused_gelu_tanh_f32_available()`
// first or route to the compiled chain.
mlx::core::array fused_gelu_tanh_f32(const mlx::core::array& x);

} // namespace mlxcel::turbo
