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

// Fused tanh-GELU custom Metal kernel launcher, f32 arithmetic.
//
// This header exposes two C++ entry points that together implement the
// "real Metal port" of the #2423 fused path:
//
//   - `fused_gelu_tanh_f32_available()` — whether the fused kernel runs on
//     this backend (Metal), read the same way the launcher decides.
//   - `fused_gelu_tanh_f32(const mlx::core::array& x)` — issue the
//     runtime-JIT Metal kernel via `mlx::core::fast::metal_kernel` and
//     return the activated tensor.
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
// (cpp/mlx_cxx_internal.h `gelu_tanh_approx`); the fused kernel may contract
// a [multiply-add] inside the compiled region, so lanes may differ from it
// by f32 rounding. The upcast and the downcast happen inside the kernel.
//
// Used by: `cpp/mlx_cxx_bridge.cpp` (`fused_gelu_tanh_f32`), which is the
// `GeluActivation::TanhF32Compiled` dispatch target.

#include <mlx/array.h>

namespace mlxcel::turbo {
// Whether the fused kernel runs on this backend rather than refusing.
//
// Custom kernels run only on the GPU stream, so this is false with
// MLXCEL_DEVICE=cpu: callers take the eager `gelu_tanh_f32` chain instead of
// the launch throwing (the same check as fused_xielu_kernel_available).
bool fused_gelu_tanh_f32_available();

// Evaluate tanh-GELU on `x` with one fused custom Metal kernel.
//
// Inputs:
// - `x`: any shape; flattened to 1-D for the launch and reshaped back on the
//   way out. f32, bfloat16 and float16 are supported (the kernel upcasts its
//   element type to f32 and rounds back at the store).
//
// Output:
// - activated array with `x`'s shape and dtype. May differ from the eager
//   `gelu_tanh_f32` chain by f32 rounding (FMA contraction inside the fused
//   kernel).
//
// Panics (throws) when there is no port on this backend — no GPU default
// device, or no Metal port in the table. Rust-side callers must gate on
// `fused_gelu_tanh_f32_available()` first or route to the eager chain.
mlx::core::array fused_gelu_tanh_f32(const mlx::core::array& x);

} // namespace mlxcel::turbo
