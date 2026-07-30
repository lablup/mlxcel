// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

// Fused residual-add + RMSNorm kernel launcher (issue #905).
//
// The per-token decode loop pays two MLX graph nodes for the residual join
// that every pre-norm transformer block performs twice:
//
//     h      = x + attn_out          // one elementwise Add kernel
//     normed = rms_norm(h, weight)   // one fast::rms_norm kernel
//
// The sum `h` is written to global memory by the first kernel and immediately
// re-read by the second, and both `h` and `normed` have to survive as separate
// MLX arrays because `h` is the residual carried into the next join. This
// launcher issues both in one dispatch: the fp32 row sum is stashed once,
// a two-stage (SIMD group, then threadgroup) sum-of-squares reduction runs over
// it, and `rsqrt(mean + eps)` is applied on the way out.
//
// MLX arrays are immutable from the graph's perspective, so the in-place
// residual update that a mutable-tensor runtime would perform is expressed as a
// two-output kernel `(normed, new_residual)` instead.
//
// `weight_bias` absorbs the Gemma `(1 + w)` weight convention without a second
// kernel or a materialised `(1 + w)` tensor: pass `0.0` for a standard RMSNorm
// and `1.0` for Gemma. The bias is folded in the weight's own dtype, exactly as
// `GemmaRMSNorm::new` pre-computes `add(ones, weight)`, so the two agree
// bit-for-bit rather than approximately.
//
// Numerics are pinned to `mlx::core::fast::rms_norm`'s Metal kernel so the
// fused and unfused paths stay interchangeable: the accumulation is fp32 over
// the dtype-rounded sum, the reciprocal square root is `precise::rsqrt`, and
// the store rounds `x * inv_mean` to the activation dtype *before* the weight
// multiply. See `fused_norm.cpp` for the per-line correspondence.
//
// Used by: `cpp/mlx_cxx_ext.cpp` (`fused_add_rms_norm`), reached from
// `mlxcel_core::layers::fused_add_rms_norm` and the Llama3 / Gemma decode
// blocks behind `MLXCEL_FUSED_ADD_RMSNORM`.

#include <mlx/array.h>

#include <vector>

namespace mlxcel::turbo {

// Run the fused residual-add + RMSNorm kernel.
//
// Inputs:
// - `x`:        `[..., D]` activation delta (the attention or MLP output).
// - `residual`: `[..., D]` residual stream, same shape and dtype as `x`.
// - `weight`:   `[D]` RMSNorm scale. May be a different float dtype than `x`;
//   the kernel converts through fp32.
// - `eps`:      RMSNorm epsilon.
// - `weight_bias`: `0.0` for a standard RMSNorm, `1.0` for the Gemma
//   `(1 + w)` convention.
//
// Outputs (in order):
// - `normed`:       `[..., D]`, dtype of `x` — `rms_norm(x + residual) * (weight_bias + weight)`.
// - `new_residual`: `[..., D]`, dtype of `x` — `x + residual`.
//
// Throws `std::invalid_argument` when the shapes or dtypes disagree, or when
// the trailing dimension does not match `weight`. Callers that cannot satisfy
// the contract should use the graph-composed fallback instead.
std::vector<mlx::core::array> fused_add_rms_norm(
    const mlx::core::array& x,
    const mlx::core::array& residual,
    const mlx::core::array& weight,
    float eps,
    float weight_bias);

// Whether the current backend has a fused-add-RMSNorm kernel at all.
//
// `mlx::core::fast::metal_kernel` throws "[metal_kernel] No Metal back-end" on
// a CPU-only build and `cuda_kernel` is equally unavailable there, so the Rust
// helper has to know before it commits to the fused path. True on Metal and
// CUDA, false on a CPU-only build.
bool fused_add_rms_norm_available();

} // namespace mlxcel::turbo
