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

// Fused absorbed-MLA single-query decode attention (issue #3121), Metal.
//
// One decode step of matrix-absorbed MLA attends `num_heads` query rows
// against one shared latent KV head:
//
//     score[h, t] = scale * (q_abs[h] . ckv[t] + q_pe[h] . kpe[t])
//     out[h]      = softmax_t(score[h, :]) @ ckv
//
// The composed path builds the rope half as a `[B, H, 1, S]` additive mask and
// hands it to MLX's SDPA, which does not fuse a 512-wide head with an array
// mask and so materializes the score, softmax and value product as separate
// kernels. At short context that chain of small launches, not the bytes it
// reads, is most of the step. This launcher does the same arithmetic in two
// kernels: a partial pass that splits the keys into two halves (one
// threadgroup per head, batch row and half; 32 simdgroups stride the keys and
// combine through threadgroup memory, f32 throughout), and a merge that
// rescales the two halves by their log-sum-exp.
//
// A second variant of the partial pass reads an affine-quantized latent
// window (issue #3156): the same body with the per-key latent load reading
// the packed codes and each group's scale and bias factored out of the score
// and the value sum, so a quantized cache is decoded without dequantizing the
// window.
//
// Used by: `cpp/mlx_cxx_bridge.cpp` (`mla_absorbed_decode_fused`,
// `mla_absorbed_decode_fused_quantized` and their `_supported` probes), the
// single-query branch of
// `mlxcel_core::mla::decode::absorbed_attention` and
// `mlxcel_core::mla::decode::fused_absorbed_decode_quantized`.

#include <mlx/array.h>

namespace mlxcel::turbo {

// Whether the fused kernel runs on this backend: a GPU default device with a
// Metal port. False on CPU, CUDA and ROCm, where the composed path runs.
bool mla_absorbed_decode_available();

// Absorbed single-query MLA attention in latent value space.
//
// Inputs:
// - `q_abs` `[B, H, 1, R]`: queries after the `W_UK` fold.
// - `q_pe`  `[B, H, 1, P]`: rotated rope queries.
// - `ckv`   `[B, 1, S, R]`: the live latent window (keys and values).
// - `kpe`   `[B, 1, S, P]`: the live rope-key window.
// - `scale`: the softmax scale applied to both score halves.
//
// `R` and `P` must be multiples of 32 with `R <= 1024`; all four inputs share
// one float dtype (f16, bf16 or f32). `ckv` and `kpe` may be strided views of
// a larger cache buffer as long as their last axis is contiguous.
//
// Output: `[B, H, 1, R]` in the inputs' dtype.
//
// Throws when there is no port on this backend or a shape is outside the
// contract; Rust callers gate on `mla_absorbed_decode_available()` and the
// shape checks in `mla::decode` first.
mlx::core::array mla_absorbed_decode(
    const mlx::core::array& q_abs,
    const mlx::core::array& q_pe,
    const mlx::core::array& ckv,
    const mlx::core::array& kpe,
    float scale);

// Whether the partial pass evaluates at this geometry on this device. A
// Metal pipeline's thread limit is fixed when the pipeline compiles, from the
// kernel's register use, and the partial pass always launches 1024 threads
// (32 simdgroups x 32 lanes, which its output exchange depends on). On an M1
// Ultra the pipeline at `R = 512` is capped at 768, so MLX throws at eval and
// the decode step fails. The answer comes from launching the real kernel once
// on a one-key window at `(dtype, R, P)` and evaluating it; it is cached per
// geometry for the process. False wherever `mla_absorbed_decode_available()`
// is false or the geometry is outside the shape contract.
bool mla_absorbed_decode_supported(mlx::core::Dtype dtype, int R, int P);

// Whether the quantized variant runs on this backend; same rule as
// `mla_absorbed_decode_available()`, against its own port table.
bool mla_absorbed_decode_quantized_available();

// `mla_absorbed_decode` over an affine-quantized latent window.
//
// Inputs that differ from the dense variant:
// - `ckv_q`  `[B, 1, S, R * bits / 32]` uint32: the packed latent, in the MLX
//   `quantize` layout (element `e` in bits `[bits * (e % (32 / bits)), +bits)`
//   of word `e / (32 / bits)`).
// - `scales`, `biases` `[B, 1, S, R / group_size]`: one affine pair per group,
//   an element reading back as `code * scale + bias`. One float dtype for
//   both, independent of the queries'.
// - `kpe` `[B, 1, S, P]`: any float dtype, independent of the queries'.
//
// `bits` must be 8 (a 4-bit latent is refused: it is not a cache mode). `R`
// must be a whole number of groups and `group_size` a multiple of `R / 32`,
// which for groups of 64 means a power-of-two rank from 64 to 1024. The four
// window arrays may be strided views with a contiguous last axis. The score,
// softmax and value sum run in f32.
//
// Output: `[B, H, 1, R]` in the queries' dtype.
//
// Throws when there is no port on this backend or an input is outside the
// contract; `mla::decode::fused_absorbed_decode_quantized` checks both first
// and treats a throw as "take the composed path".
// `mla_absorbed_decode_supported` for the quantized partial pass, whose
// pipeline also depends on the rope window's dtype `kpe_dtype`, the scale and
// bias dtype `scales_dtype`, the group size and the bit width.
bool mla_absorbed_decode_quantized_supported(
    mlx::core::Dtype dtype,
    mlx::core::Dtype kpe_dtype,
    mlx::core::Dtype scales_dtype,
    int R,
    int P,
    int group_size,
    int bits);

mlx::core::array mla_absorbed_decode_quantized(
    const mlx::core::array& q_abs,
    const mlx::core::array& q_pe,
    const mlx::core::array& ckv_q,
    const mlx::core::array& scales,
    const mlx::core::array& biases,
    const mlx::core::array& kpe,
    float scale,
    int group_size,
    int bits);

} // namespace mlxcel::turbo
