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
// Used by: `cpp/mlx_cxx_bridge.cpp` (`mla_absorbed_decode_fused`), the
// single-query branch of `mlxcel_core::mla::decode::absorbed_attention`.

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

} // namespace mlxcel::turbo
