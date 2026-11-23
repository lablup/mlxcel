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

// Inkling 2 Mini Metal kernels (issue #3359), ported from mlx-vlm
// `mlx_vlm/models/inkling2/kernels.py` and `grouped_experts.py` (MIT,
// Copyright 2026 Neywa Labs Inc, Abdur Rahim; the head preparation kernel is
// adapted there from mlx-dspark's `fused_head_prep`, MIT, Copyright 2026
// erahim3). Each launcher has a Metal port only; on CPU, CUDA and ROCm the
// Rust callers keep the ops path they replace, which every kernel matches
// bitwise.
//
// Used by: `cpp/mlx_cxx_bridge.cpp` (`inkling2_*`), reached from
// `mlxcel_core::inkling2_kernels` and the Inkling 2 model's `kernels` module.

#include <mlx/array.h>

#include <optional>

namespace mlxcel::turbo {

// Whether the three kernels run here: a GPU default device with a Metal port.
bool inkling2_kernels_available();

struct Inkling2ShortConvolution {
    mlx::core::array out;
    mlx::core::array history;
};

// Static depthwise causal convolution of `x [B, L, C]` (bf16 or f16, any
// strides) over the f32 history `[B, R, C]` (`R >= width - 1`, any strides)
// with an f32 `weight [C, width, 1]`:
//
//     out = OT(float(T((conv + x) / sqrt(width + 1))) [+ float(residual)])
//
// `residual`, when given, is `[B, L, C]` of any float dtype. `history` is the
// next history: the newest `min(width - 1 + margin, R + L)` rows of the
// history followed by `x` in f32, which is what the ops path stores.
Inkling2ShortConvolution inkling2_short_convolution(
    const mlx::core::array& history,
    const mlx::core::array& weight,
    const mlx::core::array& x,
    const std::optional<mlx::core::array>& residual,
    mlx::core::Dtype output_dtype,
    int margin);

struct Inkling2AttentionHeads {
    mlx::core::array q;
    mlx::core::array k;
    mlx::core::array v;
    mlx::core::array r;
};

// The four per-head normalizations of an Inkling 2 attention block for
// `q [1, L, H, 128]`, `k` / `v [1, L, KV, 128]` and `r [1, L, H, 16]` (bf16,
// `1 <= L <= 8`), with f32 `q_scale [H, 128]` and `r_scale [H, 16]`: l2
// clipping of q, k and r, and a gamma-free RMSNorm of v. Outputs are bf16.
Inkling2AttentionHeads inkling2_prepare_attention_heads(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    const mlx::core::array& r,
    const mlx::core::array& q_scale,
    const mlx::core::array& r_scale,
    float clip,
    float rel_clip,
    float eps);

// One grouped expert projection over expert-sorted routes. `x [rows, K]` is
// bf16; `w [E, N, K * bits / 32]` uint32 and `scales [E, N, K / group_size]`
// uint8 are a native NVFP4 (bits 4, group 16) or MXFP8 (bits 8, group 32)
// plane; `global_scale`, NVFP4 only, is the MLX-native per-expert `[E]` f32
// scale. `experts`, `source` and `destination` are `[routes]` uint32: route
// `i` uses expert `experts[i]` (non-decreasing), reads row `source[i]` of `x`
// and writes row `destination[i]` of the `[routes, N]` bf16 output.
mlx::core::array inkling2_grouped_expert_qmv(
    const mlx::core::array& x,
    const mlx::core::array& w,
    const mlx::core::array& scales,
    const std::optional<mlx::core::array>& global_scale,
    const mlx::core::array& experts,
    const mlx::core::array& source,
    const mlx::core::array& destination,
    int bits,
    int group_size);

} // namespace mlxcel::turbo
