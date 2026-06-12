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

// Fused Sparse-V SDPA kernel launcher for Turbo4Asym KV cache.
//
// This header exposes a single C++ entry point that issues the runtime-JIT
// Metal kernel via `mlx::core::fast::metal_kernel`. The kernel performs the
// "weighted-sum-then-rotate" half of attention with per-thread skipping when
// the post-softmax attention weight falls below a threshold:
//
//     out_pre[b, h, k] = Σ_t attn_weights[b, h, t]
//                          * y_hat[t, k] * v_rescale[t]
//             where v_rescale[t] = norm[t] / max(|y_hat[t]|, 1e-10)
//             (precomputed at quantize time) and the inner
//             loop short-circuits when attn_weights[b, h, t] < threshold.
//
// The caller (`mlxcel-core/cpp/mlx_cxx_bridge.cpp`) computes the FP32
// attention scores → softmax pre-stage and then applies the inverse rotation
// `D2·H·D1` *outside* the kernel on the smaller `[B, Hq, Tq, D]` output, so
// the rotation cost is independent of T (the cache length). This matches the
// "rotate after accumulation" pattern used in
// https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/turboquant.py
// (_single_tile_value_weighted_sum_kernel).
//
// Used by: `cpp/mlx_cxx_bridge.cpp` (`turbo_sparse_v_weighted_sum`).

#include <mlx/array.h>

namespace mlxcel::turbo {

// Run the fused-skip Metal kernel and return the unrotated weighted sum.
//
// Inputs:
// - `attn_weights`: `[B*Hq, Tq, Tk]` FP32 — pre-flattened post-softmax weights.
// - `v_packed`:     `[B*Hkv, Tk, D/2]` UINT8 — nibble-packed Turbo4 V indices.
// - `v_rescale`:    `[B*Hkv, Tk]` FP16 — precomputed per-token rescale factor
//   `norm[t] / max(|y_hat[t]|, 1e-10)`. promoted this from an
//   in-kernel threadgroup tree reduction to a one-time host-side precompute
//   at quantize time; eliminates `log2(Dim) + 2` per-token threadgroup
//   barriers and the kernel `tg_y_hat[Dim]` shared scratch.
// - `codebook`:     `[16]` FP32 — Lloyd-Max centroids (4-bit).
//
// Template parameters (passed to the kernel as `int` template args):
// - `Dim`: head dimension `D` (must equal `2 * v_packed.shape[-1]`).
// - `RepeatCount`: `Tq` — number of Q tokens the threadgroup handles.
// - `NRep`: `Hq / Hkv` — Q-head replication factor for grouped attention.
//   Set `1` for non-grouped attention.
//
// Scalar parameters (passed via `ScalarArg`):
// - `threshold`: alive cutoff. Any `attn_weight <= threshold` is skipped.
//
// Output:
// - `out_pre`: `[B*Hq, Tq, D]` FP32 — unrotated weighted sum. The caller
//   must apply the `D2·H·D1` inverse rotation to produce the final FP16
//   attention output.
mlx::core::array sparse_v_weighted_sum(
    const mlx::core::array& attn_weights, // [B*Hq, Tq, Tk] f32
    const mlx::core::array& v_packed,      // [B*Hkv, Tk, D/2] u8
    const mlx::core::array& v_rescale,     // [B*Hkv, Tk] f16
    const mlx::core::array& codebook,      // [16] f32
    int dim,
    int n_rep,
    float threshold);

} // namespace mlxcel::turbo
