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

// Fused dequant + SDPA Metal kernel launcher for `KVCacheMode::Turbo4Delegated`
// (follow-up to's K-side unification).
//
// The Turbo4Delegated cache stores V in two regions:
// - cold body — `[B, H_kv, T_cold, D/2]` u8 packed Turbo4 indices plus
//   `[B, H_kv, T_cold, 1]` fp16 per-token kernel rescale (`norm/|y_hat|`).
// - hot ring — `[B, H_kv, T_hot, D]` fp16 plain V tokens (no rotation).
//
// Pre- the read path materialised the full FP16 cold body (via `dequantize_v_turbo4` plus a memo) and then ran a graph-level
// `concatenate(cold_v_dequant, hot_v, axis=2)` plus standard SDPA. The memo
// alone consumed `[B, H_kv, cold_offset, D]` fp16 of working set, undoing the
// 4-bit V compression that defines the mode. retires that memo.
//
// `turbo4_delegated_sdpa_cold_weighted_sum` mirrors the Sparse-V kernel
// `sparse_v_weighted_sum` — runtime-JIT-compiled via
// `mlx::core::fast::metal_kernel`, identical input layout, identical
// "unrotated weighted sum" output contract. The host caller composes:
//
//     out = signs1 · WHT(signs2 · cold_pre) + attn_hot @ hot_v
//
// where `cold_pre` is what this launcher returns. The packed cold V is read
// directly inside the kernel; the dequantised cold V is **never** written to
// global memory. That is the whole point.
//
// The hot-side weighted sum (`attn_hot @ hot_v`) is kept in the host MLX
// graph (single small matmul, T_hot ≤ DELEGATED_HOT_MAX = 1024) — the per-
// step cost is negligible and using MLX's matmul keeps the steel-attention /
// NAX paths active.
//
// MLX C++ symbols touched (re-validate on every MLX commit bump per
// CLAUDE.md): `mlx::core::fast::metal_kernel`, `mlx::core::full`,
// `mlx::core::Shape`, `mlx::core::float32`.
//
// Used by: `cpp/mlx_cxx_bridge.cpp::turbo4_delegated_cold_weighted_sum`,
//          `cpp/mlx_cxx_bridge.cpp::turbo4_delegated_steel_sdpa`.

#include <mlx/array.h>

#include <vector>

namespace mlxcel::turbo {

// Run the fused-skip Metal kernel on the cold V body and return the unrotated
// per-coordinate weighted sum.
//
// Inputs:
// - `attn_weights_cold`: `[B*Hq, Tq, T_cold]` FP32 — pre-flattened
//   post-softmax weights restricted to the cold range. The caller is
//   responsible for slicing the full-cache softmax output to the first
//   `T_cold` columns.
// - `v_packed_cold`:     `[B*Hkv, T_cold, D/2]` UINT8 — nibble-packed Turbo4
//   V indices for the cold body.
// - `v_rescale_cold`:    `[B*Hkv, T_cold]` FP16 — precomputed per-token
//   rescale `norm[t] / max(|y_hat[t]|, 1e-10)`.
// - `codebook`:          `[16]` FP32 — Lloyd-Max centroids (4-bit).
//
// Template parameters (passed to the kernel as `int` template args):
// - `Dim`: head dimension `D` (must equal `2 * v_packed.shape[-1]`).
// - `RepeatCount`: `Tq` — number of Q tokens the threadgroup handles.
// - `NRep`: `Hq / Hkv` — Q-head replication factor for grouped attention.
//
// Scalar parameters:
// - `threshold`: alive cutoff. Any `attn_weight <= threshold` is skipped.
//   `0.0` disables skipping. The Turbo4Delegated decode path on the host
//   side currently passes whatever `MLXCEL_SPARSE_V_THRESHOLD` resolves to
//   (default `1e-6` if unset, matching the Turbo4Asym sparse-V kernel) —
//   the threshold is shared between the Turbo4Delegated cold-V kernel and
//   the Turbo4Asym sparse-V kernel via the same env var. Set the env var
//   to `0` (or `0.0`) to disable per-token skipping on this path while
//   still exercising the fused dequant.
//
// Output:
// - `out_cold_pre`: `[B*Hq, Tq, D]` FP32 — unrotated weighted sum of the
//   cold V tokens. The host caller applies the `signs1 · WHT · signs2`
//   inverse rotation to produce the rotated cold contribution.
mlx::core::array turbo4_delegated_cold_weighted_sum(
    const mlx::core::array& attn_weights_cold,  // [B*Hq, Tq, T_cold] f32
    const mlx::core::array& v_packed_cold,      // [B*Hkv, T_cold, D/2] u8
    const mlx::core::array& v_rescale_cold,     // [B*Hkv, T_cold] f16
    const mlx::core::array& codebook,           // [16] f32
    int dim,
    int n_rep,
    float threshold);

// Bulk dequantize packed Turbo4 V into rotated codec space.
//
// This is the fused equivalent of the Rust graph helper
// `dequantize_v_turbo4_rotated`: one Metal dispatch unpacks nibbles, gathers
// Lloyd-Max centroids, multiplies by the precomputed per-token rescale, and
// writes `[B, H, T, D]` FP16. It intentionally does not apply inverse WHT /
// signs; callers that use native SDPA over rotated V inverse-rotate only the
// small attention output.
mlx::core::array turbo4_delegated_bulk_dequant_rotated(
    const mlx::core::array& v_packed,   // [B, H, T, D/2] u8
    const mlx::core::array& v_rescale,  // [B, H, T, 1] f16
    const mlx::core::array& codebook,   // [16] f32
    int dim);

// Run the steel-attention-envelope fused SDPA kernel for Turbo4Delegated
// One Metal dispatch performs the entire post-Q·K SDPA inline:
// numerically stable softmax over the full (cold + hot) score range, cold-V
// dequant + weighted-sum from packed 4-bit storage, hot-V FP16 weighted-sum,
// and per-Q normalisation. The kernel returns two FP32 buffers; the host
// applies the linear `signs1 · WHT(signs2 · ·)` inverse Turbo4 rotation to the
// cold contribution and adds the hot contribution before casting to FP16.
//
// Why the rotation stays on the host (matches `sparse_v_weighted_sum`):
// the rotation is linear, so applying it to the small `[B, Hq, Tq, D]` output
// is mathematically identical to applying it per V-token, but runs once per
// decode step instead of `T_cold` times. Doing the WHT inside the kernel
// would require `log2(Dim)` extra threadgroup barriers per (B*Hq) per call —
// that defeats the point of pulling cold + hot into one dispatch.
//
// Inputs:
// - `scores`:        `[B*Hq, Tq, T_total]` FP32 — `Q·K^T * scale` with the
//   optional additive mask already added (causal positions carry `-inf` so
//   they contribute zero post-softmax). The host pre-computes this via the
//   standard MLX matmul so steel attention / NAX accelerated matmul stays
//   on the Q·K side; the kernel handles only the post-Q·K work.
// - `cold_packed`:   `[B*Hkv, T_cold, D/2]` UINT8 — nibble-packed cold V
//   indices. May be empty (`T_cold == 0`) pre-fold; in that case pass a
//   zero-shaped placeholder.
// - `cold_rescale`:  `[B*Hkv, T_cold]` FP16 — precomputed per-token cold-V
//   rescale `norm[t] / max(|y_hat[t]|, 1e-10)`.
// - `hot_v`:         `[B*Hkv, T_hot, D]` FP16 — plain FP16 hot V tokens. May
//   be empty (`T_hot == 0`) immediately after a fold.
// - `codebook`:      `[16]` FP32 — Lloyd-Max centroids.
// - `threshold`:     `[1]` FP32 — alive cutoff (compared against the
//   normalised post-softmax weight `exp(score - max) / sum`). `0.0` disables
//   skipping. Hot tokens are never skipped.
//
// Template parameters:
// - `Dim`: head dimension `D` (must be a power of 2 and equal to
//   `2 * cold_packed.shape[-1]` when `T_cold > 0`).
// - `RepeatCount`: `Tq` — Q tokens per threadgroup.
// - `NRep`: `Hq / Hkv`.
//
// Outputs (returned as a 2-element vector, in this order):
// - [0] `out_cold_pre`: `[B*Hq, Tq, D]` FP32 — unrotated cold weighted sum,
//   already divided by the softmax denominator. The host applies
//   `signs1 · WHT(signs2 · ·)` to produce the rotated cold contribution.
// - [1] `out_hot`:      `[B*Hq, Tq, D]` FP32 — hot weighted sum, already
//   divided by the same softmax denominator.
//
// The host sums `WHT_rotate(out_cold_pre) + out_hot` and casts to FP16. Both
// outputs share the same softmax denominator (single-pass numerical stability
// derived from a per-Q max + sum scan over the full T_total range) so the
// addition is bit-equivalent to a dense `softmax(scores) @ V_full`.
std::vector<mlx::core::array> turbo4_delegated_steel_sdpa(
    const mlx::core::array& scores,         // [B*Hq, Tq, T_total] f32
    const mlx::core::array& cold_packed,    // [B*Hkv, T_cold, D/2] u8
    const mlx::core::array& cold_rescale,   // [B*Hkv, T_cold]      f16
    const mlx::core::array& hot_v,          // [B*Hkv, T_hot, D]    f16
    const mlx::core::array& codebook,       // [16]                 f32
    int dim,
    int n_rep,
    int cold_offset,
    int hot_offset,
    float threshold);

} // namespace mlxcel::turbo
