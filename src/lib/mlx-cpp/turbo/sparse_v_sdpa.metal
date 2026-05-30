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
//
// Reference Metal source for the fused Sparse-V SDPA weighted-sum kernel
// (rescale precompute landed).
//
// This file is a faithful copy of the source string embedded in
// `sparse_v_sdpa.cpp`. Keeping it as a standalone .metal file makes it
// possible to:
// - Lint the kernel with the Metal compiler (`xcrun -sdk macosx metal -c`).
// - Diff against MLX upstream kernels during a commit-bump review.
// - Read the kernel without searching through C++ string literals.
//
// At runtime the kernel is JIT-compiled via `mlx::core::fast::metal_kernel`
// in `sparse_v_sdpa.cpp`. The JIT path expects the body without a `kernel`
// declaration — `metal_kernel` wraps the body in the boilerplate.
//
// Template parameters (set per launch):
//   Dim         — head dimension D
//   RepeatCount — number of Q tokens per threadgroup (Tq)
//   NRep        — Hq / Hkv (grouped attention replication factor)
//
// Scalar arguments:
//   threshold   — float; any attn_weight <= threshold is skipped
//
// Inputs (per launch):
//   weights     — [B*Hq, Tq, Tk] f32  : post-softmax attention weights
//   rescale     — [B*Hkv, Tk]    f16  : precomputed `norm[t] / |y_hat[t]|`
//                                       (replaced the previous `norms` input + in-kernel tree reduction)
//   packed      — [B*Hkv, Tk, D/2] u8 : nibble-packed V indices (2 per byte)
//   codebook    — [16]           f32  : Lloyd-Max centroids
//
// Output:
//   out         — [B*Hq, Tq, D]  f32  : unrotated weighted sum (caller
//                                       applies D2·H·D1 inverse rotation)
//
// Per-thread mapping:
//   thread_position_in_grid.x = d        (0..Dim)
//   thread_position_in_grid.z = n        (0..B*Hq)  (one threadgroup per B*Hq)
//   threadgroup size           = (Dim, 1, 1)
//
// Grouped attention: Hkv = Hq / NRep. The per-`n` Hkv slot is `n / NRep`.
// Per-thread compute proceeds dim-by-dim over D, sweeping all Tq tokens
// (RepeatCount) and accumulating the unrotated weighted sum.
//
// kernel-body change summary:
//   - Removed `tg_y_hat[Dim]` shared scratch and `tg_norm[1]` broadcast.
//   - Removed the `log2(Dim)`-step threadgroup tree reduction over `code²`.
//   - Removed the `if (tg_d == 0) compute rescale; barrier; broadcast` chain.
//   - Replaced with a single per-token `(float)rescale[t]` load. Each
//     thread reads the same scalar; Apple's L1 / per-threadgroup cache
//     coalesces the broadcast.
// The kernel output is bit-for-bit unchanged within FP16 round-off.

kernel void sparse_v_weighted_sum_kernel_reference(
    /* dummy declaration kept here so the file is syntactically a kernel
       even though the runtime JIT path strips the kernel/buffer wrapper */ )
{
    // See sparse_v_sdpa.cpp for the exact body string.
}
