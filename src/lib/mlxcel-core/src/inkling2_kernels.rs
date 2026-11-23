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

//! Safe entry points for the Inkling 2 Mini Metal kernels (issue #3359).
//!
//! Ported from mlx-vlm `mlx_vlm/models/inkling2/kernels.py` and
//! `grouped_experts.py` (MIT); the Metal bodies and their contracts are in
//! `src/lib/mlxcelverse/kernels/inkling2_kernels.{h,cpp}` and
//! `inkling2_grouped_experts.cpp`. Every function returns `None` when the
//! launcher refuses (no Metal port, a shape or dtype outside its contract), so
//! the caller keeps the ops path it would otherwise replace. The shape gates a
//! model applies on top (batch, block length, verify-only) live with the model.

use crate::{MlxArray, UniquePtr, ffi};

/// Whether the kernels run on this backend: a GPU default device with a
/// Metal port. False on CPU, CUDA and ROCm.
///
/// Used by: the Inkling 2 model's `kernels` gates.
pub fn available() -> bool {
    ffi::inkling2_kernels_available()
}

/// Fused static short convolution: `(out, next_history)`.
///
/// `history` is the f32 `[B, R, C]` convolution history (its last
/// `width - 1` rows are the context), `weight` the f32 `[C, width, 1]` taps,
/// `x` the bf16 or f16 `[B, L, C]` input. `out` is
/// `OT(float(T((conv + x) / sqrt(width + 1))) [+ float(residual)])` with `OT =
/// output_dtype`, and `next_history` the newest
/// `min(width - 1 + margin, R + L)` rows of the history followed by `x`, in
/// f32. Strided `history`, `x` and `residual` are read in place.
///
/// Used by: Inkling 2 `conv::ShortConv::forward`.
pub fn short_convolution(
    history: &MlxArray,
    weight: &MlxArray,
    x: &MlxArray,
    residual: Option<&MlxArray>,
    output_dtype: i32,
    margin: i32,
) -> Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
    let mut out = UniquePtr::null();
    let mut next = UniquePtr::null();
    let residual_ptr: *const MlxArray = residual.map_or(std::ptr::null(), |r| r as *const _);
    // SAFETY: `residual_ptr` is null or borrows `residual`, which outlives
    // the call; the launcher only reads it.
    unsafe {
        ffi::inkling2_short_convolution(
            history,
            weight,
            x,
            residual_ptr,
            output_dtype,
            margin,
            &mut out,
            &mut next,
        )
    }
    .ok()?;
    Some((out, next))
}

/// The four bf16 `(q, k, v, r)` heads of an Inkling 2 attention block after
/// l2 clipping of q (scaled by `q_scale`), k and r (scaled by `r_scale`,
/// threshold `rel_clip`), and a gamma-free RMSNorm of v (`eps`).
///
/// `q [1, L, H, 128]`, `k` / `v [1, L, KV, 128]`, `r [1, L, H, 16]` bf16
/// with `1 <= L <= 8`; f32 `q_scale [H, 128]` and `r_scale [H, 16]`.
///
/// Used by: Inkling 2 `attention::Inkling2Attention::forward`.
#[allow(clippy::too_many_arguments)]
pub fn prepare_attention_heads(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    r: &MlxArray,
    q_scale: &MlxArray,
    r_scale: &MlxArray,
    clip: f32,
    rel_clip: f32,
    eps: f32,
) -> Option<[UniquePtr<MlxArray>; 4]> {
    let mut out = [
        UniquePtr::null(),
        UniquePtr::null(),
        UniquePtr::null(),
        UniquePtr::null(),
    ];
    let [qo, ko, vo, ro] = &mut out;
    ffi::inkling2_prepare_attention_heads(
        q, k, v, r, q_scale, r_scale, clip, rel_clip, eps, qo, ko, vo, ro,
    )
    .ok()?;
    Some(out)
}

/// One grouped expert projection over expert-sorted routes: `[routes, N]`
/// bf16, row `destination[i]` holding route `i`'s product of row
/// `source[i]` of `x [rows, K]` (bf16) with expert `experts[i]`.
///
/// `w` / `scales` are a native NVFP4 (bits 4, group 16) or MXFP8 (bits 8,
/// group 32) plane, `global_scale` the MLX-native NVFP4 per-expert `[E]` f32
/// scale. `experts` (non-decreasing), `source` and `destination` are
/// `[routes]` uint32. Each output equals the `gather_qmm` row of that route.
///
/// Used by: Inkling 2 `moe::grouped_expert_projections`.
#[allow(clippy::too_many_arguments)]
pub fn grouped_expert_qmv(
    x: &MlxArray,
    w: &MlxArray,
    scales: &MlxArray,
    global_scale: Option<&MlxArray>,
    experts: &MlxArray,
    source: &MlxArray,
    destination: &MlxArray,
    bits: i32,
    group_size: i32,
) -> Option<UniquePtr<MlxArray>> {
    let global_ptr: *const MlxArray = global_scale.map_or(std::ptr::null(), |g| g as *const _);
    // SAFETY: `global_ptr` is null or borrows `global_scale`, which outlives
    // the call; the launcher only reads it.
    unsafe {
        ffi::inkling2_grouped_expert_qmv(
            x,
            w,
            scales,
            global_ptr,
            experts,
            source,
            destination,
            bits,
            group_size,
        )
    }
    .ok()
}
