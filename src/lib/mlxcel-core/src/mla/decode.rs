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

//! Stage 1 absorbed MLA decode, composed from MLX ops (issue #907).
//!
//! The absorbed score is `scale * (q_absorbed . ckv + q_pe . kpe)`. Only the
//! first term has the shape of a normal attention score against the cached
//! tensor; the second is a second contraction against a differently shaped
//! cache stream. Rather than hand-rolling the softmax, this splits the score
//! the way `src/models/deepseek_v3.rs` already does:
//!
//! ```text
//!   out = SDPA(q_absorbed, ckv, ckv, scale, mask = scale * q_pe @ kpe^T + causal)
//! ```
//!
//! SDPA computes `softmax(scale * q @ k^T + mask) @ v`, and the rope term is
//! additive in exactly that position, so the identity is exact and the fused
//! MLX attention kernel does the softmax and the value accumulation. The cost
//! is materializing the `[B, H, L, S]` rope score, which at decode is one row
//! per head. That is `H * S` floats against the `S * kv_lora_rank` the kernel
//! reads from the cache, i.e. `H / kv_lora_rank` of the KV traffic, so for
//! every shipping geometry (16 or 128 heads against rank 512) the extra term is
//! a fraction of the read the absorption just saved.
//!
//! `ckv` is passed as both K and V. That is not a shortcut: under absorption
//! the key and the value genuinely are the same latent, which is what makes the
//! cache one tensor instead of two.

use cxx::UniquePtr;

use crate::ffi::{self, MlxArray};
use crate::layers::attention;
use crate::mla::absorb::MlaAbsorbedProjections;
use crate::mla::stats::{self, MlaDecodePath};

/// Fold `W_UK` into the query: `[B, H, L, qk_nope] -> [B, H, L, kv_lora_rank]`.
///
/// The right operand is `[1, H, qk_nope, r]`, so MLX broadcasts the leading
/// batch axis and contracts per head in one batched matmul.
#[must_use]
pub fn absorb_queries(q_nope: &MlxArray, proj: &MlaAbsorbedProjections) -> UniquePtr<MlxArray> {
    ffi::matmul(q_nope, proj.w_uk())
}

/// Fold `W_UV` after the attention: `[B, H, L, kv_lora_rank] -> [B, H, L, v]`.
#[must_use]
pub fn unabsorb_output(o_latent: &MlxArray, proj: &MlaAbsorbedProjections) -> UniquePtr<MlxArray> {
    ffi::matmul(o_latent, proj.w_uv())
}

/// The rope half of the score, shaped as an additive SDPA mask.
///
/// `q_pe` is `[B, H, L, P]` and `kpe` is `[B, 1, S, P]`; the result is
/// `[B, H, L, S]`, already multiplied by `scale` because SDPA applies its own
/// `scale` only to the `q @ k^T` term. `extra` (a causal or padding mask) is
/// added on top when present, so the caller hands SDPA one mask rather than
/// two.
#[must_use]
pub fn rope_score_mask(
    q_pe: &MlxArray,
    kpe: &MlxArray,
    scale: f32,
    extra: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    // Scale through a same-dtype scalar array rather than a host-side f32
    // multiply so an f16 query stays f16 and the mask matches SDPA's dtype
    // expectation, mirroring `deepseek_v3.rs`.
    let scale_scalar = ffi::full_f32(&[1], scale, ffi::array_dtype(q_pe));
    let q_pe_scaled = ffi::multiply(q_pe, &scale_scalar);
    let kpe_t = ffi::transpose_axes(kpe, &[0, 1, 3, 2]);
    let scores = ffi::matmul(&q_pe_scaled, &kpe_t);
    match extra {
        Some(m) => ffi::add(&scores, m),
        None => scores,
    }
}

/// One absorbed MLA attention call over the latent cache.
///
/// * `q_nope` `[B, H, L, qk_nope_head_dim]` (not yet absorbed)
/// * `q_pe`   `[B, H, L, qk_rope_head_dim]` (already rotated)
/// * `ckv`    `[B, 1, S, kv_lora_rank]` (the whole live latent window)
/// * `kpe`    `[B, 1, S, qk_rope_head_dim]` (already rotated)
/// * `mask`   additive `[.., L, S]` causal or padding mask, or `None`
///
/// Returns `[B, H, L, v_head_dim]`, the per-head attention output in the
/// original value space, ready for the family's `o_proj`.
///
/// Records [`MlaDecodePath::AbsorbedComposed`] for a single-token step and
/// [`MlaDecodePath::AbsorbedPrefill`] otherwise, so a benchmark arm can prove
/// which path it ran instead of inferring it from the flag it was given.
#[must_use]
pub fn absorbed_decode(
    q_nope: &MlxArray,
    q_pe: &MlxArray,
    ckv: &MlxArray,
    kpe: &MlxArray,
    proj: &MlaAbsorbedProjections,
    scale: f32,
    mask: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    let q_len = ffi::array_shape(q_nope).get(2).copied().unwrap_or(0);
    stats::record(if q_len == 1 {
        MlaDecodePath::AbsorbedComposed
    } else {
        MlaDecodePath::AbsorbedPrefill
    });

    let q_absorbed = absorb_queries(q_nope, proj);
    let rope_mask = rope_score_mask(q_pe, kpe, scale, mask);
    let o_latent = attention(&q_absorbed, ckv, ckv, scale, Some(&rope_mask), 0.0, 0);
    unabsorb_output(&o_latent, proj)
}

/// Up-project the latent back into per-head `K_nope` and `V`.
///
/// The inverse direction of the fold, kept here because it is the reference the
/// parity tests compare against and the fallback a prefill path can use when it
/// does not want to keep the original `kv_b_proj` loaded. Returns
/// `(k_nope [B, H, S, qk_nope], v [B, H, S, v_head_dim])` from
/// `ckv [B, 1, S, kv_lora_rank]`.
///
/// The `W_UK` operand is transposed at call time. That is fine for prefill and
/// for tests, and is precisely why the decode path stores its own
/// already-transposed copy instead of doing this per step.
#[must_use]
pub fn expand_latent(
    ckv: &MlxArray,
    proj: &MlaAbsorbedProjections,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let w_uk_t = ffi::swap_axes(proj.w_uk(), -1, -2);
    let k_nope = ffi::matmul(ckv, &w_uk_t);
    let v = ffi::matmul(ckv, proj.w_uv());
    (k_nope, v)
}

#[cfg(test)]
#[path = "decode_tests.rs"]
mod decode_tests;
