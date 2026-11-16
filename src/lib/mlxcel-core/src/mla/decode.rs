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

//! Stage 1 absorbed MLA decode (issue #907).
//!
//! An unmasked single-query step runs on the fused Metal kernel
//! ([`fused_absorbed_decode`], issue #3121) wherever it can, which evaluates
//! both score halves, the softmax and the latent-value sum in one partial pass
//! plus a merge. Everything else, and every backend without the kernel, takes
//! the composed form below.
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
use crate::mla::prefill::{FoldedMask, rope_folded_attention};
use crate::mla::stats::{self, MlaDecodePath};

/// Fold `W_UK` into the query: `[B, H, L, qk_nope] -> [B, H, L, kv_lora_rank]`.
///
/// One batched contraction per head: a dense `matmul` against `[1, H, qk_nope,
/// r]`, or a `quantized_matmul` against the packed `kv_b_proj` rows when the
/// checkpoint is quantized (issue #3121).
#[must_use]
pub fn absorb_queries(q_nope: &MlxArray, proj: &MlaAbsorbedProjections) -> UniquePtr<MlxArray> {
    proj.absorb(q_nope)
}

/// Fold `W_UV` after the attention: `[B, H, L, kv_lora_rank] -> [B, H, L, v]`.
#[must_use]
pub fn unabsorb_output(o_latent: &MlxArray, proj: &MlaAbsorbedProjections) -> UniquePtr<MlxArray> {
    proj.unabsorb(o_latent)
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

/// Longest latent window the fused decode kernel takes (issue #3121).
///
/// The kernel gives each head its own threadgroups, so every head reads the
/// whole window, where the composed path's batched matmul reads it once for
/// all heads. On an M5 Max at DeepSeek-V2-Lite's geometry the fused kernel
/// measured faster at every window up to 8192 keys and slower from 16384, so
/// longer windows keep the composed path.
pub const FUSED_DECODE_MAX_KEYS: i32 = 8192;

/// Unmasked single-query absorbed attention on the fused kernel, when it
/// applies to this call.
///
/// * `q_absorbed` `[B, H, 1, kv_lora_rank]`, `q_pe` `[B, H, 1, qk_rope_head_dim]`
/// * `ckv` `[B, 1, S, kv_lora_rank]`, `kpe` `[B, 1, S, qk_rope_head_dim]`
///
/// Returns `[B, H, 1, kv_lora_rank]` and records
/// [`MlaDecodePath::AbsorbedFused`]. `None` sends the caller to its composed
/// path: the kernel is unavailable or turned off
/// ([`crate::mla::fused_decode_enabled`]), the window is longer than
/// [`FUSED_DECODE_MAX_KEYS`], or a shape or dtype is outside what the kernel
/// takes. The kernel's own shape check backs these up, and a refusal there
/// also falls back rather than aborting the step.
///
/// Used by: [`absorbed_attention`] (DeepSeek V2), DeepSeek V3's single-token
/// decode, and the shared per-row latent decode in `models::mla_batched`
/// (DeepSeek V3, V3.2, GLM-4.7-Flash, Kimi Linear, Kimi K3).
#[must_use]
pub fn fused_absorbed_decode(
    q_absorbed: &MlxArray,
    q_pe: &MlxArray,
    ckv: &MlxArray,
    kpe: &MlxArray,
    scale: f32,
) -> Option<UniquePtr<MlxArray>> {
    if !crate::mla::fused_decode_enabled() {
        return None;
    }
    let q = ffi::array_shape(q_absorbed);
    if q.len() != 4 || q[2] != 1 {
        return None;
    }
    let p = ffi::array_shape(q_pe);
    let k = ffi::array_shape(ckv);
    let (rank, rope) = (*q.get(3)?, *p.get(3)?);
    let keys = *k.get(2)?;
    let dtype = ffi::array_dtype(q_absorbed);
    let float_dtype = [
        crate::dtype::FLOAT16,
        crate::dtype::BFLOAT16,
        crate::dtype::FLOAT32,
    ]
    .contains(&dtype);
    let same_dtype = [q_pe, ckv, kpe]
        .iter()
        .all(|a| ffi::array_dtype(a) == dtype);
    if !float_dtype
        || !same_dtype
        || rank % 32 != 0
        || rope % 32 != 0
        || rank > 1024
        || keys < 1
        || keys > FUSED_DECODE_MAX_KEYS
    {
        return None;
    }
    let out = ffi::mla_absorbed_decode_fused(q_absorbed, q_pe, ckv, kpe, scale).ok()?;
    stats::record(MlaDecodePath::AbsorbedFused);
    Some(out)
}

/// Attend over the latent cache after the query-side folded projection.
///
/// This is the row-local part of absorbed MLA decode. Callers that own a
/// ragged request batch can apply [`absorb_queries`] once to the compatible
/// query rows, invoke this function for each cache row, concatenate the latent
/// outputs, and apply [`unabsorb_output`] once to that batch.
///
/// * `q_absorbed` `[B, H, L, kv_lora_rank]`
/// * `q_pe`       `[B, H, L, qk_rope_head_dim]`
/// * `ckv`        `[B, 1, S, kv_lora_rank]`
/// * `kpe`        `[B, 1, S, qk_rope_head_dim]`
///
/// Returns `[B, H, L, kv_lora_rank]` in latent value space.
///
/// Used by: [`absorbed_decode`],
/// [`crate::mla::cache::MlaLatentCache::attend_preprojected`].
#[must_use]
pub(crate) fn absorbed_attention(
    q_absorbed: &MlxArray,
    q_pe: &MlxArray,
    ckv: &MlxArray,
    kpe: &MlxArray,
    scale: f32,
    mask: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    let q_len = ffi::array_shape(q_absorbed).get(2).copied().unwrap_or(0);
    if q_len == 1
        && mask.is_none()
        && let Some(out) = fused_absorbed_decode(q_absorbed, q_pe, ckv, kpe, scale)
    {
        return out;
    }
    stats::record(if q_len == 1 {
        MlaDecodePath::AbsorbedComposed
    } else {
        MlaDecodePath::AbsorbedPrefill
    });

    if q_len > 1 {
        // A multi-token step folds the rope half into the SDPA head dim
        // instead of building the dense `[B, H, L, S]` rope-score mask, which
        // grows with the query count times the whole cache (issue #3071). A
        // single query keeps the mask form: its score row is one `[B, H, 1, S]`
        // slice, and decode output stays what it was.
        let mask = mask.map_or(FoldedMask::None, FoldedMask::Array);
        return rope_folded_attention(q_absorbed, q_pe, ckv, kpe, ckv, scale, mask);
    }
    let rope_mask = rope_score_mask(q_pe, kpe, scale, mask);
    attention(q_absorbed, ckv, ckv, scale, Some(&rope_mask), 0.0, 0)
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
/// Records [`MlaDecodePath::AbsorbedFused`] or
/// [`MlaDecodePath::AbsorbedComposed`] for a single-token step and
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
    let q_absorbed = absorb_queries(q_nope, proj);
    let o_latent = absorbed_attention(&q_absorbed, q_pe, ckv, kpe, scale, mask);
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
/// A dense fold transposes its `W_UK` operand at call time. That is fine for
/// prefill and for tests, and is precisely why the decode path stores its own
/// already-transposed copy instead of doing this per step.
#[must_use]
pub fn expand_latent(
    ckv: &MlxArray,
    proj: &MlaAbsorbedProjections,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    proj.expand(ckv)
}

#[cfg(test)]
#[path = "decode_tests.rs"]
mod decode_tests;
