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

//! Absorbed MLA decode over an affine-quantized latent (issue #3155).
//!
//! The latent `ckv` is both key and value under absorption, so one stored
//! tensor serves both contractions of a decode step. Stored as the MLX affine
//! triple, both run through `quantized_matmul` on the packed rows:
//!
//! ```text
//!   scores   = scale * quantized_matmul(q_absorbed, ckv_q, transpose = true)
//!            + scale * q_pe @ kpe^T            (+ mask)
//!   p        = softmax(scores)                 (f32)
//!   o_latent = quantized_matmul(p, ckv_q, transpose = false)
//! ```
//!
//! On Metal an unmasked step runs the same arithmetic in the fused decode
//! kernel instead, which reads the packed codes directly (issue #3156); the
//! composed form above is every other backend's path and the kernel's
//! fallback. The window is never dequantized on either path. A multi-token step and a
//! cold prefill dequantize it once ([`LatentWindow::into_dense`]) and run the
//! existing dense code, which therefore reads exactly the values a decode step
//! reads.
//!
//! Everything here is built from `quantize_weights`, `quantized_matmul` and
//! `dequantize`, the primitives the quantized weight path already runs on
//! Metal, CUDA and ROCm, so no backend needs a kernel of its own.

use cxx::UniquePtr;

use crate::cache::KVCacheMode;
use crate::cache::latent_quant::{LATENT_QUANT_GROUP_SIZE, QuantizedLatentRows};
use crate::dtype;
use crate::ffi::{self, MlxArray};
use crate::mla::MlaGeometry;
use crate::mla::decode::fused_absorbed_decode_quantized;
use crate::mla::stats::{self, MlaDecodePath};

/// MLX quantization mode of the stored latent.
const AFFINE: &str = "affine";

/// Bytes one token of an affine-quantized latent cache costs, per layer.
///
/// `rank * bits / 8` for the packed latent, an FP16 scale and an FP16 bias per
/// group, and the FP16 rope stream. 672 at 8 bits and 416 at 4 bits on
/// DeepSeek-V2-Lite (rank 512, rope 64, group 64).
///
/// Used by: `execution::kv_arch` (sizing, advisor, inspect) and the tests that
/// pin the runtime `nbytes` against it.
#[must_use]
pub const fn quantized_latent_bytes_per_token(
    kv_lora_rank: usize,
    qk_rope_head_dim: usize,
    bits: usize,
    group_size: usize,
) -> usize {
    // Saturating: `execution::kv_arch` feeds this from an untrusted
    // `config.json`, where a wrapped product would undercount the cache (the
    // unsafe direction for every budget built on it). A zero group size has no
    // meaningful rate and saturates for the same reason.
    if group_size == 0 {
        return usize::MAX;
    }
    let packed = match kv_lora_rank.checked_mul(bits) {
        Some(bit_count) => bit_count / 8,
        None => return usize::MAX,
    };
    packed
        .saturating_add((kv_lora_rank / group_size).saturating_mul(4))
        .saturating_add(qk_rope_head_dim.saturating_mul(2))
}

/// Whether a latent of this geometry can be stored at `mode`'s bit width.
///
/// The affine codec groups the latent along `kv_lora_rank`, so the rank has to
/// be a whole number of groups. Returns the operator-facing reason otherwise.
pub(crate) fn check_quantizable(geometry: &MlaGeometry, mode: KVCacheMode) -> Result<(), String> {
    if mode.latent_affine_bits().is_none() {
        return Ok(());
    }
    let group = LATENT_QUANT_GROUP_SIZE as usize;
    if geometry.kv_lora_rank == 0 || !geometry.kv_lora_rank.is_multiple_of(group) {
        return Err(format!(
            "mla: a {mode} latent cache groups the latent in {group}s and kv_lora_rank {} is not \
             a multiple of that",
            geometry.kv_lora_rank
        ));
    }
    Ok(())
}

/// The live window an MLA latent cache hands back after an update.
///
/// The contract the quantized-latent follow-ups build on (#3156 fused kernel,
/// #3157 Turbo latent, #3160-#3163 latent-pair families): a consumer matches on
/// the variant and reads the codec parameters from the window, never from the
/// model family.
pub enum LatentWindow {
    /// FP16 latent, as stored.
    Dense {
        /// `[B, 1, S, kv_lora_rank]`.
        ckv: UniquePtr<MlxArray>,
        /// `[B, 1, S, qk_rope_head_dim]`.
        kpe: UniquePtr<MlxArray>,
    },
    /// Affine-quantized latent with an FP16 rope stream.
    Quantized {
        /// Packed latent, `[B, 1, S, kv_lora_rank * bits / 32]` uint32.
        ckv_q: UniquePtr<MlxArray>,
        /// `[B, 1, S, kv_lora_rank / group_size]`.
        scales: UniquePtr<MlxArray>,
        /// `[B, 1, S, kv_lora_rank / group_size]`.
        biases: UniquePtr<MlxArray>,
        /// `[B, 1, S, qk_rope_head_dim]`.
        kpe: UniquePtr<MlxArray>,
        /// Bits per latent element.
        bits: i32,
        /// Elements per affine group.
        group_size: i32,
    },
}

impl LatentWindow {
    /// Whether the latent is stored quantized.
    #[must_use]
    pub const fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized { .. })
    }

    /// Live token count of the window.
    #[must_use]
    pub fn seq_len(&self) -> i32 {
        let kpe = match self {
            Self::Dense { kpe, .. } | Self::Quantized { kpe, .. } => kpe,
        };
        ffi::array_shape(kpe).get(2).copied().unwrap_or(0)
    }

    /// The window as dense `(ckv, kpe)`, dequantizing once when it is stored
    /// quantized.
    ///
    /// Used by: multi-token absorbed steps and the cold-prefill up-projection
    /// in `models::deepseek_v2`.
    #[must_use]
    pub fn into_dense(self) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        match self {
            Self::Dense { ckv, kpe } => (ckv, kpe),
            Self::Quantized {
                ckv_q,
                scales,
                biases,
                kpe,
                bits,
                group_size,
            } => {
                let ckv = dequantize_latent(&ckv_q, &scales, &biases, bits, group_size);
                (ffi::astype(&ckv, ffi::array_dtype(&kpe)), kpe)
            }
        }
    }

    pub(crate) fn from_rows(rows: QuantizedLatentRows, bits: i32) -> Self {
        Self::Quantized {
            ckv_q: rows.packed,
            scales: rows.scales,
            biases: rows.biases,
            kpe: rows.kpe,
            bits,
            group_size: LATENT_QUANT_GROUP_SIZE,
        }
    }
}

/// Quantize one step's latent rows `[B, 1, L, kv_lora_rank]`.
///
/// The codec runs in f32 so the group range is not rounded before the scale is
/// derived, then the scale and bias are stored FP16 (32 bytes per token at
/// rank 512). An all-zero group gets a zero bias and quantizes to zero, so it
/// reads back as zero.
pub(crate) fn quantize_latent_rows(
    ckv: &MlxArray,
    kpe: UniquePtr<MlxArray>,
    bits: i32,
) -> QuantizedLatentRows {
    let shape = ffi::array_shape(ckv);
    let (b, h, l, rank) = (shape[0], shape[1], shape[2], shape[3]);
    let flat = ffi::reshape(&ffi::astype(ckv, dtype::FLOAT32), &[b * h * l, rank]);
    let q = ffi::quantize_weights(&flat, LATENT_QUANT_GROUP_SIZE, bits);
    let unflatten = |a: UniquePtr<MlxArray>| ffi::reshape(&a, &[b, h, l, -1]);
    QuantizedLatentRows {
        packed: unflatten(ffi::quantized_weights_w(&q)),
        scales: unflatten(ffi::astype(
            &ffi::quantized_weights_scales(&q),
            dtype::FLOAT16,
        )),
        biases: unflatten(ffi::astype(
            &ffi::quantized_weights_biases(&q),
            dtype::FLOAT16,
        )),
        kpe: ffi::astype(&kpe, dtype::FLOAT16),
    }
}

/// Dequantize a packed latent window to `[B, 1, S, kv_lora_rank]`.
pub(crate) fn dequantize_latent(
    packed: &MlxArray,
    scales: &MlxArray,
    biases: &MlxArray,
    bits: i32,
    group_size: i32,
) -> UniquePtr<MlxArray> {
    let s = ffi::array_shape(packed);
    let rows = s[0] * s[1] * s[2];
    let flat = |a: &MlxArray| ffi::reshape(a, &[rows, -1]);
    let biases_flat = flat(biases);
    // SAFETY: `biases_flat` is a live array for the duration of the call; the
    // bridge reads it and does not keep the pointer.
    let dense = unsafe {
        ffi::dequantize(
            &flat(packed),
            &flat(scales),
            &*biases_flat as *const MlxArray,
            group_size,
            bits,
            AFFINE,
        )
    };
    ffi::reshape(&dense, &[s[0], s[1], s[2], -1])
}

/// `x @ dequantize(w)^T` when `transpose`, else `x @ dequantize(w)`, in f32.
fn latent_qmm(
    x: &MlxArray,
    packed: &MlxArray,
    scales: &MlxArray,
    biases: &MlxArray,
    transpose: bool,
    bits: i32,
    group_size: i32,
) -> UniquePtr<MlxArray> {
    // SAFETY: `biases` is a live array for the duration of the call; the
    // bridge reads it and does not keep the pointer.
    unsafe {
        ffi::quantized_matmul(
            x,
            packed,
            scales,
            biases as *const MlxArray,
            transpose,
            group_size,
            bits,
            AFFINE,
        )
    }
}

/// `mla::decode::rope_score_mask` for one query, with the heads on the matmul's row axis
/// so the `k_pe` window is read once rather than once per head.
///
/// Same operations in the same dtypes as `rope_score_mask`: the query is
/// scaled in its own dtype, multiplied against `kpe^T`, and the additive mask
/// is applied to the `[B, H, 1, S]` result.
fn rope_scores_by_row(
    q_pe: &MlxArray,
    kpe: &MlxArray,
    scale: f32,
    mask: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    let s = ffi::array_shape(q_pe);
    let (b, h, d) = (s[0], s[1], s[3]);
    let scale_scalar = ffi::full_f32(&[1], scale, ffi::array_dtype(q_pe));
    let q = ffi::reshape(&ffi::multiply(q_pe, &scale_scalar), &[b, 1, h, d]);
    let kpe_t = ffi::transpose_axes(kpe, &[0, 1, 3, 2]);
    let scores = ffi::reshape(&ffi::matmul(&q, &kpe_t), &[b, h, 1, -1]);
    match mask {
        Some(m) => ffi::add(&scores, m),
        None => scores,
    }
}

/// Single-query absorbed attention against a quantized latent window.
///
/// * `q_absorbed` `[B, H, 1, kv_lora_rank]`, `q_pe` `[B, H, 1, qk_rope_head_dim]`
/// * `mask` an optional additive `[.., 1, S]` mask
///
/// Returns `[B, H, 1, kv_lora_rank]` in the query's dtype. An unmasked step
/// runs on the fused Metal kernel's quantized variant where it applies
/// ([`fused_absorbed_decode_quantized`], issue #3156) and records
/// [`MlaDecodePath::AbsorbedFused`]; every other step, and every backend
/// without the kernel, takes [`composed_quantized_attention`].
///
/// The dense fused kernel and the split-KV decode both read an FP16 latent, so
/// neither is tried here; the caller reaches this function by matching on
/// [`LatentWindow::Quantized`] before either could read the window.
///
/// Used by: [`crate::mla::cache::MlaLatentCache::attend_preprojected`].
#[must_use]
#[allow(clippy::too_many_arguments)]
pub(crate) fn quantized_absorbed_attention(
    q_absorbed: &MlxArray,
    q_pe: &MlxArray,
    ckv_q: &MlxArray,
    scales: &MlxArray,
    biases: &MlxArray,
    kpe: &MlxArray,
    bits: i32,
    group_size: i32,
    scale: f32,
    mask: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    if mask.is_none()
        && let Some(out) = fused_absorbed_decode_quantized(
            q_absorbed, q_pe, ckv_q, scales, biases, kpe, bits, group_size, scale,
        )
    {
        return out;
    }
    composed_quantized_attention(
        q_absorbed, q_pe, ckv_q, scales, biases, kpe, bits, group_size, scale, mask,
    )
}

/// The composed form of [`quantized_absorbed_attention`], on
/// `quantized_matmul`; records [`MlaDecodePath::AbsorbedComposed`]. The
/// score, the softmax and the value sum run in f32: a long window puts most
/// probabilities below the FP16 normal range, and the value sum is where that
/// precision is spent.
///
/// Used by: [`quantized_absorbed_attention`], and the fused kernel's parity
/// tests as their reference.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub(crate) fn composed_quantized_attention(
    q_absorbed: &MlxArray,
    q_pe: &MlxArray,
    ckv_q: &MlxArray,
    scales: &MlxArray,
    biases: &MlxArray,
    kpe: &MlxArray,
    bits: i32,
    group_size: i32,
    scale: f32,
    mask: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    stats::record(MlaDecodePath::AbsorbedComposed);
    let out_dtype = ffi::array_dtype(q_absorbed);
    let scales = ffi::astype(scales, dtype::FLOAT32);
    let biases = ffi::astype(biases, dtype::FLOAT32);
    // Scale the query rather than the `[B, H, 1, S]` score: same product, on
    // the smaller operand.
    let q = crate::ops::multiply_scalar(&ffi::astype(q_absorbed, dtype::FLOAT32), scale);
    // The latent score keeps one batch entry per head. Folding the heads into
    // its row axis measured 3% to 13% faster but changes the summation order
    // outside the #3155 gate's near-tie bar (results doc, "Single-query decode
    // layout"); folding the value sum measured slower.
    let latent_scores = latent_qmm(&q, ckv_q, &scales, &biases, true, bits, group_size);
    let rope_scores = ffi::astype(&rope_scores_by_row(q_pe, kpe, scale, mask), dtype::FLOAT32);
    let probs = ffi::softmax(&ffi::add(&latent_scores, &rope_scores), -1);
    let o_latent = latent_qmm(&probs, ckv_q, &scales, &biases, false, bits, group_size);
    ffi::astype(&o_latent, out_dtype)
}

#[cfg(test)]
#[path = "quant_latent_tests.rs"]
mod quant_latent_tests;
