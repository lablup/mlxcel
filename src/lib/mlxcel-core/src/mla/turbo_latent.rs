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

//! Rotated-domain Turbo-4 MLA latent, evaluated and not adopted (issue #3157).
//!
//! Test-only. On DeepSeek-V2-Lite it put 45 of 512 top-1 choices off the fp16
//! latent behind 1024 tokens, the largest at a 0.719-logit gap, which fails
//! the #3155 near-tie bar of 0.11 just as 4-bit affine did, so no KV cache
//! mode reaches it and Turbo requests on an absorbed latent keep resolving to
//! the fp16 latent. Numbers, the decode cost and the hook that measured them:
//! `docs/benchmark_results/mla-turbo-latent-m3ultra-2026-11-19.md`. The code
//! stays so the parity tests pin what was measured and a later codec can be
//! compared on the same seam.
//!
//! `R = D2 H D1` is orthogonal (`H` is the orthonormal Walsh-Hadamard
//! transform, `D1`, `D2` are sign flips), so for the latent `ckv`, which is
//! both key and value under absorption,
//!
//! ```text
//!   q_abs . ckv = (R q_abs) . (R ckv)        p . ckv = R^T (p . R ckv)
//! ```
//!
//! The cache stores `R ckv` with the Turbo-4 codec: one 4-bit Lloyd-Max
//! codebook index per coordinate (`rank / 2` bytes) and one FP16 rescale per
//! token, `norm / |codebook[idx]|`, so a stored row reads back as
//! `codebook[idx] * rescale` in the rotated domain. A decode step rotates the
//! query once, attends against the gathered codebook values, and
//! inverse-rotates the `[B, H, 1, rank]` output once. One rotation serves the
//! key and the value role, so a single sign-vector pair is used: the K/V seed
//! decorrelation in `cache::turbo::quant` exists for two different tensors.
//!
//! [`RotatedAffine`] is the second arm the results doc measures: the same
//! rotation in front of the #3155 affine codec at 4 bits, decoded through
//! `quantized_absorbed_attention` on the rotated query.
//!
//! Latent and dense Turbo4 rows share the on-device index, packing and scale computation.

use cxx::UniquePtr;

use crate::cache::latent_quant::QuantizedLatentRows;
use crate::cache::turbo::quant::{TurboQuantParams, pack_rotated_turbo4, rotate_turbo4_f32};
use crate::dtype;
use crate::ffi::{self, MlxArray};
use crate::mla::quant_latent::{
    quantize_latent_rows, quantized_absorbed_attention, rope_scores_by_row,
};
use crate::mla::stats::{self, MlaDecodePath};
use crate::ops::wht;

/// Bytes one token of a Turbo-4 latent costs, per layer: `rank / 2` packed
/// indices, one FP16 rescale, and the FP16 rope stream. 386 on
/// DeepSeek-V2-Lite (rank 512, rope 64).
#[must_use]
pub(crate) const fn turbo4_latent_bytes_per_token(
    kv_lora_rank: usize,
    qk_rope_head_dim: usize,
) -> usize {
    kv_lora_rank / 2 + 2 + qk_rope_head_dim * 2
}

/// One step's (or one window's) Turbo-4 latent rows.
pub(crate) struct TurboLatentRows {
    /// `[B, 1, L, rank / 2]` uint8: low nibble the even coordinate, high
    /// nibble the odd one.
    pub packed: UniquePtr<MlxArray>,
    /// `[B, 1, L, 1]` FP16 `norm / |codebook[idx]|`; zero for an all-zero row.
    pub rescale: UniquePtr<MlxArray>,
}

/// The rotation and codebook of one latent cache.
///
/// Built from a cache's `turbo_seed` (`TURBO_DEFAULT_SEED` unless the cache
/// was given another), so a detached and re-adopted cache rebuilds the same
/// rotation from the seed alone and its stored rows stay meaningful.
pub(crate) struct TurboLatentCodec {
    rank: i32,
    signs1: UniquePtr<MlxArray>,
    signs2: UniquePtr<MlxArray>,
    centroids: UniquePtr<MlxArray>,
    boundaries: UniquePtr<MlxArray>,
}

impl TurboLatentCodec {
    /// # Panics
    ///
    /// When `kv_lora_rank` is not a power of two (the WHT's requirement).
    pub(crate) fn new(kv_lora_rank: usize, seed: u32) -> Self {
        // V-side signs only: one rotation for the one tensor.
        let params = TurboQuantParams::new(kv_lora_rank as u32, seed);
        let rank = kv_lora_rank as i32;
        let cb = &params.codebook;
        Self {
            rank,
            signs1: ffi::from_slice_f32(&params.signs1, &[1, 1, 1, rank]),
            signs2: ffi::from_slice_f32(&params.signs2, &[1, 1, 1, rank]),
            centroids: ffi::from_slice_f32(&cb.centroids, &[cb.centroids.len() as i32]),
            boundaries: ffi::from_slice_f32(&cb.boundaries, &[cb.boundaries.len() as i32]),
        }
    }

    /// `R x` in f32.
    pub(crate) fn rotate(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let x = ffi::astype(x, dtype::FLOAT32);
        rotate_turbo4_f32(&x, &self.signs1, &self.signs2)
    }

    /// `R^T x`, cast to `out_dtype`.
    pub(crate) fn inverse_rotate(&self, x: &MlxArray, out_dtype: i32) -> UniquePtr<MlxArray> {
        let x = ffi::astype(x, dtype::FLOAT32);
        let y = ffi::multiply(&wht(&ffi::multiply(&x, &self.signs2)), &self.signs1);
        ffi::astype(&y, out_dtype)
    }

    /// Quantize `[B, 1, L, rank]` latent rows.
    pub(crate) fn quantize(&self, ckv: &MlxArray) -> TurboLatentRows {
        let s = ffi::array_shape(ckv);
        assert_eq!(s[3], self.rank, "turbo latent: rank mismatch");
        let x = ffi::astype(ckv, dtype::FLOAT32);
        let norm = ffi::sqrt(&ffi::sum_axis(&ffi::multiply(&x, &x), -1, true));
        // A zero norm divides by one instead, so the row stays zero.
        let zero = ffi::full_f32(&[1], 0.0, dtype::FLOAT32);
        let one = ffi::full_f32(&[1], 1.0, dtype::FLOAT32);
        let safe = ffi::where_cond(&ffi::greater(&norm, &zero), &norm, &one);
        let unit = self.rotate(&ffi::divide(&x, &safe));
        let (packed, rescale) =
            pack_rotated_turbo4(&unit, &norm, &self.centroids, &self.boundaries);
        TurboLatentRows { packed, rescale }
    }

    /// Unscaled codebook values `[B, 1, S, rank]` in `out_dtype`.
    fn gather(&self, packed: &MlxArray, out_dtype: i32) -> UniquePtr<MlxArray> {
        let s = ffi::array_shape(packed);
        let mask = ffi::full_f32(&[1], 15.0, dtype::UINT8);
        let four = ffi::full_f32(&[1], 4.0, dtype::UINT8);
        let lo = ffi::bitwise_and(packed, &mask);
        let hi = ffi::right_shift(packed, &four);
        let idx = ffi::reshape(
            &crate::ops::stack_owned(&[lo, hi], -1),
            &[s[0], s[1], s[2], s[3] * 2],
        );
        ffi::take(&ffi::astype(&self.centroids, out_dtype), &idx, 0)
    }

    /// The stored rows in the rotated domain, f32.
    pub(crate) fn dequantize_rotated(&self, rows: &TurboLatentRows) -> UniquePtr<MlxArray> {
        let y = self.gather(&rows.packed, dtype::FLOAT32);
        ffi::multiply(&y, &ffi::astype(&rows.rescale, dtype::FLOAT32))
    }

    /// The stored rows in the latent's own basis, `[B, 1, S, rank]`.
    ///
    /// What a multi-token step and the cold prefill would read.
    pub(crate) fn dequantize(&self, rows: &TurboLatentRows, out_dtype: i32) -> UniquePtr<MlxArray> {
        self.inverse_rotate(&self.dequantize_rotated(rows), out_dtype)
    }

    /// Single-query absorbed attention against a Turbo-4 window.
    ///
    /// Same contract as `quantized_absorbed_attention`: `[B, H, 1, rank]` in
    /// the query's dtype, f32 score, softmax and value sum, and
    /// [`MlaDecodePath::AbsorbedComposed`] recorded. The rescale is folded
    /// into the score columns and the probabilities, so the gathered window
    /// is the unscaled codebook values.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn absorbed_attention(
        &self,
        q_absorbed: &MlxArray,
        q_pe: &MlxArray,
        rows: &TurboLatentRows,
        kpe: &MlxArray,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        stats::record(MlaDecodePath::AbsorbedComposed);
        let out_dtype = ffi::array_dtype(q_absorbed);
        let q = crate::ops::multiply_scalar(&self.rotate(q_absorbed), scale);
        let y = self.gather(&rows.packed, dtype::FLOAT32);
        // `[B, 1, S, 1]` -> `[B, 1, 1, S]`.
        let r = ffi::transpose_axes(&ffi::astype(&rows.rescale, dtype::FLOAT32), &[0, 1, 3, 2]);
        let y_t = ffi::transpose_axes(&y, &[0, 1, 3, 2]);
        let latent_scores = ffi::multiply(&ffi::matmul(&q, &y_t), &r);
        let rope_scores = ffi::astype(&rope_scores_by_row(q_pe, kpe, scale, mask), dtype::FLOAT32);
        let probs = ffi::softmax(&ffi::add(&latent_scores, &rope_scores), -1);
        let o_rot = ffi::matmul(&ffi::multiply(&probs, &r), &y);
        self.inverse_rotate(&o_rot, out_dtype)
    }
}

/// The second measured arm: `R ckv` stored with the #3155 affine codec.
pub(crate) struct RotatedAffine<'a> {
    pub codec: &'a TurboLatentCodec,
    pub bits: i32,
}

impl RotatedAffine<'_> {
    /// Quantize `[B, 1, L, rank]` latent rows after rotating them.
    pub(crate) fn quantize(&self, ckv: &MlxArray, kpe: UniquePtr<MlxArray>) -> QuantizedLatentRows {
        quantize_latent_rows(&self.codec.rotate(ckv), kpe, self.bits)
    }

    /// Single-query decode: rotate the query, run the #3155 composed decode,
    /// inverse-rotate the output.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn absorbed_attention(
        &self,
        q_absorbed: &MlxArray,
        q_pe: &MlxArray,
        rows: &QuantizedLatentRows,
        group_size: i32,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let out_dtype = ffi::array_dtype(q_absorbed);
        let q_rot = ffi::astype(&self.codec.rotate(q_absorbed), out_dtype);
        let o_rot = quantized_absorbed_attention(
            &q_rot,
            q_pe,
            &rows.packed,
            &rows.scales,
            &rows.biases,
            &rows.kpe,
            self.bits,
            group_size,
            scale,
            mask,
        );
        self.codec.inverse_rotate(&o_rot, out_dtype)
    }
}

#[cfg(test)]
#[path = "turbo_latent_tests.rs"]
mod turbo_latent_tests;
