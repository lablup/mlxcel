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

//! V-side 3-bit PolarQuant pipeline for `KVCacheMode::Turbo3Asym` (issue
//! #477, epic #458).
//!
//! This module mirrors [`super::quant`] (the 4-bit pipeline used by
//! `Turbo4Asym` / `Turbo4`) but uses a **3-bit** Lloyd-Max codebook and the
//! 24-bit-grouped packing layout from [`super::pack3`].
//!
//! # Why a separate module
//!
//! The 4-bit nibble-packing path can be expressed entirely on-device (a
//! `bitwise_and` + `right_shift` + `stack/reshape`) because two indices
//! always fit into one byte. The 3-bit layout splits 8 indices across 3
//! bytes, which doesn't decompose into a single MLX op — the natural
//! implementation is to read the packed bytes back to host memory, expand
//! to one-byte-per-index on CPU, and ship the dense index array back to
//! the GPU for the centroid `take()`. That host round-trip lives in this
//! module to keep [`super::quant`]'s on-device fast path uncluttered.
//!
//! # Storage layout
//!
//! For each token slot the cache stores:
//!
//! - `v_packed[..., t, :]` — `head_dim * 3 / 8` u8s. Documented bit layout
//!   is in [`super::pack3`].
//! - `v_norms[..., t, 0]`  — fp16 L2 norm of the original V vector.
//!
//! Sign vectors and codebook are reused from [`super::quant::TurboQuantParams`]
//! — Turbo3 builds its own `TurboQuantParams3` with a 3-bit codebook so the
//! existing 4-bit `TurboQuantParams` is untouched.
//!
//! # Asymmetric-only
//!
//! Issue #477 explicitly defers the symmetric `Turbo3` variant. Symmetric
//! 3-bit on dense Q4_K_M weights is catastrophic per epic #458's
//! "Quality–compression tradeoff control" section. This PR ships only
//! `KVCacheMode::Turbo3Asym` (Fp16-K + Turbo3-V). The allowlist gating in
//! [`super::allowlist`] does not apply because asymmetric Turbo* is always
//! safe.
//!
//! Used by: `KVCache::update_turbo3_asym` (cache.rs, issue #477).

use std::sync::Arc;

use cxx::UniquePtr;

use super::codebook::{nearest_centroid_indices_with_boundaries, optimal_codebook, Codebook};
use super::pack3::{pack_3bit_per_token, packed_bytes_per_token_3bit, unpack_3bit_per_token};
use super::quant::generate_signs;
use crate::dtype;
use crate::ffi;
use crate::ffi::MlxArray;
use crate::ops::wht;

/// V-side bit-width for `Turbo3Asym`. Locked to 3.
pub const V_BIT_WIDTH_3: u8 = 3;

/// Parameters for the 3-bit V-side TurboQuant pipeline.
///
/// Mirrors [`super::quant::TurboQuantParams`] but carries a 3-bit codebook
/// (8 centroids) instead of 4-bit (16 centroids). The K-side sign vectors
/// from the 4-bit struct are intentionally absent — symmetric Turbo3 is
/// out of scope for issue #477 and the asymmetric path leaves K in FP16.
///
/// One instance is built per `KVCache` at construction time when the cache
/// is in `KVCacheMode::Turbo3Asym` mode, parameterised by the V `head_dim`
/// of the runtime model and a deterministic seed. The codebook is fetched
/// from the global `OnceLock`-backed cache so multiple caches share the
/// centroids.
///
/// Used by: `KVCache::update_turbo3_asym` (cache.rs),
/// [`quantize_v_turbo3`], [`dequantize_v_turbo3`].
#[derive(Clone, Debug)]
pub struct TurboQuantParams3 {
    /// V head dimension (must be a non-zero power of two AND a multiple of 8).
    pub head_dim: u32,
    /// `±1.0` sign vector applied **before** the WHT. Length = `head_dim`.
    pub signs1: Arc<[f32]>,
    /// `±1.0` sign vector applied **after** the WHT. Length = `head_dim`.
    pub signs2: Arc<[f32]>,
    /// 3-bit Lloyd-Max codebook (8 centroids + 7 boundaries).
    pub codebook: Codebook,
}

impl TurboQuantParams3 {
    /// Build params for a `(head_dim, seed)` pair.
    ///
    /// # Panics
    ///
    /// Panics if `head_dim` is not a positive power of two — the WHT op
    /// requires it. Also panics if `head_dim` is not a multiple of 8 — the
    /// 3-bit packing layout requires byte-aligned 8-coord groups (see
    /// [`super::pack3`]). Non-power-of-two head dims need a non-WHT rotation
    /// path before they can be enabled for Turbo3.
    pub fn new(head_dim: u32, seed: u32) -> Self {
        assert!(
            head_dim > 0 && head_dim.is_power_of_two(),
            "TurboQuantParams3::new: head_dim must be a non-zero power of two; got {head_dim}"
        );
        assert!(
            head_dim.is_multiple_of(8),
            "TurboQuantParams3::new: head_dim must be a multiple of 8 \
             for the 24-bit packing layout; got {head_dim}"
        );
        let signs1 = generate_signs(head_dim as usize, seed);
        // Use a different sub-seed for signs2 so the two are independent.
        // Mirrors the 4-bit path's offset to stay deterministic across runs.
        let signs2 = generate_signs(head_dim as usize, seed.wrapping_add(0x9E37_79B9));
        let codebook = optimal_codebook(V_BIT_WIDTH_3, head_dim);
        Self {
            head_dim,
            signs1,
            signs2,
            codebook,
        }
    }
}

// ---------------------------------------------------------------------------
// Quantize / dequantize — graph + readback hybrid
// ---------------------------------------------------------------------------

/// Quantize a 4-D `[B, H, T, D]` V tensor into the packed Turbo3 representation.
///
/// Returns `(v_packed, v_norms)` where:
/// - `v_packed`: `[B, H, T, D * 3 / 8]` u8 — 24-bit-grouped 3-bit indices.
/// - `v_norms`:  `[B, H, T, 1]` fp16 — per-token L2 norm of the *original*
///   V vector (used for rescaling on dequantize).
///
/// Algorithm (mirrors [`super::quant::quantize_v_turbo4`] except for the
/// final pack step):
///
/// 1. Cast to fp32 for stable norm + rotation arithmetic.
/// 2. Per-token L2 norm extraction and unit-normalization.
/// 3. `D2 · H · D1 · x` rotation (sign × WHT × sign).
/// 4. Per-coordinate nearest-centroid lookup (3-bit → 8 centroids).
/// 5. 24-bit-grouped pack: 8 indices → 3 bytes (see [`super::pack3`]).
///
/// Used by: `KVCache::update_turbo3_asym` (cache.rs).
pub fn quantize_v_turbo3(
    v: &MlxArray,
    params: &TurboQuantParams3,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let v_f32 = ffi::astype(v, dtype::FLOAT32);
    let shape = ffi::array_shape(&v_f32);
    debug_assert_eq!(shape.len(), 4, "input must be 4-D [B, H, T, D]");
    let _b = shape[0];
    let _h = shape[1];
    let t = shape[2];
    let d = shape[3];
    debug_assert_eq!(
        d as u32, params.head_dim,
        "input last dim ({d}) must match TurboQuantParams3 head_dim ({})",
        params.head_dim
    );

    // 1. Per-token L2 norm with a `where`-guarded zero fallback (matches the
    //    Python reference's `np.where(norms > 0, norms, 1.0)` and avoids the
    //    direction-destroying `maximum(norm, 1.0)` clamp documented in the
    //    4-bit path's `small_norm_v_round_trip_recovers_within_bound` test).
    let v_sq = ffi::multiply(&v_f32, &v_f32);
    let sum_sq = ffi::sum_axis(&v_sq, -1, true);
    let norm_full = ffi::sqrt(&sum_sq);
    let zero = ffi::full_f32(&[1], 0.0, dtype::FLOAT32);
    let one = ffi::full_f32(&[1], 1.0, dtype::FLOAT32);
    let positive_mask = ffi::greater(&norm_full, &zero);
    let safe_norm = ffi::where_cond(&positive_mask, &norm_full, &one);
    let v_normalized = ffi::divide(&v_f32, &safe_norm);

    // 2. D2 · H · D1 rotation.
    let signs1_arr = ffi::from_slice_f32(&params.signs1, &[1, 1, 1, d]);
    let signs2_arr = ffi::from_slice_f32(&params.signs2, &[1, 1, 1, d]);
    let v_d1 = ffi::multiply(&v_normalized, &signs1_arr);
    let v_h = wht(&v_d1);
    let v_rot = ffi::multiply(&v_h, &signs2_arr);

    // 3. Read rotated coords back to host memory for nearest-centroid lookup.
    //    Same readback contract as the 4-bit `quantize_into_packed`.
    ffi::eval(&v_rot);
    let coord_count = (shape[0] * shape[1] * t * d) as usize;
    let v_rot_bytes = ffi::array_to_raw_bytes(&v_rot);
    debug_assert_eq!(
        v_rot_bytes.len(),
        coord_count * 4,
        "fp32 byte count mismatch"
    );
    let mut coords = Vec::with_capacity(coord_count);
    for chunk in v_rot_bytes.chunks_exact(4) {
        coords.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    let n_centroids = params.codebook.centroids.len();
    debug_assert_eq!(
        n_centroids,
        1 << V_BIT_WIDTH_3,
        "Turbo3 codebook must have 8 centroids; got {n_centroids}"
    );
    let indices_usize =
        nearest_centroid_indices_with_boundaries(&coords, &params.codebook.boundaries, n_centroids);

    // 4. Cast indices to u8 (each is in 0..8, fits trivially) and 24-bit-pack.
    debug_assert_eq!(
        d % 8,
        0,
        "head_dim must be a multiple of 8 for 3-bit packing"
    );
    let total_tokens = (shape[0] * shape[1] * t) as usize;
    let mut indices_u8 = Vec::with_capacity(indices_usize.len());
    for &idx in &indices_usize {
        indices_u8.push((idx & 0x07) as u8);
    }
    let bytes_per_token = packed_bytes_per_token_3bit(d) as usize;
    let packed_bytes = pack_3bit_per_token(&indices_u8, d, total_tokens);
    debug_assert_eq!(packed_bytes.len(), total_tokens * bytes_per_token);

    let v_packed = ffi::from_bytes(
        &packed_bytes,
        &[shape[0], shape[1], t, bytes_per_token as i32],
        dtype::UINT8,
    );

    // 5. Norms stored in fp16. Use the full-precision norm (not safe_norm)
    //    so dequantize sees the original magnitude.
    let v_norms = ffi::astype(&norm_full, dtype::FLOAT16);

    (v_packed, v_norms)
}

/// Dequantize a packed-V slice back to fp16 for the attention kernel.
///
/// Inputs:
/// - `v_packed`: `[B, H, T, D * 3 / 8]` u8 (output of [`quantize_v_turbo3`]).
/// - `v_norms`:  `[B, H, T, 1]` fp16 — per-token original-V L2 norms.
/// - `params`:   the [`TurboQuantParams3`] used at quantize time.
///
/// Returns a fresh fp16 tensor of shape `[B, H, T, D]`, identical contract
/// to [`super::quant::dequantize_v_turbo4`].
///
/// Algorithm:
/// 1. Read the packed bytes back to host memory and 24-bit-unpack into a
///    flat u8 index array.
/// 2. Materialize the indices as a `[B, H, T, D]` UINT8 MLX tensor.
/// 3. Centroid gather via `take()`.
/// 4. Norm correction (rescale to unit length), inverse rotation, rescale
///    by stored norm. Steps 3–4 are bit-identical to the 4-bit path; only
///    the unpack stage differs.
///
/// Used by: `KVCache::update_and_fetch` (Turbo3Asym mode).
pub fn dequantize_v_turbo3(
    v_packed: &MlxArray,
    v_norms: &MlxArray,
    params: &TurboQuantParams3,
) -> UniquePtr<MlxArray> {
    let packed_shape = ffi::array_shape(v_packed);
    debug_assert_eq!(packed_shape.len(), 4, "packed must be 4-D");
    let b = packed_shape[0];
    let h = packed_shape[1];
    let t = packed_shape[2];
    let bytes_per_token = packed_shape[3];
    let d = bytes_per_token * 8 / 3; // inverse of head_dim * 3 / 8
    debug_assert_eq!(
        d as u32, params.head_dim,
        "head_dim mismatch on dequantize: derived {d} from {bytes_per_token} packed bytes, \
         expected {}",
        params.head_dim
    );
    debug_assert_eq!(
        d % 8,
        0,
        "head_dim must be a multiple of 8 for 3-bit dequant; got {d}"
    );

    // 1. Pull the packed bytes back to host memory and unpack to one-byte-
    //    per-index. This is the cost of 3-bit packing not decomposing into a
    //    single MLX op (4-bit nibble-packing does — see
    //    `super::quant::dequantize_from_packed`).
    ffi::eval(v_packed);
    let raw_bytes = ffi::array_to_raw_bytes(v_packed);
    let total_tokens = (b * h * t) as usize;
    debug_assert_eq!(
        raw_bytes.len(),
        total_tokens * bytes_per_token as usize,
        "packed byte count mismatch on readback"
    );
    let indices_u8 = unpack_3bit_per_token(&raw_bytes, d, total_tokens);
    debug_assert_eq!(indices_u8.len(), total_tokens * d as usize);

    // 2. Materialize the dense index tensor on-device. UINT8 is the right
    //    dtype for `take()` (matches the 4-bit path).
    let indices_arr = ffi::from_bytes(&indices_u8, &[b, h, t, d], dtype::UINT8);

    // 3. Centroid gather: y_hat[b, h, t, k] = centroids[indices[b, h, t, k]].
    let centroids_vec: Vec<f32> = params.codebook.centroids.as_ref().to_vec();
    let centroids_arr =
        ffi::from_slice_f32(&centroids_vec, &[params.codebook.centroids.len() as i32]);
    let y_hat = ffi::take(&centroids_arr, &indices_arr, 0);

    // 4. Norm correction: rescale y_hat to unit norm so the inverse rotation
    //    sees a unit vector. Same pattern as the 4-bit path.
    let y_hat_sq = ffi::multiply(&y_hat, &y_hat);
    let y_hat_sum_sq = ffi::sum_axis(&y_hat_sq, -1, true);
    let y_hat_norm = ffi::sqrt(&y_hat_sum_sq);
    let eps = ffi::full_f32(&[1], 1e-10, dtype::FLOAT32);
    let safe_y_norm = ffi::maximum(&y_hat_norm, &eps);
    let y_hat_unit = ffi::divide(&y_hat, &safe_y_norm);

    // 5. Inverse rotation (D and H are symmetric so the transpose of D2·H·D1
    //    is D1·H·D2 — apply in reverse).
    let signs1_arr = ffi::from_slice_f32(&params.signs1, &[1, 1, 1, d]);
    let signs2_arr = ffi::from_slice_f32(&params.signs2, &[1, 1, 1, d]);
    let v_pre_h = ffi::multiply(&y_hat_unit, &signs2_arr);
    let v_pre_d1 = wht(&v_pre_h);
    let v_unit = ffi::multiply(&v_pre_d1, &signs1_arr);

    // 6. Rescale by stored original norms.
    let norms_f32 = ffi::astype(v_norms, dtype::FLOAT32);
    let v_full_f32 = ffi::multiply(&v_unit, &norms_f32);
    ffi::astype(&v_full_f32, dtype::FLOAT16)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Number of packed-V bytes needed to store `head_dim` 3-bit indices for
/// one token. Wraps [`super::pack3::packed_bytes_per_token_3bit`] so the
/// cache layer can import it without crossing module lines.
#[inline]
pub fn packed_bytes_per_token(head_dim: i32) -> i32 {
    packed_bytes_per_token_3bit(head_dim)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turbo_quant_params3_centroid_count_matches_bit_width() {
        let p = TurboQuantParams3::new(128, 42);
        assert_eq!(p.codebook.centroids.len(), 1 << V_BIT_WIDTH_3);
        assert_eq!(p.codebook.boundaries.len(), p.codebook.centroids.len() - 1);
        assert_eq!(p.signs1.len(), 128);
        assert_eq!(p.signs2.len(), 128);
    }

    /// End-to-end round-trip on `[B=1, H=1, T=4, D=128]` realistic V tensor.
    /// Per-token relative L2 error bound is wider than the 4-bit path because
    /// 3-bit Lloyd-Max gives ~6 dB more distortion (D(R=3) ≈ -17.8 dB on
    /// `N(0, 1/d)` vs -23.8 dB at R=4). Use a 25% bound to stay non-flaky.
    #[test]
    fn quantize_dequantize_round_trip_recovers_v_within_bound() {
        let head_dim: i32 = 128;
        let params = TurboQuantParams3::new(head_dim as u32, 42);

        // Same Lcg-derived deterministic test data as the 4-bit path.
        let mut state: u32 = 123;
        let mut v_data: Vec<f32> = Vec::with_capacity(4 * head_dim as usize);
        let token_scales = [0.5_f32, 1.5, 4.0, 12.0];
        for &scale in &token_scales {
            for _ in 0..head_dim {
                let mut acc = 0.0_f32;
                for _ in 0..6 {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let bit = state >> 31;
                    acc += if bit == 0 { -1.0 } else { 1.0 };
                }
                v_data.push((acc / 6.0) * scale);
            }
        }
        let v = ffi::from_slice_f32(&v_data, &[1, 1, 4, head_dim]);

        let (packed, norms) = quantize_v_turbo3(&v, &params);
        let pshape = ffi::array_shape(&packed);
        assert_eq!(pshape, vec![1_i32, 1, 4, head_dim * 3 / 8]);
        let nshape = ffi::array_shape(&norms);
        assert_eq!(nshape, vec![1_i32, 1, 4, 1]);

        let v_hat = dequantize_v_turbo3(&packed, &norms, &params);
        assert_eq!(ffi::array_dtype(&v_hat), dtype::FLOAT16);
        assert_eq!(ffi::array_shape(&v_hat), vec![1_i32, 1, 4, head_dim]);

        let v_hat_f32 = ffi::astype(&v_hat, dtype::FLOAT32);
        ffi::eval(&v_hat_f32);
        let v_hat_bytes = ffi::array_to_raw_bytes(&v_hat_f32);
        let v_hat_vec: Vec<f32> = v_hat_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // 25% per-token relative L2 bound: 3-bit Lloyd-Max distortion plus
        // fp16 round-off plus norm correction.
        let hd_usize = head_dim as usize;
        for tok in 0..4 {
            let off = tok * hd_usize;
            let mut num = 0.0_f32;
            let mut den = 0.0_f32;
            for k in 0..hd_usize {
                let diff = v_data[off + k] - v_hat_vec[off + k];
                num += diff * diff;
                den += v_data[off + k] * v_data[off + k];
            }
            let rel = (num / den.max(1e-12)).sqrt();
            assert!(
                rel < 0.25,
                "token {tok}: relative L2 error {rel:.4} exceeds 25% bound"
            );
        }
    }

    /// Determinism: same params + same V → same packed bytes.
    #[test]
    fn quantize_is_deterministic() {
        let head_dim: i32 = 64;
        let params1 = TurboQuantParams3::new(head_dim as u32, 7);
        let params2 = TurboQuantParams3::new(head_dim as u32, 7);
        assert_eq!(params1.signs1.as_ref(), params2.signs1.as_ref());

        let v_data: Vec<f32> = (0..head_dim).map(|i| (i as f32 - 32.0) * 0.05).collect();
        let v = ffi::from_slice_f32(&v_data, &[1, 1, 1, head_dim]);
        let (p1, _) = quantize_v_turbo3(&v, &params1);
        let (p2, _) = quantize_v_turbo3(&v, &params2);
        assert_eq!(ffi::array_to_raw_bytes(&p1), ffi::array_to_raw_bytes(&p2));
    }

    /// Zero V vector preserves through quant/dequant without NaN.
    #[test]
    fn zero_vector_dequantizes_to_zero() {
        let head_dim: i32 = 64;
        let params = TurboQuantParams3::new(head_dim as u32, 1);
        let v = ffi::zeros(&[1, 1, 1, head_dim], dtype::FLOAT32);
        let (packed, norms) = quantize_v_turbo3(&v, &params);
        let v_hat = dequantize_v_turbo3(&packed, &norms, &params);
        let v_hat_f32 = ffi::astype(&v_hat, dtype::FLOAT32);
        ffi::eval(&v_hat_f32);
        let bytes = ffi::array_to_raw_bytes(&v_hat_f32);
        for chunk in bytes.chunks_exact(4) {
            let val = f32::from_le_bytes(c0(chunk));
            assert!(val.abs() < 1e-3, "expected ~0, got {val}");
        }
    }

    /// Sanity: indices written to the packed buffer must all be in 0..8.
    /// Catches accidental >3-bit leakage in the pack helper.
    #[test]
    fn packed_indices_stay_in_range() {
        let head_dim: i32 = 128;
        let params = TurboQuantParams3::new(head_dim as u32, 99);
        let v_data: Vec<f32> = (0..head_dim)
            .map(|i| ((i as f32 / head_dim as f32) - 0.5) * 4.0)
            .collect();
        let v = ffi::from_slice_f32(&v_data, &[1, 1, 1, head_dim]);
        let (packed, _) = quantize_v_turbo3(&v, &params);
        ffi::eval(&packed);
        let bytes = ffi::array_to_raw_bytes(&packed);
        let total_tokens = 1_usize;
        let unpacked = unpack_3bit_per_token(&bytes, head_dim, total_tokens);
        for &idx in &unpacked {
            assert!(idx < 8, "decoded 3-bit index out of range: {idx}");
        }
    }

    /// Stored norms reflect the per-token L2 norm of the input within fp16
    /// precision. Mirrors the 4-bit path's `stored_norms_match_input_l2`.
    #[test]
    fn stored_norms_match_input_l2() {
        let head_dim: i32 = 64;
        let hd_usize = head_dim as usize;
        let params = TurboQuantParams3::new(head_dim as u32, 17);
        let mut v_data: Vec<f32> = Vec::with_capacity(2 * hd_usize);
        for k in 0..head_dim {
            v_data.push(0.5 * ((k as f32).sin()));
        }
        for k in 0..head_dim {
            v_data.push(50.0 * ((k as f32).cos()));
        }
        let v = ffi::from_slice_f32(&v_data, &[1, 1, 2, head_dim]);
        let (_, norms) = quantize_v_turbo3(&v, &params);
        let norms_f32 = ffi::astype(&norms, dtype::FLOAT32);
        ffi::eval(&norms_f32);
        let bytes = ffi::array_to_raw_bytes(&norms_f32);
        let n0 = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let n1 = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);

        let expected_n0 = (v_data[..hd_usize].iter().map(|x| x * x).sum::<f32>()).sqrt();
        let expected_n1 = (v_data[hd_usize..].iter().map(|x| x * x).sum::<f32>()).sqrt();
        assert_relative_error(n0, expected_n0, 0.02, "norm[0]");
        assert_relative_error(n1, expected_n1, 0.02, "norm[1]");
    }

    /// nbytes proxy: packed bytes + norm bytes per token.
    #[test]
    fn packed_bytes_per_token_proxy_matches_pack3() {
        for &d in &[64, 80, 96, 128, 192, 256] {
            assert_eq!(packed_bytes_per_token(d), packed_bytes_per_token_3bit(d));
        }
    }

    // -- helpers --

    fn assert_relative_error(actual: f32, expected: f32, tol: f32, label: &str) {
        let denom = expected.abs().max(1e-6);
        let err = (actual - expected).abs() / denom;
        assert!(
            err < tol,
            "{label}: |{actual} - {expected}| / |expected| = {err:.4} > {tol}"
        );
    }

    fn c0(chunk: &[u8]) -> [u8; 4] {
        [chunk[0], chunk[1], chunk[2], chunk[3]]
    }
}
