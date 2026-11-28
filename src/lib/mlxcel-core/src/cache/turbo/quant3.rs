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
//
// Portions of this file are derived from turboquant_plus
// (https://github.com/TheTom/turboquant_plus), Copyright 2026 Tom Turney,
// licensed under the Apache License, Version 2.0. See the top-level NOTICE
// file for the attribution carried forward under Apache-2.0 Section 4(d).

//! V-side 3-bit PolarQuant pipeline for `KVCacheMode::Turbo3Asym`.
//!
//! This module mirrors [`super::quant`] (the 4-bit pipeline used by
//! `Turbo4Asym` / `Turbo4`) but uses a **3-bit** Lloyd-Max codebook and the
//! 24-bit-grouped packing layout from [`super::pack3`].
//!
//! # Why a separate module
//!
//! The float half of the codec (norm, rotation, centroid gather, inverse
//! rotation, rescale) is the 4-bit pipeline's. The integer half is not: 4-bit
//! indices pack two to a byte, while the 3-bit layout spreads 8 indices over
//! 3 bytes, and Turbo3 stores the raw norm with no `v_rescale` sidecar. That
//! integer half lives here and in [`super::pack3`].
//!
//! # Everything runs on the device
//!
//! Both directions are plain lazy graph ops, with no `eval` and no copy of
//! tensor data to host memory (#3274):
//!
//! - quantize: `boundary_indices_on_device` turns the rotated coordinates
//!   into indices, and `pack_3bit_on_device` packs them;
//! - dequantize: `unpack_3bit_on_device` expands the packed bytes into the
//!   index tensor the centroid `take()` reads.
//!
//! Before #3274 both read back to the host: quantize evaluated the rotated
//! coordinates and looked the centroids up on the CPU, and dequantize
//! evaluated the packed window, unpacked it on one CPU thread and uploaded the
//! indices again, once per layer per decode step over the whole visible
//! window. The stored bytes are the same as then: only integer index, pack
//! and unpack steps moved, and every float op is unchanged and in the same
//! order. The old bodies are kept as the `#[cfg(test)]` module
//! `host_reference`, and the tests hold the device path to it bit for bit.
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
//! explicitly defers the symmetric `Turbo3` variant. Symmetric
//! 3-bit on dense Q4_K_M weights is catastrophic's
//! "Quality–compression tradeoff control" section. This PR ships only
//! `KVCacheMode::Turbo3Asym` (Fp16-K + Turbo3-V). The allowlist gating in
//! [`super::allowlist`] does not apply because asymmetric Turbo* is always
//! safe.
//!
//! Used by: `KVCache::update_turbo3_asym` (cache.rs).

use std::sync::Arc;

use cxx::UniquePtr;

use super::codebook::{Codebook, boundary_indices_on_device, optimal_codebook};
use super::pack3::{pack_3bit_on_device, packed_bytes_per_token_3bit, unpack_3bit_on_device};
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
/// out of scope and the asymmetric path leaves K in FP16.
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
// Quantize / dequantize: lazy device graphs
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
/// 4. Per-coordinate nearest-centroid lookup (3-bit → 8 centroids), on the
///    device (`boundary_indices_on_device`).
/// 5. 24-bit-grouped pack: 8 indices → 3 bytes, on the device
///    (`pack_3bit_on_device`; layout in [`super::pack3`]).
///
/// Both results are lazy: the call evaluates nothing and copies no tensor data
/// to the host. `T == 0` returns empty `[B, H, 0, D * 3 / 8]` and
/// `[B, H, 0, 1]` tensors (before #3274 a zero-token input aborted the process
/// inside the Hadamard transform).
///
/// Used by: `KVCache::update_turbo3_asym` (cache.rs).
pub fn quantize_v_turbo3(
    v: &MlxArray,
    params: &TurboQuantParams3,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let v_f32 = ffi::astype(v, dtype::FLOAT32);
    let shape = ffi::array_shape(&v_f32);
    // Unconditional, for the reason spelled out in `quant::quantize_into_packed`:
    // `params.signs1` / `params.signs2` are host slices handed to
    // `ffi::from_slice_f32(signs, &[1, 1, 1, d])` below, which reads `d` floats
    // through the raw pointer with no length check of its own. A `d` larger
    // than `signs.len()` is an out-of-bounds read rather than a wrong answer,
    // and neither `[profile.release]` nor `[profile.test-fast]` turns debug
    // assertions on, so a `debug_assert!` here would guard nothing that ships
    // or that the test gate runs. `TurboQuantParams3` has all-`pub` fields and
    // no `#[non_exhaustive]`, so a struct literal with a short sign vector is
    // constructible. The checks are O(1) against a per-call cost dominated by a
    // WHT.
    assert_eq!(shape.len(), 4, "input must be 4-D [B, H, T, D]");
    let d = shape[3];
    assert_eq!(
        d as u32, params.head_dim,
        "input last dim ({d}) must match TurboQuantParams3 head_dim ({})",
        params.head_dim
    );
    assert_eq!(
        params.signs1.len(),
        d as usize,
        "signs1 length ({}) must match head_dim ({d})",
        params.signs1.len()
    );
    assert_eq!(
        params.signs2.len(),
        d as usize,
        "signs2 length ({}) must match head_dim ({d})",
        params.signs2.len()
    );

    // No tokens: nothing to quantize. MLX's Hadamard transform rejects an
    // empty array with a C++ exception, which aborts the process across the
    // `noexcept` bridge, so the empty results are built directly.
    if shape[2] == 0 {
        return (
            ffi::zeros(
                &[shape[0], shape[1], 0, packed_bytes_per_token_3bit(d)],
                dtype::UINT8,
            ),
            ffi::zeros(&[shape[0], shape[1], 0, 1], dtype::FLOAT16),
        );
    }

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

    // 3. Nearest-centroid lookup on the device: the index of a coordinate is
    //    the number of codebook boundaries strictly below it, the rule the
    //    host `nearest_centroid_indices_with_boundaries` implements.
    let n_centroids = params.codebook.centroids.len();
    debug_assert_eq!(
        n_centroids,
        1 << V_BIT_WIDTH_3,
        "Turbo3 codebook must have 8 centroids; got {n_centroids}"
    );
    // The host lookup this replaces (`nearest_centroid_indices_with_boundaries`)
    // made both checks on every call; they stay unconditional here.
    assert!(n_centroids > 0, "n_centroids must be > 0");
    assert_eq!(
        params.codebook.boundaries.len(),
        n_centroids - 1,
        "boundaries.len() must equal n_centroids - 1"
    );
    let boundaries_arr = ffi::from_slice_f32(
        &params.codebook.boundaries,
        &[params.codebook.boundaries.len() as i32],
    );
    let indices = boundary_indices_on_device(&v_rot, &boundaries_arr);

    // 4. 24-bit-grouped pack on the device: `[B, H, T, D]` u8 indices to
    //    `[B, H, T, D * 3 / 8]` u8.
    let v_packed = pack_3bit_on_device(&indices, d);

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
/// 1. 24-bit-unpack the packed bytes into a `[B, H, T, D]` UINT8 index tensor
///    on the device (`unpack_3bit_on_device`).
/// 2. Centroid gather via `take()`.
/// 3. Norm correction (rescale to unit length), inverse rotation, rescale
///    by stored norm. Steps 2–3 are bit-identical to the 4-bit path; only
///    the unpack stage differs.
///
/// The result is lazy: the call evaluates nothing and copies no tensor data to
/// the host, so the decode step that reads the whole visible window through it
/// stays one graph. `T == 0` returns an empty `[B, H, 0, D]` tensor.
///
/// Used by: `KVCache::update_and_fetch` (Turbo3Asym mode).
pub fn dequantize_v_turbo3(
    v_packed: &MlxArray,
    v_norms: &MlxArray,
    params: &TurboQuantParams3,
) -> UniquePtr<MlxArray> {
    let packed_shape = ffi::array_shape(v_packed);
    debug_assert_eq!(packed_shape.len(), 4, "packed must be 4-D");
    let bytes_per_token = packed_shape[3];
    let d = bytes_per_token * 8 / 3; // inverse of head_dim * 3 / 8
    // Unconditional for the same reason as in `quantize_v_turbo3`: the inverse
    // rotation below hands `params.signs1` / `params.signs2` to
    // `ffi::from_slice_f32(signs, &[1, 1, 1, d])`, which reads `d` floats out
    // of a `signs.len()`-element host slice. Here `d` is derived from the
    // packed byte count, so nothing but this check ties it to the params the
    // caller passed.
    assert_eq!(
        d as u32, params.head_dim,
        "head_dim mismatch on dequantize: derived {d} from {bytes_per_token} packed bytes, \
         expected {}",
        params.head_dim
    );
    assert_eq!(
        d % 8,
        0,
        "head_dim must be a multiple of 8 for 3-bit dequant; got {d}"
    );
    assert_eq!(
        params.signs1.len(),
        d as usize,
        "signs1 length ({}) must match head_dim ({d})",
        params.signs1.len()
    );
    assert_eq!(
        params.signs2.len(),
        d as usize,
        "signs2 length ({}) must match head_dim ({d})",
        params.signs2.len()
    );

    // No tokens: nothing to dequantize. See `quantize_v_turbo3` for why the
    // empty window does not go through the Hadamard transform.
    if packed_shape[2] == 0 {
        return ffi::zeros(&[packed_shape[0], packed_shape[1], 0, d], dtype::FLOAT16);
    }

    // 1. Unpack on the device to one byte per index. UINT8 is the dtype the
    //    4-bit path hands to `take()` as well.
    let indices_arr = unpack_3bit_on_device(v_packed, d);

    // 2. Centroid gather: y_hat[b, h, t, k] = centroids[indices[b, h, t, k]].
    let centroids_vec: Vec<f32> = params.codebook.centroids.as_ref().to_vec();
    let centroids_arr =
        ffi::from_slice_f32(&centroids_vec, &[params.codebook.centroids.len() as i32]);
    let y_hat = ffi::take(&centroids_arr, &indices_arr, 0);

    // 3. Norm correction: rescale y_hat to unit norm so the inverse rotation
    //    sees a unit vector. Same pattern as the 4-bit path.
    let y_hat_sq = ffi::multiply(&y_hat, &y_hat);
    let y_hat_sum_sq = ffi::sum_axis(&y_hat_sq, -1, true);
    let y_hat_norm = ffi::sqrt(&y_hat_sum_sq);
    let eps = ffi::full_f32(&[1], 1e-10, dtype::FLOAT32);
    let safe_y_norm = ffi::maximum(&y_hat_norm, &eps);
    let y_hat_unit = ffi::divide(&y_hat, &safe_y_norm);

    // 4. Inverse rotation (D and H are symmetric so the transpose of D2·H·D1
    //    is D1·H·D2 — apply in reverse).
    let signs1_arr = ffi::from_slice_f32(&params.signs1, &[1, 1, 1, d]);
    let signs2_arr = ffi::from_slice_f32(&params.signs2, &[1, 1, 1, d]);
    let v_pre_h = ffi::multiply(&y_hat_unit, &signs2_arr);
    let v_pre_d1 = wht(&v_pre_h);
    let v_unit = ffi::multiply(&v_pre_d1, &signs1_arr);

    // 5. Rescale by stored original norms.
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
// Host reference (tests only)
// ---------------------------------------------------------------------------

/// The codec as it was before #3274, kept as the reference the device path is
/// held to bit for bit.
///
/// Both bodies are the earlier `quantize_v_turbo3` and `dequantize_v_turbo3`
/// unchanged: they evaluate the graph, copy the tensor to host memory and do
/// the index, pack and unpack steps on the CPU. Only tests may call them.
///
/// Used by: the parity tests below and the Turbo3Asym cache tests in
/// `cache/turbo_tests.rs`.
#[cfg(test)]
pub(crate) mod host_reference {
    use cxx::UniquePtr;

    use super::super::codebook::nearest_centroid_indices_with_boundaries;
    use super::super::pack3::{
        pack_3bit_per_token, packed_bytes_per_token_3bit, unpack_3bit_per_token,
    };
    use super::{TurboQuantParams3, V_BIT_WIDTH_3};
    use crate::dtype;
    use crate::ffi;
    use crate::ffi::MlxArray;
    use crate::ops::wht;

    /// Host-readback quantizer: the pre-#3274 body of
    /// [`super::quantize_v_turbo3`].
    pub(crate) fn quantize_v_turbo3_host(
        v: &MlxArray,
        params: &TurboQuantParams3,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let v_f32 = ffi::astype(v, dtype::FLOAT32);
        let shape = ffi::array_shape(&v_f32);
        // Unconditional, for the reason spelled out in `quant::quantize_into_packed`:
        // `params.signs1` / `params.signs2` are host slices handed to
        // `ffi::from_slice_f32(signs, &[1, 1, 1, d])` below, which reads `d` floats
        // through the raw pointer with no length check of its own. A `d` larger
        // than `signs.len()` is an out-of-bounds read rather than a wrong answer,
        // and neither `[profile.release]` nor `[profile.test-fast]` turns debug
        // assertions on, so a `debug_assert!` here would guard nothing that ships
        // or that the test gate runs. `TurboQuantParams3` has all-`pub` fields and
        // no `#[non_exhaustive]`, so a struct literal with a short sign vector is
        // constructible. The checks are O(1) against a per-call cost dominated by a
        // WHT and a host readback.
        assert_eq!(shape.len(), 4, "input must be 4-D [B, H, T, D]");
        let _b = shape[0];
        let _h = shape[1];
        let t = shape[2];
        let d = shape[3];
        assert_eq!(
            d as u32, params.head_dim,
            "input last dim ({d}) must match TurboQuantParams3 head_dim ({})",
            params.head_dim
        );
        assert_eq!(
            params.signs1.len(),
            d as usize,
            "signs1 length ({}) must match head_dim ({d})",
            params.signs1.len()
        );
        assert_eq!(
            params.signs2.len(),
            d as usize,
            "signs2 length ({}) must match head_dim ({d})",
            params.signs2.len()
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
        let indices_usize = nearest_centroid_indices_with_boundaries(
            &coords,
            &params.codebook.boundaries,
            n_centroids,
        );

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

    /// Host-readback dequantizer: the pre-#3274 body of
    /// [`super::dequantize_v_turbo3`].
    pub(crate) fn dequantize_v_turbo3_host(
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
        // Unconditional for the same reason as in `quantize_v_turbo3`: the inverse
        // rotation below hands `params.signs1` / `params.signs2` to
        // `ffi::from_slice_f32(signs, &[1, 1, 1, d])`, which reads `d` floats out
        // of a `signs.len()`-element host slice. Here `d` is derived from the
        // packed byte count, so nothing but this check ties it to the params the
        // caller passed.
        assert_eq!(
            d as u32, params.head_dim,
            "head_dim mismatch on dequantize: derived {d} from {bytes_per_token} packed bytes, \
             expected {}",
            params.head_dim
        );
        assert_eq!(
            d % 8,
            0,
            "head_dim must be a multiple of 8 for 3-bit dequant; got {d}"
        );
        assert_eq!(
            params.signs1.len(),
            d as usize,
            "signs1 length ({}) must match head_dim ({d})",
            params.signs1.len()
        );
        assert_eq!(
            params.signs2.len(),
            d as usize,
            "signs2 length ({}) must match head_dim ({d})",
            params.signs2.len()
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
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::pack3::unpack_3bit_per_token;
    use super::host_reference::{dequantize_v_turbo3_host, quantize_v_turbo3_host};
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

    // -- device codec against the host reference (#3274) --

    fn evaluated_bytes(arr: &MlxArray) -> Vec<u8> {
        ffi::eval(arr);
        ffi::array_to_raw_bytes(arr)
    }

    fn evaluated_f32(arr: &MlxArray) -> Vec<f32> {
        let a = ffi::astype(arr, dtype::FLOAT32);
        evaluated_bytes(&a)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c0(c)))
            .collect()
    }

    /// The rotated unit coordinates `quantize_v_turbo3` compares against the
    /// codebook boundaries, built with the same ops in the same order.
    fn rotated_unit_coords(v: &MlxArray, params: &TurboQuantParams3) -> Vec<f32> {
        let d = params.head_dim as i32;
        let v_f32 = ffi::astype(v, dtype::FLOAT32);
        let norm = ffi::sqrt(&ffi::sum_axis(&ffi::multiply(&v_f32, &v_f32), -1, true));
        let zero = ffi::full_f32(&[1], 0.0, dtype::FLOAT32);
        let one = ffi::full_f32(&[1], 1.0, dtype::FLOAT32);
        let safe = ffi::where_cond(&ffi::greater(&norm, &zero), &norm, &one);
        let unit = ffi::divide(&v_f32, &safe);
        let signs1 = ffi::from_slice_f32(&params.signs1, &[1, 1, 1, d]);
        let signs2 = ffi::from_slice_f32(&params.signs2, &[1, 1, 1, d]);
        let rot = ffi::multiply(&wht(&ffi::multiply(&unit, &signs1)), &signs2);
        evaluated_f32(&rot)
    }

    /// Deterministic `[B, H, T, D]` V data with token-varying magnitude over
    /// six decades, then a set of special tokens written over the first slots
    /// of every head (as many as fit in `t`):
    ///
    /// 0. the all-zero token (norm 0, every rotated coordinate 0),
    /// 1. a two-hot token, whose rotated coordinates are exactly `0` for half
    ///    the lanes: on the codebook's middle boundary,
    /// 2. a one-hot token (every rotated coordinate `±1/sqrt(D)`),
    /// 3. a constant token (one rotated coordinate carries all the energy),
    /// 4. a token at the smallest fp16 subnormal scale,
    /// 5. a token at the largest fp16 scale.
    ///
    /// Head `h` rotates the list by `h`, so a `T = 1` input with two heads
    /// holds both the all-zero and the on-boundary token.
    fn parity_input(b: i32, h: i32, t: i32, d: i32, seed: u32) -> Vec<f32> {
        let du = d as usize;
        let mut state = seed;
        let mut data = Vec::with_capacity((b * h * t * d) as usize);
        for token in 0..(b * h * t) {
            let scale = 10f32.powi(token % 7 - 3);
            for _ in 0..d {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                data.push(((state >> 8) as f32 / (1u32 << 23) as f32 - 1.0) * scale);
            }
        }
        let mut two_hot = vec![0.0_f32; du];
        two_hot[1] = 1.0;
        two_hot[du - 3] = 1.0;
        let mut one_hot = vec![0.0_f32; du];
        one_hot[du / 2] = -3.0;
        let specials: [Vec<f32>; 6] = [
            vec![0.0; du],
            two_hot,
            one_hot,
            vec![0.25; du],
            vec![6.0e-8; du],
            vec![6.0e4 / (du as f32).sqrt(); du],
        ];
        for bi in 0..b {
            for hi in 0..h {
                for slot in 0..(t.min(specials.len() as i32)) {
                    let which = ((slot + hi) as usize) % specials.len();
                    let off = (((bi * h + hi) * t + slot) * d) as usize;
                    data[off..off + du].copy_from_slice(&specials[which]);
                }
            }
        }
        data
    }

    /// Acceptance criterion of #3274: on fixed seeds, head dims 64, 128 and
    /// 256, `T = 1` and `T = 4096`, `B = 1`, `H = 2`, with an all-zero token
    /// and coordinates on a codebook boundary in the input, the device
    /// quantizer's packed bytes and norms are the host reference's bit for
    /// bit, and the device dequantizer's output is the host reference
    /// dequantizer's bit for bit. No tolerance anywhere.
    #[test]
    fn device_codec_is_bit_identical_to_the_host_reference() {
        for &d in &[64_i32, 128, 256] {
            let params = TurboQuantParams3::new(d as u32, 0x3274);
            // The middle boundary of the symmetric codebook is exactly zero,
            // which is what puts the all-zero and two-hot tokens' coordinates
            // on a boundary.
            assert_eq!(params.codebook.boundaries[3], 0.0, "head_dim {d}");
            for &t in &[1_i32, 4096] {
                let (b, h) = (1_i32, 2_i32);
                let data = parity_input(b, h, t, d, 0x5EED_0000 + d as u32 + t as u32);
                // The cache hands the quantizer fp16 values.
                let v = ffi::astype(&ffi::from_slice_f32(&data, &[b, h, t, d]), dtype::FLOAT16);

                // The fixture really has coordinates on a boundary: head 0
                // starts with the all-zero token, head 1 with the two-hot one.
                let rot = rotated_unit_coords(&v, &params);
                let du = d as usize;
                let token = |hi: usize, slot: usize| {
                    let off = (hi * t as usize + slot) * du;
                    &rot[off..off + du]
                };
                assert!(token(0, 0).iter().all(|&c| c == 0.0), "zero token");
                let on_boundary = token(1, 0).iter().filter(|&&c| c == 0.0).count();
                assert_eq!(on_boundary, du / 2, "two-hot token, head_dim {d}");

                let (packed, norms) = quantize_v_turbo3(&v, &params);
                let (packed_host, norms_host) = quantize_v_turbo3_host(&v, &params);
                assert_eq!(ffi::array_shape(&packed), vec![b, h, t, d * 3 / 8]);
                assert_eq!(ffi::array_dtype(&packed), dtype::UINT8);
                assert_eq!(ffi::array_shape(&norms), vec![b, h, t, 1]);
                assert_eq!(ffi::array_dtype(&norms), dtype::FLOAT16);
                let packed_bytes = evaluated_bytes(&packed);
                assert!(
                    packed_bytes == evaluated_bytes(&packed_host),
                    "packed bytes differ from the host reference (D={d}, T={t})"
                );
                assert!(
                    evaluated_bytes(&norms) == evaluated_bytes(&norms_host),
                    "norms differ from the host reference (D={d}, T={t})"
                );

                // A coordinate on the middle boundary takes the lower index.
                let indices = unpack_3bit_per_token(&packed_bytes, d, (b * h * t) as usize);
                assert!(indices[..du].iter().all(|&i| i == 3), "zero token index");
                let two_hot = &indices[t as usize * du..t as usize * du + du];
                for (k, &c) in token(1, 0).iter().enumerate() {
                    if c == 0.0 {
                        assert_eq!(two_hot[k], 3, "lane {k} sits on boundary 3");
                    }
                }

                let v_hat = dequantize_v_turbo3(&packed, &norms, &params);
                let v_hat_host = dequantize_v_turbo3_host(&packed, &norms, &params);
                assert_eq!(ffi::array_shape(&v_hat), vec![b, h, t, d]);
                assert_eq!(ffi::array_dtype(&v_hat), dtype::FLOAT16);
                assert!(
                    evaluated_bytes(&v_hat) == evaluated_bytes(&v_hat_host),
                    "dequantized V differs from the host reference (D={d}, T={t})"
                );
            }
        }
    }

    /// Every one of the seven boundaries, hit exactly, at the quantizer level.
    ///
    /// The real codebook's non-zero boundaries are not values a rotation
    /// produces on demand, so this test turns it around: it reads the rotated
    /// coordinates of a token and builds a codebook whose boundaries are seven
    /// of those very values. Each chosen coordinate then equals a boundary
    /// bit for bit and must take the lower index, on the device as on the
    /// host.
    #[test]
    fn coordinates_on_every_boundary_take_the_lower_index() {
        for &d in &[64_i32, 128, 256] {
            let base = TurboQuantParams3::new(d as u32, 0xB0DA);
            let data = parity_input(1, 1, 8, d, 0xFACE_0000 + d as u32);
            let v = ffi::from_slice_f32(&data[7 * d as usize..], &[1, 1, 1, d]);
            let rot = rotated_unit_coords(&v, &base);

            let mut distinct = rot.clone();
            distinct.sort_by(f32::total_cmp);
            distinct.dedup();
            assert!(distinct.len() >= 15, "need 15 distinct coordinates");
            // Seven boundaries spread over the sorted coordinates.
            let boundaries: Vec<f32> = (1..=7).map(|j| distinct[j * distinct.len() / 8]).collect();
            let centroids: Vec<f32> = (0..8)
                .map(|i| match i {
                    0 => boundaries[0] - 1.0,
                    7 => boundaries[6] + 1.0,
                    _ => (boundaries[i - 1] + boundaries[i]) * 0.5,
                })
                .collect();
            let params = TurboQuantParams3 {
                codebook: Codebook {
                    centroids: centroids.into(),
                    boundaries: boundaries.clone().into(),
                },
                ..base
            };

            let (packed, norms) = quantize_v_turbo3(&v, &params);
            let (packed_host, norms_host) = quantize_v_turbo3_host(&v, &params);
            let packed_bytes = evaluated_bytes(&packed);
            assert_eq!(packed_bytes, evaluated_bytes(&packed_host), "head_dim {d}");
            assert_eq!(
                evaluated_bytes(&norms),
                evaluated_bytes(&norms_host),
                "head_dim {d}"
            );

            let indices = unpack_3bit_per_token(&packed_bytes, d, 1);
            let mut hits = 0;
            for (k, &c) in rot.iter().enumerate() {
                if let Some(j) = boundaries.iter().position(|&bd| bd == c) {
                    assert_eq!(indices[k], j as u8, "lane {k} on boundary {j}");
                    hits += 1;
                }
                let expected = boundaries.iter().filter(|&&bd| bd < c).count();
                assert_eq!(indices[k] as usize, expected, "lane {k}");
            }
            assert!(hits >= 7, "every boundary is hit at least once; got {hits}");
        }
    }

    /// Batch and head axes above one, and a head dim that is not the cache's
    /// usual 128.
    #[test]
    fn device_codec_matches_host_reference_with_batch_and_heads() {
        for &(b, h, t, d) in &[
            (2_i32, 3_i32, 5_i32, 128_i32),
            (4, 8, 3, 64),
            (2, 1, 9, 256),
        ] {
            let params = TurboQuantParams3::new(d as u32, 11);
            let data = parity_input(b, h, t, d, 0xABCD_0000 + (b * h * t * d) as u32);
            let v = ffi::from_slice_f32(&data, &[b, h, t, d]);
            let (packed, norms) = quantize_v_turbo3(&v, &params);
            let (packed_host, norms_host) = quantize_v_turbo3_host(&v, &params);
            assert_eq!(evaluated_bytes(&packed), evaluated_bytes(&packed_host));
            assert_eq!(evaluated_bytes(&norms), evaluated_bytes(&norms_host));
            let v_hat = dequantize_v_turbo3(&packed, &norms, &params);
            let v_hat_host = dequantize_v_turbo3_host(&packed, &norms, &params);
            assert_eq!(evaluated_bytes(&v_hat), evaluated_bytes(&v_hat_host));
        }
    }

    /// The dequantizer on stored data no quantizer call produced: arbitrary
    /// packed bytes (every index in every lane) with norms that include zero,
    /// fp16 subnormals, the largest finite fp16 and infinity. The device and
    /// host results agree bit for bit, NaN payloads included.
    #[test]
    fn device_dequant_matches_host_reference_on_arbitrary_stored_bytes() {
        for &d in &[64_i32, 128, 256] {
            let params = TurboQuantParams3::new(d as u32, 5);
            let (b, h, t) = (1_i32, 2_i32, 40_i32);
            let bytes_per_token = packed_bytes_per_token(d);
            let mut state = 0xD15E_A5E5_u32 ^ d as u32;
            let packed: Vec<u8> = (0..(b * h * t * bytes_per_token) as usize)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (state >> 17) as u8
                })
                .collect();
            let special_norms = [0.0_f32, 6.0e-8, 1.0, 65504.0, f32::INFINITY, 1.0e-4];
            let norms: Vec<f32> = (0..(b * h * t) as usize)
                .map(|i| special_norms[i % special_norms.len()] * (1 + i / 6) as f32)
                .collect();
            let packed_arr = ffi::from_bytes(&packed, &[b, h, t, bytes_per_token], dtype::UINT8);
            let norms_arr =
                ffi::astype(&ffi::from_slice_f32(&norms, &[b, h, t, 1]), dtype::FLOAT16);
            let v_hat = dequantize_v_turbo3(&packed_arr, &norms_arr, &params);
            let v_hat_host = dequantize_v_turbo3_host(&packed_arr, &norms_arr, &params);
            assert_eq!(
                evaluated_bytes(&v_hat),
                evaluated_bytes(&v_hat_host),
                "head_dim {d}"
            );
        }
    }

    /// Zero tokens flow through both directions as zero-size tensors.
    #[test]
    fn zero_tokens_flow_through_the_device_codec() {
        for &(b, h, d) in &[(1_i32, 2_i32, 64_i32), (1, 8, 128), (2, 1, 256)] {
            let params = TurboQuantParams3::new(d as u32, 3);
            let v = ffi::zeros(&[b, h, 0, d], dtype::FLOAT16);
            let (packed, norms) = quantize_v_turbo3(&v, &params);
            assert_eq!(ffi::array_shape(&packed), vec![b, h, 0, d * 3 / 8]);
            assert_eq!(ffi::array_dtype(&packed), dtype::UINT8);
            assert_eq!(ffi::array_shape(&norms), vec![b, h, 0, 1]);
            assert_eq!(ffi::array_dtype(&norms), dtype::FLOAT16);
            assert!(evaluated_bytes(&packed).is_empty());
            assert!(evaluated_bytes(&norms).is_empty());

            let v_hat = dequantize_v_turbo3(&packed, &norms, &params);
            assert_eq!(ffi::array_shape(&v_hat), vec![b, h, 0, d]);
            assert_eq!(ffi::array_dtype(&v_hat), dtype::FLOAT16);
            assert!(evaluated_bytes(&v_hat).is_empty());
        }
    }

    /// The shapes the issue measures: a 16384-token prefill chunk through the
    /// quantizer and a 32768-token window through the dequantizer, at the
    /// Llama 3.1 8B layer shape (8 KV heads, head_dim 128). Both stay bit
    /// identical to the host reference. The device graph is a fixed number of
    /// elementwise ops, so its cost is linear in tokens.
    #[test]
    fn device_codec_matches_host_reference_at_measured_scale() {
        let (b, h, d) = (1_i32, 8_i32, 128_i32);
        let params = TurboQuantParams3::new(d as u32, 42);

        let t = 16_384_i32;
        let data = parity_input(b, h, t, d, 0x1638_4000);
        let v = ffi::astype(&ffi::from_slice_f32(&data, &[b, h, t, d]), dtype::FLOAT16);
        let (packed, norms) = quantize_v_turbo3(&v, &params);
        let (packed_host, norms_host) = quantize_v_turbo3_host(&v, &params);
        assert!(evaluated_bytes(&packed) == evaluated_bytes(&packed_host));
        assert!(evaluated_bytes(&norms) == evaluated_bytes(&norms_host));

        // A 32768-token window: the stored chunk twice over.
        let window = crate::ops::concatenate(&packed, &packed, 2);
        let window_norms = crate::ops::concatenate(&norms, &norms, 2);
        assert_eq!(ffi::array_shape(&window)[2], 32_768);
        let v_hat = dequantize_v_turbo3(&window, &window_norms, &params);
        let v_hat_host = dequantize_v_turbo3_host(&window, &window_norms, &params);
        assert!(evaluated_bytes(&v_hat) == evaluated_bytes(&v_hat_host));
    }

    /// The text of one top-level function of this file, from its `pub fn` line
    /// to the closing brace in column zero.
    fn production_fn_body<'a>(production: &'a str, signature: &str) -> &'a str {
        let start = production
            .find(signature)
            .unwrap_or_else(|| panic!("{signature} not found in quant3.rs"));
        let rest = &production[start..];
        let end = rest
            .find("\n}\n")
            .unwrap_or_else(|| panic!("end of {signature} not found"));
        &rest[..end + 3]
    }

    /// Source guard (#3274): neither codec function evaluates the graph or
    /// copies tensor data to the host. A regression that brings a readback
    /// back would still pass every parity test, because the bytes would be the
    /// same; only the source shows it.
    #[test]
    fn codec_bodies_have_no_eval_and_no_host_readback() {
        let source = include_str!("quant3.rs");
        // Cut at the first `#[cfg(test)]` attribute line: the host reference
        // module and the tests below it are allowed to read back.
        let production = source
            .split("\n#[cfg(test)]\n")
            .next()
            .expect("split yields at least one piece");
        assert!(
            production.len() < source.len(),
            "quant3.rs must keep its test-only code behind #[cfg(test)]"
        );
        for signature in ["pub fn quantize_v_turbo3(", "pub fn dequantize_v_turbo3("] {
            let body = production_fn_body(production, signature);
            assert!(
                body.contains("_on_device("),
                "{signature} does not call a device helper; was it cut correctly?"
            );
            for forbidden in [
                "ffi::eval",
                "array_to_raw_bytes",
                "async_eval",
                "from_bytes",
                "synchronize",
                "item_",
            ] {
                assert!(
                    !body.contains(forbidden),
                    "{signature} must stay on the device, but its body contains `{forbidden}`"
                );
            }
        }
        // The helpers the bodies call are held to the same rule.
        for (name, text) in [
            ("pack3.rs", include_str!("pack3.rs")),
            ("codebook.rs", include_str!("codebook.rs")),
        ] {
            let production = text.split("\n#[cfg(test)]\n").next().unwrap();
            for signature in [
                "pub(crate) fn pack_3bit_on_device(",
                "pub(crate) fn unpack_3bit_on_device(",
                "pub(crate) fn boundary_indices_on_device(",
            ] {
                let Some(start) = production.find(signature) else {
                    continue;
                };
                let rest = &production[start..];
                let body = &rest[..rest.find("\n}\n").expect("helper end") + 3];
                for forbidden in ["ffi::eval", "array_to_raw_bytes", "from_bytes"] {
                    assert!(
                        !body.contains(forbidden),
                        "{name}: {signature} contains `{forbidden}`"
                    );
                }
            }
        }
    }
}
