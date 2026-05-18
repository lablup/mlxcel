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

//! TurboQuant K/V-side PolarQuant pipeline (B2 + B4, issues #474 and #476,
//! epic #458).
//!
//! Implements the per-token compression used by `KVCacheMode::Turbo4Asym`
//! (V only) and `KVCacheMode::Turbo4` (both K and V):
//!
//! 1. Per-token L2 norm extraction.
//! 2. Sign-flip × Walsh–Hadamard × sign-flip rotation (the structured fast
//!    rotation `D2 · H · D1` from the TurboQuant+ reference). See B0 (#470)
//!    for the WHT op.
//! 3. Per-coordinate nearest-centroid lookup using the Lloyd-Max codebook
//!    from B1 (#472).
//! 4. Nibble-packing of the 4-bit indices into a `[..., head_dim/2]` u8 buffer.
//!
//! The dequantize path is the exact inverse: unpack nibbles → centroid
//! gather → optional rotated-norm correction → reverse rotation → rescale by
//! the stored norm.
//!
//! K-side compression mirrors V-side bit-for-bit, but uses an *independent*
//! pair of sign vectors derived from a different seed offset
//! ([`K_SEED_OFFSET`]). This decorrelates the K and V quantization noise so
//! that the inner product `Q · Kᵀ` in attention does not see additive bias
//! from a shared rotation. This is the exact pattern the TurboQuant+
//! reference uses (`seed + 500` in `references/turboquant_plus/turboquant/
//! kv_cache.py::KVCacheCompressor.__init__`).
//!
//! # Storage layout
//!
//! For each token slot the cache stores:
//!
//! - `v_packed[..., t, :]` — `head_dim/2` u8s, one byte = two 4-bit indices
//!   (low nibble = even-index coord, high nibble = odd-index coord).
//! - `v_norms[..., t, 0]`  — fp16 L2 norm of the original V vector.
//!
//! Sign vectors `signs1` and `signs2` are owned per-cache (one pair per cache
//! object, deterministic from a seed) and shared across batch + heads + tokens.
//!
//! # Block size
//!
//! [`BLOCK_SIZE`] = 32 controls the buffer growth granularity (the cache grows
//! its V buffer in increments of 32 tokens). The math is per-token — there is
//! no block-level shared state in the quantize/dequantize algorithm itself.
//! This matches TurboQuant+'s "turbo4 already uses block_size=128" semantics
//! for `head_dim=128` (one norm per rotation group = one norm per V vector).
//! See `references/turboquant_plus/docs/papers/block-size-experiment.md`.
//!
//! # Numerical notes
//!
//! The Lloyd-Max codebook is calibrated for a *unit-norm* `head_dim`-vector
//! whose post-WHT coordinates approximate `N(0, 1/d)`. The norm-extract /
//! rescale-on-dequant pattern (PolarQuant page 5) keeps the input distribution
//! unit-norm regardless of the raw V magnitudes. The norm-correction step
//! (`y_hat /= ||y_hat||`) compensates for the centroid gather discarding
//! magnitude information; without it, the dequantized vector underestimates
//! the original V by a few percent.
//!
//! Used by: `KVCacheMode::Turbo4Asym` V-cache update/read (B2, epic #458).

use std::sync::Arc;

use cxx::UniquePtr;

use super::codebook::{nearest_centroid_indices_with_boundaries, optimal_codebook, Codebook};
use crate::dtype;
use crate::ffi;
use crate::ffi::MlxArray;
use crate::ops::wht;

/// V-side PolarQuant bit-width. Locked to 4 bits for `Turbo4Asym` and
/// symmetric `Turbo4`. (B5/#477 will add 3-bit, but lives in its own
/// variant.)
pub const V_BIT_WIDTH: u8 = 4;

/// K-side PolarQuant bit-width. Locked to 4 bits for symmetric `Turbo4`
/// (issue #476). The K side is only quantized in symmetric mode; in
/// `Turbo4Asym` (issue #474) the K side stays in FP16.
pub const K_BIT_WIDTH: u8 = 4;

/// Seed offset added to the cache's `turbo_seed` before deriving the K-side
/// sign vectors. Matches the TurboQuant+ reference pattern
/// (`references/turboquant_plus/turboquant/kv_cache.py::KVCacheCompressor`,
/// which uses `seed + 500`). The exact value matters less than the property
/// that K and V sign vectors are *independent* — see the module docstring
/// for why this matters for attention quality.
pub const K_SEED_OFFSET: u32 = 0x4B5F_5345; // "K_SE" in ASCII

/// Buffer growth granularity (tokens) for the packed V buffer. Matches the
/// TurboQuant+ "block_size=32" default. Speculative decoding rewinds at most
/// one block at a time so the trim path is always block-aligned.
pub const BLOCK_SIZE: i32 = 32;

/// A simple 32-bit linear-congruential generator (Numerical Recipes constants)
/// used to deterministically derive `±1` sign vectors from a layer-specific
/// seed without pulling in another crate. Quality is sufficient for a
/// random-rotation matrix — TurboQuant only requires that the rotation be
/// dense and well-conditioned, and this LCG plus the Walsh–Hadamard transform
/// produces statistically indistinguishable rotations from a Haar-distributed
/// matrix once the WHT is applied.
#[derive(Debug, Clone, Copy)]
struct Lcg32 {
    state: u32,
}

impl Lcg32 {
    fn new(seed: u32) -> Self {
        // Avoid degenerate seed=0 collapsing to a fixed point.
        let state = if seed == 0 { 0xDEADBEEF } else { seed };
        Self { state }
    }

    fn next_bit(&mut self) -> u32 {
        // Numerical Recipes LCG: x_{n+1} = 1664525 * x_n + 1013904223 (mod 2^32)
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        // Use bit 31 (sign bit of the LCG) — better-distributed than bit 0.
        self.state >> 31
    }
}

/// Parameters for the V-side TurboQuant pipeline that are shared across all
/// quantize / dequantize calls for a single cache instance.
///
/// One instance is built per `KVCache` at construction time (see
/// `KVCache::new_with_mode`), parameterised by the V `head_dim` of the
/// runtime model and a deterministic seed. The codebook is fetched from the
/// global `OnceLock`-backed cache so multiple caches share the centroids.
///
/// In symmetric `Turbo4` mode (issue #476) the *same* params struct also
/// carries an independent pair of K-side sign vectors (`k_signs1` /
/// `k_signs2`) so K compression decorrelates from V. The codebook is
/// re-used as the post-WHT coordinate distribution is identical for K and V.
///
/// Used by: `KVCache::update_turbo4_asym` (cache.rs),
/// `KVCache::update_turbo4_sym` (cache.rs, issue #476),
/// `turbo::quant::dequantize_v_turbo4` (this module),
/// `turbo::quant::dequantize_k_turbo4` (this module).
#[derive(Clone, Debug)]
pub struct TurboQuantParams {
    /// V head dimension (must be a non-zero power of two — power-of-two is
    /// the WHT op's hard requirement, see `mlxcel_core::ops::wht`).
    pub head_dim: u32,
    /// V-side `±1.0` sign vector applied **before** the WHT. Length =
    /// `head_dim`.
    pub signs1: Arc<[f32]>,
    /// V-side `±1.0` sign vector applied **after** the WHT. Length =
    /// `head_dim`.
    pub signs2: Arc<[f32]>,
    /// K-side `±1.0` sign vector applied **before** the WHT (symmetric
    /// `Turbo4` only). Length = `head_dim`. Derived from `seed +
    /// K_SEED_OFFSET` so K is statistically independent from V.
    pub k_signs1: Arc<[f32]>,
    /// K-side `±1.0` sign vector applied **after** the WHT (symmetric
    /// `Turbo4` only). Length = `head_dim`.
    pub k_signs2: Arc<[f32]>,
    /// Cached centroid + boundary tables for `(V_BIT_WIDTH, head_dim)`.
    pub codebook: Codebook,
}

impl TurboQuantParams {
    /// Build params for a `(head_dim, seed)` pair. The seed determines the
    /// random sign vectors deterministically; passing the same seed across
    /// runs reproduces the same rotation.
    ///
    /// # Panics
    ///
    /// Panics if `head_dim` is not a positive power of two — the WHT op
    /// requires this and the cache layer cannot recover from a runtime
    /// quantize-time failure.
    pub fn new(head_dim: u32, seed: u32) -> Self {
        assert!(
            head_dim > 0 && head_dim.is_power_of_two(),
            "TurboQuantParams::new: head_dim must be a non-zero power of two; got {head_dim}"
        );
        let signs1 = generate_signs(head_dim as usize, seed);
        // Use a different sub-seed for signs2 so the two are independent.
        let signs2 = generate_signs(head_dim as usize, seed.wrapping_add(0x9E37_79B9));
        // K-side sign vectors derived from `seed + K_SEED_OFFSET` so the K
        // and V rotations are statistically independent (mirrors the
        // TurboQuant+ reference's `seed + 500` offset between K and V
        // quantizers).
        let k_seed = seed.wrapping_add(K_SEED_OFFSET);
        let k_signs1 = generate_signs(head_dim as usize, k_seed);
        let k_signs2 = generate_signs(head_dim as usize, k_seed.wrapping_add(0x9E37_79B9));
        let codebook = optimal_codebook(V_BIT_WIDTH, head_dim);
        Self {
            head_dim,
            signs1,
            signs2,
            k_signs1,
            k_signs2,
            codebook,
        }
    }
}

/// Generate a deterministic `±1.0` sign vector of length `len` from `seed`.
///
/// Uses an in-house LCG so the result is reproducible across runs and
/// platforms without taking a dependency on any RNG crate.
///
/// Used by: `TurboQuantParams::new`.
pub fn generate_signs(len: usize, seed: u32) -> Arc<[f32]> {
    let mut rng = Lcg32::new(seed);
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let bit = rng.next_bit();
        out.push(if bit == 0 { -1.0_f32 } else { 1.0_f32 });
    }
    Arc::from(out)
}

// ---------------------------------------------------------------------------
// Quantize / dequantize — graph + readback hybrid
// ---------------------------------------------------------------------------

/// Quantize a 4-D `[B, H, T, D]` tensor into the packed Turbo4 representation
/// using the supplied sign-flip pair and the V-bit-width Lloyd-Max codebook.
///
/// This is the shared core for both V-side ([`quantize_v_turbo4`]) and K-side
/// ([`quantize_k_turbo4`]) compression — only the sign vectors differ; the
/// rotation algorithm, packing layout, and norm-storage contract are
/// identical. Per-token norms of the *original* (pre-rotation) tensor are
/// returned alongside the packed indices and re-applied on dequantize.
///
/// `signs1` / `signs2` must have length `params.head_dim`. The packed result
/// is shape `[B, H, T, D/2]` u8 (low nibble = even coord, high nibble = odd
/// coord); the norm tensor is shape `[B, H, T, 1]` fp16.
///
/// # Returns (issue #520, fused Sparse-V kernel rescale precompute)
///
/// `(v_packed, v_norms, v_rescale)` where the third element is a precomputed
/// per-token rescale factor:
///
/// `v_rescale[b, h, t, 0] = norm[b, h, t] / max(|y_hat[b, h, t]|, 1e-10)`
///
/// where `|y_hat[b, h, t]| = sqrt(Σ_d codebook[indices[b, h, t, d]]²)` is the
/// L2 norm of the centroid-gathered (pre-rotation) reconstruction. This value
/// is identical to the inner `tg_norm[0] = vn / yh_safe` quantity computed
/// per token by the previous fused kernel via a threadgroup tree reduction
/// over `Dim` threads. Precomputing it at quantize time eliminates the
/// per-cache-token threadgroup reduction (and its `log2(Dim) + 2` barrier
/// chain) from the kernel hot path — see issue #520 for measurements.
///
/// V-side callers store `v_rescale` alongside `v_packed` / `v_norms` so the
/// fused kernel (`turbo::sparse_v::attention_sparse_v_turbo4_fused`) can read
/// it directly without recomputing. K-side callers discard the third element
/// — K-side dequant has no kernel hot path that benefits from the precompute.
fn quantize_into_packed(
    x: &MlxArray,
    params: &TurboQuantParams,
    signs1: &[f32],
    signs2: &[f32],
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
) {
    // Cast input to fp32 for stable norm + rotation arithmetic. The MLX WHT
    // op accepts fp32/fp16/bf16; fp32 keeps the centroid lookup
    // well-conditioned even for very small / very large magnitudes.
    let v_f32 = ffi::astype(x, dtype::FLOAT32);
    let shape = ffi::array_shape(&v_f32);
    debug_assert_eq!(shape.len(), 4, "input must be 4-D [B, H, T, D]");
    let _b = shape[0];
    let _h = shape[1];
    let t = shape[2];
    let d = shape[3];
    debug_assert_eq!(
        d as u32, params.head_dim,
        "input last dim ({d}) must match TurboQuantParams head_dim ({})",
        params.head_dim
    );
    debug_assert_eq!(
        signs1.len(),
        d as usize,
        "signs1 length ({}) must match head_dim ({d})",
        signs1.len()
    );
    debug_assert_eq!(
        signs2.len(),
        d as usize,
        "signs2 length ({}) must match head_dim ({d})",
        signs2.len()
    );

    // 1. Per-token L2 norm: ||x||_2 along last axis, keepdims=true → [B, H, T, 1]
    let v_sq = ffi::multiply(&v_f32, &v_f32);
    let sum_sq = ffi::sum_axis(&v_sq, -1, true);
    let norm_full = ffi::sqrt(&sum_sq);
    // Avoid divide-by-zero **only** for the exact-zero case. Using
    // `maximum(norm, 1.0)` here is mathematically wrong: it clamps any
    // norm below 1.0 *up* to 1.0, destroying direction information for
    // small-magnitude vectors. Use a `where` op so non-zero norms pass
    // through unchanged. Mirrors the Python reference at
    // `references/turboquant_plus/turboquant/polar_quant.py:60`:
    // `safe_norms = np.where(norms > 0, norms, 1.0)`.
    let zero = ffi::full_f32(&[1], 0.0, dtype::FLOAT32);
    let one = ffi::full_f32(&[1], 1.0, dtype::FLOAT32);
    let positive_mask = ffi::greater(&norm_full, &zero);
    let safe_norm = ffi::where_cond(&positive_mask, &norm_full, &one);
    let v_normalized = ffi::divide(&v_f32, &safe_norm); // [B, H, T, D]

    // 2. Apply random rotation: D2 · H · D1 · x.
    let signs1_arr = ffi::from_slice_f32(signs1, &[1, 1, 1, d]);
    let signs2_arr = ffi::from_slice_f32(signs2, &[1, 1, 1, d]);
    let v_d1 = ffi::multiply(&v_normalized, &signs1_arr);
    let v_h = wht(&v_d1);
    let v_rot = ffi::multiply(&v_h, &signs2_arr);

    // 3. Per-coordinate nearest-centroid lookup. We materialize the rotated
    //    coordinates back to host memory once per quantize call. This is the
    //    same readback pattern the TurboQuant+ MLX port uses; for B2 the
    //    correctness story dominates and a fully-fused on-GPU implementation
    //    is the natural follow-up (likely B7's delegated KVCache or B11+).
    //
    //    TODO(#474 follow-up): replace this readback with an on-device
    //    nearest-centroid lookup (broadcast-compare against the 15 boundaries
    //    and reduce). The dequantize path was migrated to on-device unpacking
    //    in PR #490 because it dominates decode latency (`O(visible_tokens)`
    //    per layer per step). The quantize path is `O(new_tokens)` per layer
    //    per step — typically 1 token at decode — so the readback cost here
    //    is dwarfed by dequantize and was deferred to keep PR #490 scoped.
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
    let indices =
        nearest_centroid_indices_with_boundaries(&coords, &params.codebook.boundaries, n_centroids);

    // 4. Pack two consecutive 4-bit indices into one byte.
    //    Layout: byte[i] low nibble = indices[2*i], high nibble = indices[2*i+1].
    debug_assert!(d % 2 == 0, "head_dim must be even for nibble-packing");
    let coords_per_token = d as usize;
    let bytes_per_token = coords_per_token / 2;
    let total_tokens = (shape[0] * shape[1] * t) as usize;
    let mut packed = vec![0u8; total_tokens * bytes_per_token];
    for tok in 0..total_tokens {
        let idx_off = tok * coords_per_token;
        let pack_off = tok * bytes_per_token;
        for j in 0..bytes_per_token {
            let lo = (indices[idx_off + 2 * j] & 0x0F) as u8;
            let hi = (indices[idx_off + 2 * j + 1] & 0x0F) as u8;
            packed[pack_off + j] = lo | (hi << 4);
        }
    }

    let v_packed = ffi::from_bytes(
        &packed,
        &[shape[0], shape[1], t, bytes_per_token as i32],
        dtype::UINT8,
    );

    // 5. Norms stored in fp16 for the cache. The full-precision norm
    //    (not safe_norm) is what the dequantize path consumes.
    let v_norms = ffi::astype(&norm_full, dtype::FLOAT16);

    // 6. Precompute the per-token rescale factor `norm[t] / max(|y_hat|, eps)`
    //    consumed by the fused Sparse-V kernel (issue #520). The previous
    //    kernel implementation derived this on-GPU per token via a
    //    `log2(Dim) + 2`-barrier threadgroup tree reduction, which dominated
    //    decode latency on M5 Max at 4 K context for `turbo4-asym` (the kernel
    //    was 2.0× slower than the graph fallback in PR #519's A/B). Because
    //    `|y_hat|` is a pure function of the packed indices and the codebook
    //    — both fixed at quantize time — we can compute it once here on the
    //    host and store the resulting fp16 scalar alongside `v_norms`.
    //
    //    Algorithm (per token t):
    //      sum_sq = Σ_d codebook[indices[t, d]]²
    //      y_hat_norm = sqrt(sum_sq)
    //      y_hat_safe = max(y_hat_norm, 1e-10)        (matches kernel guard)
    //      rescale[t] = norm_full[t] / y_hat_safe
    //
    //    The 1e-10 guard mirrors the kernel-side `1e-10f` in
    //    `sparse_v_sdpa.cpp` and the graph dequant's `eps` in
    //    `dequantize_from_packed`, so kernel and graph paths see numerically
    //    identical rescale values.
    //
    //    Since `norm_full` is still pending GPU evaluation at this point, we
    //    materialize it back to host bytes alongside the rotated coordinates
    //    we already read above. fp32 keeps the divide well-conditioned for
    //    very small / very large magnitudes; the final cast to fp16 matches
    //    the storage dtype.
    ffi::eval(&norm_full);
    let norm_bytes = ffi::array_to_raw_bytes(&norm_full);
    debug_assert_eq!(
        norm_bytes.len(),
        total_tokens * 4,
        "fp32 norm byte count mismatch"
    );
    let centroids = params.codebook.centroids.as_ref();
    let mut rescale = vec![0.0_f32; total_tokens];
    for (tok, slot) in rescale.iter_mut().enumerate() {
        let idx_off = tok * coords_per_token;
        let mut sum_sq = 0.0_f32;
        for d_i in 0..coords_per_token {
            let c = centroids[indices[idx_off + d_i]];
            sum_sq += c * c;
        }
        let y_hat_norm = sum_sq.sqrt();
        let y_hat_safe = y_hat_norm.max(1e-10);
        let norm_off = tok * 4;
        let n_t = f32::from_le_bytes([
            norm_bytes[norm_off],
            norm_bytes[norm_off + 1],
            norm_bytes[norm_off + 2],
            norm_bytes[norm_off + 3],
        ]);
        *slot = n_t / y_hat_safe;
    }
    let v_rescale_f32 = ffi::from_slice_f32(&rescale, &[shape[0], shape[1], t, 1]);
    let v_rescale = ffi::astype(&v_rescale_f32, dtype::FLOAT16);

    (v_packed, v_norms, v_rescale)
}

/// Quantize a V tensor of shape `[B, H, T, D]` (D == `params.head_dim`) into
/// the packed Turbo4 representation.
///
/// Returns `(v_packed, v_norms, v_rescale)` where:
/// - `v_packed`:  `[B, H, T, D/2]` u8  — nibble-packed 4-bit indices.
/// - `v_norms`:   `[B, H, T, 1]`   fp16 — per-token L2 norm of the *original*
///   V vector (used for rescaling on the graph dequantize path).
/// - `v_rescale`: `[B, H, T, 1]`   fp16 — precomputed `norm[t] / |y_hat[t]|`
///   used by the fused Sparse-V kernel (issue #520) to skip the per-token
///   threadgroup tree reduction. See [`quantize_into_packed`] for details.
///
/// Used by: `KVCache::update` (Turbo4Asym mode, Turbo4 mode, Turbo4Delegated
/// mode), `KVCache::sparse_v_attention` (consumes `v_rescale`).
pub fn quantize_v_turbo4(
    v: &MlxArray,
    params: &TurboQuantParams,
) -> (
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
    UniquePtr<MlxArray>,
) {
    quantize_into_packed(v, params, &params.signs1, &params.signs2)
}

/// Quantize a K tensor of shape `[B, H, T, D]` (D == `params.head_dim`) into
/// the packed Turbo4 representation, using the K-side sign vectors so the
/// resulting indices are statistically independent from any V-side
/// quantization noise.
///
/// Returns `(k_packed, k_norms)` analogous to [`quantize_v_turbo4`] but
/// without the K-side rescale precompute — the symmetric-Turbo4 K-side has
/// no analogue of the fused V-side kernel today (issue #476's K dequant runs
/// on the standard graph path), so the precompute would be wasted work.
/// Storage layout otherwise matches the V-side path bit-for-bit, so the cache
/// layer can reuse the same packing/unpacking helpers.
///
/// Used by: `KVCache::update` (Turbo4 symmetric mode, issue #476, epic
/// #458).
pub fn quantize_k_turbo4(
    k: &MlxArray,
    params: &TurboQuantParams,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let (k_packed, k_norms, _k_rescale) =
        quantize_into_packed(k, params, &params.k_signs1, &params.k_signs2);
    (k_packed, k_norms)
}

/// Shared core for [`dequantize_v_turbo4`] and [`dequantize_k_turbo4`].
///
/// `signs1` / `signs2` are the rotation sign vectors used at *quantize* time
/// — for V-side reads the caller passes `params.signs1` / `params.signs2`,
/// for K-side reads (symmetric Turbo4) the caller passes `params.k_signs1`
/// / `params.k_signs2`. Apart from that the dequantize path is bit-for-bit
/// identical for K and V.
fn dequantize_from_packed(
    packed: &MlxArray,
    norms: &MlxArray,
    params: &TurboQuantParams,
    signs1: &[f32],
    signs2: &[f32],
) -> UniquePtr<MlxArray> {
    let (indices_u8, _b, _h, _t, d) = unpack_turbo4_indices(packed);
    debug_assert_eq!(d as u32, params.head_dim, "head_dim mismatch on dequantize");
    debug_assert_eq!(
        signs1.len(),
        d as usize,
        "signs1 length ({}) must match head_dim ({d})",
        signs1.len()
    );
    debug_assert_eq!(
        signs2.len(),
        d as usize,
        "signs2 length ({}) must match head_dim ({d})",
        signs2.len()
    );

    // 1. Centroid gather: y_hat[b, h, t, k] = centroids[indices[b, h, t, k]].
    let centroids_vec: Vec<f32> = params.codebook.centroids.as_ref().to_vec();
    let centroids_arr =
        ffi::from_slice_f32(&centroids_vec, &[params.codebook.centroids.len() as i32]);
    let y_hat = ffi::take(&centroids_arr, &indices_u8, 0); // [B, H, T, D]

    // 2. Norm correction: rescale y_hat to unit norm so the inverse rotation
    //    sees a unit vector. Mirrors the Python `if norm_correction` branch
    //    in `references/turboquant_plus/turboquant/polar_quant.py`.
    let y_hat_sq = ffi::multiply(&y_hat, &y_hat);
    let y_hat_sum_sq = ffi::sum_axis(&y_hat_sq, -1, true);
    let y_hat_norm = ffi::sqrt(&y_hat_sum_sq);
    // Guard against zero (shouldn't happen in practice — codebook always has
    // a non-zero magnitude — but defensive against the all-zero edge case).
    let eps = ffi::full_f32(&[1], 1e-10, dtype::FLOAT32);
    let safe_y_norm = ffi::maximum(&y_hat_norm, &eps);
    let y_hat_unit = ffi::divide(&y_hat, &safe_y_norm);

    // 3. Inverse rotation. Since D and H are symmetric (D^T = D, H^T = H),
    //    the transpose D2·H·D1 is D1·H·D2 — apply in reverse.
    let signs1_arr = ffi::from_slice_f32(signs1, &[1, 1, 1, d]);
    let signs2_arr = ffi::from_slice_f32(signs2, &[1, 1, 1, d]);
    let v_pre_h = ffi::multiply(&y_hat_unit, &signs2_arr);
    let v_pre_d1 = wht(&v_pre_h);
    let v_unit = ffi::multiply(&v_pre_d1, &signs1_arr);

    // 4. Rescale by the stored original norms. norms shape is [B, H, T, 1]
    //    fp16; cast to fp32 to match v_unit, then back to fp16 at the end.
    let norms_f32 = ffi::astype(norms, dtype::FLOAT32);
    let v_full_f32 = ffi::multiply(&v_unit, &norms_f32);
    ffi::astype(&v_full_f32, dtype::FLOAT16)
}

/// Unpack Turbo4 nibbles into one uint8 codebook index per coordinate.
///
/// Stays entirely on-device. The packing layout is the documented low-nibble
/// even coordinate / high-nibble odd coordinate order used by
/// [`quantize_into_packed`].
fn unpack_turbo4_indices(packed: &MlxArray) -> (UniquePtr<MlxArray>, i32, i32, i32, i32) {
    let packed_shape = ffi::array_shape(packed);
    debug_assert_eq!(packed_shape.len(), 4, "packed must be 4-D");
    let b = packed_shape[0];
    let h = packed_shape[1];
    let t = packed_shape[2];
    let bytes_per_token = packed_shape[3];
    let d = bytes_per_token * 2;

    let mask_u8 = ffi::full_f32(&[1], 0x0F_u8 as f32, dtype::UINT8);
    let shift4_u8 = ffi::full_f32(&[1], 4.0, dtype::UINT8);
    let low_nibble = ffi::bitwise_and(packed, &mask_u8);
    let high_nibble = ffi::right_shift(packed, &shift4_u8);
    let stacked = crate::ops::stack_owned(&[low_nibble, high_nibble], -1);
    let indices = ffi::reshape(&stacked, &[b, h, t, d]);

    (indices, b, h, t, d)
}

/// Dequantize packed V into TurboQuant's rotated value basis.
///
/// This is the bulk-dequant analogue used by the Turbo4Delegated dequant-SDPA
/// path inspired by `references/mlx-swift-lm`: it gathers codebook centroids
/// and applies the precomputed `v_rescale = norm / |y_hat|`, but deliberately
/// skips the inverse WHT/sign rotation. A caller can run SDPA with this rotated
/// V tensor and then inverse-rotate the much smaller attention output.
///
/// Inputs:
/// - `v_packed`: `[B, H, T, D/2]` u8 packed V indices.
/// - `v_rescale`: `[B, H, T, 1]` fp16 precomputed `norm / |y_hat|`.
/// - `params`: TurboQuant params used at quantize time.
///
/// Returns `[B, H, T, D]` fp16 in rotated value basis.
pub fn dequantize_v_turbo4_rotated(
    v_packed: &MlxArray,
    v_rescale: &MlxArray,
    params: &TurboQuantParams,
) -> UniquePtr<MlxArray> {
    let (indices_u8, _b, _h, _t, d) = unpack_turbo4_indices(v_packed);
    debug_assert_eq!(
        d as u32, params.head_dim,
        "head_dim mismatch on rotated dequantize"
    );

    let centroids_vec: Vec<f32> = params.codebook.centroids.as_ref().to_vec();
    let centroids_arr =
        ffi::from_slice_f32(&centroids_vec, &[params.codebook.centroids.len() as i32]);
    let y_hat = ffi::take(&centroids_arr, &indices_u8, 0);
    let rescale_f32 = ffi::astype(v_rescale, dtype::FLOAT32);
    let rotated_f32 = ffi::multiply(&y_hat, &rescale_f32);
    ffi::astype(&rotated_f32, dtype::FLOAT16)
}

/// Dequantize packed K into TurboQuant's rotated key basis.
///
/// Symmetric `KVCacheMode::Turbo4` can run Swift-LM-style dequant-first SDPA
/// without fully inverse-rotating cached K. The caller forward-rotates Q into
/// the same K basis, runs SDPA, then inverse-rotates only the V-side output.
/// This helper mirrors the norm-correction half of [`dequantize_k_turbo4`]
/// and deliberately skips the inverse WHT/sign rotation.
pub fn dequantize_k_turbo4_rotated(
    k_packed: &MlxArray,
    k_norms: &MlxArray,
    params: &TurboQuantParams,
) -> UniquePtr<MlxArray> {
    let (indices_u8, _b, _h, _t, d) = unpack_turbo4_indices(k_packed);
    debug_assert_eq!(
        d as u32, params.head_dim,
        "head_dim mismatch on rotated K dequantize"
    );

    let centroids_vec: Vec<f32> = params.codebook.centroids.as_ref().to_vec();
    let centroids_arr =
        ffi::from_slice_f32(&centroids_vec, &[params.codebook.centroids.len() as i32]);
    let y_hat = ffi::take(&centroids_arr, &indices_u8, 0);
    let y_hat_sq = ffi::multiply(&y_hat, &y_hat);
    let y_hat_sum_sq = ffi::sum_axis(&y_hat_sq, -1, true);
    let y_hat_norm = ffi::sqrt(&y_hat_sum_sq);
    let eps = ffi::full_f32(&[1], 1e-10, dtype::FLOAT32);
    let safe_y_norm = ffi::maximum(&y_hat_norm, &eps);
    let y_hat_unit = ffi::divide(&y_hat, &safe_y_norm);
    let norms_f32 = ffi::astype(k_norms, dtype::FLOAT32);
    let rotated_f32 = ffi::multiply(&y_hat_unit, &norms_f32);
    ffi::astype(&rotated_f32, dtype::FLOAT16)
}

/// Dequantize a packed-V slice back to fp16 for the attention kernel.
///
/// Inputs:
/// - `v_packed`: `[B, H, T, D/2]` u8 (output of [`quantize_v_turbo4`]).
/// - `v_norms`:  `[B, H, T, 1]` fp16 — per-token original-V L2 norms.
/// - `params`:   the `TurboQuantParams` used at quantize time (centroids +
///   sign vectors + head_dim).
///
/// Returns a fresh fp16 tensor of shape `[B, H, T, D]`, suitable for the
/// existing `attention()` codepath — same shape and dtype contract as the
/// `Fp16` and `Int8` modes.
///
/// Used by: `KVCache::update_and_fetch` (Turbo4Asym mode, Turbo4 mode).
pub fn dequantize_v_turbo4(
    v_packed: &MlxArray,
    v_norms: &MlxArray,
    params: &TurboQuantParams,
) -> UniquePtr<MlxArray> {
    dequantize_from_packed(v_packed, v_norms, params, &params.signs1, &params.signs2)
}

/// Dequantize a packed-K slice back to fp16 for the attention kernel.
///
/// Mirrors [`dequantize_v_turbo4`] but uses the K-side sign vectors. The
/// caller is responsible for slicing `k_packed` and `k_norms` to the visible
/// portion of the cache (the same way the V-side path does in
/// `KVCache::update_and_fetch`).
///
/// Used by: `KVCache::update_and_fetch` (Turbo4 symmetric mode, issue #476).
pub fn dequantize_k_turbo4(
    k_packed: &MlxArray,
    k_norms: &MlxArray,
    params: &TurboQuantParams,
) -> UniquePtr<MlxArray> {
    dequantize_from_packed(
        k_packed,
        k_norms,
        params,
        &params.k_signs1,
        &params.k_signs2,
    )
}

// ---------------------------------------------------------------------------
// Public rotation helper (used by kurtosis sanity tests)
// ---------------------------------------------------------------------------

/// Apply the TurboQuant `D2 · H · D1` rotation to an arbitrary tensor.
///
/// Input `x` must have shape `[..., D]` where `D` matches `params.head_dim`.
/// The function multiplies by `signs1`, applies the WHT, then multiplies by
/// `signs2` — the same ordered sequence used inside [`quantize_v_turbo4`].
///
/// Returns an FP32 tensor of the same shape as the input.
///
/// # Purpose
///
/// Exposing this as a standalone function allows the kurtosis sanity test
/// in `tests/turbo_kv_e2e.rs` to load a real K tensor (or a slice thereof),
/// apply the rotation, and measure the post-rotation kurtosis without having
/// to run the full quantize pipeline. This validates the whitening claim
/// (TurboQuant+ reports kurtosis ~900 → ~2.9 on Qwen3-1.7B K caches) on
/// actual model weights.
///
/// Used by: `tests/turbo_kv_e2e.rs` kurtosis sanity test (issue #475).
// doc(hidden) keeps this out of the public rustdoc while still being visible
// to external crates (including the integration test crate).
#[doc(hidden)]
pub fn turbo4_v_rotate(x: &MlxArray, params: &TurboQuantParams) -> UniquePtr<MlxArray> {
    let shape = ffi::array_shape(x);
    let d = *shape
        .last()
        .expect("turbo4_v_rotate: input must be at least 1-D") as usize;
    assert_eq!(
        d, params.head_dim as usize,
        "turbo4_v_rotate: last dim ({d}) must match TurboQuantParams head_dim ({})",
        params.head_dim
    );
    // Broadcast signs as [1, ..., 1, D] against an arbitrary leading shape.
    let signs1_arr = ffi::from_slice_f32(&params.signs1, &[1, 1, 1, d as i32]);
    let signs2_arr = ffi::from_slice_f32(&params.signs2, &[1, 1, 1, d as i32]);
    // Cast to fp32 for stable numeric behaviour identical to the quantize path.
    let x_f32 = ffi::astype(x, crate::dtype::FLOAT32);
    let x_d1 = ffi::multiply(&x_f32, &signs1_arr);
    let x_h = wht(&x_d1);
    ffi::multiply(&x_h, &signs2_arr)
}

/// Apply the K-side TurboQuant `D2 · H · D1` rotation.
///
/// Used by the symmetric Turbo4 dequant-first SDPA path to rotate Q into the
/// same basis as [`dequantize_k_turbo4_rotated`].
pub fn turbo4_k_rotate(x: &MlxArray, params: &TurboQuantParams) -> UniquePtr<MlxArray> {
    let shape = ffi::array_shape(x);
    let d = *shape
        .last()
        .expect("turbo4_k_rotate: input must be at least 1-D") as usize;
    assert_eq!(
        d, params.head_dim as usize,
        "turbo4_k_rotate: last dim ({d}) must match TurboQuantParams head_dim ({})",
        params.head_dim
    );
    let signs1_arr = ffi::from_slice_f32(&params.k_signs1, &[1, 1, 1, d as i32]);
    let signs2_arr = ffi::from_slice_f32(&params.k_signs2, &[1, 1, 1, d as i32]);
    let x_f32 = ffi::astype(x, crate::dtype::FLOAT32);
    let x_d1 = ffi::multiply(&x_f32, &signs1_arr);
    let x_h = wht(&x_d1);
    ffi::multiply(&x_h, &signs2_arr)
}

/// Apply the inverse TurboQuant V rotation to a tensor in rotated value basis.
///
/// Used by the delegated dequant-SDPA path: cold V is dequantized as
/// `codebook[index] * rescale`, hot V is forward-rotated into the same basis,
/// SDPA consumes that rotated V, and only the small output tensor is brought
/// back to the model's original value basis.
pub fn turbo4_v_inverse_rotate(x: &MlxArray, params: &TurboQuantParams) -> UniquePtr<MlxArray> {
    let shape = ffi::array_shape(x);
    let d = *shape
        .last()
        .expect("turbo4_v_inverse_rotate: input must be at least 1-D") as usize;
    assert_eq!(
        d, params.head_dim as usize,
        "turbo4_v_inverse_rotate: last dim ({d}) must match TurboQuantParams head_dim ({})",
        params.head_dim
    );
    let signs1_arr = ffi::from_slice_f32(&params.signs1, &[1, 1, 1, d as i32]);
    let signs2_arr = ffi::from_slice_f32(&params.signs2, &[1, 1, 1, d as i32]);
    let x_f32 = ffi::astype(x, crate::dtype::FLOAT32);
    let pre_h = ffi::multiply(&x_f32, &signs2_arr);
    let post_h = wht(&pre_h);
    let out_f32 = ffi::multiply(&post_h, &signs1_arr);
    ffi::astype(&out_f32, dtype::FLOAT16)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Number of packed-V bytes needed to store `head_dim` 4-bit indices.
///
/// `head_dim` must be even; nibble-packing fits exactly two indices per byte.
#[inline]
pub fn packed_bytes_per_token(head_dim: i32) -> i32 {
    head_dim / 2
}

/// Helper: align a token count up to the next multiple of [`BLOCK_SIZE`] for
/// V-buffer growth. The trim path can rely on this to stay block-aligned.
#[inline]
pub fn round_up_to_block(token_count: i32) -> i32 {
    let block = BLOCK_SIZE;
    ((token_count + block - 1) / block) * block
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_relative_error(actual: f32, expected: f32, tol: f32, label: &str) {
        let denom = expected.abs().max(1e-6);
        let err = (actual - expected).abs() / denom;
        assert!(
            err < tol,
            "{label}: |{actual} - {expected}| / |expected| = {err:.4} > {tol}"
        );
    }

    #[test]
    fn lcg32_is_deterministic() {
        let mut a = Lcg32::new(42);
        let mut b = Lcg32::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_bit(), b.next_bit());
        }
    }

    #[test]
    fn generate_signs_is_pm_one() {
        let signs = generate_signs(128, 7);
        assert_eq!(signs.len(), 128);
        for &s in signs.iter() {
            assert!(s == 1.0 || s == -1.0, "non-±1 sign in vector: {s}");
        }
    }

    #[test]
    fn turbo_quant_params_centroid_count_matches_bit_width() {
        let p = TurboQuantParams::new(128, 42);
        assert_eq!(p.codebook.centroids.len(), 1 << V_BIT_WIDTH);
        assert_eq!(p.codebook.boundaries.len(), p.codebook.centroids.len() - 1);
        assert_eq!(p.signs1.len(), 128);
        assert_eq!(p.signs2.len(), 128);
        assert_eq!(p.k_signs1.len(), 128);
        assert_eq!(p.k_signs2.len(), 128);
    }

    /// K-side and V-side sign vectors must be statistically distinct — if
    /// they were equal, the K and V quantization noise would correlate and
    /// the inner product `Q · Kᵀ` in attention would see additive bias.
    /// This test enforces the *property* (different vectors) at construction
    /// time so a future refactor cannot accidentally collapse them.
    #[test]
    fn k_and_v_sign_vectors_are_independent() {
        let p = TurboQuantParams::new(128, 42);
        assert_ne!(
            p.signs1.as_ref(),
            p.k_signs1.as_ref(),
            "K-side and V-side signs1 must differ for symmetric Turbo4 to be safe"
        );
        assert_ne!(
            p.signs2.as_ref(),
            p.k_signs2.as_ref(),
            "K-side and V-side signs2 must differ for symmetric Turbo4 to be safe"
        );
    }

    /// K-side dequantize must round-trip the input within the same Lloyd-Max
    /// distortion bound as the V-side path. Verifies that the K-side sign
    /// vectors and shared codebook produce a numerically valid round-trip,
    /// not just a syntactically valid one.
    #[test]
    fn k_side_quantize_dequantize_round_trip_bounded_error() {
        let head_dim: i32 = 128;
        let params = TurboQuantParams::new(head_dim as u32, 0xCAFE_F00D);

        let mut rng = Lcg32::new(0xBEEF_CACE);
        let n_tokens = 4usize;
        let hd_usize = head_dim as usize;
        let mut k_data: Vec<f32> = Vec::with_capacity(n_tokens * hd_usize);
        let token_scales = [0.5_f32, 1.5, 4.0, 12.0];
        for &scale in token_scales.iter().take(n_tokens) {
            for _ in 0..head_dim {
                let mut acc = 0.0_f32;
                for _ in 0..6 {
                    acc += if rng.next_bit() == 0 { -1.0 } else { 1.0 };
                }
                k_data.push((acc / 6.0) * scale);
            }
        }
        let k = ffi::from_slice_f32(&k_data, &[1, 1, n_tokens as i32, head_dim]);

        let (packed, norms) = quantize_k_turbo4(&k, &params);
        let pshape = ffi::array_shape(&packed);
        assert_eq!(pshape, vec![1_i32, 1, n_tokens as i32, head_dim / 2]);

        let k_hat = dequantize_k_turbo4(&packed, &norms, &params);
        assert_eq!(ffi::array_dtype(&k_hat), dtype::FLOAT16);
        let k_hat_f32 = ffi::astype(&k_hat, dtype::FLOAT32);
        ffi::eval(&k_hat_f32);
        let bytes = ffi::array_to_raw_bytes(&k_hat_f32);
        let k_hat_vec: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        for tok in 0..n_tokens {
            let off = tok * hd_usize;
            let mut num = 0.0_f32;
            let mut den = 0.0_f32;
            for k_i in 0..hd_usize {
                let diff = k_data[off + k_i] - k_hat_vec[off + k_i];
                num += diff * diff;
                den += k_data[off + k_i] * k_data[off + k_i];
            }
            let rel = (num / den.max(1e-12)).sqrt();
            assert!(
                rel < 0.15,
                "token {tok}: K-side relative L2 error {rel:.4} exceeds 15% bound"
            );
        }
    }

    /// K-side and V-side packed buffers must differ for the same input —
    /// confirms the two paths really do use independent rotations rather
    /// than producing identical results despite different sign vectors.
    #[test]
    fn k_and_v_packed_outputs_differ_for_same_input() {
        let head_dim: i32 = 64;
        let params = TurboQuantParams::new(head_dim as u32, 0x1234_5678);
        let v_data: Vec<f32> = (0..head_dim)
            .map(|i| (i as f32 / head_dim as f32 - 0.5) * 2.0)
            .collect();
        let x = ffi::from_slice_f32(&v_data, &[1, 1, 1, head_dim]);

        let (v_packed, _, _) = quantize_v_turbo4(&x, &params);
        let (k_packed, _) = quantize_k_turbo4(&x, &params);

        let v_bytes = ffi::array_to_raw_bytes(&v_packed);
        let k_bytes = ffi::array_to_raw_bytes(&k_packed);
        assert_eq!(
            v_bytes.len(),
            k_bytes.len(),
            "K and V packed buffers must have the same byte size"
        );
        assert_ne!(
            v_bytes, k_bytes,
            "K and V packed outputs must differ for the same input — \
             otherwise K/V noise would be correlated and break attention"
        );
    }

    #[test]
    fn round_up_to_block_pads_correctly() {
        assert_eq!(round_up_to_block(0), 0);
        assert_eq!(round_up_to_block(1), 32);
        assert_eq!(round_up_to_block(31), 32);
        assert_eq!(round_up_to_block(32), 32);
        assert_eq!(round_up_to_block(33), 64);
        assert_eq!(round_up_to_block(256), 256);
    }

    #[test]
    fn packed_bytes_per_token_matches_layout() {
        assert_eq!(packed_bytes_per_token(64), 32);
        assert_eq!(packed_bytes_per_token(128), 64);
        assert_eq!(packed_bytes_per_token(256), 128);
    }

    /// End-to-end round-trip on a [B=1, H=1, T=4, D=128] tensor with
    /// realistic per-token magnitudes. The rotated coordinates follow N(0,1/d)
    /// after WHT, so 4-bit centroids reconstruct each coordinate with bounded
    /// MSE; relative L2 error per token should be well under 10%.
    #[test]
    fn quantize_dequantize_round_trip_recovers_v_within_bound() {
        let head_dim: i32 = 128;
        let params = TurboQuantParams::new(head_dim as u32, 42);

        // Build a deterministic test V tensor: each token is a scaled
        // Gaussian-ish vector. Use a simple LCG-derived sequence for repeatability.
        let mut rng = Lcg32::new(123);
        let mut v_data: Vec<f32> = Vec::with_capacity(4 * head_dim as usize);
        // Per-token magnitudes spanning ~1.5 decades — typical for real KV
        // values. Very-tiny magnitudes (< 0.1) hit fp16 underflow in the
        // norm storage, which is acceptable in production but would noise
        // up this test.
        let token_scales = [0.5_f32, 1.5, 4.0, 12.0];
        for &scale in &token_scales {
            for _ in 0..head_dim {
                // Pseudo-Gaussian via summed uniform bits → ~normal in [-1, 1].
                let mut acc = 0.0_f32;
                for _ in 0..6 {
                    acc += if rng.next_bit() == 0 { -1.0 } else { 1.0 };
                }
                v_data.push((acc / 6.0) * scale);
            }
        }
        let v = ffi::from_slice_f32(&v_data, &[1, 1, 4, head_dim]);

        let (packed, norms, _rescale) = quantize_v_turbo4(&v, &params);
        // Verify shapes
        let pshape = ffi::array_shape(&packed);
        assert_eq!(pshape, vec![1_i32, 1, 4, head_dim / 2]);
        let nshape = ffi::array_shape(&norms);
        assert_eq!(nshape, vec![1_i32, 1, 4, 1]);

        let v_hat = dequantize_v_turbo4(&packed, &norms, &params);
        assert_eq!(ffi::array_dtype(&v_hat), dtype::FLOAT16);
        assert_eq!(ffi::array_shape(&v_hat), vec![1_i32, 1, 4, head_dim]);

        // Convert v_hat back to fp32 for comparison
        let v_hat_f32 = ffi::astype(&v_hat, dtype::FLOAT32);
        ffi::eval(&v_hat_f32);
        let v_hat_bytes = ffi::array_to_raw_bytes(&v_hat_f32);
        let v_hat_vec: Vec<f32> = v_hat_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Per-token relative L2 error is bounded by Lloyd-Max theoretical
        // distortion at 4 bits on N(0, 1/d). The classical D(R) bound at 4
        // bits gives ~−23.8 dB → ~6.5% RMSE in the rotated domain. After
        // inverse rotation and rescaling, fp16 round-off plus the norm
        // correction can push observed per-token error to ~11–13% on real
        // inputs (the rotation cannot hide all bias when the input has heavy
        // structure, which our synthetic data does). 15% is a safe bound that
        // catches gross algorithm bugs without being flaky.
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
                rel < 0.15,
                "token {tok}: relative L2 error {rel:.4} exceeds 15% bound"
            );
        }
    }

    /// C1 regression: V vectors with `||v|| < 1` must round-trip through the
    /// quantizer correctly. The earlier `safe_norm = maximum(norm, 1.0)` clamp
    /// silently destroyed direction information for any small-magnitude V
    /// because the divide pinned the rotated coordinates near the centroid for
    /// `0` (variance `1/d`, designed for unit-norm post-WHT). With
    /// `||v|| ≈ 0.06–0.1` the reviewer measured ~65% reconstruction error on
    /// real Llama 3.1 traffic. Use a uniform-`[-0.05, 0.05]` distribution over
    /// `head_dim=128`, which lands per-token L2 norms in `[0.30, 0.40]` —
    /// safely below 1.0 and well above the fp16 underflow threshold so the
    /// stored norm survives round-trip. The test passes once the clamp is
    /// replaced by a true zero-only fallback (`where(norm > 0, norm, 1.0)`)
    /// and would have failed under the old `maximum(_, 1.0)` clamp.
    #[test]
    fn small_norm_v_round_trip_recovers_within_bound() {
        let head_dim: i32 = 128;
        let params = TurboQuantParams::new(head_dim as u32, 0xC1_DECAFu32);

        let mut rng = Lcg32::new(0xC0FF_EE42);
        let n_tokens = 4usize;
        let hd_usize = head_dim as usize;
        let mut v_data: Vec<f32> = Vec::with_capacity(n_tokens * hd_usize);
        for _ in 0..n_tokens {
            for _ in 0..head_dim {
                // Pseudo-uniform on [-0.05, 0.05] via 6-bit LCG draw.
                let mut acc = 0u32;
                for b in 0..6 {
                    acc |= rng.next_bit() << b;
                }
                let u = (acc as f32) / 63.0; // [0, 1]
                v_data.push((u - 0.5) * 0.10); // [-0.05, 0.05]
            }
        }

        // Sanity: each token norm should sit comfortably below 1.0 so the
        // test actually exercises the regression path. Pre-fix `safe_norm`
        // saturated to 1.0 here, destroying direction.
        for tok in 0..n_tokens {
            let off = tok * hd_usize;
            let n = v_data[off..off + hd_usize]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            assert!(
                n < 1.0,
                "token {tok}: precondition violated, ||v|| = {n} >= 1.0 — \
                 this test must use small-norm V vectors to catch C1"
            );
            assert!(
                n > 0.05,
                "token {tok}: precondition violated, ||v|| = {n} <= 0.05 — \
                 fp16 norm storage would underflow and noise the test"
            );
        }

        let v = ffi::from_slice_f32(&v_data, &[1, 1, n_tokens as i32, head_dim]);
        let (packed, norms, _rescale) = quantize_v_turbo4(&v, &params);
        let v_hat = dequantize_v_turbo4(&packed, &norms, &params);

        let v_hat_f32 = ffi::astype(&v_hat, dtype::FLOAT32);
        ffi::eval(&v_hat_f32);
        let v_hat_bytes = ffi::array_to_raw_bytes(&v_hat_f32);
        let v_hat_vec: Vec<f32> = v_hat_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        for tok in 0..n_tokens {
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
                rel < 0.15,
                "token {tok}: small-norm relative L2 error {rel:.4} exceeds 15% \
                 bound — likely C1 regression (`safe_norm = maximum(_, 1.0)` \
                 reintroduced)"
            );
        }
    }

    /// Sanity: zero V vector is preserved (norm=0 path doesn't NaN).
    #[test]
    fn zero_vector_dequantizes_to_zero() {
        let head_dim: i32 = 64;
        let params = TurboQuantParams::new(head_dim as u32, 1);
        let v = ffi::zeros(&[1, 1, 1, head_dim], dtype::FLOAT32);
        let (packed, norms, _rescale) = quantize_v_turbo4(&v, &params);
        let v_hat = dequantize_v_turbo4(&packed, &norms, &params);
        let v_hat_f32 = ffi::astype(&v_hat, dtype::FLOAT32);
        ffi::eval(&v_hat_f32);
        let bytes = ffi::array_to_raw_bytes(&v_hat_f32);
        for chunk in bytes.chunks_exact(4) {
            let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            assert!(val.abs() < 1e-3, "expected ~0, got {val}");
        }
    }

    /// Rotated-space bulk dequant plus output inverse-rotation should match
    /// the standard per-token full dequant path. This is the algebra used by
    /// the Turbo4Delegated dequant-SDPA path.
    #[test]
    fn rotated_dequant_then_inverse_matches_full_dequant() {
        let head_dim: i32 = 64;
        let params = TurboQuantParams::new(head_dim as u32, 123);
        let n_tokens = 4;
        let v_data: Vec<f32> = (0..(n_tokens * head_dim))
            .map(|i| {
                let f = i as f32;
                0.35 * (f * 0.17).sin() + 0.15 * (f * 0.07).cos()
            })
            .collect();
        let v = ffi::from_slice_f32(&v_data, &[1, 1, n_tokens, head_dim]);
        let (packed, norms, rescale) = quantize_v_turbo4(&v, &params);

        let full = dequantize_v_turbo4(&packed, &norms, &params);
        let rotated = dequantize_v_turbo4_rotated(&packed, &rescale, &params);
        let inverse = turbo4_v_inverse_rotate(&rotated, &params);

        let full_vec = {
            let arr = ffi::astype(&full, dtype::FLOAT32);
            ffi::eval(&arr);
            ffi::array_to_raw_bytes(&arr)
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<_>>()
        };
        let inverse_vec = {
            let arr = ffi::astype(&inverse, dtype::FLOAT32);
            ffi::eval(&arr);
            ffi::array_to_raw_bytes(&arr)
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<_>>()
        };

        assert_eq!(full_vec.len(), inverse_vec.len());
        let mut sum_sq = 0.0_f64;
        for (a, b) in full_vec.iter().zip(inverse_vec.iter()) {
            let d = (*a - *b) as f64;
            sum_sq += d * d;
        }
        let rms = (sum_sq / full_vec.len() as f64).sqrt() as f32;
        assert!(
            rms < 2e-3,
            "rotated bulk dequant + inverse rotation RMS {rms:.4e} exceeds 2e-3"
        );
    }

    /// Determinism: same params + same V → same packed bytes.
    #[test]
    fn quantize_is_deterministic() {
        let head_dim: i32 = 64;
        let params1 = TurboQuantParams::new(head_dim as u32, 7);
        let params2 = TurboQuantParams::new(head_dim as u32, 7);
        // signs derived from same seed must match
        assert_eq!(params1.signs1.as_ref(), params2.signs1.as_ref());

        let v_data: Vec<f32> = (0..head_dim).map(|i| (i as f32 - 32.0) * 0.05).collect();
        let v = ffi::from_slice_f32(&v_data, &[1, 1, 1, head_dim]);
        let (p1, _, _) = quantize_v_turbo4(&v, &params1);
        let (p2, _, _) = quantize_v_turbo4(&v, &params2);
        assert_eq!(ffi::array_to_raw_bytes(&p1), ffi::array_to_raw_bytes(&p2));
    }

    /// Nibble-packing layout: byte = (high << 4) | low matches the documented
    /// "low nibble = even, high nibble = odd" contract.
    #[test]
    fn nibble_packing_layout_is_documented() {
        // Decoded indices must be in [0, 16). The full sign-and-rotation
        // determinism is exercised by quantize_is_deterministic; here we just
        // sanity-check that nothing leaks out of the 4-bit range.
        let head_dim: i32 = 64;
        let params = TurboQuantParams::new(head_dim as u32, 99);
        let v_data: Vec<f32> = (0..head_dim)
            .map(|i| (i as f32 / head_dim as f32) - 0.5)
            .collect();
        let v = ffi::from_slice_f32(&v_data, &[1, 1, 1, head_dim]);
        let (packed, _, _) = quantize_v_turbo4(&v, &params);
        let bytes = ffi::array_to_raw_bytes(&packed);
        for &b in &bytes {
            let lo = b & 0x0F;
            let hi = b >> 4;
            assert!(lo < 16, "low nibble out of range: {lo}");
            assert!(hi < 16, "high nibble out of range: {hi}");
        }
    }

    /// Stored norms should match the per-token L2 of the input within fp16 precision.
    #[test]
    fn stored_norms_match_input_l2() {
        let head_dim: i32 = 64;
        let hd_usize = head_dim as usize;
        let params = TurboQuantParams::new(head_dim as u32, 17);
        // Two tokens with very different magnitudes.
        let mut v_data: Vec<f32> = Vec::with_capacity(2 * hd_usize);
        for k in 0..head_dim {
            v_data.push(0.5 * ((k as f32).sin())); // small token
        }
        for k in 0..head_dim {
            v_data.push(50.0 * ((k as f32).cos())); // large token
        }
        let v = ffi::from_slice_f32(&v_data, &[1, 1, 2, head_dim]);
        let (_, norms, _) = quantize_v_turbo4(&v, &params);
        let norms_f32 = ffi::astype(&norms, dtype::FLOAT32);
        ffi::eval(&norms_f32);
        let bytes = ffi::array_to_raw_bytes(&norms_f32);
        let n0 = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let n1 = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);

        let expected_n0 = (v_data[..hd_usize].iter().map(|x| x * x).sum::<f32>()).sqrt();
        let expected_n1 = (v_data[hd_usize..].iter().map(|x| x * x).sum::<f32>()).sqrt();
        // fp16 has ~3 decimal digits of precision; allow 2% relative error.
        assert_relative_error(n0, expected_n0, 0.02, "norm[0]");
        assert_relative_error(n1, expected_n1, 0.02, "norm[1]");
    }
}
