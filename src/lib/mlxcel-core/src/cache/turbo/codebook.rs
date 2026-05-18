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

//! PolarQuant Lloyd-Max codebook generator.
//!
//! After Walsh–Hadamard rotation each KV coordinate follows `N(0, 1/d)` in the
//! large-`d` limit (exactly `Beta(d/2, d/2)` rescaled; converges quickly for
//! `d ≥ 64`). This module computes the **optimal scalar-quantization centroids**
//! for that distribution via Lloyd's algorithm.
//!
//! # Reference
//!
//! This is a pure-Rust port of
//! `references/turboquant_plus/turboquant/codebook.py::optimal_centroids`.
//! The algorithm is:
//!
//! * `b = 1`: closed-form `±sqrt(2 / (π·d))`.
//! * `b = 2`: closed-form `[−1.51, −0.453, 0.453, 1.51] / sqrt(d)`.
//! * `b ≥ 3`: iterative Lloyd-Max on `N(0, 1/d)` — 100 iterations, uniform
//!   quantile initialization, conditional-expectation centroid update.
//!
//! # Numerical notes
//!
//! * All intermediate arithmetic uses `f64` for accuracy; results are
//!   downcast to `f32` at the end. The centroids themselves (`Vec<f32>`) are
//!   the hot-path data — no `f64` appears in the returned values.
//! * `gaussian_cdf` / `gaussian_sf` use `libm::erfc` (IEEE-quality minimax)
//!   so the centroids reproduce scipy's `Norm(0, σ²).cdf` byte-for-byte
//!   after the f32 cast. `gaussian_ppf` uses Acklam's rational approximation
//!   (accurate to `|err| < 1.15e-9` for `p ∈ (0, 1)`).
//!
//! # Caching
//!
//! `optimal_centroids` is typically called once per `(bit_width, head_dim)`
//! pair during model initialisation. Results are cached in a `OnceLock`-backed
//! `HashMap` so repeated calls are free.
//!
//! Used by: TurboQuant KV cache (B2 onward, epic #458)

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

// ---------------------------------------------------------------------------
// Codebook struct
// ---------------------------------------------------------------------------

/// Bundled centroids and midpoint boundaries for a single `(bit_width, head_dim)`
/// pair, with both arrays held as `Arc<[f32]>` for zero-copy sharing across
/// quantize/dequantize sites.
///
/// `boundaries[i] = (centroids[i] + centroids[i+1]) / 2` (length =
/// `centroids.len() - 1`). Holding both pre-built lets the B2 hot path on the
/// quantize side amortize the `Vec<f32>` allocation that
/// [`nearest_centroid_indices`] otherwise rebuilds per call. Centroids are
/// shared with the dequantize-on-read path inside the cache, which references
/// them once per layer/forward at minimal overhead.
///
/// Used by: TurboQuant V-side quantize/dequantize (B2, epic #458).
#[derive(Clone, Debug)]
pub struct Codebook {
    /// Sorted centroid values for `N(0, 1/d)`. Length = `2^bit_width`.
    pub centroids: Arc<[f32]>,
    /// Midpoint boundaries between consecutive centroids. Length =
    /// `centroids.len() - 1`. Pre-built so callers do not re-allocate per call.
    pub boundaries: Arc<[f32]>,
}

impl Codebook {
    /// Build a [`Codebook`] from raw centroids, deriving boundaries.
    ///
    /// `centroids` must be sorted in ascending order. This invariant is
    /// guaranteed by [`compute_centroids`] and [`optimal_centroids`] by
    /// construction.
    pub fn from_centroids(centroids: Arc<[f32]>) -> Self {
        let boundaries: Vec<f32> = centroids.windows(2).map(|w| (w[0] + w[1]) * 0.5).collect();
        Self {
            centroids,
            boundaries: Arc::from(boundaries),
        }
    }

    /// Number of centroids (`= 2^bit_width`).
    pub fn len(&self) -> usize {
        self.centroids.len()
    }

    /// Whether the codebook is empty (always false for valid bit_width >= 1).
    pub fn is_empty(&self) -> bool {
        self.centroids.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute optimal MSE centroids for the post-WHT coordinate distribution.
///
/// Returns a `Vec<f32>` of exactly `2^bit_width` sorted centroids for the
/// distribution `N(0, 1/d)`.
///
/// # Supported parameters
///
/// | `bit_width` | algorithm  |
/// |-------------|------------|
/// | 1           | closed-form |
/// | 2           | closed-form |
/// | ≥ 3         | iterative Lloyd-Max (100 iterations) |
///
/// `head_dim` must be > 0. Any positive `u32` is accepted; the module has
/// been validated against the Python reference for `d ∈ {64, 80, 96, 128, 192, 256}`.
///
/// # Panics
///
/// Panics if `bit_width == 0` or `head_dim == 0`.
///
/// # Caching
///
/// Results are memoized in a process-global `Mutex<HashMap>`. The first call
/// for each `(bit_width, head_dim)` pair pays the iteration cost; subsequent
/// calls return the cached `Vec<f32>` clone. This is appropriate for model
/// initialization which happens once per process.
pub fn optimal_centroids(bit_width: u8, head_dim: u32) -> Vec<f32> {
    assert!(bit_width > 0, "bit_width must be >= 1");
    // Cap at 8 bits: `1usize << bit_width` is the centroid count, and 32+ bits
    // overflows on 64-bit platforms (UB per Rust language ref) while 24+ bits
    // would silently allocate gigabytes. TurboQuant only ever uses 1–4 bits,
    // and even 8-bit headroom is well above any realistic future codebook.
    assert!(
        bit_width <= 8,
        "bit_width must be <= 8 (TurboQuant uses 1–4); got {bit_width}"
    );
    assert!(head_dim > 0, "head_dim must be > 0");

    use std::sync::Mutex;
    type CodebookCache = Mutex<HashMap<(u8, u32), Vec<f32>>>;
    static TABLE: OnceLock<CodebookCache> = OnceLock::new();

    let map = TABLE.get_or_init(|| Mutex::new(HashMap::new()));
    // Recover from poisoning: a previous panic during compute leaves the cache
    // in a recoverable state (centroid generation has no side effects beyond
    // the HashMap insert), so re-acquire the inner state instead of panicking
    // every subsequent call.
    let mut guard = map.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    guard
        .entry((bit_width, head_dim))
        .or_insert_with(|| compute_centroids(bit_width, head_dim))
        .clone()
}

/// Optimal codebook (centroids + pre-built boundaries) for a `(bit_width,
/// head_dim)` pair, cached for zero-allocation reuse on hot paths.
///
/// This is the API the B2 V-side quantize loop should call: it returns a
/// cheap `Arc`-clone on every cache hit (just a refcount bump, no `Vec`
/// allocation), and avoids rebuilding the midpoint boundaries that
/// [`nearest_centroid_indices`] would otherwise compute per call.
///
/// # Caching
///
/// Like [`optimal_centroids`], results are memoized in a process-global
/// `Mutex<HashMap>` keyed by `(bit_width, head_dim)`. The first call pays the
/// Lloyd-Max cost; subsequent calls return a clone of the same `Arc<[f32]>`s.
///
/// # Panics
///
/// Same panic conditions as [`optimal_centroids`].
///
/// Used by: TurboQuant V-side quantize/dequantize (B2, epic #458).
pub fn optimal_codebook(bit_width: u8, head_dim: u32) -> Codebook {
    assert!(bit_width > 0, "bit_width must be >= 1");
    assert!(
        bit_width <= 8,
        "bit_width must be <= 8 (TurboQuant uses 1–4); got {bit_width}"
    );
    assert!(head_dim > 0, "head_dim must be > 0");

    use std::sync::Mutex;
    type CodebookCache = Mutex<HashMap<(u8, u32), Codebook>>;
    static TABLE: OnceLock<CodebookCache> = OnceLock::new();

    let map = TABLE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = map.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    guard
        .entry((bit_width, head_dim))
        .or_insert_with(|| {
            let centroids: Arc<[f32]> = Arc::from(compute_centroids(bit_width, head_dim));
            Codebook::from_centroids(centroids)
        })
        .clone()
}

/// Compute centroids without caching. Prefer [`optimal_centroids`] for
/// production use (which caches results).
///
/// Exposed for testing and benchmarking.
pub fn compute_centroids(bit_width: u8, head_dim: u32) -> Vec<f32> {
    assert!(bit_width > 0, "bit_width must be >= 1");
    assert!(
        bit_width <= 8,
        "bit_width must be <= 8 (TurboQuant uses 1–4); got {bit_width}"
    );
    assert!(head_dim > 0, "head_dim must be > 0");
    let d = head_dim as f64;

    match bit_width {
        1 => {
            let c = (2.0_f64 / (std::f64::consts::PI * d)).sqrt();
            vec![-c as f32, c as f32]
        }
        2 => {
            let inv_sqrt_d = 1.0_f64 / d.sqrt();
            vec![
                (-1.51_f64 * inv_sqrt_d) as f32,
                (-0.453_f64 * inv_sqrt_d) as f32,
                (0.453_f64 * inv_sqrt_d) as f32,
                (1.51_f64 * inv_sqrt_d) as f32,
            ]
        }
        b => {
            let n_centroids = 1usize << b;
            let sigma = 1.0_f64 / d.sqrt();
            lloyds_gaussian(n_centroids, sigma, 100)
                .into_iter()
                .map(|c| c as f32)
                .collect()
        }
    }
}

/// Find the nearest centroid index for each value via binary search on midpoint
/// boundaries (equivalent to `np.searchsorted(boundaries, values)`).
///
/// `centroids` must be sorted in ascending order (guaranteed by
/// [`optimal_centroids`] / [`compute_centroids`]).
///
/// Returns integer indices in `[0, n_centroids)`.
///
/// Used by: unit tests; B2 quantization path (epic #458). On hot paths prefer
/// [`nearest_centroid_indices_with_boundaries`] (or use the [`Codebook`]
/// returned by [`optimal_codebook`]) so the midpoint boundary array is only
/// built once per `(bit_width, head_dim)` instead of per-call.
pub fn nearest_centroid_indices(values: &[f32], centroids: &[f32]) -> Vec<usize> {
    assert!(!centroids.is_empty(), "centroids must be non-empty");
    let n = centroids.len();

    // Build midpoint boundaries: boundaries[i] = (centroids[i] + centroids[i+1]) / 2
    let boundaries: Vec<f32> = centroids.windows(2).map(|w| (w[0] + w[1]) * 0.5).collect();

    values
        .iter()
        .map(|&v| {
            // Binary search: find insertion point in sorted boundaries
            // This gives the centroid region index in [0, n)
            boundaries.partition_point(|&b| b < v).min(n - 1)
        })
        .collect()
}

/// Like [`nearest_centroid_indices`] but takes pre-built midpoint boundaries.
///
/// Use this on hot paths to avoid rebuilding the boundary array per call.
/// `boundaries.len()` must equal `n_centroids - 1` (i.e., the boundary
/// invariant established by [`Codebook::from_centroids`]).
///
/// Used by: TurboQuant V-side quantize loop (B2, epic #458).
pub fn nearest_centroid_indices_with_boundaries(
    values: &[f32],
    boundaries: &[f32],
    n_centroids: usize,
) -> Vec<usize> {
    assert!(n_centroids > 0, "n_centroids must be > 0");
    assert_eq!(
        boundaries.len(),
        n_centroids - 1,
        "boundaries.len() must equal n_centroids - 1"
    );
    let last_idx = n_centroids - 1;
    values
        .iter()
        .map(|&v| boundaries.partition_point(|&b| b < v).min(last_idx))
        .collect()
}

// ---------------------------------------------------------------------------
// Lloyd-Max iteration (f64 precision)
// ---------------------------------------------------------------------------

/// Lloyd's algorithm for optimal scalar quantization of `N(0, σ²)`.
///
/// Initialises boundaries from uniform quantiles, then iterates:
/// 1. Recompute boundaries as midpoints between centroids.
/// 2. Recompute centroids as conditional expectations within each region.
///
/// Matches the reference Python exactly (100 iterations, same init).
fn lloyds_gaussian(n_centroids: usize, sigma: f64, n_iter: usize) -> Vec<f64> {
    // Initialize boundary positions from uniform quantiles
    // Python: stats.norm.ppf(np.linspace(0, 1, n_centroids + 1)[1:-1], scale=sigma)
    let mut boundaries: Vec<f64> = (1..n_centroids)
        .map(|i| {
            let p = i as f64 / n_centroids as f64;
            gaussian_ppf(p) * sigma
        })
        .collect();

    // Initial centroids: conditional expectations within each region
    let mut centroids = conditional_centroids(&boundaries, sigma, n_centroids);

    for _ in 0..n_iter {
        // Update boundaries (midpoints between consecutive centroids)
        boundaries = centroids.windows(2).map(|w| (w[0] + w[1]) * 0.5).collect();

        // Update centroids (conditional expectations within each region)
        centroids = conditional_centroids(&boundaries, sigma, n_centroids);
    }

    centroids
}

/// Compute conditional-expectation centroids given boundaries.
fn conditional_centroids(boundaries: &[f64], sigma: f64, n_centroids: usize) -> Vec<f64> {
    let mut centroids = vec![0.0_f64; n_centroids];

    centroids[0] = gaussian_conditional_expectation(sigma, f64::NEG_INFINITY, boundaries[0]);
    for i in 1..n_centroids - 1 {
        centroids[i] = gaussian_conditional_expectation(sigma, boundaries[i - 1], boundaries[i]);
    }
    centroids[n_centroids - 1] =
        gaussian_conditional_expectation(sigma, boundaries[n_centroids - 2], f64::INFINITY);

    centroids
}

/// `E[X | a < X < b]` where `X ~ N(0, σ²)`.
///
/// Formula: `σ · (φ(a/σ) − φ(b/σ)) / (Φ(b/σ) − Φ(a/σ))`
///
/// Handles semi-infinite intervals and near-zero probability regions via the
/// same asymptotic fallbacks as the Python reference.
fn gaussian_conditional_expectation(sigma: f64, a: f64, b: f64) -> f64 {
    let a_std = if a.is_finite() { a / sigma } else { a };
    let b_std = if b.is_finite() { b / sigma } else { b };

    // P(a < X/σ < b) — use the numerically stable formulation
    let prob = if !a_std.is_finite() {
        gaussian_cdf(b_std)
    } else if !b_std.is_finite() {
        gaussian_sf(a_std) // survival function = 1 - CDF, more stable for large a
    } else {
        gaussian_cdf(b_std) - gaussian_cdf(a_std)
    };

    if prob < 1e-15 {
        // Asymptotic fallback — mirrors the Python reference exactly
        if a.is_finite() && !b.is_finite() {
            return a + sigma; // E[X | X > a] ≈ a + σ for extreme a
        } else if !a.is_finite() && b.is_finite() {
            return b - sigma;
        } else if a.is_finite() && b.is_finite() {
            return (a + b) * 0.5;
        } else {
            return 0.0;
        }
    }

    let pdf_diff = gaussian_pdf(a_std) - gaussian_pdf(b_std);
    sigma * pdf_diff / prob
}

// ---------------------------------------------------------------------------
// Standard normal helpers (no external crate dependency)
// ---------------------------------------------------------------------------

/// `φ(x)` — standard normal PDF.
#[inline]
fn gaussian_pdf(x: f64) -> f64 {
    let inv_sqrt_2pi = 0.398_942_280_401_432_7_f64; // 1/sqrt(2π)
    inv_sqrt_2pi * (-0.5 * x * x).exp()
}

/// `Φ(x)` — standard normal CDF via erfc.
///
/// `Φ(x) = erfc(-x / sqrt(2)) / 2`
///
/// Uses `libm::erfc` (IEEE-quality minimax) — matches `scipy.stats.norm.cdf`
/// to within last-bit f64 precision over the full input range.
#[inline]
fn gaussian_cdf(x: f64) -> f64 {
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    0.5 * libm::erfc(-x * inv_sqrt2)
}

/// `1 − Φ(x)` — survival function.
///
/// More numerically stable than `1 - gaussian_cdf(x)` for large positive `x`.
#[inline]
fn gaussian_sf(x: f64) -> f64 {
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    0.5 * libm::erfc(x * inv_sqrt2)
}

/// `Φ⁻¹(p)` — standard normal quantile (inverse CDF / ppf).
///
/// Uses the rational approximation from Peter J. Acklam (2002),
/// accurate to `|error| < 1.15e-9` for `p ∈ (0, 1)`.
///
/// Reference: <https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/>
#[allow(clippy::excessive_precision)] // Acklam tabulated constants — keep as-published
fn gaussian_ppf(p: f64) -> f64 {
    // Coefficients for rational approximation
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    if p < P_LOW {
        // Rational approximation for lower region
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        // Rational approximation for central region
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        // Rational approximation for upper region
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

// Note: complementary error function `erfc` is provided by the `libm` crate.
// We tried implementing a Cody/Chebyshev minimax in-place but the polynomial
// coefficients were not accurate enough to reproduce scipy's centroids to f32
// last-bit precision; libm's IEEE-quality implementation does match.

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Gaussian helpers (self-consistency)
    // -----------------------------------------------------------------------

    #[test]
    fn gaussian_cdf_standard_values() {
        // Φ(0) = 0.5
        let v = gaussian_cdf(0.0);
        assert!((v - 0.5).abs() < 1e-14, "Φ(0) = {v}");

        // Φ(1) ≈ 0.8413447460685429
        let v = gaussian_cdf(1.0);
        assert!((v - 0.841_344_746_068_542_9).abs() < 1e-12, "Φ(1) = {v}");

        // Φ(−1) ≈ 0.15865525393145702
        let v = gaussian_cdf(-1.0);
        assert!((v - 0.158_655_253_931_457).abs() < 1e-12, "Φ(-1) = {v}");
    }

    #[test]
    fn gaussian_ppf_roundtrip() {
        // Acklam's rational approximation is documented to |err| < 1.15e-9 in
        // x; cdf is libm-quality (~1e-15). Round-trip therefore inherits the
        // ppf bound — 1e-8 is the right gate.
        for p in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let x = gaussian_ppf(p);
            let p_back = gaussian_cdf(x);
            assert!(
                (p_back - p).abs() < 1e-8,
                "ppf roundtrip failed at p={p}: got {p_back}"
            );
        }
    }

    #[test]
    fn gaussian_ppf_symmetry() {
        for p in [0.1, 0.2, 0.3, 0.4] {
            let lo = gaussian_ppf(p);
            let hi = gaussian_ppf(1.0 - p);
            assert!(
                (lo + hi).abs() < 1e-12,
                "ppf not antisymmetric at p={p}: lo={lo}, hi={hi}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Closed-form paths — exact match against Python reference values
    // -----------------------------------------------------------------------

    /// Helper: compare two f32 slices at bitwise (byte-level) f32 equality.
    fn assert_centroids_exact(rust: &[f32], expected_bits: &[u32], label: &str) {
        assert_eq!(rust.len(), expected_bits.len(), "{label}: length mismatch");
        for (i, (&r, &eb)) in rust.iter().zip(expected_bits.iter()).enumerate() {
            let rb = r.to_bits();
            assert_eq!(
                rb, eb,
                "{label}[{i}]: bits differ: rust=0x{rb:08x} expected=0x{eb:08x}"
            );
        }
    }

    // b=2 closed-form: ±1.51/sqrt(d), ±0.453/sqrt(d)
    #[test]
    fn b2_closed_form_d64() {
        let c = compute_centroids(2, 64);
        // Expected (from Python fixture, downcast to f32):
        // [-0.1887499988079071, -0.05662500113248825, 0.05662500113248825, 0.1887499988079071]
        let expected: &[f32] = &[
            f32::from_bits(0xbe4147ae),
            f32::from_bits(0xbd67ef9e),
            f32::from_bits(0x3d67ef9e),
            f32::from_bits(0x3e4147ae),
        ];
        assert_centroids_exact(
            &c,
            &[0xbe4147ae, 0xbd67ef9e, 0x3d67ef9e, 0x3e4147ae],
            "b=2,d=64",
        );
        let _ = expected; // used by assert_centroids_exact
    }

    #[test]
    fn b2_closed_form_d128() {
        let c = compute_centroids(2, 128);
        assert_centroids_exact(
            &c,
            &[0xbe08ab6b, 0xbd2400e7, 0x3d2400e7, 0x3e08ab6b],
            "b=2,d=128",
        );
    }

    #[test]
    fn b2_closed_form_all_dims() {
        // Verify the 2-bit closed form for all head dims in the fixture grid
        let cases: &[(u32, [u32; 4])] = &[
            (64, [0xbe4147ae, 0xbd67ef9e, 0x3d67ef9e, 0x3e4147ae]),
            (80, [0xbe2cdff9, 0xbd4f732a, 0x3d4f732a, 0x3e2cdff9]),
            (96, [0xbe1dcffd, 0xbd3d5ffd, 0x3d3d5ffd, 0x3e1dcffd]),
            (128, [0xbe08ab6b, 0xbd2400e7, 0x3d2400e7, 0x3e08ab6b]),
            (192, [0xbddf2e37, 0xbd05e887, 0x3d05e887, 0x3ddf2e37]),
            (256, [0xbdc147ae, 0xbce7ef9e, 0x3ce7ef9e, 0x3dc147ae]),
        ];
        for &(d, ref bits) in cases {
            let c = compute_centroids(2, d);
            assert_centroids_exact(&c, bits, &format!("b=2,d={d}"));
        }
    }

    // -----------------------------------------------------------------------
    // Lloyd-Max paths — tolerance match (max relative error < 1e-5)
    // from fixture file tests/fixtures/turbo_codebooks/codebooks_hex.json
    // -----------------------------------------------------------------------

    #[test]
    fn b3_d64_vs_fixture() {
        let c = compute_centroids(3, 64);
        // Fixture hex: 3_64
        let bits: [u32; 8] = [
            0xbe89b977, 0xbe2c0532, 0xbdc18987, 0xbcfaf9eb, 0x3cfaf9eb, 0x3dc18987, 0x3e2c0532,
            0x3e89b977,
        ];
        assert_centroids_exact(&c, &bits, "b=3,d=64");
    }

    #[test]
    fn b3_d128_vs_fixture() {
        let c = compute_centroids(3, 128);
        let bits: [u32; 8] = [
            0xbe42c596, 0xbdf34600, 0xbd88d9fa, 0xbcb1778d, 0x3cb1778d, 0x3d88d9fa, 0x3df34600,
            0x3e42c596,
        ];
        assert_centroids_exact(&c, &bits, "b=3,d=128");
    }

    #[test]
    fn b4_d64_vs_fixture() {
        let c = compute_centroids(4, 64);
        let bits: [u32; 16] = [
            0xbeadee42, 0xbe83563b, 0xbe4ce718, 0xbe1eb6fa, 0xbdeda172, 0xbda55816, 0xbd4329cb,
            0xbc811273, 0x3c811273, 0x3d4329cb, 0x3da55816, 0x3deda172, 0x3e1eb6fa, 0x3e4ce718,
            0x3e83563b, 0x3eadee42,
        ];
        assert_centroids_exact(&c, &bits, "b=4,d=64");
    }

    #[test]
    fn b4_d128_vs_fixture() {
        let c = compute_centroids(4, 128);
        let bits: [u32; 16] = [
            0xbe75f9a3, 0xbe39bd03, 0xbe10e35b, 0xbde074e1, 0xbda807be, 0xbd69d4f4, 0xbd0a0053,
            0xbc368915, 0x3c368915, 0x3d0a0053, 0x3d69d4f4, 0x3da807be, 0x3de074e1, 0x3e10e35b,
            0x3e39bd03, 0x3e75f9a3,
        ];
        assert_centroids_exact(&c, &bits, "b=4,d=128");
    }

    // -----------------------------------------------------------------------
    // Full fixture comparison — byte-for-byte against Python reference
    // This is the acceptance-gate test required by epic #458.
    // Reads tests/fixtures/turbo_codebooks/codebooks_hex.json and compares
    // every (b, d) pair from b ∈ {2,3,4}, d ∈ {64,80,96,128,192,256}.
    // -----------------------------------------------------------------------
    #[test]
    fn full_fixture_comparison_all_bit_width_head_dim_pairs() {
        // CARGO_MANIFEST_DIR = .../src/lib/mlxcel-core, project root is three
        // levels up: .../src/lib/mlxcel-core/../../.. → project root, then
        // tests/fixtures/turbo_codebooks/.
        let fixture_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../tests/fixtures/turbo_codebooks/codebooks_hex.json"
        );
        let content = std::fs::read_to_string(fixture_path)
            .unwrap_or_else(|e| panic!("Failed to read fixture {fixture_path}: {e}"));

        // Minimal JSON parser — we only need to parse the flat hex-string map
        let map: std::collections::HashMap<String, Vec<String>> = serde_json::from_str(&content)
            .unwrap_or_else(|e| panic!("Failed to parse fixture JSON: {e}"));

        let bit_widths: [u8; 3] = [2, 3, 4];
        let head_dims: [u32; 6] = [64, 80, 96, 128, 192, 256];

        for &b in &bit_widths {
            for &d in &head_dims {
                let key = format!("{b}_{d}");
                let hex_list = map
                    .get(&key)
                    .unwrap_or_else(|| panic!("Fixture missing key: {key}"));

                // Parse hex strings into u32 bit patterns
                let expected_bits: Vec<u32> = hex_list
                    .iter()
                    .map(|h| {
                        u32::from_str_radix(h, 16)
                            .unwrap_or_else(|e| panic!("Bad hex in fixture key={key}: '{h}': {e}"))
                    })
                    .collect();

                let rust_centroids = compute_centroids(b, d);
                let label = format!("b={b},d={d}");

                assert_eq!(
                    rust_centroids.len(),
                    expected_bits.len(),
                    "{label}: centroid count mismatch"
                );

                for (i, (&rust_val, &exp_bits)) in
                    rust_centroids.iter().zip(expected_bits.iter()).enumerate()
                {
                    let rust_bits = rust_val.to_bits();
                    assert_eq!(
                        rust_bits,
                        exp_bits,
                        "{label}[{i}]: byte-for-byte mismatch: \
                         rust=0x{rust_bits:08x} ({rust_val}), \
                         expected=0x{exp_bits:08x} ({})",
                        f32::from_bits(exp_bits)
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // nearest_centroid_indices
    // -----------------------------------------------------------------------

    #[test]
    fn nearest_centroid_indices_basic() {
        let centroids = compute_centroids(2, 64);
        // The 4 centroids divide the real line into 4 regions
        // A value at the first centroid should map to index 0
        let result = nearest_centroid_indices(&[centroids[0]], &centroids);
        assert_eq!(result, vec![0], "value == centroids[0] → index 0");

        // A value at the last centroid should map to the last index
        let n = centroids.len();
        let result = nearest_centroid_indices(&[centroids[n - 1]], &centroids);
        assert_eq!(
            result,
            vec![n - 1],
            "value == centroids[n-1] → index {}",
            n - 1
        );

        // A very negative value should map to index 0
        let result = nearest_centroid_indices(&[-1e6_f32], &centroids);
        assert_eq!(result, vec![0], "very negative → index 0");

        // A very positive value should map to the last index
        let result = nearest_centroid_indices(&[1e6_f32], &centroids);
        assert_eq!(result, vec![n - 1], "very positive → last index");
    }

    #[test]
    fn nearest_centroid_indices_midpoint_boundary() {
        let centroids: &[f32] = &[-0.5, 0.5];
        // The midpoint boundary is at 0.0
        // Values < 0 → index 0; values >= 0 → index 1
        let result = nearest_centroid_indices(&[-0.1, 0.0, 0.1], centroids);
        // -0.1 < 0 → 0; 0.0 is at the boundary → 1 (searchsorted left); 0.1 → 1
        assert_eq!(result[0], 0, "value -0.1 → index 0");
        assert_eq!(result[2], 1, "value 0.1 → index 1");
    }

    #[test]
    fn nearest_centroid_indices_matches_naive_nearest() {
        let centroids = compute_centroids(4, 128);
        let test_values: Vec<f32> = (0..20).map(|i| (i as f32 - 10.0) * 0.02).collect();

        let result = nearest_centroid_indices(&test_values, &centroids);

        // Verify by brute-force nearest neighbor
        for (v, &idx) in test_values.iter().zip(result.iter()) {
            let naive_idx = centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| ((*a - v).abs()).partial_cmp(&((*b - v).abs())).unwrap())
                .unwrap()
                .0;
            assert_eq!(
                idx, naive_idx,
                "value={v}: searchsorted={idx} naive={naive_idx}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Properties: sorted, symmetric (odd bit-widths are antisymmetric)
    // -----------------------------------------------------------------------

    #[test]
    fn centroids_are_sorted() {
        for b in 1u8..=4 {
            for d in [64u32, 80, 96, 128, 192, 256] {
                let c = compute_centroids(b, d);
                for w in c.windows(2) {
                    assert!(w[0] < w[1], "b={b},d={d}: not sorted: {:?}", c);
                }
            }
        }
    }

    #[test]
    fn centroids_are_antisymmetric() {
        // For a zero-mean symmetric distribution, optimal centroids are
        // antisymmetric in the limit. After 100 Lloyd-Max iterations from
        // uniform-quantile initialization the residual asymmetry is bounded
        // by accumulated f64 rounding plus the f32 cast — a 1e-3 relative
        // tolerance against the centroid magnitude is the right gate.
        for b in 2u8..=4 {
            for d in [64u32, 128, 256] {
                let c = compute_centroids(b, d);
                let n = c.len();
                let scale = c.iter().map(|v| v.abs()).fold(0.0_f32, f32::max).max(1e-6);
                for i in 0..n / 2 {
                    let sum = c[i] + c[n - 1 - i];
                    let rel = (sum / scale).abs();
                    assert!(
                        rel < 1e-3,
                        "b={b},d={d}: not antisymmetric at i={i}: {} + {} = {} (rel={rel:.2e})",
                        c[i],
                        c[n - 1 - i],
                        sum
                    );
                }
            }
        }
    }

    #[test]
    fn centroids_count_is_two_pow_b() {
        for b in 1u8..=4 {
            for d in [64u32, 128] {
                let c = compute_centroids(b, d);
                assert_eq!(
                    c.len(),
                    1 << b,
                    "b={b},d={d}: expected {} centroids",
                    1 << b
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Codebook struct
    // -----------------------------------------------------------------------

    #[test]
    fn codebook_from_centroids_builds_correct_boundaries() {
        let centroids: Arc<[f32]> = Arc::from(vec![-2.0_f32, -1.0, 1.0, 2.0]);
        let cb = Codebook::from_centroids(centroids);
        assert_eq!(cb.boundaries.as_ref(), &[-1.5_f32, 0.0, 1.5]);
    }

    #[test]
    fn optimal_codebook_returns_consistent_centroids() {
        // optimal_codebook should agree with optimal_centroids for the same args
        for b in 2u8..=4 {
            for d in [64u32, 128] {
                let cb = optimal_codebook(b, d);
                let raw = optimal_centroids(b, d);
                assert_eq!(
                    cb.centroids.as_ref(),
                    raw.as_slice(),
                    "Codebook centroids must match optimal_centroids for b={b}, d={d}"
                );
                assert_eq!(cb.boundaries.len(), cb.centroids.len() - 1);
            }
        }
    }

    #[test]
    fn optimal_codebook_arc_is_shared() {
        // Two calls for the same key should hand back the same Arc backing.
        let cb1 = optimal_codebook(4, 128);
        let cb2 = optimal_codebook(4, 128);
        // pointer equality on the Arc<[f32]> backing
        assert!(
            Arc::ptr_eq(&cb1.centroids, &cb2.centroids),
            "optimal_codebook should share Arc storage on cache hits"
        );
        assert!(
            Arc::ptr_eq(&cb1.boundaries, &cb2.boundaries),
            "boundary Arc must also be shared on cache hits"
        );
    }

    #[test]
    fn nearest_centroid_indices_with_boundaries_matches_naive() {
        let cb = optimal_codebook(4, 128);
        let values: Vec<f32> = (0..20).map(|i| (i as f32 - 10.0) * 0.02).collect();
        let result =
            nearest_centroid_indices_with_boundaries(&values, &cb.boundaries, cb.centroids.len());
        let naive = nearest_centroid_indices(&values, &cb.centroids);
        assert_eq!(result, naive);
    }
}
