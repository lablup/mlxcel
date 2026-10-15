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

//! Test-only failure diagnostics for the `Turbo4Delegated` attention parity
//! tests in `cache/turbo_tests.rs` (#1712).
//!
//! `delegated_steel_envelope_matches_cold_only_fused_over_200_steps` failed
//! twice with the identical value `RMS 4.6824e-1` at step 0 and then never
//! again, and the assertion message carried nothing that could tell a wrong
//! kernel output from diverged cache state or from a one-off device fault.
//! [`diagnose_delegated_mismatch`] builds that classification at the moment of
//! failure: which side is wrong against the graph reference, whether the two
//! caches still hold the same state, whether every path reproduces in-process
//! on the same inputs, where the largest deviation sits, and how far the
//! common "a contribution was dropped" outputs sit from the reference, so a
//! single recurrence is enough to classify it.
//!
//! Everything here is graph-level or goes through the public launchers, so it
//! compiles and runs on every backend; the Metal-only launchers simply report
//! "not dispatched" off macOS.

use std::fmt::{self, Write as _};

use cxx::UniquePtr;

use super::KVCache;
use super::turbo::{TurboQuantParams, sparse_v};
use crate::dtype;
use crate::ffi::{self, MlxArray};

/// Read an array back as a flat `f32` vector.
pub(super) fn flatten_fp32(arr: &MlxArray) -> Vec<f32> {
    let a = ffi::astype(arr, dtype::FLOAT32);
    ffi::eval(&a);
    ffi::array_to_raw_bytes(&a)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Root mean square of `values` (the distance of `values` from all zeros).
pub(super) fn rms_of(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = values.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    (sum_sq / values.len() as f64).sqrt() as f32
}

/// Element-wise comparison of two flat outputs of the same `[.., D]` layout.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ParityDiff {
    /// RMS of `lhs - rhs`; the quantity the parity assertions bound.
    pub rms: f32,
    /// Largest `|lhs - rhs|`.
    pub max_abs: f32,
    /// Flat index of `max_abs`.
    pub max_idx: usize,
    /// Head dimension used to split `max_idx` into `(row, d)`.
    pub dim: usize,
    /// `lhs[max_idx]`.
    pub lhs: f32,
    /// `rhs[max_idx]`.
    pub rhs: f32,
    /// Whether either side holds a NaN or an infinity.
    pub non_finite: bool,
}

/// Compare `lhs` against `rhs`. Lengths must match; the caller asserts that
/// first so a shape mismatch keeps its own message.
pub(super) fn parity_diff(lhs: &[f32], rhs: &[f32], dim: usize) -> ParityDiff {
    debug_assert_eq!(lhs.len(), rhs.len(), "parity_diff needs equal lengths");
    let mut sum_sq = 0.0_f64;
    let mut max_abs = 0.0_f32;
    let mut max_idx = 0_usize;
    let mut non_finite = false;
    for (i, (x, y)) in lhs.iter().zip(rhs.iter()).enumerate() {
        non_finite |= !x.is_finite() || !y.is_finite();
        let d = x - y;
        sum_sq += f64::from(d) * f64::from(d);
        if d.abs() > max_abs {
            max_abs = d.abs();
            max_idx = i;
        }
    }
    let n = lhs.len().min(rhs.len());
    let rms = if n == 0 {
        0.0
    } else {
        (sum_sq / n as f64).sqrt() as f32
    };
    ParityDiff {
        rms,
        max_abs,
        max_idx,
        dim: dim.max(1),
        lhs: lhs.get(max_idx).copied().unwrap_or(0.0),
        rhs: rhs.get(max_idx).copied().unwrap_or(0.0),
        non_finite,
    }
}

impl fmt::Display for ParityDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "rms={:.4e} max_abs={:.4e} at flat[{}] (row {}, d {}): {:.6} vs {:.6}{}",
            self.rms,
            self.max_abs,
            self.max_idx,
            self.max_idx / self.dim,
            self.max_idx % self.dim,
            self.lhs,
            self.rhs,
            if self.non_finite { " NON-FINITE" } else { "" }
        )
    }
}

/// The kernel inputs `KVCache::update_and_turbo4_delegated_attention` slices
/// out of an already-updated `Turbo4Delegated` cache, built the same way the
/// wrapper builds them.
pub(super) struct DelegatedKernelInputs {
    pub k_slice: UniquePtr<MlxArray>,
    pub v_packed: Option<UniquePtr<MlxArray>>,
    pub v_rescale: Option<UniquePtr<MlxArray>>,
    pub hot_v: Option<UniquePtr<MlxArray>>,
    pub cold_offset: i32,
    pub hot_offset: i32,
}

impl DelegatedKernelInputs {
    /// Slice the visible kernel inputs out of `cache`.
    ///
    /// Panics when the cache is not an updated `Turbo4Delegated` cache, which
    /// is a test-construction error.
    pub(super) fn from_cache(cache: &KVCache) -> Self {
        let cold_offset = cache.cold_offset();
        let hot_offset = cache.seq_len() - cold_offset;
        let k_buf = cache
            .keys
            .as_ref()
            .expect("unified keys must exist after update on Turbo4Delegated");
        let ks = ffi::array_shape(k_buf);
        let k_slice = ffi::slice(
            k_buf,
            &[0, 0, 0, 0],
            &[ks[0], ks[1], cache.seq_len(), ks[3]],
        );
        let leading = |arr: &MlxArray, len: i32| {
            let s = ffi::array_shape(arr);
            ffi::slice(arr, &[0, 0, 0, 0], &[s[0], s[1], len, s[3]])
        };
        let v_packed = (cold_offset > 0).then(|| {
            let vp = cache
                .v_packed
                .as_ref()
                .expect("v_packed must exist when cold_offset > 0");
            leading(vp, cold_offset)
        });
        let v_rescale = (cold_offset > 0).then(|| {
            let vr = cache
                .v_rescale
                .as_ref()
                .expect("v_rescale must exist when cold_offset > 0");
            leading(vr, cold_offset)
        });
        let hot_v = (hot_offset > 0).then(|| {
            let hv = cache
                .values
                .as_ref()
                .expect("hot values must exist when hot_offset > 0");
            leading(hv, hot_offset)
        });
        Self {
            k_slice,
            v_packed,
            v_rescale,
            hot_v,
            cold_offset,
            hot_offset,
        }
    }
}

/// One of the delegated attention routes, callable on explicit inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DelegatedPath {
    /// `attention_turbo4_delegated_dequant_sdpa`, the wrapper's default route.
    DequantSdpa,
    /// `attention_turbo4_delegated_steel`, the steel-envelope Metal kernel.
    Steel,
    /// `attention_turbo4_delegated_fused`, the cold-only kernel composition.
    ColdOnlyFused,
}

impl DelegatedPath {
    pub(super) const ALL: [Self; 3] = [Self::DequantSdpa, Self::Steel, Self::ColdOnlyFused];

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::DequantSdpa => "dequant-sdpa",
            Self::Steel => "steel",
            Self::ColdOnlyFused => "cold-only",
        }
    }

    /// Run this route on `inputs`. `None` means the launcher declined to
    /// dispatch (non-macOS, `MLXCEL_SPARSE_V_KERNEL=0`, or a non-power-of-2
    /// head dim).
    pub(super) fn run(
        self,
        inputs: &DelegatedKernelInputs,
        q: &MlxArray,
        params: &TurboQuantParams,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> Option<UniquePtr<MlxArray>> {
        let k = &inputs.k_slice;
        let vp = inputs.v_packed.as_deref();
        let vr = inputs.v_rescale.as_deref();
        let hv = inputs.hot_v.as_deref();
        let (cold, hot) = (inputs.cold_offset, inputs.hot_offset);
        match self {
            Self::DequantSdpa => sparse_v::attention_turbo4_delegated_dequant_sdpa(
                q, k, vp, vr, hv, params, cold, hot, scale, mask,
            ),
            Self::Steel => sparse_v::attention_turbo4_delegated_steel(
                q,
                k,
                vp,
                vr,
                hv,
                params,
                cold,
                hot,
                scale,
                mask,
                sparse_v::threshold(),
            ),
            Self::ColdOnlyFused => sparse_v::attention_turbo4_delegated_fused(
                q,
                k,
                vp,
                vr,
                hv,
                params,
                cold,
                hot,
                scale,
                mask,
                sparse_v::threshold(),
            ),
        }
    }
}

/// The call that failed: what was compared and on which inputs.
pub(super) struct MismatchContext<'a> {
    pub step: usize,
    pub head_dim: usize,
    pub q: &'a MlxArray,
    pub scale: f32,
    pub mask: Option<&'a MlxArray>,
    /// The outputs the assertion compared, labelled, as flat `f32`.
    pub observed: &'a [(&'a str, &'a [f32])],
}

/// Largest `|a - b|` over two arrays of identical shape, or a note when the
/// shapes differ.
fn max_abs_between(a: &MlxArray, b: &MlxArray) -> String {
    let (sa, sb) = (ffi::array_shape(a), ffi::array_shape(b));
    if sa != sb {
        return format!("shape {sa:?} vs {sb:?}");
    }
    let (fa, fb) = (flatten_fp32(a), flatten_fp32(b));
    let max = fa
        .iter()
        .zip(fb.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max);
    format!("{max:.3e}")
}

fn optional_max_abs(a: Option<&MlxArray>, b: Option<&MlxArray>) -> String {
    match (a, b) {
        (Some(a), Some(b)) => max_abs_between(a, b),
        (None, None) => "absent on both".to_string(),
        _ => "present on one side only".to_string(),
    }
}

/// Copy of `v` with rows `[from, to)` of axis 2 replaced by zeros.
fn zero_rows(v: &MlxArray, from: i32, to: i32) -> UniquePtr<MlxArray> {
    let s = ffi::array_shape(v);
    let zeros = ffi::zeros(&[s[0], s[1], to - from, s[3]], ffi::array_dtype(v));
    let head = (from > 0).then(|| ffi::slice(v, &[0, 0, 0, 0], &[s[0], s[1], from, s[3]]));
    let tail = (to < s[2]).then(|| ffi::slice(v, &[0, 0, to, 0], &[s[0], s[1], s[2], s[3]]));
    let mut out = match head {
        Some(h) => crate::concatenate(&h, &zeros, 2),
        None => zeros,
    };
    if let Some(t) = tail {
        out = crate::concatenate(&out, &t, 2);
    }
    out
}

/// Build the classification report for a failed delegated parity assertion.
///
/// `cache_a` and `cache_b` are the two caches the compared outputs came from,
/// both already updated for this step; they were fed identical tokens, so
/// their visible state must match. The graph reference is taken from
/// `cache_b` through `delegated_graph_attention`, which shares no custom
/// kernel with any of the routes under test.
pub(super) fn diagnose_delegated_mismatch(
    ctx: &MismatchContext<'_>,
    cache_a: &mut KVCache,
    cache_b: &mut KVCache,
) -> String {
    let mut r = String::new();
    let _ = writeln!(r, "--- #1712 diagnostics (step {}) ---", ctx.step);
    let _ = writeln!(
        r,
        "gates: dequant_sdpa_enabled={} sparse_v_kernel_enabled={} sparse_v_threshold={:e} \
         fp16_fast_path(a,b)=({},{})",
        sparse_v::turbo4_delegated_dequant_sdpa_enabled(),
        sparse_v::kernel_enabled(),
        sparse_v::threshold(),
        cache_a.delegated_fp16_fast_path,
        cache_b.delegated_fp16_fast_path,
    );

    let in_a = DelegatedKernelInputs::from_cache(cache_a);
    let in_b = DelegatedKernelInputs::from_cache(cache_b);
    let _ = writeln!(
        r,
        "offsets: a cold={} hot={} | b cold={} hot={}",
        in_a.cold_offset, in_a.hot_offset, in_b.cold_offset, in_b.hot_offset
    );
    let _ = writeln!(
        r,
        "cache state a vs b (max |delta|): K={} v_packed={} v_rescale={} hot_v={}",
        max_abs_between(&in_a.k_slice, &in_b.k_slice),
        optional_max_abs(in_a.v_packed.as_deref(), in_b.v_packed.as_deref()),
        optional_max_abs(in_a.v_rescale.as_deref(), in_b.v_rescale.as_deref()),
        optional_max_abs(in_a.hot_v.as_deref(), in_b.hot_v.as_deref()),
    );

    let reference = cache_b.delegated_graph_attention(ctx.q, ctx.scale, ctx.mask);
    let ref_flat = flatten_fp32(&reference);
    let _ = writeln!(
        r,
        "graph reference (cache b): rms={:.4e}, which is also how far an all-zero output sits from it",
        rms_of(&ref_flat)
    );
    for (label, flat) in ctx.observed {
        if flat.len() == ref_flat.len() {
            let _ = writeln!(
                r,
                "observed {label} vs graph reference: {}",
                parity_diff(flat, &ref_flat, ctx.head_dim)
            );
        }
    }

    // Signatures of the wrong answers a dropped contribution would produce, so
    // a recurrence's RMS can be matched against them without a reproduction.
    let (k_full, v_full) = cache_b.fetch_turbo4_delegated();
    let cold = in_b.cold_offset;
    let total = cold + in_b.hot_offset;
    let mut signature = |name: &str, out: UniquePtr<MlxArray>| {
        let flat = flatten_fp32(&out);
        if flat.len() == ref_flat.len() {
            let _ = writeln!(
                r,
                "signature {name}: rms vs graph reference = {:.4e}",
                parity_diff(&flat, &ref_flat, ctx.head_dim).rms
            );
        }
    };
    if cold > 0 && total > cold {
        signature(
            "hot V dropped",
            crate::layers::attention(
                ctx.q,
                &k_full,
                &zero_rows(&v_full, cold, total),
                ctx.scale,
                ctx.mask,
                0.0,
                0,
            ),
        );
        signature(
            "cold V dropped",
            crate::layers::attention(
                ctx.q,
                &k_full,
                &zero_rows(&v_full, 0, cold),
                ctx.scale,
                ctx.mask,
                0.0,
                0,
            ),
        );
    }

    // In-process re-run of every route on both caches' current inputs. A
    // route that reproduces its first answer here is deterministic on these
    // inputs; one that now matches the reference was a transient fault.
    let params_a = cache_a.turbo_params.clone();
    let params_b = cache_b.turbo_params.clone();
    for (side, inputs, params) in [("a", &in_a, &params_a), ("b", &in_b, &params_b)] {
        let Some(params) = params.as_ref() else {
            let _ = writeln!(r, "rerun on cache {side}: turbo_params unset, skipped");
            continue;
        };
        for path in DelegatedPath::ALL {
            match path.run(inputs, ctx.q, params, ctx.scale, ctx.mask) {
                Some(out) => {
                    let flat = flatten_fp32(&out);
                    let mut line = format!("rerun {} on cache {side}:", path.label());
                    if flat.len() == ref_flat.len() {
                        let _ = write!(
                            line,
                            " vs graph reference {}",
                            parity_diff(&flat, &ref_flat, ctx.head_dim)
                        );
                    }
                    for (label, obs) in ctx.observed {
                        if obs.len() == flat.len() {
                            let _ = write!(
                                line,
                                "; vs observed {label} rms={:.4e}",
                                parity_diff(&flat, obs, ctx.head_dim).rms
                            );
                        }
                    }
                    let _ = writeln!(r, "{line}");
                }
                None => {
                    let _ = writeln!(r, "rerun {} on cache {side}: not dispatched", path.label());
                }
            }
        }
    }
    r
}
