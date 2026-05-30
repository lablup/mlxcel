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

//! `InterpolateOp` — Axis A operation #9.
//!
//! Blends two donor checkpoints tensor-by-tensor and writes the
//! result back into the base [`WeightMap`]. Supported methods:
//!
//! - **LERP** (linear): `out = (1 - t) * a + t * b`.
//! - **SLERP** (spherical-linear): treat each tensor as a flat
//!   vector in `R^N`, compute the angle `θ` between the two donor
//!   vectors, and interpolate angularly:
//!   `out = sin((1-t)θ)/sin(θ) * a + sin(tθ)/sin(θ) * b`.
//!
//! ## Numerical stability — SLERP epsilon and parallel-vector fallback
//!
//! SLERP is unstable when `sin(θ)` is near zero, i.e. when the two
//! donor vectors are nearly parallel (`cos θ ≈ 1`) or nearly
//! anti-parallel (`cos θ ≈ -1`). Both regimes produce a near-zero
//! denominator and would amplify rounding noise into the output. We
//! treat both as the same fallback case: if `|cos θ| > 1 - ε` for
//! [`SLERP_PARALLEL_EPSILON`] = `1e-6`, we silently fall back to LERP
//! on that tensor.
//!
//! The `1e-6` threshold is conservative for f32 arithmetic — the
//! resulting `sin θ` is at minimum `sqrt(2*ε) ≈ 1.4e-3`, which keeps
//! the division safely outside f32 denormal range. Anything tighter
//! risks NaNs when two donor tensors agree exactly on direction; the
//! issue body explicitly calls out this edge case ("when the two
//! tensors are (nearly) parallel — fall back to LERP with a
//! configurable epsilon").
//!
//! ## Dtype policy
//!
//! - **Float dtypes (`f32`, `f16`, `bf16`)**: the actual math is
//!   carried out in `f32` for numerical accuracy and cast back to
//!   the base tensor's dtype before being written into the map. This
//!   matches the bf16 → f16 conversion policy in the consolidated
//!   loader (`docs/apple-silicon-precision.md`) and avoids losing
//!   precision when blending two bf16 checkpoints.
//! - **Quantized layouts (`u8`/`u32`/integer-coded tensors)**:
//!   interpolation is *not* defined on the packed quantized
//!   representation. The first-cut policy (§"Scope")
//!   is to require the base, source_a, and source_b to share the
//!   exact same dtype. Any integer dtype encountered errors with a
//!   clear message directing the user to either dequantize ahead of
//!   time or supply matching float donors. A future iteration can
//!   add a dequant → interp → requant path once the requantization
//!   infrastructure exists.
//!
//! ## Donor checkpoint loading
//!
//! Donor `.safetensors` files are loaded on every `apply` invocation.
//! This is intentional: surgery runs at most once per model load, so
//! caching donors across calls would only matter if the same op were
//! reused for multiple loads — a workflow that is not on the current
//! roadmap, and is also explicitly opt-out-able by passing
//! pre-resolved donors through [`InterpolateOp::with_donors`] in
//! unit tests.
//!
//! Avoiding the cache also sidesteps a `Send + Sync` issue: the
//! underlying [`UniquePtr<MlxArray>`] in the donor [`WeightMap`] is
//! not thread-safe by default, and the [`SurgeryOp`] trait requires
//! `Send + Sync`. Reloading the file each time keeps the op fully
//! stateless and trivially threadsafe.

use std::path::{Path, PathBuf};

use globset::{Glob, GlobMatcher};
use mlxcel_core::weights::{load_safetensors, WeightMap};
use mlxcel_core::{MlxArray, UniquePtr};

use crate::config::InterpolateMethod;
use crate::{SurgeryError, SurgeryOp};

/// `|cos θ|` above this threshold is treated as "parallel /
/// anti-parallel" and falls back to LERP. See module-level docs for
/// the numerical reasoning.
pub const SLERP_PARALLEL_EPSILON: f32 = 1e-6;

/// Numeric label used in [`SurgeryError::DtypeMismatch`] for a dtype
/// integer code. Mirrors the constants in [`mlxcel_core::dtype`]
/// without taking a public dependency on that module's exhaustive
/// list — surgery only needs the names, not the codes themselves.
fn dtype_label(code: i32) -> String {
    match code {
        0 => "bool".to_string(),
        1 => "uint8".to_string(),
        2 => "uint16".to_string(),
        3 => "uint32".to_string(),
        4 => "uint64".to_string(),
        5 => "int8".to_string(),
        6 => "int16".to_string(),
        7 => "int32".to_string(),
        8 => "int64".to_string(),
        9 => "float16".to_string(),
        10 => "float32".to_string(),
        11 => "float64".to_string(),
        12 => "bfloat16".to_string(),
        13 => "complex64".to_string(),
        other => format!("dtype#{other}"),
    }
}

/// `true` iff `code` is a float dtype that this op can interpolate
/// in. Used to gate the "first cut" quantized policy described in
/// the module-level docs.
///
/// The valid range `9..=12` covers `float16`, `float32`, `float64`,
/// and `bfloat16` per [`mlxcel_core::dtype`]. Any other dtype code
/// (integer / boolean / packed quantized payload) is rejected.
fn is_float_dtype(code: i32) -> bool {
    // Used by: InterpolateOp::apply — guards against integer /
    // quantized donor inputs.
    matches!(code, 9..=12)
}

/// Interpolate two donor checkpoints into the base [`WeightMap`].
///
/// One `InterpolateOp` is materialized per `op: interpolate` block
/// in a parsed surgery YAML (see [`crate::config::OpSpec::Interpolate`]).
/// The two donor `.safetensors` files are loaded on every `apply`
/// invocation — see the module-level docs for rationale.
///
/// Used by: `materialize_op` in [`crate::config`] (factory hook),
/// [`crate::SurgeryPipeline`] (registration), unit tests in this
/// module (`apply_with_donors` test path)
#[derive(Debug)]
pub struct InterpolateOp {
    /// Compiled glob matcher derived from the YAML `pattern` field.
    matcher: GlobMatcher,
    /// Source pattern string, preserved for diagnostic messages.
    pattern: String,
    /// Donor A safetensors path. Loaded on every `apply` call.
    source_a: PathBuf,
    /// Donor B safetensors path. Loaded on every `apply` call.
    source_b: PathBuf,
    /// Mixing ratio `t`. Validated to `[0.0, 1.0]` by the config
    /// parser; defended again here against direct construction.
    ratio: f32,
    /// Interpolation method (SLERP or LERP).
    method: InterpolateMethod,
}

impl InterpolateOp {
    /// Construct an `InterpolateOp` for the given pattern and donor
    /// files.
    ///
    /// `pattern` is parsed via `globset::Glob`. Path resolution and
    /// existence checks are done by the YAML parser; this
    /// constructor trusts the caller and only defends against
    /// obviously invalid input (malformed glob, out-of-range ratio,
    /// non-finite ratio).
    pub fn new(
        pattern: &str,
        source_a: PathBuf,
        source_b: PathBuf,
        ratio: f32,
        method: InterpolateMethod,
    ) -> Result<Self, SurgeryError> {
        let glob = Glob::new(pattern).map_err(|e| {
            SurgeryError::Other(anyhow::anyhow!(
                "interpolate: malformed glob pattern {pattern:?}: {e}"
            ))
        })?;
        if !ratio.is_finite() || !(0.0..=1.0).contains(&ratio) {
            return Err(SurgeryError::Other(anyhow::anyhow!(
                "interpolate: ratio must be a finite number in [0.0, 1.0], got {ratio}"
            )));
        }
        Ok(Self {
            matcher: glob.compile_matcher(),
            pattern: pattern.to_string(),
            source_a,
            source_b,
            ratio,
            method,
        })
    }

    /// Path the op was configured with. Useful for diagnostics.
    pub fn source_a(&self) -> &Path {
        self.source_a.as_path()
    }

    /// Path the op was configured with. Useful for diagnostics.
    pub fn source_b(&self) -> &Path {
        self.source_b.as_path()
    }

    /// Glob pattern this op was configured with. Useful for logging.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Apply the operation with explicitly-provided donor weight
    /// maps. This is the in-memory path used by both `SurgeryOp::apply`
    /// (which loads donors from disk first) and by the unit tests
    /// (which inject synthesized donors directly).
    ///
    /// Splitting this out from `SurgeryOp::apply` keeps the FFI
    /// loading logic isolated and lets tests cover SLERP / LERP
    /// numerics without touching the file system or contending with
    /// the [`UniquePtr<MlxArray>`] `!Send` / `!Sync` constraint that
    /// would otherwise leak into the public op state.
    pub(crate) fn apply_with_donors(
        &self,
        weights: &mut WeightMap,
        donor_a: &WeightMap,
        donor_b: &WeightMap,
    ) -> Result<(), SurgeryError> {
        let matching_keys: Vec<String> = weights
            .keys()
            .filter(|k| self.matcher.is_match(k.as_str()))
            .cloned()
            .collect();

        if matching_keys.is_empty() {
            return Err(SurgeryError::Other(anyhow::anyhow!(
                "interpolate: glob pattern {pattern:?} matched zero tensors in the base WeightMap",
                pattern = self.pattern,
            )));
        }

        for key in matching_keys {
            let base = weights
                .get(&key)
                .expect("key from matching_keys must still be present");
            let donor_a_tensor = donor_a
                .get(&key)
                .ok_or_else(|| SurgeryError::TensorNotFound(format!("source_a[{key}]")))?;
            let donor_b_tensor = donor_b
                .get(&key)
                .ok_or_else(|| SurgeryError::TensorNotFound(format!("source_b[{key}]")))?;

            assert_shape_matches(&key, "source_a", base, donor_a_tensor)?;
            assert_shape_matches(&key, "source_b", base, donor_b_tensor)?;
            assert_dtype_matches(&key, "source_a", base, donor_a_tensor)?;
            assert_dtype_matches(&key, "source_b", base, donor_b_tensor)?;

            let base_dtype = mlxcel_core::array_dtype(base);
            if !is_float_dtype(base_dtype) {
                return Err(SurgeryError::Other(anyhow::anyhow!(
                    "interpolate: tensor {key:?} has non-float dtype {} — quantized layouts are not yet supported; please dequantize donors first",
                    dtype_label(base_dtype)
                )));
            }

            let out = interpolate_tensor(
                donor_a_tensor,
                donor_b_tensor,
                self.ratio,
                self.method,
                base_dtype,
            );
            weights.insert(key, out);
        }
        Ok(())
    }
}

impl SurgeryOp for InterpolateOp {
    fn apply(&self, weights: &mut WeightMap, _cfg: &serde_json::Value) -> Result<(), SurgeryError> {
        let donor_a = load_safetensors(&self.source_a).map_err(|e| {
            SurgeryError::Other(anyhow::anyhow!(
                "interpolate: failed to load source_a {}: {e}",
                self.source_a.display(),
            ))
        })?;
        let donor_b = load_safetensors(&self.source_b).map_err(|e| {
            SurgeryError::Other(anyhow::anyhow!(
                "interpolate: failed to load source_b {}: {e}",
                self.source_b.display(),
            ))
        })?;
        self.apply_with_donors(weights, &donor_a, &donor_b)
    }

    fn name(&self) -> &'static str {
        "interpolate"
    }
}

/// Verify two tensors share an identical shape, reporting through
/// [`SurgeryError::ShapeMismatch`] if not.
fn assert_shape_matches(
    key: &str,
    donor_label: &str,
    base: &MlxArray,
    donor: &MlxArray,
) -> Result<(), SurgeryError> {
    let base_shape: Vec<usize> = mlxcel_core::array_shape(base)
        .iter()
        .map(|&d| d as usize)
        .collect();
    let donor_shape: Vec<usize> = mlxcel_core::array_shape(donor)
        .iter()
        .map(|&d| d as usize)
        .collect();
    if base_shape != donor_shape {
        return Err(SurgeryError::ShapeMismatch {
            key: format!("{key} ({donor_label})"),
            expected: base_shape,
            actual: donor_shape,
        });
    }
    Ok(())
}

/// Verify two tensors share identical dtypes, reporting through
/// [`SurgeryError::DtypeMismatch`] if not.
fn assert_dtype_matches(
    key: &str,
    donor_label: &str,
    base: &MlxArray,
    donor: &MlxArray,
) -> Result<(), SurgeryError> {
    let base_dtype = mlxcel_core::array_dtype(base);
    let donor_dtype = mlxcel_core::array_dtype(donor);
    if base_dtype != donor_dtype {
        return Err(SurgeryError::DtypeMismatch {
            key: format!("{key} ({donor_label})"),
            expected: dtype_label(base_dtype),
            actual: dtype_label(donor_dtype),
        });
    }
    Ok(())
}

/// Apply [`InterpolateMethod`] to a single tensor pair, returning a
/// fresh [`UniquePtr<MlxArray>`] in the requested output dtype.
///
/// The math is performed in f32 regardless of the input dtype so
/// bf16 / f16 donors do not accumulate rounding error during the
/// dot-product / norm steps. The output is cast back to
/// `out_dtype` via `astype` before being returned to the caller.
fn interpolate_tensor(
    a: &MlxArray,
    b: &MlxArray,
    ratio: f32,
    method: InterpolateMethod,
    out_dtype: i32,
) -> UniquePtr<MlxArray> {
    // Cast both donors to f32 for the math. `astype` is a no-op when
    // the input is already f32. FLOAT32 = 10 (mirrors
    // mlxcel_core::dtype::FLOAT32).
    let a_f32 = mlxcel_core::astype(a, 10);
    let b_f32 = mlxcel_core::astype(b, 10);

    let blended_f32 = match method {
        InterpolateMethod::Lerp => lerp_in_f32(&a_f32, &b_f32, ratio),
        InterpolateMethod::Slerp => slerp_in_f32(&a_f32, &b_f32, ratio),
    };

    // Convert back to the base dtype (no-op if already f32).
    if out_dtype == 10 {
        blended_f32
    } else {
        mlxcel_core::astype(&blended_f32, out_dtype)
    }
}

/// LERP in f32: `out = (1 - t) * a + t * b`.
///
/// Scalar coefficients are materialized once each, dispatch is
/// element-wise — same shape contract as the inputs.
fn lerp_in_f32(a: &MlxArray, b: &MlxArray, t: f32) -> UniquePtr<MlxArray> {
    // Use mlxcel-core's policy helper so the scalar broadcast adopts
    // the input dtype (here always f32 because we cast above).
    let one_minus_t = mlxcel_core::multiply_scalar(a, 1.0 - t);
    let t_b = mlxcel_core::multiply_scalar(b, t);
    mlxcel_core::add(&one_minus_t, &t_b)
}

/// SLERP in f32 with parallel-vector LERP fallback.
///
/// Operates on the *flat* representation: regardless of the input
/// tensor rank we treat both operands as vectors in `R^N` for the
/// dot product and norm computations, which matches the standard
/// "flatten → unit-normalize → angularly interpolate → re-magnify"
/// recipe described in the issue body. Element-wise multiplications
/// and additions naturally preserve the original tensor shape.
fn slerp_in_f32(a: &MlxArray, b: &MlxArray, t: f32) -> UniquePtr<MlxArray> {
    // dot = sum(a * b)
    let dot_tensor = {
        let prod = mlxcel_core::multiply(a, b);
        mlxcel_core::sum_all(&prod)
    };
    // ||a||^2 = sum(a * a), ||b||^2 = sum(b * b)
    let norm_a_sq_tensor = {
        let sq = mlxcel_core::square(a);
        mlxcel_core::sum_all(&sq)
    };
    let norm_b_sq_tensor = {
        let sq = mlxcel_core::square(b);
        mlxcel_core::sum_all(&sq)
    };

    // Materialize scalars on the host so we can branch on the angle.
    // `eval` is implicit in `item_f32` for the returned scalar.
    let dot = mlxcel_core::item_f32(&dot_tensor);
    let norm_a_sq = mlxcel_core::item_f32(&norm_a_sq_tensor);
    let norm_b_sq = mlxcel_core::item_f32(&norm_b_sq_tensor);

    // Either zero-magnitude tensor leaves SLERP undefined; fall back
    // to LERP. This naturally also handles the degenerate case where
    // both donors are all-zero.
    if !norm_a_sq.is_finite()
        || !norm_b_sq.is_finite()
        || norm_a_sq <= f32::MIN_POSITIVE
        || norm_b_sq <= f32::MIN_POSITIVE
    {
        return lerp_in_f32(a, b, t);
    }

    let norm_a = norm_a_sq.sqrt();
    let norm_b = norm_b_sq.sqrt();
    let cos_theta_raw = dot / (norm_a * norm_b);
    // Clamp to the valid range of acos to guard against rounding
    // that pushes the cosine slightly outside [-1, 1].
    let cos_theta = cos_theta_raw.clamp(-1.0, 1.0);

    if cos_theta.abs() >= 1.0 - SLERP_PARALLEL_EPSILON {
        // Parallel or anti-parallel — division by sin(theta) is
        // unstable, fall back to LERP. The anti-parallel case
        // (cos ≈ -1) deliberately reuses LERP rather than picking an
        // arbitrary great-circle direction; the issue explicitly
        // calls this out as the desired behavior.
        return lerp_in_f32(a, b, t);
    }

    let theta = cos_theta.acos();
    let sin_theta = theta.sin();
    let coeff_a = ((1.0 - t) * theta).sin() / sin_theta;
    let coeff_b = (t * theta).sin() / sin_theta;

    let scaled_a = mlxcel_core::multiply_scalar(a, coeff_a);
    let scaled_b = mlxcel_core::multiply_scalar(b, coeff_b);
    mlxcel_core::add(&scaled_a, &scaled_b)
}

#[cfg(test)]
mod tests {
    //! Unit tests covering the acceptance criteria in
    //!
    //! - (a-1) SLERP on a small known fixture.
    //! - (a-2) LERP on a small fixture.
    //! - (a-3) `ratio=0` returns donor A, `ratio=1` returns donor B.
    //! - (a-4) Parallel-vector edge — SLERP falls back to LERP.
    //! - (a-5) Shape mismatch / dtype mismatch / zero-match errors.

    use super::*;
    use mlxcel_core::dtype;
    use mlxcel_core::weights::WeightMap;
    use std::collections::HashMap;

    /// Helper: build a 1D float32 tensor from a Rust slice.
    fn t_f32_1d(data: &[f32]) -> UniquePtr<MlxArray> {
        let shape = [data.len() as i32];
        mlxcel_core::from_slice_f32(data, &shape)
    }

    /// Helper: read back a tensor's contents as `Vec<f32>` for
    /// assertions.
    fn read_f32(t: &MlxArray) -> Vec<f32> {
        // Force evaluation, then convert to raw bytes and reinterpret
        // as f32. MLX's `array_to_raw_bytes` flushes the lazy graph
        // and returns the tensor's contiguous byte representation,
        // which for f32 tensors is `len * 4` little-endian bytes.
        mlxcel_core::eval(t);
        let bytes = mlxcel_core::array_to_raw_bytes(t);
        assert!(
            bytes.len().is_multiple_of(4),
            "f32 byte count must be multiple of 4"
        );
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Helper: build a donor `WeightMap` with one key + tensor.
    fn donor_with(name: &str, data: &[f32]) -> WeightMap {
        let mut map: HashMap<String, UniquePtr<MlxArray>> = HashMap::new();
        map.insert(name.to_string(), t_f32_1d(data));
        map
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: {actual:?} vs {expected:?}",
        );
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "index {i}: got {a}, expected {e} (tol {tol})",
            );
        }
    }

    /// Helper: build a stub op pointing at the synthetic donor
    /// paths (the real file system is never touched because the
    /// tests call `apply_with_donors` directly).
    fn stub_op(pattern: &str, ratio: f32, method: InterpolateMethod) -> InterpolateOp {
        InterpolateOp::new(
            pattern,
            PathBuf::from("<test-donor-a>"),
            PathBuf::from("<test-donor-b>"),
            ratio,
            method,
        )
        .expect("test constructor")
    }

    // --- (a-2) LERP correctness on a small fixture. -------------------

    #[test]
    fn lerp_matches_closed_form() {
        // a = [1, 2, 3], b = [10, 20, 30], t = 0.25
        // out = 0.75*a + 0.25*b = [0.75+2.5, 1.5+5.0, 2.25+7.5] =
        //       [3.25, 6.5, 9.75]
        let mut base = WeightMap::new();
        base.insert("w".to_string(), t_f32_1d(&[1.0, 2.0, 3.0]));
        let donor_a = donor_with("w", &[1.0, 2.0, 3.0]);
        let donor_b = donor_with("w", &[10.0, 20.0, 30.0]);
        let op = stub_op("w", 0.25, InterpolateMethod::Lerp);
        op.apply_with_donors(&mut base, &donor_a, &donor_b)
            .expect("lerp apply");
        let out = read_f32(base.get("w").unwrap());
        assert_close(&out, &[3.25, 6.5, 9.75], 1e-5);
    }

    // --- (a-1) SLERP correctness on a small fixture. ------------------

    #[test]
    fn slerp_matches_closed_form_orthogonal_vectors() {
        // Orthogonal unit vectors: a = [1, 0], b = [0, 1], t = 0.5.
        // theta = pi/2, sin(theta) = 1.
        // coeff_a = sin(pi/4)/1 = sqrt(2)/2
        // coeff_b = sin(pi/4)/1 = sqrt(2)/2
        // out = [sqrt(2)/2, sqrt(2)/2]
        let mut base = WeightMap::new();
        base.insert("w".to_string(), t_f32_1d(&[1.0, 0.0]));
        let donor_a = donor_with("w", &[1.0, 0.0]);
        let donor_b = donor_with("w", &[0.0, 1.0]);
        let op = stub_op("w", 0.5, InterpolateMethod::Slerp);
        op.apply_with_donors(&mut base, &donor_a, &donor_b)
            .expect("slerp apply");
        let out = read_f32(base.get("w").unwrap());
        let half_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        assert_close(&out, &[half_sqrt2, half_sqrt2], 1e-5);
    }

    #[test]
    fn slerp_matches_closed_form_at_quarter() {
        // a = [1, 0], b = [0, 1], t = 0.25.
        // theta = pi/2.
        // coeff_a = sin(0.75 * pi/2)/sin(pi/2) = sin(3pi/8)
        // coeff_b = sin(0.25 * pi/2)/sin(pi/2) = sin(pi/8)
        let mut base = WeightMap::new();
        base.insert("w".to_string(), t_f32_1d(&[1.0, 0.0]));
        let donor_a = donor_with("w", &[1.0, 0.0]);
        let donor_b = donor_with("w", &[0.0, 1.0]);
        let op = stub_op("w", 0.25, InterpolateMethod::Slerp);
        op.apply_with_donors(&mut base, &donor_a, &donor_b)
            .expect("slerp apply");
        let out = read_f32(base.get("w").unwrap());

        let pi = std::f32::consts::PI;
        let expected_a = (3.0 * pi / 8.0).sin();
        let expected_b = (pi / 8.0).sin();
        assert_close(&out, &[expected_a, expected_b], 1e-5);
    }

    // --- (a-3) Endpoint behavior: ratio=0 -> a, ratio=1 -> b. ---------

    #[test]
    fn lerp_ratio_zero_returns_source_a() {
        let mut base = WeightMap::new();
        base.insert("w".to_string(), t_f32_1d(&[0.0, 0.0, 0.0]));
        let donor_a = donor_with("w", &[1.0, 2.0, 3.0]);
        let donor_b = donor_with("w", &[10.0, 20.0, 30.0]);
        let op = stub_op("w", 0.0, InterpolateMethod::Lerp);
        op.apply_with_donors(&mut base, &donor_a, &donor_b)
            .expect("lerp apply");
        let out = read_f32(base.get("w").unwrap());
        assert_close(&out, &[1.0, 2.0, 3.0], 1e-6);
    }

    #[test]
    fn lerp_ratio_one_returns_source_b() {
        let mut base = WeightMap::new();
        base.insert("w".to_string(), t_f32_1d(&[0.0, 0.0, 0.0]));
        let donor_a = donor_with("w", &[1.0, 2.0, 3.0]);
        let donor_b = donor_with("w", &[10.0, 20.0, 30.0]);
        let op = stub_op("w", 1.0, InterpolateMethod::Lerp);
        op.apply_with_donors(&mut base, &donor_a, &donor_b)
            .expect("lerp apply");
        let out = read_f32(base.get("w").unwrap());
        assert_close(&out, &[10.0, 20.0, 30.0], 1e-6);
    }

    #[test]
    fn slerp_ratio_zero_returns_source_a() {
        let mut base = WeightMap::new();
        base.insert("w".to_string(), t_f32_1d(&[0.0, 0.0]));
        let donor_a = donor_with("w", &[1.0, 0.0]);
        let donor_b = donor_with("w", &[0.0, 1.0]);
        let op = stub_op("w", 0.0, InterpolateMethod::Slerp);
        op.apply_with_donors(&mut base, &donor_a, &donor_b)
            .expect("slerp apply");
        let out = read_f32(base.get("w").unwrap());
        assert_close(&out, &[1.0, 0.0], 1e-5);
    }

    #[test]
    fn slerp_ratio_one_returns_source_b() {
        let mut base = WeightMap::new();
        base.insert("w".to_string(), t_f32_1d(&[0.0, 0.0]));
        let donor_a = donor_with("w", &[1.0, 0.0]);
        let donor_b = donor_with("w", &[0.0, 1.0]);
        let op = stub_op("w", 1.0, InterpolateMethod::Slerp);
        op.apply_with_donors(&mut base, &donor_a, &donor_b)
            .expect("slerp apply");
        let out = read_f32(base.get("w").unwrap());
        assert_close(&out, &[0.0, 1.0], 1e-5);
    }

    // --- (a-4) Parallel-vector edge: SLERP gracefully falls back. -----

    #[test]
    fn slerp_parallel_vectors_falls_back_to_lerp() {
        // Two donors aligned on the same direction; LERP and SLERP
        // must produce numerically the same output.
        let mut base = WeightMap::new();
        base.insert("w".to_string(), t_f32_1d(&[0.0, 0.0, 0.0]));
        let donor_a = donor_with("w", &[1.0, 2.0, 3.0]);
        let donor_b = donor_with("w", &[2.0, 4.0, 6.0]); // exactly 2x donor a
        let op = stub_op("w", 0.5, InterpolateMethod::Slerp);
        op.apply_with_donors(&mut base, &donor_a, &donor_b)
            .expect("slerp apply with parallel donors");
        let out = read_f32(base.get("w").unwrap());
        // LERP at t=0.5 of [1,2,3] and [2,4,6] is [1.5,3,4.5].
        assert_close(&out, &[1.5, 3.0, 4.5], 1e-5);
        // Critical: output must contain no NaNs.
        for v in &out {
            assert!(v.is_finite(), "parallel-vector SLERP must not NaN: {v}");
        }
    }

    #[test]
    fn slerp_anti_parallel_vectors_falls_back_to_lerp() {
        // a = +x, b = -x: cos(theta) = -1, sin(theta) ≈ 0. Must not
        // explode.
        let mut base = WeightMap::new();
        base.insert("w".to_string(), t_f32_1d(&[0.0, 0.0, 0.0]));
        let donor_a = donor_with("w", &[1.0, 2.0, 3.0]);
        let donor_b = donor_with("w", &[-1.0, -2.0, -3.0]);
        let op = stub_op("w", 0.5, InterpolateMethod::Slerp);
        op.apply_with_donors(&mut base, &donor_a, &donor_b)
            .expect("slerp apply with anti-parallel donors");
        let out = read_f32(base.get("w").unwrap());
        // LERP at t=0.5 of [1,2,3] and [-1,-2,-3] is [0,0,0].
        assert_close(&out, &[0.0, 0.0, 0.0], 1e-5);
        for v in &out {
            assert!(v.is_finite(), "anti-parallel SLERP must not NaN: {v}");
        }
    }

    #[test]
    fn slerp_zero_magnitude_donor_falls_back_to_lerp() {
        // ||a|| = 0 makes the cosine undefined; must fall back to
        // LERP rather than producing NaN.
        let mut base = WeightMap::new();
        base.insert("w".to_string(), t_f32_1d(&[0.0, 0.0]));
        let donor_a = donor_with("w", &[0.0, 0.0]);
        let donor_b = donor_with("w", &[1.0, 1.0]);
        let op = stub_op("w", 0.5, InterpolateMethod::Slerp);
        op.apply_with_donors(&mut base, &donor_a, &donor_b)
            .expect("slerp apply with zero-magnitude donor");
        let out = read_f32(base.get("w").unwrap());
        // LERP at t=0.5 of [0,0] and [1,1] is [0.5,0.5].
        assert_close(&out, &[0.5, 0.5], 1e-5);
        for v in &out {
            assert!(v.is_finite(), "zero-magnitude SLERP must not NaN: {v}");
        }
    }

    // --- (a-5) Error cases. -------------------------------------------

    #[test]
    fn zero_match_pattern_errors() {
        // Pattern matches nothing in the base WeightMap.
        let mut base = WeightMap::new();
        base.insert("model.foo.weight".to_string(), t_f32_1d(&[1.0, 2.0]));
        let donor_a = donor_with("model.bar.weight", &[1.0, 2.0]);
        let donor_b = donor_with("model.bar.weight", &[3.0, 4.0]);
        let op = stub_op("model.bar.*", 0.5, InterpolateMethod::Lerp);
        let err = op
            .apply_with_donors(&mut base, &donor_a, &donor_b)
            .expect_err("zero-match must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("matched zero tensors"),
            "error must say zero-match: {msg}",
        );
    }

    #[test]
    fn shape_mismatch_errors() {
        let mut base = WeightMap::new();
        base.insert("w".to_string(), t_f32_1d(&[1.0, 2.0, 3.0]));
        let donor_a = donor_with("w", &[1.0, 2.0]); // wrong shape
        let donor_b = donor_with("w", &[3.0, 4.0, 5.0]);
        let op = stub_op("w", 0.5, InterpolateMethod::Lerp);
        let err = op
            .apply_with_donors(&mut base, &donor_a, &donor_b)
            .expect_err("shape mismatch must error");
        match err {
            SurgeryError::ShapeMismatch {
                ref key,
                ref expected,
                ref actual,
            } => {
                assert!(key.contains("source_a"));
                assert_eq!(expected, &vec![3usize]);
                assert_eq!(actual, &vec![2usize]);
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn dtype_mismatch_errors() {
        // Build a base tensor in f32 and an i32 donor — same shape,
        // different dtype. The op must reject this before invoking
        // any arithmetic.
        let mut base = WeightMap::new();
        base.insert("w".to_string(), t_f32_1d(&[1.0, 2.0]));
        let mut donor_a: HashMap<String, UniquePtr<MlxArray>> = HashMap::new();
        donor_a.insert("w".to_string(), mlxcel_core::from_slice_i32(&[1, 2], &[2]));
        let donor_b = donor_with("w", &[3.0, 4.0]);
        let op = stub_op("w", 0.5, InterpolateMethod::Lerp);
        let err = op
            .apply_with_donors(&mut base, &donor_a, &donor_b)
            .expect_err("dtype mismatch must error");
        match err {
            SurgeryError::DtypeMismatch {
                ref key,
                ref expected,
                ref actual,
            } => {
                assert!(key.contains("source_a"), "key context lost: {key}");
                assert_eq!(expected, "float32");
                assert_eq!(actual, "int32");
            }
            other => panic!("expected DtypeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn quantized_dtype_is_rejected() {
        // All three tensors are u32 (a stand-in for quantized
        // packed weights). The op must refuse rather than
        // interpolating packed integer codes.
        let mut base = WeightMap::new();
        base.insert(
            "w".to_string(),
            mlxcel_core::from_slice_u32(&[1, 2, 3], &[3]),
        );
        let mut donor_a: HashMap<String, UniquePtr<MlxArray>> = HashMap::new();
        donor_a.insert(
            "w".to_string(),
            mlxcel_core::from_slice_u32(&[1, 2, 3], &[3]),
        );
        let mut donor_b: HashMap<String, UniquePtr<MlxArray>> = HashMap::new();
        donor_b.insert(
            "w".to_string(),
            mlxcel_core::from_slice_u32(&[10, 20, 30], &[3]),
        );
        let op = stub_op("w", 0.5, InterpolateMethod::Lerp);
        let err = op
            .apply_with_donors(&mut base, &donor_a, &donor_b)
            .expect_err("quantized must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("quantized") || msg.contains("non-float"),
            "error must mention quantized policy: {msg}"
        );
    }

    #[test]
    fn missing_donor_key_errors() {
        // Donor A has the matching tensor; donor B is missing it.
        let mut base = WeightMap::new();
        base.insert("w".to_string(), t_f32_1d(&[1.0, 2.0]));
        let donor_a = donor_with("w", &[1.0, 2.0]);
        let donor_b = donor_with("other", &[3.0, 4.0]);
        let op = stub_op("w", 0.5, InterpolateMethod::Lerp);
        let err = op
            .apply_with_donors(&mut base, &donor_a, &donor_b)
            .expect_err("missing donor key must error");
        match err {
            SurgeryError::TensorNotFound(ref k) => {
                assert!(k.contains("source_b"));
                assert!(k.contains("w"));
            }
            other => panic!("expected TensorNotFound, got {other:?}"),
        }
    }

    // --- Constructor validation. --------------------------------------

    #[test]
    fn constructor_rejects_out_of_range_ratio() {
        let err = InterpolateOp::new(
            "w",
            PathBuf::from("<a>"),
            PathBuf::from("<b>"),
            1.5,
            InterpolateMethod::Lerp,
        )
        .expect_err("ratio > 1 must fail");
        let msg = format!("{err}");
        assert!(msg.contains("ratio"), "{msg}");
    }

    #[test]
    fn constructor_rejects_malformed_glob() {
        let err = InterpolateOp::new(
            "model.layers.{0",
            PathBuf::from("<a>"),
            PathBuf::from("<b>"),
            0.5,
            InterpolateMethod::Lerp,
        )
        .expect_err("malformed glob must fail");
        let msg = format!("{err}");
        assert!(msg.contains("glob") || msg.contains("pattern"), "{msg}");
    }

    // --- Multi-tensor coverage: glob matches multiple keys. -----------

    #[test]
    fn glob_matches_multiple_tensors() {
        let mut base = WeightMap::new();
        base.insert("model.layers.0.w".to_string(), t_f32_1d(&[1.0, 2.0]));
        base.insert("model.layers.1.w".to_string(), t_f32_1d(&[3.0, 4.0]));
        base.insert("model.other.weight".to_string(), t_f32_1d(&[100.0]));

        let mut donor_a: HashMap<String, UniquePtr<MlxArray>> = HashMap::new();
        donor_a.insert("model.layers.0.w".to_string(), t_f32_1d(&[10.0, 20.0]));
        donor_a.insert("model.layers.1.w".to_string(), t_f32_1d(&[30.0, 40.0]));
        let mut donor_b: HashMap<String, UniquePtr<MlxArray>> = HashMap::new();
        donor_b.insert("model.layers.0.w".to_string(), t_f32_1d(&[100.0, 200.0]));
        donor_b.insert("model.layers.1.w".to_string(), t_f32_1d(&[300.0, 400.0]));

        let op = stub_op("model.layers.*.w", 0.5, InterpolateMethod::Lerp);
        op.apply_with_donors(&mut base, &donor_a, &donor_b)
            .expect("multi-tensor lerp apply");

        let l0 = read_f32(base.get("model.layers.0.w").unwrap());
        let l1 = read_f32(base.get("model.layers.1.w").unwrap());
        let other = read_f32(base.get("model.other.weight").unwrap());

        // LERP at t=0.5 of [10,20] and [100,200] is [55,110].
        assert_close(&l0, &[55.0, 110.0], 1e-5);
        // LERP at t=0.5 of [30,40] and [300,400] is [165, 220].
        assert_close(&l1, &[165.0, 220.0], 1e-5);
        // Non-matching tensor is untouched.
        assert_close(&other, &[100.0], 1e-6);
    }

    // --- bf16 input dtype: math is done in f32 and cast back. --------

    #[test]
    fn bf16_inputs_produce_bf16_output() {
        let a_f32 = t_f32_1d(&[1.0, 2.0, 3.0]);
        let b_f32 = t_f32_1d(&[10.0, 20.0, 30.0]);
        let a_bf16 = mlxcel_core::astype(&a_f32, dtype::BFLOAT16);
        let b_bf16 = mlxcel_core::astype(&b_f32, dtype::BFLOAT16);

        let mut base = WeightMap::new();
        let base_bf16 = mlxcel_core::astype(&t_f32_1d(&[0.0, 0.0, 0.0]), dtype::BFLOAT16);
        base.insert("w".to_string(), base_bf16);
        let mut donor_a: HashMap<String, UniquePtr<MlxArray>> = HashMap::new();
        donor_a.insert("w".to_string(), a_bf16);
        let mut donor_b: HashMap<String, UniquePtr<MlxArray>> = HashMap::new();
        donor_b.insert("w".to_string(), b_bf16);

        let op = stub_op("w", 0.5, InterpolateMethod::Lerp);
        op.apply_with_donors(&mut base, &donor_a, &donor_b)
            .expect("bf16 lerp apply");
        let out_arr = base.get("w").unwrap();
        assert_eq!(
            mlxcel_core::array_dtype(out_arr),
            dtype::BFLOAT16,
            "output dtype must match the base tensor"
        );
        // Round-trip back through f32 for the value check; bf16 has
        // ~3 digits of precision so use a looser tolerance.
        let out_f32 = mlxcel_core::astype(out_arr, dtype::FLOAT32);
        let out = read_f32(&out_f32);
        assert_close(&out, &[5.5, 11.0, 16.5], 1e-1);
    }
}
