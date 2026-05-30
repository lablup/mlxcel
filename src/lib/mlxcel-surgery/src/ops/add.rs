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

//! `AddOp` — task-vector addition (axis A6).
//!
//! For every tensor key in the in-memory `WeightMap` whose name
//! matches the glob `pattern`, look up the same-named tensor in an
//! external "task vector" safetensors file, validate that the shape
//! is identical, and rewrite the base tensor as
//!
//! ```text
//! base += alpha * delta
//! ```
//!
//! This is the canonical Ilharco-et-al. weight-space-arithmetic
//! operation; see `docs_internal/architecture/structural-finetuning-overview-20260419.md`
//! §3.2 (#2 Add — task vector) for background.
//!
//! ## Behavior summary
//!
//! - **Shape mismatch**: returns [`SurgeryError::ShapeMismatch`].
//! - **Donor missing the key**: returns [`SurgeryError::TensorNotFound`].
//! - **Zero matched keys**: returns [`SurgeryError::Other`] explaining
//!   the pattern matched nothing (parallel with `ScaleOp`).
//! - **Integer (i.e. quantized) base tensor**: returns
//!   [`SurgeryError::Other`] with a clear message. The first cut of
//!   A6 only supports floating-point base tensors and floating-point
//!   task vectors. Quantized models can still add deltas to the
//!   adjacent `.scales` / `.biases` tensors (which are float), or
//!   first dequantize via a future surgery op.
//! - **Donor dtype differs from base dtype** (both float): the donor
//!   tensor is cast to the base dtype before adding. This lets users
//!   ship f32 task vectors against f16/bf16 model weights without
//!   pre-converting on disk.
//! - **`alpha == 0.0`**: no-op. The base tensor is left untouched, no
//!   donor tensor is materialized for the matching key, and the donor
//!   load itself is skipped (since `alpha * delta == 0` for every
//!   matching key). This makes "compile in a surgery op but
//!   temporarily disable it" cheap.
//! - **`alpha == 1.0`**: skip the scalar multiplication and add the
//!   donor directly. Shaves one graph node per matched key.
//!
//! ## Donor caching
//!
//! The donor file is loaded once per `apply()` invocation. MLX's
//! safetensors loader is mmap-backed so loading is effectively free
//! beyond the OS page cache. Caching the loaded donor across multiple
//! `AddOp`s that share a source path would shave a syscall per op;
//! it's an obvious follow-up but not in scope for the first cut.
//!
//! Used by: `mlxcel_surgery::config::materialize_op`,
//! `SurgeryPipeline` (via the `SurgeryOp` trait), the consolidated
//! text / VLM weight loaders (transitively, via the
//! `WeightTransform` hook).

use std::path::{Path, PathBuf};

use globset::{Glob, GlobMatcher};
use mlxcel_core::dtype as mlx_dtype;
use mlxcel_core::weights::{load_safetensors, WeightMap};
use mlxcel_core::{
    add as mlx_add, array_dtype, array_shape, astype, copy as mlx_copy, multiply_scalar, MlxArray,
    UniquePtr,
};

use crate::{SurgeryError, SurgeryOp};

/// `add(W) = W + alpha * source[key]` over every tensor whose key
/// matches `pattern`.
///
/// Constructed by [`crate::config::parse_config_str`] /
/// [`crate::config::parse_config_file`] from a parsed `OpSpec::Add`
/// entry, or directly in tests / library callers that want to bypass
/// YAML.
///
/// `AddOp` is intentionally cheap to clone — only an arc'd
/// `GlobMatcher`, a `PathBuf`, and an `f32`. The donor safetensors
/// payload itself is **not** owned; it is reloaded on every
/// `apply()`. See the module docs for the caching follow-up.
///
/// Used by: `OpSpec::Add` materialization,
/// `SurgeryPipeline::push`, unit / integration tests.
#[derive(Debug, Clone)]
pub struct AddOp {
    /// Compiled glob matcher. Kept alongside the source pattern for
    /// error-message clarity.
    matcher: GlobMatcher,
    /// Raw pattern string, retained so error messages can quote the
    /// user-facing form (e.g. `pattern "model.layers.*.mlp.*"`).
    pattern_src: String,
    /// Absolute path to a safetensors file containing the task vector.
    source_path: PathBuf,
    /// Coefficient on the source term (`base += alpha * source`).
    /// `alpha == 0.0` triggers the no-op fast path.
    alpha: f32,
}

impl AddOp {
    /// Build an `AddOp` from a raw glob pattern, source path, and alpha.
    ///
    /// Returns an error if the pattern is not a valid `globset::Glob`
    /// or if `alpha` is not finite. The source file is **not**
    /// stat'd here — that check belongs to the YAML config layer (in
    /// `crate::config::materialize_op`) so unit tests can construct
    /// an `AddOp` against any path. Callers that build `AddOp`
    /// directly are responsible for path validation.
    pub fn new(
        pattern: &str,
        source_path: impl Into<PathBuf>,
        alpha: f32,
    ) -> Result<Self, SurgeryError> {
        if !alpha.is_finite() {
            return Err(SurgeryError::Other(anyhow::anyhow!(
                "AddOp: alpha must be finite, got {alpha}"
            )));
        }
        let matcher = Glob::new(pattern)
            .map_err(|e| {
                SurgeryError::Other(anyhow::anyhow!(
                    "AddOp: malformed glob pattern {pattern:?}: {e}"
                ))
            })?
            .compile_matcher();
        Ok(Self {
            matcher,
            pattern_src: pattern.to_string(),
            source_path: source_path.into(),
            alpha,
        })
    }

    /// Build an `AddOp` from an already-compiled glob matcher. Used
    /// by the YAML config layer so the matcher is not compiled twice
    /// (once in `materialize_op` for validation, once here).
    pub(crate) fn from_compiled(
        matcher: GlobMatcher,
        pattern_src: String,
        source_path: PathBuf,
        alpha: f32,
    ) -> Self {
        debug_assert!(
            alpha.is_finite(),
            "AddOp::from_compiled requires finite alpha; config layer must validate"
        );
        Self {
            matcher,
            pattern_src,
            source_path,
            alpha,
        }
    }

    /// Source-file path the donor was constructed against. Exposed
    /// for tests and tracing.
    pub fn source_path(&self) -> &Path {
        &self.source_path
    }

    /// Coefficient on the task vector. Exposed for tests and tracing.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// The raw glob pattern string. Exposed for tests, tracing, and
    /// error messages.
    pub fn pattern(&self) -> &str {
        &self.pattern_src
    }
}

impl SurgeryOp for AddOp {
    fn apply(&self, weights: &mut WeightMap, _cfg: &serde_json::Value) -> Result<(), SurgeryError> {
        // Snapshot matched keys up front so we can mutate the map
        // afterwards without iterating a borrowed view. `keys` is
        // also handy for the "zero matches" diagnostic below.
        let matched: Vec<String> = weights
            .keys()
            .filter(|k| self.matcher.is_match(k.as_str()))
            .cloned()
            .collect();

        if matched.is_empty() {
            return Err(SurgeryError::Other(anyhow::anyhow!(
                "AddOp pattern {:?} matched zero tensor keys in the weight map; \
                 either the pattern is wrong or the model name layout differs \
                 from what was assumed (see source: {})",
                self.pattern_src,
                self.source_path.display(),
            )));
        }

        // alpha == 0.0 is the no-op fast path. Skip loading the donor
        // safetensors entirely — `base += 0 * delta == base` for every
        // matching key, regardless of donor content. Only the
        // zero-match diagnostic above still has to fire so users
        // notice typo'd patterns.
        if self.alpha == 0.0 {
            return Ok(());
        }

        // Load the donor (task vector) safetensors. MLX manages the
        // mmap and lazy materialization internally, so this is cheap
        // beyond the initial open + header parse.
        let donor: WeightMap = load_safetensors(&self.source_path).map_err(|msg| {
            SurgeryError::Other(anyhow::anyhow!(
                "AddOp: failed to load task-vector source {}: {msg}",
                self.source_path.display(),
            ))
        })?;

        for key in matched {
            // We've just snapshotted `matched` from the same map, so
            // the key is guaranteed to be present. Use `remove` so we
            // own the base array for the duration of the rewrite —
            // MLX arrays are not `Copy` and the WeightMap stores
            // `UniquePtr<MlxArray>`.
            let base = weights
                .remove(&key)
                .expect("key was sampled from `weights` and not mutated since");

            // Look up the donor tensor *before* doing any work so we
            // can surface `TensorNotFound` cleanly while still
            // restoring the base into the map. The pipeline
            // short-circuits on the first error, but other tests (and
            // the consolidated loader's diagnostic flow) inspect the
            // weight map after a failed op, so we keep `weights`
            // consistent on every error path.
            let delta = match donor.get(&key) {
                Some(d) => d,
                None => {
                    weights.insert(key.clone(), base);
                    return Err(SurgeryError::TensorNotFound(key));
                }
            };

            let updated = match apply_add_to_tensor(&key, &base, delta, self.alpha) {
                Ok(new_arr) => new_arr,
                Err(e) => {
                    weights.insert(key, base);
                    return Err(e);
                }
            };
            weights.insert(key, updated);
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "add"
    }
}

/// Single-key add: validate shape, dtype kinds, cast donor if
/// needed, scale by `alpha` if needed, and add.
fn apply_add_to_tensor(
    key: &str,
    base: &UniquePtr<MlxArray>,
    delta: &UniquePtr<MlxArray>,
    alpha: f32,
) -> Result<UniquePtr<MlxArray>, SurgeryError> {
    let base_shape = array_shape(base);
    let delta_shape = array_shape(delta);
    if base_shape != delta_shape {
        return Err(SurgeryError::ShapeMismatch {
            key: key.to_string(),
            expected: base_shape.iter().map(|d| *d as usize).collect(),
            actual: delta_shape.iter().map(|d| *d as usize).collect(),
        });
    }

    let base_dtype = array_dtype(base);
    let delta_dtype = array_dtype(delta);

    // Quantized base tensors are packed integers (uint8/uint16/uint32
    // depending on MLX's storage of the affine packing). Adding a
    // float task vector to packed bits would corrupt the
    // dequantization. Surface a focused error here — the user-facing
    // workaround is to point the AddOp pattern at the matching
    // `.scales` and `.biases` keys instead, or to apply the surgery
    // before quantization.
    if !is_floating_point(base_dtype) {
        return Err(SurgeryError::Other(anyhow::anyhow!(
            "AddOp: base tensor {key:?} has non-floating dtype {} (likely a packed \
             quantized weight). Task-vector addition is only supported on \
             floating-point base tensors in A6; apply the AddOp to the matching \
             `.scales` / `.biases` keys instead, or dequantize first.",
            dtype_label(base_dtype),
        )));
    }
    if !is_floating_point(delta_dtype) {
        return Err(SurgeryError::DtypeMismatch {
            key: key.to_string(),
            expected: dtype_label(base_dtype).to_string(),
            actual: dtype_label(delta_dtype).to_string(),
        });
    }

    // Cast the donor to the base dtype when they differ so the
    // subsequent `add` stays in the base precision and matches what
    // the model graph will consume.
    let delta_cast = if delta_dtype == base_dtype {
        mlx_copy(delta)
    } else {
        astype(delta, base_dtype)
    };

    let scaled = if alpha == 1.0 {
        delta_cast
    } else {
        // multiply_scalar materializes the scalar in the base dtype
        // (it reads the input dtype), so the resulting array also
        // stays in base precision.
        multiply_scalar(&delta_cast, alpha)
    };

    Ok(mlx_add(base, &scaled))
}

/// Is the given MLX dtype code a floating-point type? Mirrors
/// `mlxcel_core::dtype::{FLOAT16, FLOAT32, FLOAT64, BFLOAT16}`.
fn is_floating_point(dtype: i32) -> bool {
    matches!(
        dtype,
        mlx_dtype::FLOAT16 | mlx_dtype::FLOAT32 | mlx_dtype::FLOAT64 | mlx_dtype::BFLOAT16,
    )
}

/// Render an MLX dtype code as a short human-readable label.
/// Used in `SurgeryError::DtypeMismatch` / `SurgeryError::Other`
/// messages so users see `"bfloat16"` instead of an opaque integer
/// code.
fn dtype_label(dtype: i32) -> &'static str {
    match dtype {
        mlx_dtype::BOOL => "bool",
        mlx_dtype::UINT8 => "uint8",
        mlx_dtype::UINT16 => "uint16",
        mlx_dtype::UINT32 => "uint32",
        mlx_dtype::UINT64 => "uint64",
        mlx_dtype::INT8 => "int8",
        mlx_dtype::INT16 => "int16",
        mlx_dtype::INT32 => "int32",
        mlx_dtype::INT64 => "int64",
        mlx_dtype::FLOAT16 => "float16",
        mlx_dtype::FLOAT32 => "float32",
        mlx_dtype::FLOAT64 => "float64",
        mlx_dtype::BFLOAT16 => "bfloat16",
        mlx_dtype::COMPLEX64 => "complex64",
        _ => "<unknown-dtype>",
    }
}
