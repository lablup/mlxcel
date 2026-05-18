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

//! `ScaleOp` — multiply every matched tensor by a scalar factor.
//!
//! YAML shape (#369 — A3):
//!
//! ```yaml
//! - op: scale
//!   pattern: "model.layers.*.self_attn.o_proj.weight"
//!   factor: 1.2
//! ```
//!
//! ## Quantized layout handling
//!
//! mlxcel stores affine-quantized (4bit/8bit) tensors as a triplet:
//! the packed quantized payload at `<prefix>.weight` (uint32 codes),
//! per-group scales at `<prefix>.scales`, and per-group biases at
//! `<prefix>.biases` (`affine` mode only). The effective fp tensor is
//! `dequant = q * scales + biases`, so multiplying the effective
//! weight by a scalar `factor` is mathematically equivalent to
//! multiplying both `scales` and `biases` by `factor` while leaving
//! the packed codes alone:
//!
//! ```text
//! factor * (q * scales + biases)
//!     = q * (factor * scales) + (factor * biases)
//! ```
//!
//! For `mxfp4` / `nvfp4` / `mxfp8` modes there is no `biases` tensor;
//! NVFP4 is fully dequantized to bf16 by `load_text_weights` before
//! the surgery hook runs, so by the time `ScaleOp::apply` sees the
//! map the only remaining quantized formats are pure affine plus
//! `mxfp4` / `mxfp8` (scales without biases). Both are handled by the
//! same "scale the metadata, leave the codes alone" branch below.
//!
//! Plain fp16/bf16/f32 tensors are multiplied directly.

use std::collections::BTreeSet;

use globset::{Glob, GlobMatcher};
use mlxcel_core::weights::WeightMap;

use crate::config::OpSpec;
use crate::{SurgeryError, SurgeryOp};

/// Per-tensor scalar multiplication (`W := factor * W`).
///
/// The `pattern` is a `globset`-compatible glob matched against the
/// keys of the [`WeightMap`]. Pattern matches resolve to *effective*
/// tensors before mutation: a match on `<prefix>.weight` of an
/// affine-quantized layer is rewritten to mutate `<prefix>.scales`
/// (and `<prefix>.biases` if present), never the packed codes
/// themselves. See module docs.
///
/// `ScaleOp` is stateless across `apply` calls and is therefore
/// `Send + Sync` to satisfy [`SurgeryOp`].
///
/// Used by: SurgeryPipeline, mlxcel-surgery YAML factory
/// (`config::materialize_op`)
#[derive(Debug)]
pub struct ScaleOp {
    /// Source glob (kept for diagnostics — never matched against, the
    /// compiled `matcher` is the runtime authority).
    pattern: String,
    /// Pre-compiled glob matcher.
    matcher: GlobMatcher,
    /// Scalar multiplier. Validated `finite` at parse time.
    factor: f32,
}

impl ScaleOp {
    /// Construct a `ScaleOp` directly from its component parts.
    ///
    /// Used by tests that want to build a `ScaleOp` without going
    /// through the YAML factory. Production callers should go through
    /// [`Self::from_spec`] (called from `config::materialize_op`) so
    /// the same validation rules are applied uniformly.
    ///
    /// Returns an error if the glob is malformed or `factor` is not
    /// finite.
    pub fn new(pattern: impl Into<String>, factor: f32) -> Result<Self, SurgeryError> {
        let pattern = pattern.into();
        if !factor.is_finite() {
            return Err(SurgeryError::Other(anyhow::anyhow!(
                "scale: factor must be finite, got {factor}"
            )));
        }
        let matcher = Glob::new(&pattern)
            .map(|g| g.compile_matcher())
            .map_err(|e| {
                SurgeryError::Other(anyhow::anyhow!(
                    "scale: malformed glob pattern {pattern:?}: {e}"
                ))
            })?;
        Ok(Self {
            pattern,
            matcher,
            factor,
        })
    }

    /// Build a `ScaleOp` from the parsed YAML `OpSpec::Scale` variant.
    /// Returns an error if the variant is not `Scale` (a programming
    /// error from the factory side — kept defensive so the factory's
    /// `match` stays exhaustive at the type level).
    pub fn from_spec(spec: OpSpec) -> Result<Self, SurgeryError> {
        match spec {
            OpSpec::Scale { pattern, factor } => Self::new(pattern, factor),
            other => Err(SurgeryError::Other(anyhow::anyhow!(
                "scale: from_spec called with non-Scale OpSpec ({other:?})"
            ))),
        }
    }

    /// Read-only access to the source glob string (for diagnostics).
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Read-only access to the configured factor.
    pub fn factor(&self) -> f32 {
        self.factor
    }
}

impl SurgeryOp for ScaleOp {
    /// Multiply every effective tensor whose key matches the glob by
    /// the configured scalar.
    ///
    /// See the module docstring for the quantized-layout routing rule.
    fn apply(&self, weights: &mut WeightMap, _cfg: &serde_json::Value) -> Result<(), SurgeryError> {
        // Stage 1 — resolve every glob hit on the *as-loaded* key set
        // to one or more *effective* mutation targets. This isolates
        // the iteration from the subsequent mutation, lets us
        // deduplicate when the same logical target is reached via
        // multiple input keys (e.g. a pattern of `"*"` would match
        // `.weight`, `.scales`, and `.biases` of the same affine
        // triplet), and surfaces the zero-match error before any
        // tensor is touched.

        // Sorted for deterministic iteration order across runs — the
        // raw HashMap key order is non-deterministic but the order in
        // which we apply scale is observable through tensor-eval
        // sequencing in tests.
        let mut matched_input_keys: Vec<&String> = weights
            .keys()
            .filter(|k| self.matcher.is_match(k.as_str()))
            .collect();
        matched_input_keys.sort();

        if matched_input_keys.is_empty() {
            return Err(SurgeryError::Other(anyhow::anyhow!(
                "scale: pattern {:?} matched zero tensors",
                self.pattern,
            )));
        }

        // Build the set of *effective* target keys. Using BTreeSet so
        // the eventual mutation order is deterministic and so dedup
        // is cheap.
        let mut targets: BTreeSet<String> = BTreeSet::new();
        for key in &matched_input_keys {
            for target in resolve_effective_targets(key, weights) {
                targets.insert(target);
            }
        }

        if targets.is_empty() {
            // Should not be reachable: every matched input key
            // resolves to at least one effective target (itself in
            // the fallback case). Keep the guard as defense in depth
            // so callers get a clear error rather than a silent
            // no-op if the resolver is ever extended incorrectly.
            return Err(SurgeryError::Other(anyhow::anyhow!(
                "scale: pattern {:?} matched {} key(s) but resolved to zero effective targets",
                self.pattern,
                matched_input_keys.len(),
            )));
        }

        // Stage 2 — mutate every effective target in place.
        // We snapshot dtype/shape per target so we can assert
        // preservation after the multiply (defensive: the FFI helper
        // already preserves them, but a regression in `multiply_scalar`
        // would otherwise silently corrupt the load).
        for target_key in targets {
            let (orig_dtype, orig_shape) = {
                let tensor = weights.get(&target_key).ok_or_else(|| {
                    // Targets are derived from current keys; a missing
                    // target indicates a concurrent mutation, which is
                    // impossible behind `&mut WeightMap` but worth a
                    // clear error rather than a panic.
                    SurgeryError::TensorNotFound(target_key.clone())
                })?;
                (
                    mlxcel_core::array_dtype(tensor),
                    mlxcel_core::array_shape(tensor),
                )
            };

            let scaled = {
                let tensor = weights.get(&target_key).expect("checked above");
                mlxcel_core::multiply_scalar(tensor, self.factor)
            };

            // Sanity check dtype / shape preservation. `multiply_scalar`
            // upcasts the scalar into the input dtype, so the result
            // dtype should match input dtype exactly. If MLX ever
            // changes that contract, surface it explicitly so we do
            // not silently produce a dtype-shifted WeightMap.
            let new_dtype = mlxcel_core::array_dtype(&scaled);
            if new_dtype != orig_dtype {
                return Err(SurgeryError::DtypeMismatch {
                    key: target_key.clone(),
                    expected: format!("{orig_dtype}"),
                    actual: format!("{new_dtype}"),
                });
            }
            let new_shape = mlxcel_core::array_shape(&scaled);
            if new_shape != orig_shape {
                return Err(SurgeryError::ShapeMismatch {
                    key: target_key.clone(),
                    expected: orig_shape.into_iter().map(|d| d as usize).collect(),
                    actual: new_shape.into_iter().map(|d| d as usize).collect(),
                });
            }

            weights.insert(target_key, scaled);
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "scale"
    }
}

/// Resolve the "effective" targets for a matched input key.
///
/// For affine-quantized layers, scaling the *effective* fp tensor
/// requires scaling the bf16 scales (and biases if present), not the
/// packed integer codes. Plain fp tensors and stand-alone
/// scales/biases that the user explicitly targets are passed through
/// unchanged.
///
/// Returns at least one entry (the input key in the fallback case).
fn resolve_effective_targets(input_key: &str, weights: &WeightMap) -> Vec<String> {
    // Heuristic — a key is the packed payload of an affine-quantized
    // layer iff it ends in `.weight` AND a sibling `.scales` exists
    // in the same map. This matches the convention enforced by the
    // existing `QuantizedEmbedding::from_weights` and `QuantizedLinear`
    // loaders in `mlxcel-core/src/layers.rs`.
    if let Some(prefix) = input_key.strip_suffix(".weight") {
        let scales_key = format!("{prefix}.scales");
        if weights.contains_key(&scales_key) {
            let biases_key = format!("{prefix}.biases");
            let mut out = Vec::with_capacity(2);
            out.push(scales_key);
            if weights.contains_key(&biases_key) {
                out.push(biases_key);
            }
            return out;
        }
    }

    // Default: the matched key *is* the effective target. This
    // branch handles plain fp tensors, RMSNorm gain weights, and
    // user-driven explicit targeting of `.scales` / `.biases`.
    vec![input_key.to_string()]
}

#[cfg(test)]
#[path = "scale_tests.rs"]
mod tests;
