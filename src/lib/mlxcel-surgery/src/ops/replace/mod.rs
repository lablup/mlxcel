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

//! `ReplaceOp` — substitute base-model tensors with the matching
//! tensors from an external donor `.safetensors` checkpoint
//!
//! ## Semantics
//!
//! - Every tensor key in [`crate::WeightMap`] that matches `pattern`
//!   is replaced with the equivalent tensor from the donor file.
//! - `pattern` and `source_key` both use simple `*` wildcards.
//!   Wildcard capture is positional: the N-th `*` in `pattern`
//!   captures a substring of the matched key; that substring is
//!   substituted in for the N-th `*` of `source_key` to form the
//!   donor key. The number of `*` in `pattern` and `source_key`
//!   must match — this is checked at construction time.
//! - Single-tensor (no wildcards) case: `pattern` and `source_key`
//!   are used verbatim. They may differ — e.g. base
//!   `model.embed_tokens.weight` replaced by donor
//!   `embedding.weight`.
//! - When the base tensor is quantized (sibling `.scales` and/or
//!   `.biases` keys exist in the base map), the operation runs
//!   atomically over the `.weight` plus its present sibling keys.
//!   All sibling shapes and dtypes must match between base and
//!   donor, otherwise the op fails before mutating anything. This
//!   enforces the "donor must share the exact quantization layout"
//!   requirement with no implicit re-quantization.
//! - On any error, [`crate::SurgeryError`] is returned and the
//!   underlying [`WeightMap`] is left unchanged ("atomic on
//!   error").
//!
//! ## Wildcard substitution example
//!
//! ```yaml
//! - op: replace
//!   pattern: "model.layers.*.self_attn.q_proj.weight"
//!   source: "./donor.safetensors"
//!   source_key: "donor.transformer.h.*.attn.q.weight"
//! ```
//!
//! When the base map contains
//! `model.layers.0.self_attn.q_proj.weight`, the captured fragment
//! `"0"` is substituted into the donor template, yielding the donor
//! key `donor.transformer.h.0.attn.q.weight`.
//!
//! ## Used by
//!
//! [`crate::config::materialize_op`] — registered for
//! [`crate::config::OpSpec::Replace`].

use std::collections::HashSet;
use std::path::PathBuf;

use mlxcel_core::{MlxArray, UniquePtr};

use crate::{SurgeryError, SurgeryOp, WeightMap};

mod wildcard;

use wildcard::WildcardPattern;

#[cfg(test)]
mod quant_tests;
#[cfg(test)]
mod tests;

/// Concrete `SurgeryOp` implementing tensor substitution from an
/// external donor safetensors file.
///
/// Construct via [`ReplaceOp::new`] from a parsed YAML spec; the
/// constructor validates that `pattern` and `source_key` carry the
/// same number of `*` wildcards. Apply via the [`SurgeryOp::apply`]
/// trait method.
///
/// The donor file is opened lazily on the first `apply` call so
/// surgery configurations can be parsed (and rejected for syntax
/// errors) even when the donor file lives on slow / cold storage.
///
/// Used by: [`crate::config::materialize_op`]
pub struct ReplaceOp {
    /// Parsed `pattern`. Split into literal segments around each
    /// `*` wildcard. Used both to match keys in the base map and to
    /// extract positional captures.
    pattern: WildcardPattern,
    /// Parsed `source_key`. Must have the same wildcard count as
    /// `pattern` — validated by the constructor.
    source_key: WildcardPattern,
    /// Absolute path to the donor safetensors file. Resolved by
    /// [`crate::config::materialize_op`] before this struct is
    /// constructed.
    source_path: PathBuf,
}

impl std::fmt::Debug for ReplaceOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReplaceOp")
            .field("pattern", &self.pattern.original())
            .field("source_key", &self.source_key.original())
            .field("source_path", &self.source_path)
            .finish()
    }
}

impl ReplaceOp {
    /// Build a `ReplaceOp` from the parsed YAML spec.
    ///
    /// Returns an error if `pattern` or `source_key` is empty, or
    /// if they disagree on wildcard count. The donor file is *not*
    /// opened here — it is opened on first `apply` so configurations
    /// can be parsed even when the donor file is on slow / cold
    /// storage.
    ///
    /// The `source_path` should already be resolved to an absolute
    /// path by the caller ([`crate::config::materialize_op`]).
    pub fn new(
        pattern: &str,
        source_key: &str,
        source_path: PathBuf,
    ) -> Result<Self, SurgeryError> {
        if pattern.is_empty() {
            return Err(SurgeryError::Other(anyhow::anyhow!(
                "replace: pattern must not be empty"
            )));
        }
        if source_key.is_empty() {
            return Err(SurgeryError::Other(anyhow::anyhow!(
                "replace: source_key must not be empty"
            )));
        }
        let pat = WildcardPattern::parse(pattern);
        let key = WildcardPattern::parse(source_key);
        if pat.wildcard_count() != key.wildcard_count() {
            return Err(SurgeryError::Other(anyhow::anyhow!(
                "replace: pattern has {} wildcard(s) but source_key has {}; \
                 source_key must use the same number of `*` so captured \
                 fragments can be substituted positionally \
                 (pattern={pattern:?}, source_key={source_key:?})",
                pat.wildcard_count(),
                key.wildcard_count(),
            )));
        }
        Ok(Self {
            pattern: pat,
            source_key: key,
            source_path,
        })
    }
}

impl SurgeryOp for ReplaceOp {
    fn apply(&self, weights: &mut WeightMap, _cfg: &serde_json::Value) -> Result<(), SurgeryError> {
        // 1) Collect base keys to replace. Sorting keeps the
        //    operation deterministic across `HashMap` iteration
        //    order so error messages are stable.
        let mut matches: Vec<(String, Vec<String>)> = weights
            .keys()
            .filter_map(|k| {
                self.pattern
                    .match_with_captures(k)
                    .map(|caps| (k.clone(), caps))
            })
            .collect();
        matches.sort_by(|a, b| a.0.cmp(&b.0));

        if matches.is_empty() {
            return Err(SurgeryError::Other(anyhow::anyhow!(
                "replace: pattern {:?} matched no keys in the weight map",
                self.pattern.original(),
            )));
        }

        // 2) Build the substitution plan. For each matched base key
        //    we always need the matching donor `.weight` plus its
        //    `.scales` / `.biases` siblings if the base map carries
        //    them — quantized layouts must be replaced atomically.
        let mut plans: Vec<PlannedReplacement> = Vec::with_capacity(matches.len());
        let mut donor_keys_needed: HashSet<String> = HashSet::new();
        for (base_key, captures) in &matches {
            let donor_key = self.source_key.render(captures);
            donor_keys_needed.insert(donor_key.clone());

            let mut sibling_keys = Vec::new();
            for suffix in QUANT_SIBLING_SUFFIXES {
                let base_sibling = format!("{base_key}{suffix}");
                if weights.contains_key(&base_sibling) {
                    let donor_sibling = format!("{donor_key}{suffix}");
                    donor_keys_needed.insert(donor_sibling.clone());
                    sibling_keys.push((base_sibling, donor_sibling));
                }
            }
            plans.push(PlannedReplacement {
                base_key: base_key.clone(),
                donor_key,
                sibling_keys,
            });
        }

        // 3) Verify the donor file exists before reaching into FFI.
        //    This gives a clean Io error rather than a generic
        //    "failed to load safetensors" diagnostic.
        if !self.source_path.exists() {
            return Err(SurgeryError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "replace: donor safetensors file does not exist: {}",
                    self.source_path.display()
                ),
            )));
        }
        let donor_path_str = self.source_path.to_str().ok_or_else(|| {
            SurgeryError::Other(anyhow::anyhow!(
                "replace: donor path is not valid UTF-8: {}",
                self.source_path.display()
            ))
        })?;

        // 4) Load only the donor tensors we actually need. Filtering
        //    keeps memory bounded for multi-GB donor files when only
        //    a handful of tensors are being replaced.
        let donor_keys_needed_capture = donor_keys_needed.clone();
        let donor_map: WeightMap =
            mlxcel_core::weights::load_safetensors_filtered(donor_path_str, move |name| {
                donor_keys_needed_capture.contains(name)
            })
            .map_err(|e| {
                SurgeryError::Other(anyhow::anyhow!(
                    "replace: failed to load donor safetensors {donor_path_str}: {e}"
                ))
            })?;

        // 5) Verify every requested donor key was provided. We do
        //    this before mutating the base map so that an error
        //    leaves `weights` unchanged.
        let mut missing: Vec<&str> = donor_keys_needed
            .iter()
            .filter(|k| !donor_map.contains_key(k.as_str()))
            .map(|s| s.as_str())
            .collect();
        if !missing.is_empty() {
            missing.sort();
            return Err(SurgeryError::TensorNotFound(format!(
                "replace: donor {} is missing tensor(s): {}",
                self.source_path.display(),
                missing.join(", "),
            )));
        }

        // 6) Shape + dtype checks for every planned replacement.
        //    Atomic-on-error: detected before any mutation.
        for plan in &plans {
            let base_w = weights
                .get(&plan.base_key)
                .expect("base_key came from weights.keys()");
            let donor_w = donor_map
                .get(&plan.donor_key)
                .expect("donor_key existence checked above");
            check_compatible(&plan.base_key, base_w, donor_w)?;
            for (base_sibling, donor_sibling) in &plan.sibling_keys {
                let base_s = weights.get(base_sibling).expect("sibling in weights");
                let donor_s = donor_map
                    .get(donor_sibling)
                    .expect("sibling existence checked above");
                check_compatible(base_sibling, base_s, donor_s)?;
            }
        }

        // 7) All checks passed. Perform the substitution. We copy
        //    each donor tensor so the donor `WeightMap` can be
        //    dropped (when it goes out of scope below) without
        //    invalidating the base map entries.
        for plan in plans {
            let donor_w = donor_map
                .get(&plan.donor_key)
                .expect("donor_key in donor_map");
            let copied = mlxcel_core::copy(donor_w);
            weights.insert(plan.base_key, copied);
            for (base_sibling, donor_sibling) in plan.sibling_keys {
                let donor_s = donor_map
                    .get(&donor_sibling)
                    .expect("donor sibling in donor_map");
                let copied = mlxcel_core::copy(donor_s);
                weights.insert(base_sibling, copied);
            }
        }

        Ok(())
    }

    fn name(&self) -> &'static str {
        "replace"
    }
}

/// One planned tensor substitution, computed up front so error
/// detection is fully separated from mutation.
struct PlannedReplacement {
    base_key: String,
    donor_key: String,
    /// `(base_sibling_key, donor_sibling_key)` pairs for the
    /// quantized `.scales` / `.biases` siblings, when present.
    sibling_keys: Vec<(String, String)>,
}

/// Quantization sibling suffixes attached to a packed `.weight` key
/// in MLX safetensors checkpoints. The replace op picks these up
/// automatically when a base `.weight` is matched so quantized
/// layouts stay coherent.
const QUANT_SIBLING_SUFFIXES: &[&str] = &[".scales", ".biases"];

/// Verify that `donor` is a drop-in substitute for `base` — same
/// shape and same dtype. The exact-equality requirement keeps the
/// contract simple: no implicit re-quantization, no
/// implicit precision conversion.
///
/// Used by: [`ReplaceOp::apply`]
pub(crate) fn check_compatible(
    key: &str,
    base: &UniquePtr<MlxArray>,
    donor: &UniquePtr<MlxArray>,
) -> Result<(), SurgeryError> {
    let base_shape: Vec<usize> = mlxcel_core::array_shape(base)
        .into_iter()
        .map(|d| d as usize)
        .collect();
    let donor_shape: Vec<usize> = mlxcel_core::array_shape(donor)
        .into_iter()
        .map(|d| d as usize)
        .collect();
    if base_shape != donor_shape {
        return Err(SurgeryError::ShapeMismatch {
            key: key.to_string(),
            expected: base_shape,
            actual: donor_shape,
        });
    }
    let base_dt = mlxcel_core::array_dtype(base);
    let donor_dt = mlxcel_core::array_dtype(donor);
    if base_dt != donor_dt {
        return Err(SurgeryError::DtypeMismatch {
            key: key.to_string(),
            expected: dtype_label(base_dt).to_string(),
            actual: dtype_label(donor_dt).to_string(),
        });
    }
    Ok(())
}

/// Human-readable label for an MLX dtype code. Mirrors the names in
/// [`mlxcel_core::dtype`] (`bool`, `float16`, `uint32`, ...) so error
/// messages match what users see in tooling such as `safetensors-cli`.
/// Falls back to `"unknown"` if a future dtype constant is observed,
/// keeping the diagnostic path robust against MLX dtype additions.
pub(crate) fn dtype_label(code: i32) -> &'static str {
    use mlxcel_core::dtype;
    match code {
        c if c == dtype::BOOL => "bool",
        c if c == dtype::UINT8 => "uint8",
        c if c == dtype::UINT16 => "uint16",
        c if c == dtype::UINT32 => "uint32",
        c if c == dtype::UINT64 => "uint64",
        c if c == dtype::INT8 => "int8",
        c if c == dtype::INT16 => "int16",
        c if c == dtype::INT32 => "int32",
        c if c == dtype::INT64 => "int64",
        c if c == dtype::FLOAT16 => "float16",
        c if c == dtype::FLOAT32 => "float32",
        c if c == dtype::FLOAT64 => "float64",
        c if c == dtype::BFLOAT16 => "bfloat16",
        c if c == dtype::COMPLEX64 => "complex64",
        _ => "unknown",
    }
}
