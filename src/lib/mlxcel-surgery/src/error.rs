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

//! Crate-level error type used by every `SurgeryOp` and by
//! `SurgeryPipeline` when it routes errors out through the
//! `WeightTransform::apply` hook.

use thiserror::Error;

/// Errors that can arise while applying a surgery operation.
///
/// Variants cover the common failure modes for the future op set
/// (A5–A9): missing weight key, shape mismatch when materializing a
/// donor tensor, dtype mismatch when interpolating between
/// checkpoints, I/O failure when loading an external source file,
/// and a free-form `Other` for anything that does not yet have a
/// dedicated variant (this composes with `anyhow::Error` via
/// `#[from]`).
///
/// Used by: SurgeryOp::apply, SurgeryPipeline (WeightTransform impl)
#[derive(Debug, Error)]
pub enum SurgeryError {
    /// A weight key referenced by the operation did not exist in
    /// the [`crate::WeightMap`] (or, for source-driven ops, in the
    /// external donor checkpoint).
    #[error("tensor key not found: {0}")]
    TensorNotFound(String),

    /// An operation expected a tensor with `expected` shape but the
    /// runtime tensor had `actual`.
    #[error("shape mismatch for {key}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Tensor key the mismatch was detected on.
        key: String,
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        actual: Vec<usize>,
    },

    /// An operation expected a particular dtype but the runtime
    /// tensor had a different one (e.g. attempting f16 interpolation
    /// over a quantized donor checkpoint without an explicit dequant
    /// step).
    #[error("dtype mismatch for {key}: expected {expected}, got {actual}")]
    DtypeMismatch {
        /// Tensor key the mismatch was detected on.
        key: String,
        /// Expected dtype label.
        expected: String,
        /// Actual dtype label.
        actual: String,
    },

    /// I/O failure when reading an external donor file.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// A failure from the config layer or any
    /// downstream that has chosen to surface its errors as
    /// `anyhow::Error`.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_not_found_display() {
        let err = SurgeryError::TensorNotFound("model.embed_tokens.weight".to_string());
        assert_eq!(
            format!("{err}"),
            "tensor key not found: model.embed_tokens.weight"
        );
    }

    #[test]
    fn shape_mismatch_display_includes_both_shapes() {
        let err = SurgeryError::ShapeMismatch {
            key: "model.layers.0.self_attn.q_proj.weight".to_string(),
            expected: vec![4096, 4096],
            actual: vec![2048, 4096],
        };
        let s = format!("{err}");
        assert!(s.contains("model.layers.0.self_attn.q_proj.weight"));
        assert!(s.contains("[4096, 4096]"));
        assert!(s.contains("[2048, 4096]"));
    }

    #[test]
    fn from_io_error_is_transparent() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "not here");
        let surgery_err: SurgeryError = io_err.into();
        match surgery_err {
            SurgeryError::Io(_) => {}
            other => panic!("expected Io variant, got {other:?}"),
        }
    }

    #[test]
    fn from_anyhow_error_routes_to_other() {
        let any: anyhow::Error = anyhow::anyhow!("synthetic config failure");
        let surgery_err: SurgeryError = any.into();
        match surgery_err {
            SurgeryError::Other(inner) => {
                assert_eq!(inner.to_string(), "synthetic config failure");
            }
            other => panic!("expected Other variant, got {other:?}"),
        }
    }
}
