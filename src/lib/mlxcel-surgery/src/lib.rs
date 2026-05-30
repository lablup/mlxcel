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

//! `mlxcel-surgery` — Axis A "weight-load surgery" framework
//! (A5–A9).
//!
//! This crate is the structural fine-tuning machinery. It owns:
//!
//! - [`SurgeryOp`] — the trait every individual surgical operation
//!   (scale / add / prune / replace / interpolate, see A5–A9)
//!   implements. Operations transform an in-memory [`WeightMap`] in
//!   place.
//! - [`SurgeryPipeline`] — an ordered collection of `SurgeryOp` that
//!   itself implements [`mlxcel_core::weights::WeightTransform`]
//!   (A1). Plugging a `SurgeryPipeline` into
//!   `mlxcel::models::load_text_weights(path, Some(&pipeline))` (or
//!   the VLM equivalent) is the entire integration surface with the
//!   load path.
//! - [`SurgeryError`] — crate-level error type. It composes with
//!   `anyhow::Error` (via `#[from]`) so the consolidated loader's
//!   `Result<_, String>` adapter can convert it without ceremony.
//!
//! ## Isolation
//!
//! The crate is intentionally a leaf node in the workspace graph. It
//! depends only on `mlxcel-core` (for the `WeightMap` /
//! `WeightTransform` types and the underlying `MlxArray` ops).
//! Consumers opt in by enabling the top-level `mlxcel` crate's
//! `surgery` feature; when the feature is off, the crate is not
//! linked into `mlxcel` at all.
//!
//! See `docs_internal/architecture/structural-finetuning-overview-20260419.md`
//! sections §3.2, §3.3, and §5 for the architectural rationale.

use std::sync::Arc;

pub use mlxcel_core::weights::{WeightMap, WeightTransform};

pub mod config;
mod error;
pub mod ops;
mod pipeline;

pub use config::{
    parse_config_file, parse_config_str, InterpolateMethod, OpSpec, PruneGranularity,
    SurgeryConfig, SUPPORTED_SCHEMA_VERSION,
};
pub use error::SurgeryError;
pub use ops::{AddOp, InterpolateOp, PruneOp, PruneSelector, ReplaceOp, ScaleOp};
pub use pipeline::SurgeryPipeline;

/// A single weight-load surgical operation.
///
/// Implementations transform an in-memory [`WeightMap`] in place.
/// They observe the parsed `config.json` ([`serde_json::Value`]) so
/// they can locate quantization metadata, layer counts, head
/// dimensions, etc. without depending on every model-specific
/// `ModelArgs` struct.
///
/// The trait is `Send + Sync` because pipelines may be cloned by
/// reference across threads (for example, the surgery pipeline is
/// shared by every model load worker). Implementations must remain
/// stateless across `apply` calls — any per-load state belongs in
/// the [`WeightMap`] itself.
///
/// Used by: SurgeryPipeline, future concrete ops (A5–A9 — Scale, Add,
/// Prune, Replace, Interpolate)
pub trait SurgeryOp: Send + Sync {
    /// Apply this operation to the in-memory weight map.
    ///
    /// The operation must not hold references into `weights` after
    /// return — by the time `apply` returns, ownership of every
    /// tensor in the map is unchanged from the caller's perspective.
    fn apply(&self, weights: &mut WeightMap, cfg: &serde_json::Value) -> Result<(), SurgeryError>;

    /// Short name for logging / metrics (e.g. `"scale"`, `"add"`,
    /// `"prune"`). Used by tracing spans and CLI status output.
    fn name(&self) -> &'static str;
}

/// Boxed, refcounted `SurgeryOp`. The pipeline stores these directly
/// so individual operations can be referenced (and unit-tested) from
/// elsewhere without losing trait-object dispatch.
pub type SharedSurgeryOp = Arc<dyn SurgeryOp>;

#[cfg(test)]
mod tests {
    //! Unit tests for the public surface.
    //!
    //! Covers acceptance criterion (a) — pipeline ordering and error
    //! propagation on a synthetic [`WeightMap`] — and (b) — the
    //! pipeline is invoked through A1's `WeightTransform` hook with
    //! no-op semantics yielding bit-exactness vs the `None` path.

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::*;

    /// Mock op that increments a shared counter on every `apply`.
    /// Used to prove the pipeline invokes ops in registration order
    /// and exactly once per pipeline `apply` call.
    struct CountingOp {
        tag: u32,
        counter: Arc<AtomicUsize>,
        order_log: Arc<std::sync::Mutex<Vec<u32>>>,
    }

    impl SurgeryOp for CountingOp {
        fn apply(
            &self,
            _weights: &mut WeightMap,
            _cfg: &serde_json::Value,
        ) -> Result<(), SurgeryError> {
            self.counter.fetch_add(1, Ordering::SeqCst);
            self.order_log.lock().unwrap().push(self.tag);
            Ok(())
        }

        fn name(&self) -> &'static str {
            "counting"
        }
    }

    /// Mock op that always fails — exercises pipeline short-circuiting
    /// and error propagation.
    struct FailingOp;

    impl SurgeryOp for FailingOp {
        fn apply(
            &self,
            _weights: &mut WeightMap,
            _cfg: &serde_json::Value,
        ) -> Result<(), SurgeryError> {
            Err(SurgeryError::TensorNotFound("synthetic".to_string()))
        }

        fn name(&self) -> &'static str {
            "failing"
        }
    }

    #[test]
    fn empty_pipeline_is_no_op() {
        let pipeline = SurgeryPipeline::new();
        assert!(pipeline.is_empty());
        assert_eq!(pipeline.len(), 0);

        let mut weights = WeightMap::new();
        // Apply through the WeightTransform trait — this is the same
        // entry point A1's `load_text_weights` will use.
        let res = <SurgeryPipeline as WeightTransform>::apply(
            &pipeline,
            &mut weights,
            &serde_json::Value::Null,
        );
        assert!(res.is_ok(), "empty pipeline must succeed");
        assert!(weights.is_empty(), "empty pipeline must not touch weights");
    }

    #[test]
    fn pipeline_applies_ops_in_registration_order() {
        let counter = Arc::new(AtomicUsize::new(0));
        let order_log = Arc::new(std::sync::Mutex::new(Vec::new()));

        let mut pipeline = SurgeryPipeline::new();
        for tag in [1u32, 2, 3] {
            pipeline.push(Arc::new(CountingOp {
                tag,
                counter: counter.clone(),
                order_log: order_log.clone(),
            }));
        }
        assert_eq!(pipeline.len(), 3);
        assert!(!pipeline.is_empty());

        let mut weights = WeightMap::new();
        <SurgeryPipeline as WeightTransform>::apply(
            &pipeline,
            &mut weights,
            &serde_json::Value::Null,
        )
        .expect("pipeline must succeed");

        assert_eq!(
            counter.load(Ordering::SeqCst),
            3,
            "each op runs exactly once"
        );
        let recorded = order_log.lock().unwrap().clone();
        assert_eq!(
            recorded,
            vec![1, 2, 3],
            "ops must run in registration order"
        );
    }

    #[test]
    fn pipeline_short_circuits_on_error() {
        let counter = Arc::new(AtomicUsize::new(0));
        let order_log = Arc::new(std::sync::Mutex::new(Vec::new()));

        let mut pipeline = SurgeryPipeline::new();
        pipeline.push(Arc::new(CountingOp {
            tag: 1,
            counter: counter.clone(),
            order_log: order_log.clone(),
        }));
        pipeline.push(Arc::new(FailingOp));
        pipeline.push(Arc::new(CountingOp {
            tag: 3,
            counter: counter.clone(),
            order_log: order_log.clone(),
        }));

        let mut weights = WeightMap::new();
        let result = <SurgeryPipeline as WeightTransform>::apply(
            &pipeline,
            &mut weights,
            &serde_json::Value::Null,
        );
        assert!(result.is_err(), "pipeline with failing op must error");

        // First counting op ran, failing op returned an error, third
        // op never ran.
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "only the first op should have executed",
        );
        assert_eq!(order_log.lock().unwrap().as_slice(), &[1u32]);
    }

    #[test]
    fn weight_transform_signature_matches_a1() {
        // Compile-time check that `SurgeryPipeline` is a drop-in for
        // any `&dyn WeightTransform` callsite — this is exactly what
        // the consolidated text/VLM loaders take.
        fn accepts_transform(_t: &dyn WeightTransform) {}
        let pipeline = SurgeryPipeline::new();
        accepts_transform(&pipeline);
    }
}
