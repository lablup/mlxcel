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

//! `SurgeryPipeline` — ordered collection of `SurgeryOp` that itself
//! implements the [`mlxcel_core::weights::WeightTransform`] hook
//! (A1).

use mlxcel_core::weights::{WeightMap, WeightTransform};

use crate::{SharedSurgeryOp, SurgeryOp};

/// Ordered list of [`SurgeryOp`] applied in registration order.
///
/// `SurgeryPipeline` is the only public type that gets handed to the
/// consolidated weight loaders. It is constructed either
/// programmatically (`SurgeryPipeline::new()` followed by `push`) or
/// by the config layer, and then passed to
/// `mlxcel::models::load_text_weights(path, Some(&pipeline))` or the
/// VLM equivalent.
///
/// The struct is intentionally cheap to clone (each op is wrapped in
/// `Arc<dyn SurgeryOp>`), so a parsed pipeline can be shared by
/// every concurrent model loader without re-parsing the config.
///
/// Used by: future `mlxcel::surgery` integration glue (CLI flag —
/// A4), text and VLM consolidated loaders (via the WeightTransform hook)
#[derive(Default, Clone)]
pub struct SurgeryPipeline {
    ops: Vec<SharedSurgeryOp>,
}

impl SurgeryPipeline {
    /// Construct an empty pipeline. Applying an empty pipeline is a
    /// no-op and is therefore bit-exact identical to the
    /// `transform = None` path in the consolidated loaders.
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Append `op` to the end of the pipeline. Ops apply in
    /// registration order during [`WeightTransform::apply`].
    pub fn push(&mut self, op: SharedSurgeryOp) {
        self.ops.push(op);
    }

    /// Number of registered operations.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// `true` if no operations are registered. The hook is then a
    /// guaranteed no-op.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Iterate registered operations in apply order (used by tracing
    /// helpers and by the config layer's `Display`).
    pub fn ops(&self) -> impl Iterator<Item = &dyn SurgeryOp> {
        self.ops.iter().map(|op| op.as_ref())
    }
}

impl WeightTransform for SurgeryPipeline {
    /// Apply every registered [`SurgeryOp`] in order. Short-circuits
    /// on the first error and surfaces the underlying
    /// [`crate::SurgeryError`] as a `String` so the consolidated
    /// loader's `Result<_, String>` API stays uniform.
    fn apply(&self, weights: &mut WeightMap, cfg: &serde_json::Value) -> Result<(), String> {
        for op in &self.ops {
            op.apply(weights, cfg)
                .map_err(|e| format!("surgery op `{}` failed: {e}", op.name()))?;
        }
        Ok(())
    }
}

/// Convenience constructor: build a pipeline from any iterator of
/// shared ops. Useful for the config layer which yields
/// ops one at a time as it parses YAML.
impl FromIterator<SharedSurgeryOp> for SurgeryPipeline {
    fn from_iter<I: IntoIterator<Item = SharedSurgeryOp>>(iter: I) -> Self {
        Self {
            ops: iter.into_iter().collect(),
        }
    }
}

/// Convenience constructor: build a pipeline from a `Vec`.
impl From<Vec<SharedSurgeryOp>> for SurgeryPipeline {
    fn from(ops: Vec<SharedSurgeryOp>) -> Self {
        Self { ops }
    }
}

impl std::fmt::Debug for SurgeryPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SurgeryPipeline")
            .field(
                "ops",
                &self.ops.iter().map(|op| op.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    /// No-op stub used to populate a pipeline so we can verify
    /// `Debug`, `Clone`, and `ops()` iteration without depending on
    /// the test fixtures defined in `lib.rs`.
    struct NoOp(&'static str);

    impl SurgeryOp for NoOp {
        fn apply(
            &self,
            _weights: &mut WeightMap,
            _cfg: &serde_json::Value,
        ) -> Result<(), crate::SurgeryError> {
            Ok(())
        }
        fn name(&self) -> &'static str {
            self.0
        }
    }

    #[test]
    fn new_pipeline_is_empty_and_default_matches() {
        assert!(SurgeryPipeline::new().is_empty());
        assert!(SurgeryPipeline::default().is_empty());
    }

    #[test]
    fn from_iter_and_from_vec_preserve_order() {
        let ops: Vec<Arc<dyn SurgeryOp>> = vec![
            Arc::new(NoOp("a")),
            Arc::new(NoOp("b")),
            Arc::new(NoOp("c")),
        ];

        let from_iter: SurgeryPipeline = ops.clone().into_iter().collect();
        let from_vec = SurgeryPipeline::from(ops);

        let names_from_iter: Vec<&'static str> = from_iter.ops().map(|op| op.name()).collect();
        let names_from_vec: Vec<&'static str> = from_vec.ops().map(|op| op.name()).collect();
        assert_eq!(names_from_iter, vec!["a", "b", "c"]);
        assert_eq!(names_from_vec, vec!["a", "b", "c"]);
    }

    #[test]
    fn debug_lists_op_names_in_order() {
        let mut pipeline = SurgeryPipeline::new();
        pipeline.push(Arc::new(NoOp("scale")));
        pipeline.push(Arc::new(NoOp("add")));
        let dbg = format!("{pipeline:?}");
        assert!(dbg.contains("scale"));
        assert!(dbg.contains("add"));
        assert!(dbg.find("scale").unwrap() < dbg.find("add").unwrap());
    }

    #[test]
    fn clone_shares_ops_via_arc() {
        let mut pipeline = SurgeryPipeline::new();
        let op: SharedSurgeryOp = Arc::new(NoOp("scale"));
        pipeline.push(op.clone());

        let cloned = pipeline.clone();
        assert_eq!(cloned.len(), pipeline.len());
        // Arc count is 3: the original local + pipeline + cloned pipeline
        assert_eq!(Arc::strong_count(&op), 3);
    }

    #[test]
    fn empty_pipeline_apply_returns_ok_unit() {
        let pipeline = SurgeryPipeline::new();
        let mut weights = WeightMap::new();
        let result = <SurgeryPipeline as WeightTransform>::apply(
            &pipeline,
            &mut weights,
            &serde_json::Value::Null,
        );
        assert!(result.is_ok());
    }
}
