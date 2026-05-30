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

//! `PruneOp` — Axis A "Operation: Prune".
//!
//! Zero-masks selected slices of model weight tensors at one of three
//! structural granularities:
//!
//! - [`PruneSelector::Layer`]: every tensor matching the glob pattern
//!   AND whose key contains `model.layers.<id>.` is wholly zeroed for
//!   each id in `layer_ids`.
//! - [`PruneSelector::AttentionHead`]: tensors matching the pattern
//!   are recognized as Q / O / K / V projections (and their quantized
//!   `.scales` / `.biases` companions). The slice of the OUT axis (for
//!   Q / K / V) or the IN axis (for O) corresponding to each head id
//!   is zeroed. **GQA policy**: head ids are interpreted as **Q head
//!   ids**. K and V projections are deliberately skipped (with a
//!   warning) because in GQA the same KV head is shared across multiple
//!   Q heads — zeroing it would silently kill more Q heads than the
//!   user asked for. See the module-level GQA section below.
//! - [`PruneSelector::MlpChannel`]: tensors are recognized as
//!   `up_proj` / `gate_proj` (OUT axis = intermediate) and `down_proj`
//!   (IN axis = intermediate); the slice corresponding to each
//!   `channel_ids` entry is zeroed along the appropriate axis.
//!
//! ## GQA decision
//!
//! For Grouped-Query Attention models (e.g. Llama 3+, Qwen 2.5, Gemma 3),
//! `num_attention_heads > num_key_value_heads`. A single KV head feeds
//! `num_attention_heads / num_key_value_heads` Q heads. The user gives
//! Q head ids in `head_ids` (this matches the dominant mental model:
//! "kill the contribution of head N"). The implementation:
//!
//! 1. Zeroes the corresponding OUT slice of `q_proj` and the IN slice
//!    of `o_proj` for each listed Q head id.
//! 2. **Skips** `k_proj` and `v_proj` tensors entirely, even when the
//!    glob pattern matches them. A one-line `eprintln!` warning is
//!    emitted per skipped tensor so the user can verify the policy.
//! 3. Skips `q_norm` and `k_norm` tensors — head-dim norms are shared
//!    across heads (Qwen 3 / DeepSeek style), so per-head zeroing of
//!    these is not meaningful.
//!
//! The alternative policy ("zero a KV head only if no Q head still
//! consumes it") is rejected for two reasons: (a) the common case is
//! that **every** KV head is shared by at least two Q heads, so a
//! literal interpretation would almost always be a no-op for KV; and
//! (b) zeroing a KV head also kills every Q head that maps to it,
//! which violates the principle of least surprise.
//!
//! ## Quantization handling
//!
//! For Affine 4-bit / 8-bit quantization (`mlx-community/*-4bit`,
//! `*-8bit`):
//! - `.weight` holds packed integers. Each output row is independent,
//!   so zeroing rows is byte-level safe. The packed value 0 always
//!   represents q=0 in MLX's affine packing.
//! - `.scales` holds the per-group dequantization scale.
//! - `.biases` holds the per-group dequantization bias.
//! - Dequantized value = `q * scale + bias`. Setting q=0 AND bias=0
//!   yields dequant=0 regardless of `scale`. The op therefore zeros
//!   `weight`, `scales`, **and** `biases` for each affected row.
//!
//! For MXFP4 / NVFP4 / MXFP8 (no biases, scale-only): zeroing
//! `weight` and `scales` is sufficient (dequant = packed * scale = 0).
//!
//! The IN-axis prune (o_proj, down_proj) requires that the slice
//! boundary be aligned to `group_size`, otherwise the op cannot zero
//! a partial group without disturbing untargeted columns. The op
//! detects misalignment and returns a clear error rather than
//! silently zeroing a wider region.
//!
//! ## Module layout
//!
//! The implementation is split into four files to keep each below
//! the project's 500-line soft target:
//!
//! - `mod.rs` (this file): public types ([`PruneOp`],
//!   [`PruneSelector`]), the [`crate::SurgeryOp`] trait impl, and the
//!   YAML factory hook ([`build_from_yaml`]).
//! - `model_dims.rs`: read attention / MLP dimensions from
//!   `config.json` and validate id lists against them.
//! - `granularity.rs`: per-granularity pruners
//!   ([`granularity::prune_layers`], [`granularity::prune_attention_heads`],
//!   [`granularity::prune_mlp_channels`]) and key-suffix classifiers.
//! - `tensor_ops.rs`: low-level tensor-zeroing primitives over the
//!   `mlxcel-core` FFI ([`tensor_ops::zero_axis0_rows`], etc.).
//! - `tests.rs`: unit tests for the above.
//!
//! Used by: [`crate::config::materialize_op`] (when an `OpSpec::Prune`
//! is parsed from YAML), [`crate::SurgeryPipeline`] (via the
//! [`crate::SurgeryOp`] trait object).

use std::sync::Arc;

use globset::{Glob, GlobMatcher};

use crate::config::PruneGranularity;
use crate::{SharedSurgeryOp, SurgeryError, SurgeryOp, WeightMap};

mod granularity;
mod model_dims;
mod tensor_ops;

#[cfg(test)]
mod tests;

use model_dims::ModelDims;

/// Selector portion of a [`PruneOp`].
///
/// `Layer` carries `layer_ids`, `AttentionHead` carries Q-head ids,
/// `MlpChannel` carries channel ids. The parser ([`crate::config`])
/// validates that the correct list is non-empty for the chosen
/// granularity at YAML-load time, so consumers can treat the field as
/// authoritative.
#[derive(Debug, Clone)]
pub enum PruneSelector {
    /// Zero every tensor in the listed transformer blocks.
    Layer { layer_ids: Vec<usize> },
    /// Zero the Q / O slices for the listed Q head ids. K / V left
    /// untouched per the GQA policy documented at module level.
    AttentionHead { head_ids: Vec<usize> },
    /// Zero the IN / OUT slices of the listed MLP intermediate
    /// channel ids.
    MlpChannel { channel_ids: Vec<usize> },
}

impl PruneSelector {
    /// Short label for log messages.
    fn label(&self) -> &'static str {
        match self {
            Self::Layer { .. } => "layer",
            Self::AttentionHead { .. } => "attention_head",
            Self::MlpChannel { .. } => "mlp_channel",
        }
    }
}

/// Compiled [`crate::OpSpec::Prune`] ready to run.
///
/// Stores the compiled glob and the validated selector. The pattern
/// string is retained for log/Debug clarity; the matcher is what the
/// op uses on every key.
///
/// Used by: [`crate::config::materialize_op`], unit tests in this
/// module
pub struct PruneOp {
    /// Original glob pattern, kept for human-readable logging.
    pattern_src: String,
    matcher: GlobMatcher,
    selector: PruneSelector,
}

impl PruneOp {
    /// Construct a [`PruneOp`] from a parsed glob and selector.
    ///
    /// `pattern` must be a syntactically valid glob — for the YAML
    /// path the parser has already validated this, but the function
    /// re-checks so direct callers (tests, programmatic users) get a
    /// clear error rather than a panic.
    ///
    /// Used by: [`crate::config::materialize_op`] (YAML path), unit
    /// tests, future programmatic API users
    pub fn new(pattern: &str, selector: PruneSelector) -> Result<Self, SurgeryError> {
        let glob = Glob::new(pattern).map_err(|e| {
            SurgeryError::Other(anyhow::anyhow!(
                "prune: malformed glob pattern {pattern:?}: {e}"
            ))
        })?;
        Ok(Self {
            pattern_src: pattern.to_string(),
            matcher: glob.compile_matcher(),
            selector,
        })
    }

    /// Convenience constructor that returns a [`SharedSurgeryOp`] (the
    /// pipeline's storage type). Used by the YAML factory and tests
    /// that need to push directly into a pipeline.
    pub fn into_shared(self) -> SharedSurgeryOp {
        Arc::new(self)
    }
}

impl SurgeryOp for PruneOp {
    fn apply(&self, weights: &mut WeightMap, cfg: &serde_json::Value) -> Result<(), SurgeryError> {
        // Collect matching keys first so we can mutate the map safely
        // and so we can detect the "pattern matches zero tensors"
        // error before doing any work.
        let matched: Vec<String> = weights
            .keys()
            .filter(|k| self.matcher.is_match(k.as_str()))
            .cloned()
            .collect();
        if matched.is_empty() {
            return Err(SurgeryError::Other(anyhow::anyhow!(
                "prune: pattern {:?} matched zero tensors",
                self.pattern_src
            )));
        }

        let model = ModelDims::from_config(cfg)?;
        model_dims::validate_ids(&self.selector, &model)?;

        match &self.selector {
            PruneSelector::Layer { layer_ids } => {
                granularity::prune_layers(weights, &matched, layer_ids)
            }
            PruneSelector::AttentionHead { head_ids } => {
                granularity::prune_attention_heads(weights, &matched, &model, head_ids)
            }
            PruneSelector::MlpChannel { channel_ids } => {
                granularity::prune_mlp_channels(weights, &matched, &model, channel_ids)
            }
        }
    }

    fn name(&self) -> &'static str {
        "prune"
    }
}

impl std::fmt::Debug for PruneOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PruneOp")
            .field("pattern", &self.pattern_src)
            .field("granularity", &self.selector.label())
            .field("selector", &self.selector)
            .finish()
    }
}

/// Convenience: build a [`PruneOp`] from the parsed [`PruneGranularity`]
/// enum the YAML layer hands out and the matching id list.
///
/// The id-list shape is determined by the granularity; the parser has
/// already enforced that the correct field is `Some(..)` and
/// non-empty, so this helper takes the three Option<Vec<usize>> and
/// trusts that exactly one is `Some` based on `granularity`.
///
/// Used by: [`crate::config::materialize_op`]
pub(crate) fn build_from_yaml(
    pattern: &str,
    granularity: PruneGranularity,
    head_ids: Option<Vec<usize>>,
    channel_ids: Option<Vec<usize>>,
    layer_ids: Option<Vec<usize>>,
) -> Result<PruneOp, SurgeryError> {
    let selector = match granularity {
        PruneGranularity::Layer => PruneSelector::Layer {
            // Parser guarantees Some(non_empty).
            layer_ids: layer_ids
                .ok_or_else(|| internal_err("layer prune missing layer_ids after validation"))?,
        },
        PruneGranularity::AttentionHead => PruneSelector::AttentionHead {
            head_ids: head_ids.ok_or_else(|| {
                internal_err("attention_head prune missing head_ids after validation")
            })?,
        },
        PruneGranularity::MlpChannel => PruneSelector::MlpChannel {
            channel_ids: channel_ids.ok_or_else(|| {
                internal_err("mlp_channel prune missing channel_ids after validation")
            })?,
        },
    };
    PruneOp::new(pattern, selector)
}

fn internal_err(msg: &str) -> SurgeryError {
    SurgeryError::Other(anyhow::anyhow!("prune (internal): {msg}"))
}
