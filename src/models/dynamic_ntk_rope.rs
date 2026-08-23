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

//! Dynamic NTK RoPE, the schedule the InternLM families declare.
//!
//! # What the schedule is
//!
//! The reference is the checkpoint's own remote code:
//! `modeling_internlm3.py` builds its rotary embedding through
//! `ROPE_INIT_FUNCTIONS[rope_type]`, and for `"dynamic"` that is transformers'
//! [`_compute_dynamic_ntk_parameters`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py).
//! With `f = factor`, `d = dims`, `M = max_position_embeddings`:
//!
//! ```text
//! seq_len  = max(L + offset, M)
//! base_eff = base * ((f * seq_len / M) - (f - 1)) ^ (d / (d - 2))
//! angle(p, j) = p * base_eff ^ (-2j / d)          for 0 <= j < d / 2
//! ```
//!
//! The position `p` enters that angle unscaled. Nothing in the dynamic branch
//! divides or multiplies a position; the whole adjustment lives in the base,
//! and the clamp means the base does not move at all until the sequence passes
//! `max_position_embeddings`. `"linear"` is the other way round: the base is
//! untouched and the position becomes `p / f`, which MLX's `fast::rope`
//! expresses as `scale = 1 / f` because it multiplies the position by `scale`.
//!
//! # Why this module exists
//!
//! Both InternLM families had this wrong, in two different ways, and both were
//! ported faithfully from mlx-lm, which has it wrong in the same place
//! ([`mlx_lm/models/internlm3.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/internlm3.py)
//! and
//! [`mlx_lm/models/internlm2.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/internlm2.py)
//! both end their `rope_scale` expression in `else 2.0` and then reuse that
//! same number as the NTK factor). Keeping the decision in one place is what
//! stops the next family from picking one of the two halves and dropping the
//! other:
//!
//! * `internlm3` parsed the block, matched only `"linear"`, and fell through to
//!   `unwrap_or(2.0)` for everything else including `"dynamic"` and the absent
//!   block, so every query and key was rotated at twice its true position at
//!   every context length (issue #1324).
//! * `internlm2` never declared `rope_scaling` in its `ModelArgs` at all, so
//!   its checkpoints' live `{"type": "dynamic", "factor": 2.0}` was dropped at
//!   deserialization and the base never left `rope_theta`.
//!
//! # Why the block is read through [`RopeScalingSpec`]
//!
//! The key naming the scheme is spelled `type` by the InternLM2 checkpoints and
//! `rope_type` by the InternLM3 ones, and some conversions carry both. A
//! derived struct with `#[serde(rename = "type", alias = "rope_type")]` turns
//! the both-keys case into a hard `duplicate field` parse error, which is why
//! `rope_utils::RopeScalingSpec` reads the block as a JSON map instead. This
//! module builds on that reader rather than on a second one.

use crate::models::rope_utils::{RopeScalingSpec, is_usable_scalar, printable_label};
use mlxcel_core::{MlxArray, UniquePtr};

/// The three `rope_scaling` schemes the InternLM families accept.
///
/// Anything else is a load error rather than a silent fallback; see
/// [`DynamicNtkRope::from_scaling`].
// Used by: InternLM3, InternLM2
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DynamicNtkRopeMode {
    /// No block, or `"default"`. Unscaled positions on the plain base.
    Default,
    /// Positions divided by `factor`; the base never moves.
    Linear { factor: f32 },
    /// Unscaled positions; the base grows once the sequence passes
    /// `max_position_embeddings`.
    Dynamic { factor: f32 },
}

/// A resolved rotary schedule: everything `fast_rope` needs except the tensor.
///
/// One of these is built per attention block at load time. It owns no MLX
/// array, so duplicating it across layers costs nothing and each block stays
/// independently constructible.
// Used by: InternLM3, InternLM2
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicNtkRope {
    dims: i32,
    base: f32,
    traditional: bool,
    max_position_embeddings: usize,
    mode: DynamicNtkRopeMode,
}

impl DynamicNtkRope {
    /// Resolve a `rope_scaling` block, or name why it cannot be served.
    ///
    /// * absent, or `type` / `rope_type` in `{None, "default"}`:
    ///   [`DynamicNtkRopeMode::Default`].
    /// * `"linear"` / `"dynamic"`: require a positive finite `factor`.
    /// * anything else: `Err`.
    ///
    /// # Why an unimplemented scheme is an error here and a warning in `rope_utils`
    ///
    /// [`crate::models::rope_utils::RopeScalingKind::resolve`] degrades to the
    /// plain table with a warning, because the args it serves are also what
    /// several VLM loaders parse a `text_config` into, and at least one
    /// checkpoint that loads today (`models/internvl3-1b`) declares a scheme
    /// that path does not implement. Erroring there would take a working model
    /// offline.
    ///
    /// Neither InternLM `ModelArgs` is reachable that way. Both are only
    /// constructed from a top-level `config.json` through the `ConfigBacked`
    /// route in `src/model_metadata.rs`; `src/loading/vlm_internvl.rs` parses
    /// its `text_config` into `llama3::ModelArgs`, not into either of these. So
    /// an unimplemented scheme here can only mean a checkpoint whose positions
    /// would be computed wrongly for its whole context, and refusing to load it
    /// is also exactly what both upstream `__post_init__` implementations do
    /// (`raise ValueError` on anything outside `{"linear", "dynamic"}`).
    ///
    /// `model_label` is only used to name the offending checkpoint in the
    /// error. It is checkpoint-controlled text, so it goes through
    /// [`printable_label`], as does the scheme name.
    // Used by: InternLM3, InternLM2
    pub fn from_scaling(
        dims: i32,
        base: f32,
        traditional: bool,
        max_position_embeddings: usize,
        scaling: Option<&RopeScalingSpec>,
        model_label: &str,
    ) -> Result<Self, String> {
        let mode = Self::resolve_mode(scaling, max_position_embeddings, model_label)?;
        Ok(Self {
            dims,
            base,
            traditional,
            max_position_embeddings,
            mode,
        })
    }

    fn resolve_mode(
        scaling: Option<&RopeScalingSpec>,
        max_position_embeddings: usize,
        model_label: &str,
    ) -> Result<DynamicNtkRopeMode, String> {
        let Some(spec) = scaling else {
            return Ok(DynamicNtkRopeMode::Default);
        };

        let who = printable_label(model_label);
        let kind = spec.rope_type();
        match kind {
            "default" => Ok(DynamicNtkRopeMode::Default),
            "linear" | "dynamic" => {
                // Upstream indexes `rope_scaling["factor"]` with no default, so
                // a block without one is malformed. Defaulting it to 1.0 would
                // build the identity schedule and hide the malformed config,
                // which is the silence this module exists to remove.
                let Some(factor) = spec.factor else {
                    return Err(format!(
                        "{who}: rope_scaling type \"{kind}\" names no numeric factor"
                    ));
                };
                if !is_usable_scalar(factor) {
                    return Err(format!(
                        "{who}: rope_scaling factor {factor} is not a positive finite number"
                    ));
                }
                if kind == "linear" {
                    return Ok(DynamicNtkRopeMode::Linear { factor });
                }
                // `max_position_embeddings` is the divisor of the base
                // schedule. Zero would make every effective base `inf` and
                // every logit `NaN`, with nothing on the path throwing.
                if max_position_embeddings == 0 {
                    return Err(format!(
                        "{who}: rope_scaling type \"dynamic\" needs a non-zero \
                         max_position_embeddings"
                    ));
                }
                Ok(DynamicNtkRopeMode::Dynamic { factor })
            }
            other => Err(format!(
                "{who}: rope_scaling type \"{}\" is not implemented for the InternLM families; \
                 only \"linear\" and \"dynamic\" are",
                printable_label(other)
            )),
        }
    }

    /// The resolved scheme.
    // Used by: InternLM3, InternLM2 (tests)
    pub fn mode(&self) -> DynamicNtkRopeMode {
        self.mode
    }

    /// The position scale to hand `fast_rope`.
    ///
    /// `1.0` for `Default` **and** for `Dynamic`: the dynamic schedule adjusts
    /// the base, never the position. Only `Linear` scales, and MLX multiplies
    /// the position by `scale`, so dividing by `factor` is `1 / factor`.
    // Used by: InternLM3, InternLM2
    pub fn scale(&self) -> f32 {
        match self.mode {
            DynamicNtkRopeMode::Default | DynamicNtkRopeMode::Dynamic { .. } => 1.0,
            DynamicNtkRopeMode::Linear { factor } => 1.0 / factor,
        }
    }

    /// The rotary base for a forward whose last position is `seq_len - 1`.
    ///
    /// `seq_len` is `L + offset`, the number of positions the cache will hold
    /// after this forward. For `Default` and `Linear` the base is constant. For
    /// `Dynamic` it is the NTK-rescaled base, computed from `seq_len` clamped
    /// up to `max_position_embeddings`, so it equals the plain base for every
    /// sequence at or below that length and grows monotonically past it.
    ///
    /// `factor` is screened positive at construction and the clamp puts
    /// `seq_len >= max_position_embeddings`, so the ratio is always at least
    /// `1.0` and `powf` cannot produce a NaN here.
    // Used by: InternLM3, InternLM2
    pub fn base_for(&self, seq_len: i32) -> f32 {
        let DynamicNtkRopeMode::Dynamic { factor } = self.mode else {
            return self.base;
        };

        // `max_position_embeddings` is a `usize` from the config and the clamp
        // has to happen in a type that holds both. Saturating rather than
        // `as`-truncating keeps an absurd config from wrapping to a negative
        // clamp instead of simply never triggering the rescale.
        let max_pos = i64::try_from(self.max_position_embeddings).unwrap_or(i64::MAX);
        let clamped = (seq_len as i64).max(max_pos);

        let ratio = (factor * (clamped as f32) / (max_pos as f32)) - (factor - 1.0);
        let power = (self.dims as f32) / ((self.dims - 2) as f32);
        self.base * ratio.powf(power)
    }

    /// Apply the schedule to a `[B, n_heads, L, head_dim]` tensor.
    ///
    /// `offset` is the cache offset (the position of the first row of `x`) and
    /// `seq_len` is `L + offset`.
    // Used by: InternLM3, InternLM2
    pub fn apply(&self, x: &MlxArray, offset: i32, seq_len: i32) -> UniquePtr<MlxArray> {
        mlxcel_core::fast_rope(
            x,
            self.dims,
            self.traditional,
            self.base_for(seq_len),
            self.scale(),
            offset,
        )
    }
}

#[cfg(test)]
#[path = "dynamic_ntk_rope_tests.rs"]
mod tests;
