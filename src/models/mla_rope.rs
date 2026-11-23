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

//! The decoupled RoPE of the DeepSeek-style MLA attentions.
//!
//! Upstream builds the rotation of `q_pe` / `k_pe` with `initialize_rope(dims=
//! qk_rope_head_dim, base=rope_theta, traditional=True, scaling_config=
//! rope_scaling)`, which for every shipped DeepSeek-V3 / Kimi-K2 / LongCat /
//! Xing checkpoint is `YarnRoPE`. Rotating with plain `base^(2i/d)`
//! frequencies matches YaRN only on the high-frequency pairs the correction
//! ramp leaves alone; past `original_max_position_embeddings` the
//! low-frequency pairs rotate `factor` times too fast and long-context output
//! degrades (#3178, #3277, #3295).
//!
//! The frequency table comes from the shared [`RopeScalingKind::resolve`], so
//! the `--rope-*` / `--yarn-*` overrides apply here as they do everywhere
//! else. It is built once per model and duplicated into each attention block
//! (and indexer). An absent block, or one that resolves to the default scheme,
//! keeps the exact plain `fast_rope(.., base, 1.0, offset)` call.
//!
//! Used by: DeepSeek-V3 / Kimi-K2 / JoyAI (`deepseek_v3::DeepSeekV3Attention`,
//! and through it Kimi-VL's text model, the DeepSeek-V3 pipeline stage
//! executor, Xing4.0 and Youtu), DeepSeek V3.2 / GLM-5 MLA and DSA indexer
//! (`deepseek_v32_rope::DsaRope`), LongCat-Flash ngram MLA
//! (`longcat_flash_ngram::LongcatFlashMLA`), and LongCat-Flash-Lite-Sparse
//! MLA and LSA indexer (`longcat_flash_sparse`).
//!
//! Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/layers/rope.py (YarnRoPE)

use std::collections::HashMap;

use mlxcel_core::utils::slice_axis;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::rope_utils::{RopeScalingKind, RopeScalingSpec};

/// One resolved RoPE for the first `dims` features of the tensors it rotates.
pub(crate) struct MlaRope {
    kind: RopeScalingKind,
    dims: i32,
    base: f32,
}

impl MlaRope {
    /// Resolve a parsed `rope_scaling` block for `dims` rotated features.
    ///
    /// `deepseek_yarn` is upstream's alias for `yarn` (`initialize_rope`
    /// builds one `YarnRoPE` for both), so it is renamed before the shared
    /// resolver sees it.
    ///
    /// `train_ctx` is the fallback original context when the block names
    /// none. Upstream's `YarnRoPE` defaults a missing
    /// `original_max_position_embeddings` to 4096, not to
    /// `max_position_embeddings`, which is what `None` gives; each family
    /// keeps the choice it made before the type was shared.
    pub(crate) fn resolve(
        spec: Option<&RopeScalingSpec>,
        dims: usize,
        base: f32,
        train_ctx: Option<f32>,
        model_label: &str,
    ) -> Self {
        let spec = spec.map(|spec| {
            if spec.rope_type.as_deref() == Some("deepseek_yarn") {
                RopeScalingSpec {
                    rope_type: Some("yarn".to_string()),
                    ..spec.clone()
                }
            } else {
                spec.clone()
            }
        });
        let kind = RopeScalingKind::resolve(spec.as_ref(), dims, base, train_ctx, model_label);
        Self {
            kind,
            dims: dims as i32,
            // The plain rotation's base has to follow a `--rope-freq-base`
            // override the same way the table does (see `resolve_base`).
            base: crate::models::rope_overrides::resolve_base(base),
        }
    }

    /// [`Self::resolve`] for a config that keeps the block as a JSON map.
    pub(crate) fn from_block(
        block: Option<&HashMap<String, serde_json::Value>>,
        dims: usize,
        base: f32,
        train_ctx: Option<f32>,
        model_label: &str,
    ) -> Self {
        let spec = block.map(|block| RopeScalingSpec::from_lookup(|key| block.get(key)));
        Self::resolve(spec.as_ref(), dims, base, train_ctx, model_label)
    }

    /// Plain `base^(2i/d)` frequencies, with no block and no override.
    ///
    /// For a family that has to keep the bare `fast_rope(.., base, 1.0,
    /// offset)` call whatever its block says (Youtu, which refuses any block
    /// that does not reduce to the identity).
    pub(crate) fn plain(dims: usize, base: f32) -> Self {
        Self {
            kind: RopeScalingKind::Default,
            dims: dims as i32,
            base,
        }
    }

    /// A rope from an already resolved scheme.
    #[cfg(test)]
    pub(crate) fn from_kind(kind: RopeScalingKind, dims: usize, base: f32) -> Self {
        Self {
            kind,
            dims: dims as i32,
            base,
        }
    }

    /// A per-layer handle on the same table (see [`RopeScalingKind::duplicate`]).
    pub(crate) fn duplicate(&self) -> Self {
        Self {
            kind: self.kind.duplicate(),
            dims: self.dims,
            base: self.base,
        }
    }

    /// Whether the scheme is `linear`, which rotates at a position scale
    /// rather than through a table.
    pub(crate) fn is_linear(&self) -> bool {
        matches!(self.kind, RopeScalingKind::Linear { .. })
    }

    /// Rotate the first `dims` features of `x` (`[B, H, L, D]`, `D >= dims`)
    /// at position `offset`.
    pub(crate) fn apply(
        &self,
        x: &MlxArray,
        traditional: bool,
        offset: i32,
    ) -> UniquePtr<MlxArray> {
        let scaled = self.magnitude(x);
        let x = scaled.as_deref().unwrap_or(x);
        match self.kind.freqs() {
            Some(freqs) => {
                mlxcel_core::fast_rope_with_freqs(x, self.dims, traditional, 1.0, offset, freqs)
            }
            None => mlxcel_core::fast_rope(
                x,
                self.dims,
                traditional,
                self.base,
                self.kind.scale(),
                offset,
            ),
        }
    }

    /// [`Self::apply`] with one position per batch row.
    pub(crate) fn apply_batched(
        &self,
        x: &MlxArray,
        traditional: bool,
        offsets: &[i32],
    ) -> UniquePtr<MlxArray> {
        let scaled = self.magnitude(x);
        let x = scaled.as_deref().unwrap_or(x);
        match self.kind.freqs() {
            Some(freqs) => mlxcel_core::fast_rope_batched_with_freqs(
                x,
                self.dims,
                traditional,
                1.0,
                offsets,
                freqs,
            ),
            None => mlxcel_core::fast_rope_batched(
                x,
                self.dims,
                traditional,
                self.base,
                self.kind.scale(),
                offsets,
            ),
        }
    }

    /// Upstream `YarnRoPE.__call__` scales `x[..., :dims]` by its magnitude
    /// multiplier `yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor,
    /// mscale_all_dim)` before rotating. That is `1.0` for the shipped
    /// checkpoints (`mscale == mscale_all_dim`), in which case `x` passes
    /// through untouched. It is independent of the softmax-scale correction
    /// the attentions fold in from `mscale_all_dim`, which upstream applies as
    /// well. `None` means "use `x` as is".
    fn magnitude(&self, x: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        let mscale = self.kind.attn_scale();
        if mscale == 1.0 {
            return None;
        }
        let shape = mlxcel_core::array_shape(x);
        let width = shape.last().copied().unwrap_or(0);
        let scaled = if width > self.dims {
            let head = slice_axis(x, -1, 0, self.dims);
            let tail = slice_axis(x, -1, self.dims, width);
            let head = mlxcel_core::multiply_scalar(&head, mscale);
            mlxcel_core::concatenate(&head, &tail, -1)
        } else {
            mlxcel_core::multiply_scalar(x, mscale)
        };
        Some(scaled)
    }

    /// The precomputed frequency table, when the scheme has one.
    #[cfg(test)]
    pub(crate) fn freqs(&self) -> Option<&MlxArray> {
        self.kind.freqs()
    }

    /// The magnitude multiplier applied before the rotation.
    #[cfg(test)]
    pub(crate) fn mscale(&self) -> f32 {
        self.kind.attn_scale()
    }
}

#[cfg(test)]
#[path = "mla_rope_tests.rs"]
mod tests;
