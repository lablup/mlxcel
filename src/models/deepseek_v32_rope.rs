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

//! The RoPE shared by the DeepSeek V3.2 MLA attention and its DSA indexer.
//!
//! Upstream builds both with `initialize_rope(..., scaling_config=
//! args.rope_scaling)`, which for the DeepSeek-V3.2 checkpoints is `YarnRoPE`
//! (`factor` 40 over `original_max_position_embeddings` 4096). Before #3178
//! both used plain RoPE, which matches YaRN only on the high-frequency pairs
//! the correction ramp leaves alone; past 4096 positions the low-frequency
//! pairs rotate 40x too fast and long-context output degrades.
//!
//! The table comes from the shared [`RopeScalingKind::resolve`] (so the
//! `--rope-*` / `--yarn-*` overrides apply here as they do everywhere else),
//! and is built once per model then duplicated into each attention block and
//! indexer. An absent block, or one that resolves to the default scheme, keeps
//! the exact plain `fast_rope(.., base, 1.0, offset)` call the model made
//! before.
//!
//! Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/rope_utils.py (YarnRoPE)

use mlxcel_core::utils::slice_axis;
use mlxcel_core::{MlxArray, UniquePtr};

use super::ModelArgs;
use crate::models::rope_utils::{RopeScalingKind, RopeScalingSpec};

/// One resolved RoPE for `qk_rope_head_dim` rotated dims.
pub(super) struct DsaRope {
    kind: RopeScalingKind,
    dims: i32,
    base: f32,
}

impl DsaRope {
    /// Resolve the checkpoint's `rope_scaling` block for this model.
    ///
    /// `deepseek_yarn` is upstream's alias for `yarn` (`initialize_rope`
    /// accepts both), so it is renamed before the shared resolver sees it.
    /// No training-context fallback is passed: upstream's `YarnRoPE` defaults a
    /// missing `original_max_position_embeddings` to 4096, not to
    /// `max_position_embeddings`, and the shared resolver does the same when
    /// given `None`.
    ///
    /// Refuses a `qk_rope_head_dim` that is zero, odd, or wider than the
    /// indexer head (`index_head_dim`, the narrower of the two tensors this
    /// rotates) before the YaRN table is built, so a malformed config fails
    /// at load instead of panicking or reading out of range in the kernel.
    pub(super) fn from_args(args: &ModelArgs) -> Result<Self, String> {
        let dims = args.qk_rope_head_dim;
        if dims == 0 || !dims.is_multiple_of(2) {
            return Err(format!(
                "qk_rope_head_dim must be a positive even number, got {dims}"
            ));
        }
        let widest = args.index_head_dim.min(args.qk_nope_head_dim + dims);
        if dims > widest {
            return Err(format!(
                "qk_rope_head_dim ({dims}) exceeds the head dim it rotates ({widest}); check qk_nope_head_dim and index_head_dim"
            ));
        }
        let spec = args.rope_scaling.as_ref().map(|spec| {
            if spec.rope_type.as_deref() == Some("deepseek_yarn") {
                RopeScalingSpec {
                    rope_type: Some("yarn".to_string()),
                    ..spec.clone()
                }
            } else {
                spec.clone()
            }
        });
        let kind =
            RopeScalingKind::resolve(spec.as_ref(), dims, args.rope_theta, None, &args.model_type);
        Ok(Self {
            kind,
            dims: dims as i32,
            // The plain rotation's base has to follow a `--rope-freq-base`
            // override the same way the table does (see `resolve_base`).
            base: crate::models::rope_overrides::resolve_base(args.rope_theta),
        })
    }

    /// A per-layer handle on the same table (see [`RopeScalingKind::duplicate`]).
    pub(super) fn duplicate(&self) -> Self {
        Self {
            kind: self.kind.duplicate(),
            dims: self.dims,
            base: self.base,
        }
    }

    /// Rotate the first `dims` features of `x` (`[B, H, L, D]`, `D >= dims`)
    /// at position `offset`.
    pub(super) fn apply(
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
    pub(super) fn apply_batched(
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
    /// `load_mla_attention` folds in from `mscale_all_dim`, which upstream
    /// applies as well. `None` means "use `x` as is".
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
    pub(super) fn freqs(&self) -> Option<&MlxArray> {
        self.kind.freqs()
    }

    /// The magnitude multiplier applied before the rotation.
    #[cfg(test)]
    pub(super) fn mscale(&self) -> f32 {
        self.kind.attn_scale()
    }
}
