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

//! Shared Llama4 helper functions for RoPE, mask construction and weight
//! loading.
//!
//! These helpers are reused across the Llama4 attention/cache path and are
//! easier to review and test in isolation than when buried deep in the main
//! model file.

use mlxcel_core::MlxArray;
use mlxcel_core::UniquePtr;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;

use crate::models::llama4::TextArgs;
use crate::models::rope_utils::RopeScalingKind;

/// The rotary embedding of a Llama 4 RoPE layer (#3120).
///
/// Port of the rope upstream builds in `Attention.__init__`
/// ([`mlx_lm/models/llama4.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/llama4.py)):
/// `initialize_rope(head_dim, rope_theta, traditional=True,
/// scaling_config=rope_scaling, max_position_embeddings=...)`. Two things
/// differ from a Llama 3.x rotation and both are load-bearing: the rotation
/// pairs adjacent dimensions (`traditional`), and the checkpoint's
/// `rope_scaling` block (`llama3`, factor 16 on Scout and Maverick) selects a
/// banded frequency table. Before #3120 the attention rotated the two halves of
/// the head with plain `rope_theta` frequencies, which disagreed with mlx-lm on
/// 46% of decided positions.
///
/// The table comes from the shared [`RopeScalingKind::resolve`], so a
/// `--rope-*` runtime override reaches Llama 4 the way it reaches every family
/// on that path.
pub(crate) struct Llama4Rope {
    dims: i32,
    base: f32,
    scale: f32,
    /// Multiplies Q and K before the rotation. `1.0` for every scheme but YaRN.
    mscale: f32,
    freqs: Option<UniquePtr<MlxArray>>,
}

/// Upstream passes `traditional=True` for every Llama 4 rope layer.
const LLAMA4_ROPE_TRADITIONAL: bool = true;

impl Llama4Rope {
    /// Resolve the rotation a checkpoint declares.
    pub(crate) fn from_args(args: &TextArgs) -> Self {
        let kind = RopeScalingKind::resolve(
            args.rope_scaling.as_ref(),
            args.head_dim,
            args.rope_theta,
            Some(args.max_position_embeddings as f32),
            &args.model_type,
        );
        let scale = kind.scale();
        let mscale = kind.attn_scale();
        let freqs = match kind {
            RopeScalingKind::Llama3 { freqs } | RopeScalingKind::Yarn { freqs, .. } => Some(freqs),
            RopeScalingKind::Default | RopeScalingKind::Linear { .. } => None,
        };
        Self {
            dims: args.head_dim as i32,
            base: crate::models::rope_overrides::resolve_base(args.rope_theta),
            scale,
            mscale,
            freqs,
        }
    }

    fn pre_scale(&self, x: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        ((self.mscale - 1.0).abs() > f32::EPSILON)
            .then(|| mlxcel_core::multiply_scalar(x, self.mscale))
    }

    /// Rotate `[batch, heads, seq, head_dim]` starting at position `offset`.
    // Used by: Llama 4 `CxxAttention` (forward_llama4, forward, forward_debug).
    pub(crate) fn apply(&self, x: &MlxArray, offset: i32) -> UniquePtr<MlxArray> {
        let scaled = self.pre_scale(x);
        let x = scaled.as_deref().unwrap_or(x);
        match self.freqs.as_ref() {
            Some(freqs) => mlxcel_core::fast_rope_with_freqs(
                x,
                self.dims,
                LLAMA4_ROPE_TRADITIONAL,
                self.scale,
                offset,
                freqs,
            ),
            None => mlxcel_core::fast_rope(
                x,
                self.dims,
                LLAMA4_ROPE_TRADITIONAL,
                self.base,
                self.scale,
                offset,
            ),
        }
    }

    /// Rotate a `[batch, heads, 1, head_dim]` decode step, row `b` at
    /// position `offsets[b]`.
    // Used by: Llama 4 `CxxAttention::forward_batched_decode_rows`.
    pub(crate) fn apply_batched(&self, x: &MlxArray, offsets: &[i32]) -> UniquePtr<MlxArray> {
        let scaled = self.pre_scale(x);
        let x = scaled.as_deref().unwrap_or(x);
        match self.freqs.as_ref() {
            Some(freqs) => mlxcel_core::fast_rope_batched_with_freqs(
                x,
                self.dims,
                LLAMA4_ROPE_TRADITIONAL,
                self.scale,
                offsets,
                freqs,
            ),
            None => mlxcel_core::fast_rope_batched(
                x,
                self.dims,
                LLAMA4_ROPE_TRADITIONAL,
                self.base,
                self.scale,
                offsets,
            ),
        }
    }
}

/// Build the additive chunked attention mask used by Llama4 chunked caches.
///
/// A one-row prefill no longer builds it (issue #3100, see
/// `ChunkedKVCache::attend`); batched rows, speculative verify blocks and
/// `MLXCEL_PREFILL_CAUSAL_MODE=0` keep it.
pub(crate) fn create_chunked_attention_mask(
    seq_len: i32,
    start_position: i32,
    offset: i32,
    chunk_size: usize,
) -> UniquePtr<MlxArray> {
    let end = offset + seq_len;
    let chunk_size = chunk_size as i32;

    let linds = mlxcel_core::arange_i32(start_position, end, 1);
    let rinds = mlxcel_core::arange_i32(offset, end, 1);
    let rinds = mlxcel_core::reshape(&rinds, &[seq_len, 1]);
    let visible_len = end - start_position;

    let chunk_size_f = mlxcel_core::full_f32(&[1], chunk_size as f32, mlxcel_core::dtype::FLOAT32);
    let linds_f = mlxcel_core::astype(&linds, mlxcel_core::dtype::FLOAT32);
    let rinds_f = mlxcel_core::astype(&rinds, mlxcel_core::dtype::FLOAT32);
    let linds_block = mlxcel_core::floor_divide(&linds_f, &chunk_size_f);
    let rinds_block = mlxcel_core::floor_divide(&rinds_f, &chunk_size_f);
    let linds_block = mlxcel_core::reshape(&linds_block, &[1, visible_len]);

    let block_diff = mlxcel_core::subtract(&rinds_block, &linds_block);
    let block_pos = mlxcel_core::abs(&block_diff);
    let zero_f = mlxcel_core::full_f32(&[1], 0.0, mlxcel_core::dtype::FLOAT32);
    let same_block = mlxcel_core::equal(&block_pos, &zero_f);

    let linds_reshaped = mlxcel_core::reshape(&linds, &[1, visible_len]);
    let rinds_orig = mlxcel_core::arange_i32(offset, end, 1);
    let rinds_reshaped = mlxcel_core::reshape(&rinds_orig, &[seq_len, 1]);
    let causal = mlxcel_core::less_equal(&linds_reshaped, &rinds_reshaped);
    let bool_mask = mlxcel_core::logical_and(&same_block, &causal);

    let zeros = mlxcel_core::zeros(&[seq_len, visible_len], mlxcel_core::dtype::FLOAT32);
    let neg_inf = mlxcel_core::full_f32(&[1], f32::NEG_INFINITY, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::where_cond(&bool_mask, &zeros, &neg_inf)
}

/// Get a deep copy of a named weight from the weight map.
pub(crate) fn get_weight_copy(
    weights: &WeightMap,
    name: &str,
) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|weight| mlxcel_core::copy(weight.as_ref().unwrap()))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

/// Load a possibly quantized linear layer from Llama4 weights.
pub(crate) fn load_quantized_linear(
    weights: &WeightMap,
    prefix: &str,
    args: &TextArgs,
) -> Result<UnifiedLinear, String> {
    UnifiedLinear::from_weights(weights, prefix, args.group_size(), args.bits())
}
