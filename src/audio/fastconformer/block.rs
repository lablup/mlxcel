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

//! Conformer block: macaron feed-forwards, rel-pos attention and the causal
//! convolution module.
//!
//! Port of `FeedForward`, `ConformerConvolution` and `ConformerBlock` in
//! `mlx_audio/stt/models/nemotron_asr/conformer.py`:
//!
//! ```text
//! x = x + 0.5 * ff1(LN(x))
//! x = x + attn(LN(x), pos_emb, mask)
//! x = x + conv(LN(x))
//! x = x + 0.5 * ff2(LN(x))
//! x = LN_out(x)
//! ```
//!
//! The convolution module is `pointwise_conv1` (d -> 2d, k1) -> GLU ->
//! time padding `(left, right)` (`(k - 1, 0)` for `"causal"`) -> depthwise
//! conv (groups d) -> LayerNorm (stored under `batch_norm`, NeMo's name) ->
//! SiLU -> `pointwise_conv2`.

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::attention::RelPositionMultiHeadAttention;
use super::config::ConformerArgs;
use super::layers::{LayerNorm, conv1d_weight, glu_last_axis, maybe_copy_weight};

struct FeedForward {
    linear1: UnifiedLinear,
    linear2: UnifiedLinear,
}

impl FeedForward {
    fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            linear1: UnifiedLinear::from_weights(weights, &format!("{prefix}.linear1"), 64, 4)?,
            linear2: UnifiedLinear::from_weights(weights, &format!("{prefix}.linear2"), 64, 4)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.linear2
            .forward(&mlxcel_core::silu(&self.linear1.forward(x)))
    }
}

struct Conv1d {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
    groups: i32,
}

impl Conv1d {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        in_per_group: i32,
        groups: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: conv1d_weight(weights, &format!("{prefix}.weight"), in_per_group)?,
            bias: maybe_copy_weight(weights, &format!("{prefix}.bias")),
            groups,
        })
    }

    fn forward(&self, x: &MlxArray, what: &str) -> Result<UniquePtr<MlxArray>, String> {
        let y = mlxcel_core::try_conv1d(x, &self.weight, 1, 0, 1, self.groups)
            .map_err(|e| format!("FastConformer {what} conv1d failed: {e}"))?;
        Ok(match &self.bias {
            Some(b) => mlxcel_core::add(&y, b),
            None => y,
        })
    }
}

struct ConformerConvolution {
    pointwise_conv1: Conv1d,
    depthwise_conv: Conv1d,
    norm: LayerNorm,
    pointwise_conv2: Conv1d,
    pad: (i32, i32),
}

impl ConformerConvolution {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        args: &ConformerArgs,
    ) -> Result<Self, String> {
        let d = args.d_model as i32;
        let (left, right) = args.conv_padding()?;
        Ok(Self {
            pointwise_conv1: Conv1d::from_weights(
                weights,
                &format!("{prefix}.pointwise_conv1"),
                d,
                1,
            )?,
            depthwise_conv: Conv1d::from_weights(
                weights,
                &format!("{prefix}.depthwise_conv"),
                1,
                d,
            )?,
            norm: LayerNorm::from_weights(weights, &format!("{prefix}.batch_norm"))?,
            pointwise_conv2: Conv1d::from_weights(
                weights,
                &format!("{prefix}.pointwise_conv2"),
                d,
                1,
            )?,
            pad: (left as i32, right as i32),
        })
    }

    fn forward(&self, x: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let h = self.pointwise_conv1.forward(x, "pointwise1")?;
        let h = glu_last_axis(&h);
        let h = mlxcel_core::pad(&h, &[0, 0, self.pad.0, self.pad.1, 0, 0], 0.0);
        let h = self.depthwise_conv.forward(&h, "depthwise")?;
        let h = mlxcel_core::silu(&self.norm.forward(&h));
        self.pointwise_conv2.forward(&h, "pointwise2")
    }
}

pub struct ConformerBlock {
    norm_feed_forward1: LayerNorm,
    feed_forward1: FeedForward,
    norm_self_att: LayerNorm,
    self_attn: RelPositionMultiHeadAttention,
    norm_conv: LayerNorm,
    conv: ConformerConvolution,
    norm_feed_forward2: LayerNorm,
    feed_forward2: FeedForward,
    norm_out: LayerNorm,
}

impl ConformerBlock {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        args: &ConformerArgs,
    ) -> Result<Self, String> {
        let norm = |name: &str| LayerNorm::from_weights(weights, &format!("{prefix}.{name}"));
        Ok(Self {
            norm_feed_forward1: norm("norm_feed_forward1")?,
            feed_forward1: FeedForward::from_weights(weights, &format!("{prefix}.feed_forward1"))?,
            norm_self_att: norm("norm_self_att")?,
            self_attn: RelPositionMultiHeadAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                args.n_heads,
                args.d_model,
            )?,
            norm_conv: norm("norm_conv")?,
            conv: ConformerConvolution::from_weights(weights, &format!("{prefix}.conv"), args)?,
            norm_feed_forward2: norm("norm_feed_forward2")?,
            feed_forward2: FeedForward::from_weights(weights, &format!("{prefix}.feed_forward2"))?,
            norm_out: norm("norm_out")?,
        })
    }

    /// `x: [B, T, d]` to `[B, T, d]`.
    pub fn forward(
        &self,
        x: &MlxArray,
        pos_emb: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let half = |h: UniquePtr<MlxArray>| mlxcel_core::multiply_scalar(&h, 0.5);
        let ff1 = self
            .feed_forward1
            .forward(&self.norm_feed_forward1.forward(x));
        let x = mlxcel_core::add(x, &half(ff1));
        let attn = self
            .self_attn
            .forward(&self.norm_self_att.forward(&x), pos_emb, mask);
        let x = mlxcel_core::add(&x, &attn);
        let conv = self.conv.forward(&self.norm_conv.forward(&x))?;
        let x = mlxcel_core::add(&x, &conv);
        let ff2 = self
            .feed_forward2
            .forward(&self.norm_feed_forward2.forward(&x));
        let x = mlxcel_core::add(&x, &half(ff2));
        Ok(self.norm_out.forward(&x))
    }
}
