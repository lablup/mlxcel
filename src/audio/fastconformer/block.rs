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

    /// Cache-aware causal step: prepend the cached last `conv_left` GLU
    /// frames (zeros on the first call) instead of zero padding, run the
    /// depthwise conv unpadded, and return the new cache tail.
    fn stream(
        &self,
        x: &MlxArray,
        cache: Option<&MlxArray>,
        conv_left: i32,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>), String> {
        let h = self.pointwise_conv1.forward(x, "pointwise1")?;
        let g = glu_last_axis(&h);
        let g_shape = mlxcel_core::array_shape(&g);
        let zeros;
        let cache = match cache {
            Some(c) => c,
            None => {
                zeros = mlxcel_core::zeros(
                    &[g_shape[0], conv_left, g_shape[2]],
                    mlxcel_core::array_dtype(&g),
                );
                &*zeros
            }
        };
        let din = mlxcel_core::concatenate(cache, &g, 1);
        let len = mlxcel_core::array_shape(&din)[1];
        let next = mlxcel_core::slice(
            &din,
            &[0, len - conv_left, 0],
            &[g_shape[0], len, g_shape[2]],
        );
        let h = self.depthwise_conv.forward(&din, "depthwise")?;
        let h = mlxcel_core::silu(&self.norm.forward(&h));
        Ok((self.pointwise_conv2.forward(&h, "pointwise2")?, next))
    }
}

/// Output of [`ConformerBlock::stream`].
pub struct BlockStreamOutput {
    pub output: UniquePtr<MlxArray>,
    /// Last `left_cache` attention-input frames (`None` when `left_cache == 0`).
    pub attn_cache: Option<UniquePtr<MlxArray>>,
    /// Last `conv_left` GLU-output frames.
    pub conv_cache: UniquePtr<MlxArray>,
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

    /// Cache-aware step (`_stream_block` in the reference `streaming.py`):
    /// the new frames `x: [B, c, d]` attend to `attn_cache ++ LN(x)` with
    /// `pos_emb` built for that window length, and the causal conv continues
    /// from `conv_cache`. The causal convolution module is required.
    pub fn stream(
        &self,
        x: &MlxArray,
        pos_emb: &MlxArray,
        attn_cache: Option<&MlxArray>,
        conv_cache: Option<&MlxArray>,
        left_cache: usize,
        conv_left: usize,
    ) -> Result<BlockStreamOutput, String> {
        if self.conv.pad != (conv_left as i32, 0) {
            return Err(format!(
                "streaming needs a causal conv module (pad ({conv_left}, 0)), got {:?}",
                self.conv.pad
            ));
        }
        let half = |h: UniquePtr<MlxArray>| mlxcel_core::multiply_scalar(&h, 0.5);
        let ff1 = self
            .feed_forward1
            .forward(&self.norm_feed_forward1.forward(x));
        let residual = mlxcel_core::add(x, &half(ff1));

        let xn = self.norm_self_att.forward(&residual);
        let kv = match attn_cache {
            Some(cache) => mlxcel_core::concatenate(cache, &xn, 1),
            None => mlxcel_core::copy(&xn),
        };
        let attn = self.self_attn.stream(&xn, &kv, pos_emb)?;
        let residual = mlxcel_core::add(&residual, &attn);
        let kv_shape = mlxcel_core::array_shape(&kv);
        let attn_next = (left_cache > 0).then(|| {
            let start = (kv_shape[1] - left_cache as i32).max(0);
            mlxcel_core::slice(&kv, &[0, start, 0], &kv_shape)
        });

        let (conv, conv_next) = self.conv.stream(
            &self.norm_conv.forward(&residual),
            conv_cache,
            conv_left as i32,
        )?;
        let residual = mlxcel_core::add(&residual, &conv);
        let ff2 = self
            .feed_forward2
            .forward(&self.norm_feed_forward2.forward(&residual));
        let residual = mlxcel_core::add(&residual, &half(ff2));
        Ok(BlockStreamOutput {
            output: self.norm_out.forward(&residual),
            attn_cache: attn_next,
            conv_cache: conv_next,
        })
    }
}
