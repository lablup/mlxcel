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

use mlxcel_core::layers::LayerNorm;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::super::linear::UnifiedLinear;

use super::super::config::TrackerConfig;
use super::super::sam_components::{
    Conv2dLayer, LayerNorm2d, RopeAttention, load_array, load_layer_norm,
};

struct DownsampleConvBlock {
    conv: Conv2dLayer,
    layer_norm: LayerNorm2d,
}

impl DownsampleConvBlock {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        stride: i32,
        padding: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            conv: Conv2dLayer::from_weights(
                weights,
                &format!("{prefix}.conv"),
                stride,
                padding,
                1,
                true,
            )?,
            layer_norm: LayerNorm2d::from_weights(weights, &format!("{prefix}.layer_norm"))?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        super::super::activation::gelu(&self.layer_norm.forward(&self.conv.forward(x)))
    }
}

struct SimpleMaskDownsampler {
    layers: Vec<DownsampleConvBlock>,
    final_conv: Conv2dLayer,
}

impl SimpleMaskDownsampler {
    fn from_weights(
        weights: &WeightMap,
        config: &TrackerConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(4);
        for index in 0..4 {
            layers.push(DownsampleConvBlock::from_weights(
                weights,
                &format!("{prefix}.layers.{index}"),
                config.mask_downsampler_stride as i32,
                config.mask_downsampler_padding as i32,
            )?);
        }
        Ok(Self {
            layers,
            final_conv: Conv2dLayer::from_weights(
                weights,
                &format!("{prefix}.final_conv"),
                1,
                0,
                1,
                true,
            )?,
        })
    }

    fn forward(&self, masks: &MlxArray) -> UniquePtr<MlxArray> {
        let mut x = mlxcel_core::copy(masks);
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        self.final_conv.forward(&x)
    }
}

struct CxBlock {
    depthwise_conv: Conv2dLayer,
    layer_norm: LayerNorm2d,
    pointwise_conv1: UnifiedLinear,
    pointwise_conv2: UnifiedLinear,
    scale: UniquePtr<MlxArray>,
}

impl CxBlock {
    fn from_weights(
        weights: &WeightMap,
        config: &TrackerConfig,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let dim = config.memory_fuser_embed_dim as i32;
        Ok(Self {
            depthwise_conv: Conv2dLayer::from_weights(
                weights,
                &format!("{prefix}.depthwise_conv"),
                1,
                config.memory_fuser_padding as i32,
                dim,
                true,
            )?,
            layer_norm: LayerNorm2d::from_weights(weights, &format!("{prefix}.layer_norm"))?,
            pointwise_conv1: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.pointwise_conv1"),
                group_size,
                bits,
            )?,
            pointwise_conv2: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.pointwise_conv2"),
                group_size,
                bits,
            )?,
            scale: load_array(weights, &format!("{prefix}.scale"))?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let residual = mlxcel_core::copy(x);
        let hidden = self.depthwise_conv.forward(x);
        let hidden = self.layer_norm.forward(&hidden);
        let shape = mlxcel_core::array_shape(&hidden);
        let hidden = mlxcel_core::reshape(&hidden, &[shape[0] * shape[1] * shape[2], shape[3]]);
        let hidden = self.pointwise_conv1.forward(&hidden);
        let hidden = self
            .pointwise_conv2
            .forward(&super::super::activation::gelu(&hidden));
        let hidden = mlxcel_core::reshape(&hidden, &shape);
        mlxcel_core::add(&residual, &mlxcel_core::multiply(&hidden, &self.scale))
    }
}

struct MemoryFuser {
    layers: Vec<CxBlock>,
}

impl MemoryFuser {
    fn from_weights(
        weights: &WeightMap,
        config: &TrackerConfig,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(config.memory_fuser_num_layers);
        for index in 0..config.memory_fuser_num_layers {
            layers.push(CxBlock::from_weights(
                weights,
                config,
                &format!("{prefix}.layers.{index}"),
                group_size,
                bits,
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mut x = mlxcel_core::copy(x);
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }
}

/// Encodes current-frame features and mask logits into compact spatial memory.
pub struct MemoryEncoder {
    mask_downsampler: SimpleMaskDownsampler,
    memory_fuser: MemoryFuser,
    feature_projection: Conv2dLayer,
    projection: Conv2dLayer,
}

impl MemoryEncoder {
    pub(super) fn from_weights(
        weights: &WeightMap,
        config: &TrackerConfig,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            mask_downsampler: SimpleMaskDownsampler::from_weights(
                weights,
                config,
                &format!("{prefix}.mask_downsampler"),
            )?,
            memory_fuser: MemoryFuser::from_weights(
                weights,
                config,
                &format!("{prefix}.memory_fuser"),
                group_size,
                bits,
            )?,
            feature_projection: Conv2dLayer::from_weights(
                weights,
                &format!("{prefix}.feature_projection"),
                1,
                0,
                1,
                true,
            )?,
            projection: Conv2dLayer::from_weights(
                weights,
                &format!("{prefix}.projection"),
                1,
                0,
                1,
                true,
            )?,
        })
    }

    pub fn forward(
        &self,
        features: &MlxArray,
        masks: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let mask_features = self.mask_downsampler.forward(masks);
        let features = self.feature_projection.forward(features);
        let feature_shape = mlxcel_core::array_shape(&features);
        let mask_shape = mlxcel_core::array_shape(&mask_features);
        if feature_shape != mask_shape {
            return Err(format!(
                "SAM3 memory feature/mask shape mismatch: {feature_shape:?} versus {mask_shape:?}"
            ));
        }
        let fused = self
            .memory_fuser
            .forward(&mlxcel_core::add(&features, &mask_features));
        Ok(self.projection.forward(&fused))
    }
}

struct MemoryAttentionLayer {
    self_attn: RopeAttention,
    cross_attn_image: RopeAttention,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
    layer_norm3: LayerNorm,
    linear1: UnifiedLinear,
    linear2: UnifiedLinear,
}

impl MemoryAttentionLayer {
    fn from_weights(
        weights: &WeightMap,
        config: &TrackerConfig,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let d = config.memory_attention_hidden_size as i32;
        let feat_sizes = config.memory_attention_rope_feat_sizes.as_slice();
        if feat_sizes.len() != 2 {
            return Err(format!(
                "SAM3 memory RoPE needs two feature sizes, got {feat_sizes:?}"
            ));
        }
        let rope_args = (feat_sizes[0] as i32, feat_sizes[1] as i32);
        Ok(Self {
            self_attn: RopeAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                d,
                config.memory_attention_num_attention_heads as i32,
                config.memory_attention_downsample_rate as i32,
                d,
                rope_args,
                config.memory_attention_rope_theta,
                false,
                group_size,
                bits,
            )?,
            cross_attn_image: RopeAttention::from_weights(
                weights,
                &format!("{prefix}.cross_attn_image"),
                d,
                config.memory_attention_num_attention_heads as i32,
                config.memory_attention_downsample_rate as i32,
                config.memory_encoder_output_channels as i32,
                rope_args,
                config.memory_attention_rope_theta,
                true,
                group_size,
                bits,
            )?,
            layer_norm1: load_layer_norm(weights, &format!("{prefix}.layer_norm1"))?,
            layer_norm2: load_layer_norm(weights, &format!("{prefix}.layer_norm2"))?,
            layer_norm3: load_layer_norm(weights, &format!("{prefix}.layer_norm3"))?,
            linear1: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.linear1"),
                group_size,
                bits,
            )?,
            linear2: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.linear2"),
                group_size,
                bits,
            )?,
        })
    }

    fn forward(&self, src: &MlxArray, memory: &MlxArray) -> UniquePtr<MlxArray> {
        let attended = self.self_attn.forward(src, src, src);
        let src = self.layer_norm1.forward(&mlxcel_core::add(src, &attended));
        let attended = self.cross_attn_image.forward(&src, memory, memory);
        let src = self.layer_norm2.forward(&mlxcel_core::add(&src, &attended));
        let hidden = self.linear1.forward(&src);
        let hidden = self.linear2.forward(&mlxcel_core::relu(&hidden));
        self.layer_norm3.forward(&mlxcel_core::add(&src, &hidden))
    }
}

/// Transformer layers that condition current features on the memory bank.
pub struct MemoryAttention {
    layers: Vec<MemoryAttentionLayer>,
    layer_norm: LayerNorm,
}

impl MemoryAttention {
    pub(super) fn from_weights(
        weights: &WeightMap,
        config: &TrackerConfig,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(config.memory_attention_num_layers);
        for index in 0..config.memory_attention_num_layers {
            layers.push(MemoryAttentionLayer::from_weights(
                weights,
                config,
                &format!("{prefix}.layers.{index}"),
                group_size,
                bits,
            )?);
        }
        Ok(Self {
            layers,
            layer_norm: load_layer_norm(weights, &format!("{prefix}.layer_norm"))?,
        })
    }

    pub fn forward(&self, src: &MlxArray, memory: &MlxArray) -> UniquePtr<MlxArray> {
        let mut src = mlxcel_core::copy(src);
        for layer in &self.layers {
            src = layer.forward(&src, memory);
        }
        self.layer_norm.forward(&src)
    }
}

pub(super) fn nearest_resize_nhwc(
    x: &MlxArray,
    target_height: i32,
    target_width: i32,
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    if shape[1] == target_height && shape[2] == target_width {
        return mlxcel_core::copy(x);
    }
    let rows: Vec<i32> = (0..target_height)
        .map(|row| row * shape[1] / target_height)
        .collect();
    let columns: Vec<i32> = (0..target_width)
        .map(|column| column * shape[2] / target_width)
        .collect();
    let rows = mlxcel_core::from_slice_i32(&rows, &[target_height]);
    let columns = mlxcel_core::from_slice_i32(&columns, &[target_width]);
    let x = mlxcel_core::take(x, &rows, 1);
    mlxcel_core::take(&x, &columns, 2)
}
