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

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::attention::SamAttention;

const LAYER_NORM_EPS: f32 = 1e-5;
const LAYER_NORM_2D_EPS: f32 = 1e-6;

pub(crate) fn load_array(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|value| mlxcel_core::copy(value))
        .ok_or_else(|| format!("SAM3 tracker weight missing: {name}"))
}

pub(crate) fn load_layer_norm(weights: &WeightMap, prefix: &str) -> Result<LayerNorm, String> {
    Ok(LayerNorm::new(
        load_array(weights, &format!("{prefix}.weight"))?,
        Some(load_array(weights, &format!("{prefix}.bias"))?),
        LAYER_NORM_EPS,
    ))
}

/// Channels-last convolution loaded from an already-sanitized checkpoint.
pub(crate) struct Conv2dLayer {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
    stride: i32,
    padding: i32,
    groups: i32,
}

impl Conv2dLayer {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        stride: i32,
        padding: i32,
        groups: i32,
        bias: bool,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: load_array(weights, &format!("{prefix}.weight"))?,
            bias: if bias {
                Some(load_array(weights, &format!("{prefix}.bias"))?)
            } else {
                None
            },
            stride,
            padding,
            groups,
        })
    }

    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let y = mlxcel_core::conv2d(
            x,
            &self.weight,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            1,
            1,
            self.groups,
        );
        match &self.bias {
            Some(bias) => mlxcel_core::add(&y, bias),
            None => y,
        }
    }
}

/// Layer normalization over the channel axis of an NHWC tensor.
pub struct LayerNorm2d {
    norm: LayerNorm,
}

impl LayerNorm2d {
    pub(crate) fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            norm: LayerNorm::new(
                load_array(weights, &format!("{prefix}.weight"))?,
                Some(load_array(weights, &format!("{prefix}.bias"))?),
                LAYER_NORM_2D_EPS,
            ),
        })
    }

    #[must_use]
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.norm.forward(x)
    }
}

pub(crate) struct MlpBlock {
    proj_in: UnifiedLinear,
    proj_out: UnifiedLinear,
}

impl MlpBlock {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            proj_in: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.proj_in"),
                group_size,
                bits,
            )?,
            proj_out: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.proj_out"),
                group_size,
                bits,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.proj_out
            .forward(&mlxcel_core::relu(&self.proj_in.forward(x)))
    }
}

pub(crate) struct TwoWayAttentionBlock {
    skip_first_layer_pe: bool,
    self_attn: SamAttention,
    layer_norm1: LayerNorm,
    cross_attn_token_to_image: SamAttention,
    layer_norm2: LayerNorm,
    mlp: MlpBlock,
    layer_norm3: LayerNorm,
    cross_attn_image_to_token: SamAttention,
    layer_norm4: LayerNorm,
}

impl TwoWayAttentionBlock {
    #[allow(clippy::too_many_arguments)]
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        hidden_size: i32,
        num_heads: i32,
        attention_downsample_rate: i32,
        skip_first_layer_pe: bool,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            skip_first_layer_pe,
            self_attn: SamAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                hidden_size,
                num_heads,
                1,
                group_size,
                bits,
            )?,
            layer_norm1: load_layer_norm(weights, &format!("{prefix}.layer_norm1"))?,
            cross_attn_token_to_image: SamAttention::from_weights(
                weights,
                &format!("{prefix}.cross_attn_token_to_image"),
                hidden_size,
                num_heads,
                attention_downsample_rate,
                group_size,
                bits,
            )?,
            layer_norm2: load_layer_norm(weights, &format!("{prefix}.layer_norm2"))?,
            mlp: MlpBlock::from_weights(weights, &format!("{prefix}.mlp"), group_size, bits)?,
            layer_norm3: load_layer_norm(weights, &format!("{prefix}.layer_norm3"))?,
            cross_attn_image_to_token: SamAttention::from_weights(
                weights,
                &format!("{prefix}.cross_attn_image_to_token"),
                hidden_size,
                num_heads,
                attention_downsample_rate,
                group_size,
                bits,
            )?,
            layer_norm4: load_layer_norm(weights, &format!("{prefix}.layer_norm4"))?,
        })
    }

    fn forward(
        &self,
        queries: &MlxArray,
        keys: &MlxArray,
        query_pe: &MlxArray,
        key_pe: &MlxArray,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let queries = if self.skip_first_layer_pe {
            self.self_attn.forward(queries, queries, queries)
        } else {
            let q = mlxcel_core::add(queries, query_pe);
            mlxcel_core::add(queries, &self.self_attn.forward(&q, &q, queries))
        };
        let queries = self.layer_norm1.forward(&queries);
        let q = mlxcel_core::add(&queries, query_pe);
        let k = mlxcel_core::add(keys, key_pe);
        let queries = mlxcel_core::add(
            &queries,
            &self.cross_attn_token_to_image.forward(&q, &k, keys),
        );
        let queries = self.layer_norm2.forward(&queries);
        let queries = mlxcel_core::add(&queries, &self.mlp.forward(&queries));
        let queries = self.layer_norm3.forward(&queries);
        let q = mlxcel_core::add(keys, key_pe);
        let k = mlxcel_core::add(&queries, query_pe);
        let keys = mlxcel_core::add(
            keys,
            &self.cross_attn_image_to_token.forward(&q, &k, &queries),
        );
        (queries, self.layer_norm4.forward(&keys))
    }
}

pub(crate) struct TwoWayTransformer {
    layers: Vec<TwoWayAttentionBlock>,
    final_attn_token_to_image: SamAttention,
    layer_norm_final_attn: LayerNorm,
}

impl TwoWayTransformer {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        hidden_size: i32,
        num_heads: i32,
        num_layers: usize,
        attention_downsample_rate: i32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(num_layers);
        for index in 0..num_layers {
            layers.push(TwoWayAttentionBlock::from_weights(
                weights,
                &format!("{prefix}.layers.{index}"),
                hidden_size,
                num_heads,
                attention_downsample_rate,
                index == 0,
                group_size,
                bits,
            )?);
        }
        Ok(Self {
            layers,
            final_attn_token_to_image: SamAttention::from_weights(
                weights,
                &format!("{prefix}.final_attn_token_to_image"),
                hidden_size,
                num_heads,
                attention_downsample_rate,
                group_size,
                bits,
            )?,
            layer_norm_final_attn: load_layer_norm(
                weights,
                &format!("{prefix}.layer_norm_final_attn"),
            )?,
        })
    }

    pub(crate) fn forward(
        &self,
        image_embedding: &MlxArray,
        image_pe: &MlxArray,
        point_embedding: &MlxArray,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let mut queries = mlxcel_core::copy(point_embedding);
        let mut keys = mlxcel_core::copy(image_embedding);
        for layer in &self.layers {
            (queries, keys) = layer.forward(&queries, &keys, point_embedding, image_pe);
        }
        let q = mlxcel_core::add(&queries, point_embedding);
        let k = mlxcel_core::add(&keys, image_pe);
        let attended = self.final_attn_token_to_image.forward(&q, &k, &keys);
        let queries = self
            .layer_norm_final_attn
            .forward(&mlxcel_core::add(&queries, &attended));
        (queries, keys)
    }
}
