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
//! SAM3 pixel/mask decoder and text-query dot-product scoring head.

use mlxcel_core::layers::LayerNorm;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::linear::UnifiedLinear;

use super::config::DetectorMaskDecoderConfig;
use super::decoder::{MultiheadAttention, layer_norm};
use crate::vision::dense::interpolate::resize_nearest_nhwc;

fn weight(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|value| mlxcel_core::copy(value))
        .ok_or_else(|| format!("SAM3 weight not found: {key}"))
}

struct Conv2d {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
    padding: i32,
}

impl Conv2d {
    fn from_weights(weights: &WeightMap, prefix: &str, padding: i32) -> Result<Self, String> {
        Ok(Self {
            weight: weight(weights, &format!("{prefix}.weight"))?,
            bias: weights
                .get(&format!("{prefix}.bias"))
                .map(|value| mlxcel_core::copy(value)),
            padding,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let y = mlxcel_core::conv2d(x, &self.weight, 1, 1, self.padding, self.padding, 1, 1, 1);
        match &self.bias {
            Some(bias) => mlxcel_core::add(&y, bias),
            None => y,
        }
    }
}

/// PyTorch-compatible GroupNorm over channels-last feature maps.
struct GroupNorm {
    weight: UniquePtr<MlxArray>,
    bias: UniquePtr<MlxArray>,
    groups: i32,
    eps: f32,
}

impl GroupNorm {
    fn from_weights(weights: &WeightMap, prefix: &str, groups: i32) -> Result<Self, String> {
        Ok(Self {
            weight: weight(weights, &format!("{prefix}.weight"))?,
            bias: weight(weights, &format!("{prefix}.bias"))?,
            groups,
            eps: 1e-5,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let dtype = mlxcel_core::array_dtype(x);
        let x = mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32);
        let s = mlxcel_core::array_shape(&x);
        let (b, channels) = (s[0], s[s.len() - 1]);
        let spatial: i32 = s[1..s.len() - 1].iter().product();
        let group_width = channels / self.groups;
        let x = mlxcel_core::reshape(&x, &[b, spatial, self.groups, group_width]);
        let x = mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]);
        let x = mlxcel_core::reshape(&x, &[b, self.groups, spatial * group_width]);
        // SAFETY: null affine pointers request normalization only; affine is
        // applied after restoring NHWC so checkpoint vectors broadcast by C.
        let x = unsafe {
            mlxcel_core::fast_layer_norm(&x, std::ptr::null(), std::ptr::null(), self.eps)
        };
        let x = mlxcel_core::reshape(&x, &[b, self.groups, spatial, group_width]);
        let x = mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]);
        let x = mlxcel_core::reshape(&x, &s);
        let x = mlxcel_core::add(&mlxcel_core::multiply(&x, &self.weight), &self.bias);
        mlxcel_core::astype(&x, dtype)
    }
}

struct PixelDecoder {
    convolutions: Vec<Conv2d>,
    norms: Vec<GroupNorm>,
}

impl PixelDecoder {
    fn from_weights(weights: &WeightMap, prefix: &str, stages: usize) -> Result<Self, String> {
        let mut convolutions = Vec::with_capacity(stages);
        let mut norms = Vec::with_capacity(stages);
        for i in 0..stages {
            convolutions.push(Conv2d::from_weights(
                weights,
                &format!("{prefix}.conv_layers.{i}"),
                1,
            )?);
            norms.push(GroupNorm::from_weights(
                weights,
                &format!("{prefix}.norms.{i}"),
                8,
            )?);
        }
        Ok(Self {
            convolutions,
            norms,
        })
    }

    fn forward(&self, features: &[UniquePtr<MlxArray>]) -> Result<UniquePtr<MlxArray>, String> {
        let Some(coarsest) = features.last() else {
            return Err("SAM3 pixel decoder requires at least one feature level".into());
        };
        if features.len().saturating_sub(1) > self.convolutions.len() {
            return Err(format!(
                "SAM3 pixel decoder received {} feature transitions but has {} stages",
                features.len().saturating_sub(1),
                self.convolutions.len()
            ));
        }
        let mut x = mlxcel_core::copy(coarsest);
        for (i, skip) in features[..features.len() - 1].iter().rev().enumerate() {
            let s = mlxcel_core::array_shape(skip);
            x = resize_nearest_nhwc(&x, (s[1] as usize, s[2] as usize))
                .map_err(|error| error.to_string())?;
            x = mlxcel_core::add(&x, skip);
            x = self.convolutions[i].forward(&x);
            x = mlxcel_core::relu(&self.norms[i].forward(&x));
        }
        Ok(x)
    }
}

struct MaskEmbedder {
    layers: Vec<UnifiedLinear>,
}

impl MaskEmbedder {
    fn from_weights(weights: &WeightMap, prefix: &str, gs: i32, bits: i32) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(3);
        for i in 0..3 {
            layers.push(UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.layers.{i}"),
                gs,
                bits,
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mut y = mlxcel_core::copy(x);
        for (index, layer) in self.layers.iter().enumerate() {
            y = layer.forward(&y);
            if index + 1 != self.layers.len() {
                y = mlxcel_core::relu(&y);
            }
        }
        y
    }
}

pub struct MaskDecoderOutput {
    pub pred_masks: UniquePtr<MlxArray>,
    pub semantic_segmentation: UniquePtr<MlxArray>,
}

pub struct MaskDecoder {
    pixels: PixelDecoder,
    mask_embedder: MaskEmbedder,
    prompt_attn: MultiheadAttention,
    prompt_norm: LayerNorm,
    semantic: Conv2d,
    instance: Conv2d,
}

impl MaskDecoder {
    pub fn from_weights(
        weights: &WeightMap,
        cfg: &DetectorMaskDecoderConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            pixels: PixelDecoder::from_weights(
                weights,
                &format!("{prefix}.pixel_decoder"),
                cfg.num_upsampling_stages,
            )?,
            mask_embedder: MaskEmbedder::from_weights(
                weights,
                &format!("{prefix}.mask_embedder"),
                gs,
                bits,
            )?,
            prompt_attn: MultiheadAttention::from_weights(
                weights,
                &format!("{prefix}.prompt_cross_attn"),
                cfg.hidden_size,
                cfg.num_attention_heads,
                gs,
                bits,
            )?,
            prompt_norm: layer_norm(
                weights,
                &format!("{prefix}.prompt_cross_attn_norm"),
                cfg.layer_norm_eps,
            )?,
            semantic: Conv2d::from_weights(weights, &format!("{prefix}.semantic_projection"), 0)?,
            instance: Conv2d::from_weights(weights, &format!("{prefix}.instance_projection"), 0)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        object_queries: &MlxArray,
        backbone_features: &[UniquePtr<MlxArray>],
        encoder_hidden: Option<&MlxArray>,
        prompt_features: Option<&MlxArray>,
        prompt_mask: Option<&MlxArray>,
    ) -> Result<MaskDecoderOutput, String> {
        let encoder = match (encoder_hidden, prompt_features) {
            (Some(hidden), Some(prompt)) => {
                let normed = self.prompt_norm.forward(hidden);
                let mask = prompt_mask.map(|mask| {
                    let mask = mlxcel_core::astype(mask, mlxcel_core::dtype::FLOAT32);
                    let mask = mlxcel_core::subtract(&mlxcel_core::full_like(&mask, 1.0), &mask);
                    let mask = mlxcel_core::multiply_scalar(&mask, -1e9);
                    mlxcel_core::expand_dims_multi(&mask, &[1, 2])
                });
                let attended = self
                    .prompt_attn
                    .forward(&normed, prompt, prompt, mask.as_deref());
                Some(mlxcel_core::add(hidden, &attended))
            }
            (Some(hidden), None) => Some(mlxcel_core::copy(hidden)),
            _ => None,
        };

        let mut features: Vec<UniquePtr<MlxArray>> = backbone_features
            .iter()
            .map(|feature| mlxcel_core::copy(feature))
            .collect();
        if let (Some(hidden), Some(coarsest)) = (encoder.as_deref(), features.last_mut()) {
            let s = mlxcel_core::array_shape(coarsest);
            let spatial = s[1] * s[2];
            let hidden = mlxcel_core::slice(hidden, &[0, 0, 0], &[s[0], spatial, s[3]]);
            *coarsest = mlxcel_core::reshape(&hidden, &[s[0], s[1], s[2], s[3]]);
        }
        let pixels = self.pixels.forward(&features)?;
        let instance = self.instance.forward(&pixels);
        let mask_embeddings = self.mask_embedder.forward(object_queries);
        let s = mlxcel_core::array_shape(&instance);
        let flat = mlxcel_core::reshape(&instance, &[s[0], s[1] * s[2], s[3]]);
        let flat = mlxcel_core::transpose_axes(&flat, &[0, 2, 1]);
        let masks = mlxcel_core::matmul(&mask_embeddings, &flat);
        let masks = mlxcel_core::reshape(&masks, &[s[0], -1, s[1], s[2]]);
        let semantic = self.semantic.forward(&pixels);
        Ok(MaskDecoderOutput {
            pred_masks: masks,
            semantic_segmentation: mlxcel_core::transpose_axes(&semantic, &[0, 3, 1, 2]),
        })
    }
}

pub struct DotProductScoring {
    query_proj: UnifiedLinear,
    text_proj: UnifiedLinear,
    text_mlp1: UnifiedLinear,
    text_mlp2: UnifiedLinear,
    text_norm: LayerNorm,
    scale: f32,
}

impl DotProductScoring {
    pub fn from_weights(
        weights: &WeightMap,
        hidden_size: usize,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            query_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.query_proj"),
                gs,
                bits,
            )?,
            text_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.text_proj"),
                gs,
                bits,
            )?,
            text_mlp1: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.text_mlp.layer1"),
                gs,
                bits,
            )?,
            text_mlp2: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.text_mlp.layer2"),
                gs,
                bits,
            )?,
            text_norm: layer_norm(weights, &format!("{prefix}.text_mlp_out_norm"), 1e-5)?,
            scale: 1.0 / (hidden_size as f32).sqrt(),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &MlxArray,
        text: &MlxArray,
        text_mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let processed = self
            .text_mlp2
            .forward(&mlxcel_core::relu(&self.text_mlp1.forward(text)));
        let processed = self.text_norm.forward(&mlxcel_core::add(text, &processed));
        let pooled = if let Some(mask) = text_mask {
            let valid = mlxcel_core::expand_dims(
                &mlxcel_core::astype(mask, mlxcel_core::dtype::FLOAT32),
                -1,
            );
            let count = mlxcel_core::sum_axis(&valid, 1, false);
            let count = mlxcel_core::maximum(&count, &mlxcel_core::full_like(&count, 1.0));
            let total = mlxcel_core::sum_axis(&mlxcel_core::multiply(&processed, &valid), 1, false);
            mlxcel_core::divide(&total, &count)
        } else {
            mlxcel_core::mean_axis(&processed, 1, false)
        };
        let text = self.text_proj.forward(&pooled);
        let text = mlxcel_core::expand_dims_multi(&text, &[0, 3]);
        let queries = self.query_proj.forward(hidden_states);
        let scores =
            mlxcel_core::multiply_scalar(&mlxcel_core::matmul(&queries, &text), self.scale);
        mlxcel_core::clip(
            &scores,
            &mlxcel_core::full_like(&scores, -12.0),
            &mlxcel_core::full_like(&scores, 12.0),
        )
    }
}
