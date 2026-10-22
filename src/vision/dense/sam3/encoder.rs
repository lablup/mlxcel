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

//! SAM 3 DETR encoder with text cross-attention.

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::DetrEncoderConfig;

/// Multi-head attention with the checkpoint's separate q/k/v/o projections.
pub struct MultiheadAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl MultiheadAttention {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        hidden_size: usize,
        num_heads: usize,
        quantization: (i32, i32),
    ) -> Result<Self, String> {
        if num_heads == 0 || !hidden_size.is_multiple_of(num_heads) {
            return Err(format!(
                "{prefix}: hidden size {hidden_size} is not divisible by {num_heads} heads"
            ));
        }
        let (group_size, bits) = quantization;
        let linear = |name: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{name}"), group_size, bits)
        };
        let head_dim = (hidden_size / num_heads) as i32;
        Ok(Self {
            q_proj: linear("q_proj")?,
            k_proj: linear("k_proj")?,
            v_proj: linear("v_proj")?,
            o_proj: linear("o_proj")?,
            num_heads: num_heads as i32,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// `query`, `key`, and `value` are batch-first sequences.
    #[must_use]
    pub fn forward(
        &self,
        query: &MlxArray,
        key: &MlxArray,
        value: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let query_shape = mlxcel_core::array_shape(query);
        let key_shape = mlxcel_core::array_shape(key);
        let (batch, query_len, key_len) = (query_shape[0], query_shape[1], key_shape[1]);
        let q = split_heads(
            self.q_proj.forward(query),
            batch,
            query_len,
            self.num_heads,
            self.head_dim,
        );
        let k = split_heads(
            self.k_proj.forward(key),
            batch,
            key_len,
            self.num_heads,
            self.head_dim,
        );
        let v = split_heads(
            self.v_proj.forward(value),
            batch,
            key_len,
            self.num_heads,
            self.head_dim,
        );
        let mask_ptr = mask.map_or(std::ptr::null(), |m| m as *const MlxArray);
        // SAFETY: q/k/v and the optional borrowed mask remain alive for this call.
        let attended = unsafe {
            mlxcel_core::fast_scaled_dot_product_attention(&q, &k, &v, self.scale, mask_ptr)
        };
        let attended = mlxcel_core::transpose_axes(&attended, &[0, 2, 1, 3]);
        let attended = mlxcel_core::reshape(
            &attended,
            &[batch, query_len, self.num_heads * self.head_dim],
        );
        self.o_proj.forward(&attended)
    }
}

fn split_heads(
    x: UniquePtr<MlxArray>,
    batch: i32,
    length: i32,
    heads: i32,
    head_dim: i32,
) -> UniquePtr<MlxArray> {
    let x = mlxcel_core::reshape(&x, &[batch, length, heads, head_dim]);
    mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3])
}

/// Two-layer feed-forward network used by DETR and geometry blocks.
pub struct Mlp {
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
    activation: Activation,
}

#[derive(Debug, Clone, Copy)]
enum Activation {
    Relu,
    Gelu,
}

impl Mlp {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        activation: &str,
        quantization: (i32, i32),
    ) -> Result<Self, String> {
        let activation = match activation {
            "relu" => Activation::Relu,
            "gelu" => Activation::Gelu,
            other => return Err(format!("{prefix}: unsupported activation {other}")),
        };
        let (group_size, bits) = quantization;
        Ok(Self {
            fc1: UnifiedLinear::from_weights(weights, &format!("{prefix}.fc1"), group_size, bits)?,
            fc2: UnifiedLinear::from_weights(weights, &format!("{prefix}.fc2"), group_size, bits)?,
            activation,
        })
    }

    #[must_use]
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let hidden = self.fc1.forward(x);
        let hidden = match self.activation {
            Activation::Relu => mlxcel_core::relu(&hidden),
            Activation::Gelu => mlxcel_core::gelu(&hidden),
        };
        self.fc2.forward(&hidden)
    }
}

struct DetrEncoderLayer {
    self_attn: MultiheadAttention,
    cross_attn: MultiheadAttention,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
    layer_norm3: LayerNorm,
    mlp: Mlp,
}

impl DetrEncoderLayer {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &DetrEncoderConfig,
        quantization: (i32, i32),
    ) -> Result<Self, String> {
        Ok(Self {
            self_attn: MultiheadAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                config.hidden_size,
                config.num_attention_heads,
                quantization,
            )?,
            cross_attn: MultiheadAttention::from_weights(
                weights,
                &format!("{prefix}.cross_attn"),
                config.hidden_size,
                config.num_attention_heads,
                quantization,
            )?,
            layer_norm1: load_layer_norm(
                weights,
                &format!("{prefix}.layer_norm1"),
                config.layer_norm_eps,
            )?,
            layer_norm2: load_layer_norm(
                weights,
                &format!("{prefix}.layer_norm2"),
                config.layer_norm_eps,
            )?,
            layer_norm3: load_layer_norm(
                weights,
                &format!("{prefix}.layer_norm3"),
                config.layer_norm_eps,
            )?,
            mlp: Mlp::from_weights(
                weights,
                &format!("{prefix}.mlp"),
                &config.hidden_act,
                quantization,
            )?,
        })
    }

    fn forward(
        &self,
        src: &MlxArray,
        pos: &MlxArray,
        text_memory: &MlxArray,
        text_mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let hidden = self.layer_norm1.forward(src);
        let hidden_with_pos = mlxcel_core::add(&hidden, pos);
        let attended = self
            .self_attn
            .forward(&hidden_with_pos, &hidden_with_pos, &hidden, None);
        let src = mlxcel_core::add(src, &attended);

        let cross_mask = text_mask.map(|mask| padding_mask(mask, mlxcel_core::array_dtype(&src)));
        let hidden = self.layer_norm2.forward(&src);
        let attended =
            self.cross_attn
                .forward(&hidden, text_memory, text_memory, cross_mask.as_deref());
        let src = mlxcel_core::add(&src, &attended);

        let hidden = self.layer_norm3.forward(&src);
        mlxcel_core::add(&src, &self.mlp.forward(&hidden))
    }
}

/// Six-layer pre-norm DETR image encoder with per-layer text fusion.
pub struct DetrEncoder {
    layers: Vec<DetrEncoderLayer>,
}

impl DetrEncoder {
    /// `prefix` is normally `detector_model.detr_encoder`.
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &DetrEncoderConfig,
        quantization: (i32, i32),
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(config.num_layers);
        for index in 0..config.num_layers {
            layers.push(DetrEncoderLayer::from_weights(
                weights,
                &format!("{prefix}.layers.{index}"),
                config,
                quantization,
            )?);
        }
        Ok(Self { layers })
    }

    /// Encode flattened image features `(B, HW, D)` against `(B, T, D)` text.
    #[must_use]
    pub fn forward(
        &self,
        src: &MlxArray,
        pos: &MlxArray,
        text_memory: &MlxArray,
        text_mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut output = mlxcel_core::copy(src);
        for layer in &self.layers {
            output = layer.forward(&output, pos, text_memory, text_mask);
            mlxcel_core::eval(&output);
        }
        output
    }
}

pub(crate) fn load_layer_norm(
    weights: &WeightMap,
    prefix: &str,
    eps: f32,
) -> Result<LayerNorm, String> {
    let weight_name = format!("{prefix}.weight");
    let weight = weights
        .get(&weight_name)
        .map(|value| mlxcel_core::copy(value))
        .ok_or_else(|| format!("Weight not found: {weight_name}"))?;
    let bias = weights
        .get(&format!("{prefix}.bias"))
        .map(|value| mlxcel_core::copy(value));
    Ok(LayerNorm::new(weight, bias, eps))
}

fn padding_mask(mask: &MlxArray, dtype: i32) -> UniquePtr<MlxArray> {
    let valid = mlxcel_core::astype(mask, dtype);
    let one = mlxcel_core::full_f32(&[1], 1.0, dtype);
    let blocked = mlxcel_core::multiply_scalar(&mlxcel_core::subtract(&one, &valid), -1e9);
    mlxcel_core::expand_dims_multi(&blocked, &[1, 2])
}
