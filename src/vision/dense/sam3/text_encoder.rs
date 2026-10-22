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

//! CLIP text transformer used by the SAM 3 detector.
//!
//! Unlike contrastive CLIP inference, SAM 3 keeps every final normalized token
//! and projects the sequence to the DETR width; it never pools the EOS token.
//!
//! Used by: Sam3, Sam31.

use mlxcel_core::layers::{LayerNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_causal_mask, create_causal_padding_mask};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::TextEncoderConfig;
use super::encoder::load_layer_norm;

struct ClipAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    out_proj: UnifiedLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl ClipAttention {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &TextEncoderConfig,
        quantization: (i32, i32),
    ) -> Result<Self, String> {
        if config.num_attention_heads == 0
            || !config
                .hidden_size
                .is_multiple_of(config.num_attention_heads)
        {
            return Err(format!(
                "{prefix}: hidden size {} is not divisible by {} heads",
                config.hidden_size, config.num_attention_heads
            ));
        }
        let (group_size, bits) = quantization;
        let linear = |name: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{name}"), group_size, bits)
        };
        let head_dim = (config.hidden_size / config.num_attention_heads) as i32;
        Ok(Self {
            q_proj: linear("q_proj")?,
            k_proj: linear("k_proj")?,
            v_proj: linear("v_proj")?,
            out_proj: linear("out_proj")?,
            num_heads: config.num_attention_heads as i32,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(&self, x: &MlxArray, mask: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (batch, length) = (shape[0], shape[1]);
        let project = |linear: &UnifiedLinear| {
            let projected = linear.forward(x);
            let projected =
                mlxcel_core::reshape(&projected, &[batch, length, self.num_heads, self.head_dim]);
            mlxcel_core::transpose_axes(&projected, &[0, 2, 1, 3])
        };
        let q = project(&self.q_proj);
        let k = project(&self.k_proj);
        let v = project(&self.v_proj);
        // SAFETY: q/k/v/mask remain alive for the duration of the FFI call.
        let attended = unsafe {
            mlxcel_core::fast_scaled_dot_product_attention(
                &q,
                &k,
                &v,
                self.scale,
                mask as *const MlxArray,
            )
        };
        let attended = mlxcel_core::transpose_axes(&attended, &[0, 2, 1, 3]);
        let attended =
            mlxcel_core::reshape(&attended, &[batch, length, self.num_heads * self.head_dim]);
        self.out_proj.forward(&attended)
    }
}

struct ClipMlp {
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
}

impl ClipMlp {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        quantization: (i32, i32),
    ) -> Result<Self, String> {
        let (group_size, bits) = quantization;
        Ok(Self {
            fc1: UnifiedLinear::from_weights(weights, &format!("{prefix}.fc1"), group_size, bits)?,
            fc2: UnifiedLinear::from_weights(weights, &format!("{prefix}.fc2"), group_size, bits)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.fc2.forward(&mlxcel_core::gelu(&self.fc1.forward(x)))
    }
}

struct ClipEncoderLayer {
    self_attn: ClipAttention,
    layer_norm1: LayerNorm,
    mlp: ClipMlp,
    layer_norm2: LayerNorm,
}

impl ClipEncoderLayer {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &TextEncoderConfig,
        quantization: (i32, i32),
    ) -> Result<Self, String> {
        Ok(Self {
            self_attn: ClipAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                config,
                quantization,
            )?,
            layer_norm1: load_layer_norm(
                weights,
                &format!("{prefix}.layer_norm1"),
                config.layer_norm_eps,
            )?,
            mlp: ClipMlp::from_weights(weights, &format!("{prefix}.mlp"), quantization)?,
            layer_norm2: load_layer_norm(
                weights,
                &format!("{prefix}.layer_norm2"),
                config.layer_norm_eps,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray, mask: &MlxArray) -> UniquePtr<MlxArray> {
        let attended = self.self_attn.forward(&self.layer_norm1.forward(x), mask);
        let x = mlxcel_core::add(x, &attended);
        mlxcel_core::add(&x, &self.mlp.forward(&self.layer_norm2.forward(&x)))
    }
}

/// CLIP text model plus both projections present in the SAM 3 checkpoint.
pub struct TextEncoder {
    token_embedding: UnifiedEmbedding,
    position_embedding: UnifiedEmbedding,
    layers: Vec<ClipEncoderLayer>,
    final_layer_norm: LayerNorm,
    clip_projection: UnifiedLinear,
    detector_projection: UnifiedLinear,
    max_positions: usize,
}

impl TextEncoder {
    /// Load the real checkpoint layout rooted at `detector_model`.
    pub fn from_weights(
        weights: &WeightMap,
        detector_prefix: &str,
        config: &TextEncoderConfig,
        quantization: (i32, i32),
    ) -> Result<Self, String> {
        let (group_size, bits) = quantization;
        let text_prefix = format!("{detector_prefix}.text_encoder");
        let model_prefix = format!("{text_prefix}.text_model");
        let embeddings_prefix = format!("{model_prefix}.embeddings");
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            layers.push(ClipEncoderLayer::from_weights(
                weights,
                &format!("{model_prefix}.encoder.layers.{index}"),
                config,
                quantization,
            )?);
        }
        Ok(Self {
            token_embedding: UnifiedEmbedding::from_weights(
                weights,
                &format!("{embeddings_prefix}.token_embedding"),
                group_size,
                bits,
            )?,
            position_embedding: UnifiedEmbedding::from_weights(
                weights,
                &format!("{embeddings_prefix}.position_embedding"),
                group_size,
                bits,
            )?,
            layers,
            final_layer_norm: load_layer_norm(
                weights,
                &format!("{model_prefix}.final_layer_norm"),
                config.layer_norm_eps,
            )?,
            clip_projection: UnifiedLinear::from_weights(
                weights,
                &format!("{text_prefix}.text_projection"),
                group_size,
                bits,
            )?,
            detector_projection: UnifiedLinear::from_weights(
                weights,
                &format!("{detector_prefix}.text_projection"),
                group_size,
                bits,
            )?,
            max_positions: config.max_position_embeddings,
        })
    }

    /// Return final normalized per-token CLIP features `(B, T, hidden_size)`.
    #[must_use]
    pub fn forward(
        &self,
        input_ids: &MlxArray,
        attention_mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let input_shape = mlxcel_core::array_shape(input_ids);
        let length = input_shape[1];
        debug_assert!(length as usize <= self.max_positions);
        let token = self.token_embedding.forward(input_ids);
        let position_ids = mlxcel_core::arange_i32(0, length, 1);
        let position = self.position_embedding.forward(&position_ids);
        let mut hidden = mlxcel_core::add(&token, &position);
        let mask = causal_padding_mask(length, attention_mask, mlxcel_core::array_dtype(&hidden));
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &mask);
        }
        self.final_layer_norm.forward(&hidden)
    }

    /// Map every token from CLIP hidden width to the detector width (256).
    #[must_use]
    pub fn project(&self, hidden_states: &MlxArray) -> UniquePtr<MlxArray> {
        self.detector_projection.forward(hidden_states)
    }

    /// The checkpoint's contrastive CLIP projection (1024 -> 512).
    #[must_use]
    pub fn project_clip(&self, hidden_states: &MlxArray) -> UniquePtr<MlxArray> {
        self.clip_projection.forward(hidden_states)
    }
}

fn causal_padding_mask(
    length: i32,
    attention_mask: Option<&MlxArray>,
    dtype: i32,
) -> UniquePtr<MlxArray> {
    let mask = attention_mask.map_or_else(
        || create_causal_mask(length, 0),
        |padding| create_causal_padding_mask(padding, 0),
    );
    mlxcel_core::astype(&mask, dtype)
}
