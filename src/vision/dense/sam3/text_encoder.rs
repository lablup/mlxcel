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

use mlxcel_core::layers::{LayerNorm, UnifiedEmbedding};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::linear::UnifiedLinear;

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
        self.fc2
            .forward(&super::activation::gelu(&self.fc1.forward(x)))
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
        let component = |name: &str| {
            if detector_prefix.is_empty() {
                name.to_owned()
            } else {
                format!("{detector_prefix}.{name}")
            }
        };
        let text_prefix = component("text_encoder");
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
                &component("text_projection"),
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

    #[cfg(test)]
    pub(crate) fn detector_projection_dtype(&self) -> Option<i32> {
        self.detector_projection.dense_weight_dtype()
    }

    #[cfg(test)]
    pub(crate) fn diagnostic_stages(
        &self,
        input_ids: &MlxArray,
        attention_mask: &MlxArray,
    ) -> Vec<(String, UniquePtr<MlxArray>)> {
        let input_shape = mlxcel_core::array_shape(input_ids);
        let length = input_shape[1];
        let token = self.token_embedding.forward(input_ids);
        let position_ids = mlxcel_core::arange_i32(0, length, 1);
        let position = self.position_embedding.forward(&position_ids);
        let mut hidden = mlxcel_core::add(&token, &position);
        let mask = causal_padding_mask(
            length,
            Some(attention_mask),
            mlxcel_core::array_dtype(&hidden),
        );
        let mut stages = vec![
            ("embeddings_output".into(), mlxcel_core::copy(&hidden)),
            ("layer_0_mask".into(), mlxcel_core::copy(&mask)),
        ];

        for (index, layer) in self.layers.iter().enumerate() {
            if index != 0 {
                hidden = layer.forward(&hidden, &mask);
                stages.push((
                    format!("layer_{index}_residual2"),
                    mlxcel_core::copy(&hidden),
                ));
                continue;
            }

            let normalized = layer.layer_norm1.forward(&hidden);
            stages.push(("layer_0_ln1_output".into(), mlxcel_core::copy(&normalized)));
            let shape = mlxcel_core::array_shape(&normalized);
            let (batch, sequence) = (shape[0], shape[1]);
            let project = |linear: &UnifiedLinear| {
                let raw = linear.forward(&normalized);
                let split = mlxcel_core::reshape(
                    &raw,
                    &[
                        batch,
                        sequence,
                        layer.self_attn.num_heads,
                        layer.self_attn.head_dim,
                    ],
                );
                (raw, mlxcel_core::transpose_axes(&split, &[0, 2, 1, 3]))
            };
            let (q_raw, q) = project(&layer.self_attn.q_proj);
            let (k_raw, k) = project(&layer.self_attn.k_proj);
            let (v_raw, v) = project(&layer.self_attn.v_proj);
            stages.push(("layer_0_q_output".into(), q_raw));
            stages.push(("layer_0_k_output".into(), k_raw));
            stages.push(("layer_0_v_output".into(), v_raw));
            // SAFETY: q/k/v/mask remain alive for the duration of the FFI call.
            let attended = unsafe {
                mlxcel_core::fast_scaled_dot_product_attention(
                    &q,
                    &k,
                    &v,
                    layer.self_attn.scale,
                    &*mask as *const MlxArray,
                )
            };
            let attended = mlxcel_core::transpose_axes(&attended, &[0, 2, 1, 3]);
            let attended = mlxcel_core::reshape(
                &attended,
                &[
                    batch,
                    sequence,
                    layer.self_attn.num_heads * layer.self_attn.head_dim,
                ],
            );
            stages.push((
                "layer_0_attention_projection_input_0".into(),
                mlxcel_core::copy(&attended),
            ));
            let projected = layer.self_attn.out_proj.forward(&attended);
            stages.push((
                "layer_0_attention_projection_output".into(),
                mlxcel_core::copy(&projected),
            ));
            hidden = mlxcel_core::add(&hidden, &projected);
            stages.push(("layer_0_residual1".into(), mlxcel_core::copy(&hidden)));
            let normalized = layer.layer_norm2.forward(&hidden);
            stages.push(("layer_0_ln2_output".into(), mlxcel_core::copy(&normalized)));
            let fc1 = layer.mlp.fc1.forward(&normalized);
            stages.push(("layer_0_fc1_output".into(), mlxcel_core::copy(&fc1)));
            let activated = super::activation::gelu(&fc1);
            stages.push(("layer_0_gelu".into(), mlxcel_core::copy(&activated)));
            let fc2 = layer.mlp.fc2.forward(&activated);
            stages.push(("layer_0_fc2_output".into(), mlxcel_core::copy(&fc2)));
            hidden = mlxcel_core::add(&hidden, &fc2);
            stages.push(("layer_0_residual2".into(), mlxcel_core::copy(&hidden)));
        }
        let final_norm = self.final_layer_norm.forward(&hidden);
        stages.push(("final_norm_output".into(), mlxcel_core::copy(&final_norm)));
        stages.push((
            "detector_projection_output".into(),
            self.detector_projection.forward(&final_norm),
        ));
        stages
    }
}

fn causal_padding_mask(
    length: i32,
    attention_mask: Option<&MlxArray>,
    dtype: i32,
) -> UniquePtr<MlxArray> {
    // Pinned mlx-vlm creates an f32 0/-inf causal mask, casts it to the
    // activation dtype, then adds a -1e9 padding-key bias in that dtype. Keep
    // that operation order: in bf16 the padding sentinel rounds to -998244352
    // while the causal sentinel remains -inf. In f16, -1e9 becomes -inf and
    // `0 * -inf` would create NaNs, so the load-time f16 compatibility mode
    // uses the shared safe 0/-inf mask builder instead.
    if dtype == mlxcel_core::dtype::FLOAT16 {
        let mask = attention_mask.map_or_else(
            || mlxcel_core::utils::create_causal_mask(length, 0),
            |padding| mlxcel_core::utils::create_causal_padding_mask(padding, 0),
        );
        return mlxcel_core::astype(&mask, dtype);
    }

    let causal = mlxcel_core::utils::create_causal_mask(length, 0);
    let causal = mlxcel_core::astype(&causal, dtype);
    let Some(attention_mask) = attention_mask else {
        return causal;
    };
    let shape = mlxcel_core::array_shape(attention_mask);
    let valid = mlxcel_core::astype(attention_mask, dtype);
    let padding = mlxcel_core::subtract(&mlxcel_core::ones_like(&valid), &valid);
    let padding = mlxcel_core::multiply_scalar(&padding, -1e9);
    let padding = mlxcel_core::reshape(&padding, &[shape[0], 1, 1, length]);
    mlxcel_core::add(&causal, &padding)
}
