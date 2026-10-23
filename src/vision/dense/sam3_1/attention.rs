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

//! Decoupled SAM 3.1 memory attention.

use mlxcel_core::layers::LayerNorm;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::super::sam3::linear::UnifiedLinear;
use super::super::sam3::position::{apply_rotary_enc_1d, init_2d_freqs};
use super::super::sam3::sam_components::load_layer_norm;
use super::config::TrackerConfig;

struct RopeAttention {
    heads: i32,
    head_dim: i32,
    scale: f32,
    repeat_keys: bool,
    cos: UniquePtr<MlxArray>,
    sin: UniquePtr<MlxArray>,
}

impl RopeAttention {
    fn new(config: &TrackerConfig, repeat_keys: bool) -> Self {
        let hidden = config.memory_attention_hidden_size as i32;
        let heads = config.memory_attention_num_attention_heads as i32;
        let head_dim = hidden / heads;
        let (cos, sin) = init_2d_freqs(
            head_dim as usize,
            config.memory_attention_rope_feat_sizes[0],
            config.memory_attention_rope_feat_sizes[1],
            config.memory_attention_rope_theta,
        );
        Self {
            heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            repeat_keys,
            cos,
            sin,
        }
    }

    fn forward(
        &self,
        query: &MlxArray,
        key: &MlxArray,
        value: &MlxArray,
        exclude_key_tokens: i32,
    ) -> UniquePtr<MlxArray> {
        let q_shape = mlxcel_core::array_shape(query);
        let k_shape = mlxcel_core::array_shape(key);
        let q = mlxcel_core::reshape(query, &[q_shape[0], q_shape[1], self.heads, self.head_dim]);
        let k = mlxcel_core::reshape(key, &[k_shape[0], k_shape[1], self.heads, self.head_dim]);
        let v = mlxcel_core::reshape(value, &[k_shape[0], k_shape[1], self.heads, self.head_dim]);
        let rope_key_len = k_shape[1] - exclude_key_tokens;
        let k_rope = mlxcel_core::slice(
            &k,
            &[0, 0, 0, 0],
            &[k_shape[0], rope_key_len, self.heads, self.head_dim],
        );
        let (q, k_rope) = apply_rotary_enc_1d(&q, &k_rope, &self.cos, &self.sin, self.repeat_keys);
        let k = if exclude_key_tokens > 0 {
            let tail = mlxcel_core::slice(
                &k,
                &[0, rope_key_len, 0, 0],
                &[k_shape[0], k_shape[1], self.heads, self.head_dim],
            );
            mlxcel_core::concatenate(&k_rope, &tail, 1)
        } else {
            k_rope
        };
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);
        // SAFETY: null selects unmasked attention and all arrays outlive the call.
        let output = unsafe {
            mlxcel_core::fast_scaled_dot_product_attention(&q, &k, &v, self.scale, std::ptr::null())
        };
        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        mlxcel_core::reshape(
            &output,
            &[q_shape[0], q_shape[1], self.heads * self.head_dim],
        )
    }
}

struct MemoryAttentionLayer {
    self_q: UnifiedLinear,
    self_k: UnifiedLinear,
    self_v: UnifiedLinear,
    self_out: UnifiedLinear,
    cross_q: UnifiedLinear,
    cross_k: UnifiedLinear,
    cross_v: UnifiedLinear,
    cross_out: UnifiedLinear,
    image_q: UnifiedLinear,
    image_k: UnifiedLinear,
    linear1: UnifiedLinear,
    linear2: UnifiedLinear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    self_rope: RopeAttention,
    cross_rope: RopeAttention,
}

impl MemoryAttentionLayer {
    fn from_weights(
        weights: &WeightMap,
        config: &TrackerConfig,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let linear = |name: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{name}"), group_size, bits)
        };
        Ok(Self {
            self_q: linear("self_attn_q_proj")?,
            self_k: linear("self_attn_k_proj")?,
            self_v: linear("self_attn_v_proj")?,
            self_out: linear("self_attn_out_proj")?,
            cross_q: linear("cross_attn_q_proj")?,
            cross_k: linear("cross_attn_k_proj")?,
            cross_v: linear("cross_attn_v_proj")?,
            cross_out: linear("cross_attn_out_proj")?,
            image_q: linear("image_cross_attn_q_proj")?,
            image_k: linear("image_cross_attn_k_proj")?,
            linear1: linear("linear1")?,
            linear2: linear("linear2")?,
            norm1: load_layer_norm(weights, &format!("{prefix}.norm1"))?,
            norm2: load_layer_norm(weights, &format!("{prefix}.norm2"))?,
            norm3: load_layer_norm(weights, &format!("{prefix}.norm3"))?,
            self_rope: RopeAttention::new(config, false),
            cross_rope: RopeAttention::new(config, true),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        image: &MlxArray,
        source: &MlxArray,
        memory_image: &MlxArray,
        memory: &MlxArray,
        memory_image_position: Option<&MlxArray>,
        exclude_key_tokens: i32,
    ) -> UniquePtr<MlxArray> {
        let normalized = self.norm1.forward(source);
        let attended = self.self_rope.forward(
            &self.self_q.forward(&normalized),
            &self.self_k.forward(&normalized),
            &self.self_v.forward(&normalized),
            0,
        );
        let source = mlxcel_core::add(source, &self.self_out.forward(&attended));

        let normalized = self.norm2.forward(&source);
        let query = mlxcel_core::add(
            &self.image_q.forward(image),
            &self.cross_q.forward(&normalized),
        );
        let mut key = mlxcel_core::add(
            &self.image_k.forward(memory_image),
            &self.cross_k.forward(memory),
        );
        if let Some(position) = memory_image_position {
            key = mlxcel_core::add(&key, position);
        }
        let attended = self.cross_rope.forward(
            &query,
            &key,
            &self.cross_v.forward(memory),
            exclude_key_tokens,
        );
        let source = mlxcel_core::add(&source, &self.cross_out.forward(&attended));

        let normalized = self.norm3.forward(&source);
        let feed_forward = self.linear2.forward(&super::super::sam3::activation::gelu(
            &self.linear1.forward(&normalized),
        ));
        let output = mlxcel_core::add(&source, &feed_forward);
        mlxcel_core::eval(&output);
        output
    }
}

pub(crate) struct MemoryAttention {
    layers: Vec<MemoryAttentionLayer>,
    norm: LayerNorm,
}

impl MemoryAttention {
    pub(crate) fn from_weights(
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
            norm: load_layer_norm(weights, &format!("{prefix}.layer_norm"))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        image: &MlxArray,
        source: &MlxArray,
        memory_image: &MlxArray,
        memory: &MlxArray,
        source_position: Option<&MlxArray>,
        memory_position: Option<&MlxArray>,
        memory_image_position: Option<&MlxArray>,
        exclude_key_tokens: i32,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let mut source = source_position.map_or_else(
            || mlxcel_core::copy(source),
            |position| mlxcel_core::add(source, &mlxcel_core::multiply_scalar(position, 0.1)),
        );
        let memory_shape = mlxcel_core::array_shape(memory);
        let image_shape = mlxcel_core::array_shape(memory_image);
        let mut memory_image = mlxcel_core::copy(memory_image);
        let mut image_position = memory_image_position.map(mlxcel_core::copy);
        if image_shape[1] != memory_shape[1] {
            if image_shape[1] > memory_shape[1] {
                return Err("SAM 3.1 image memory is longer than mask memory".into());
            }
            let padding = memory_shape[1] - image_shape[1];
            if padding != exclude_key_tokens {
                return Err(format!(
                    "SAM 3.1 memory/image length gap {padding} does not match {exclude_key_tokens} pointer tokens"
                ));
            }
            memory_image = mlxcel_core::concatenate(
                &memory_image,
                &mlxcel_core::zeros(
                    &[image_shape[0], padding, image_shape[2]],
                    mlxcel_core::array_dtype(&memory_image),
                ),
                1,
            );
            if let Some(position) = image_position.as_ref() {
                let memory_position = memory_position.ok_or_else(|| {
                    "SAM 3.1 pointer positions need mask-memory positions".to_owned()
                })?;
                let tail = mlxcel_core::slice(
                    memory_position,
                    &[0, memory_shape[1] - padding, 0],
                    &[1, memory_shape[1], memory_shape[2]],
                );
                image_position = Some(mlxcel_core::concatenate(position, &tail, 1));
            }
        }
        for layer in &self.layers {
            source = layer.forward(
                image,
                &source,
                &memory_image,
                memory,
                image_position.as_deref(),
                exclude_key_tokens,
            );
        }
        Ok(self.norm.forward(&source))
    }
}
