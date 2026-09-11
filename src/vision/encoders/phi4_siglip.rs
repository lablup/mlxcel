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

//! Phi4-SigLIP vision encoder.
//!
//! Used by: Phi4-SigLIP VLM

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

use crate::vision::encoders::qwen2_vl::concat_many;

#[derive(Debug, Clone, Deserialize)]
pub struct Phi4SigLipVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_num_patches")]
    pub num_patches: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
}

fn default_num_channels() -> usize {
    3
}

fn default_image_size() -> usize {
    512
}

fn default_patch_size() -> usize {
    16
}

fn default_num_patches() -> usize {
    256
}

fn default_layer_norm_eps() -> f32 {
    1e-6
}

struct Phi4SigLipAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    out_proj: UnifiedLinear,
    num_heads: i32,
    scale: f32,
}

impl Phi4SigLipAttention {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        num_heads: usize,
        dims: usize,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let q_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.q_proj", prefix), group_size, bits)?;
        let k_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.k_proj", prefix), group_size, bits)?;
        let v_proj =
            UnifiedLinear::from_weights(weights, &format!("{}.v_proj", prefix), group_size, bits)?;
        let out_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{}.out_proj", prefix),
            group_size,
            bits,
        )?;
        let head_dim = dims / num_heads;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: num_heads as i32,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0];
        let seq_len = shape[1];

        let queries = self.q_proj.forward(x);
        let keys = self.k_proj.forward(x);
        let values = self.v_proj.forward(x);
        let head_dim = mlxcel_core::array_shape(&queries)[2] / self.num_heads;

        let queries = mlxcel_core::reshape(&queries, &[batch, seq_len, self.num_heads, head_dim]);
        let queries = mlxcel_core::transpose_axes(&queries, &[0, 2, 1, 3]);
        let keys = mlxcel_core::reshape(&keys, &[batch, seq_len, self.num_heads, head_dim]);
        let keys = mlxcel_core::transpose_axes(&keys, &[0, 2, 1, 3]);
        let values = mlxcel_core::reshape(&values, &[batch, seq_len, self.num_heads, head_dim]);
        let values = mlxcel_core::transpose_axes(&values, &[0, 2, 1, 3]);

        let output = unsafe {
            mlxcel_core::layers::attention_from_ptr(
                &queries,
                &keys,
                &values,
                self.scale,
                std::ptr::null(),
                0.0,
                0,
            )
        };
        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[batch, seq_len, self.num_heads * head_dim]);
        self.out_proj.forward(&output)
    }
}

struct Phi4SigLipMlp {
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
}

impl Phi4SigLipMlp {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let fc1 =
            UnifiedLinear::from_weights(weights, &format!("{}.fc1", prefix), group_size, bits)?;
        let fc2 =
            UnifiedLinear::from_weights(weights, &format!("{}.fc2", prefix), group_size, bits)?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let x = self.fc1.forward(x);
        let x = mlxcel_core::gelu_approx(&x);
        self.fc2.forward(&x)
    }
}

struct Phi4SigLipEncoderLayer {
    self_attn: Phi4SigLipAttention,
    layer_norm1: LayerNorm,
    mlp: Phi4SigLipMlp,
    layer_norm2: LayerNorm,
}

impl Phi4SigLipEncoderLayer {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Phi4SigLipVisionConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            self_attn: Phi4SigLipAttention::from_weights(
                weights,
                &format!("{}.self_attn", prefix),
                config.num_attention_heads,
                config.hidden_size,
                group_size,
                bits,
            )?,
            layer_norm1: load_layer_norm(
                weights,
                &format!("{}.layer_norm1", prefix),
                config.layer_norm_eps,
            )?,
            mlp: Phi4SigLipMlp::from_weights(
                weights,
                &format!("{}.mlp", prefix),
                group_size,
                bits,
            )?,
            layer_norm2: load_layer_norm(
                weights,
                &format!("{}.layer_norm2", prefix),
                config.layer_norm_eps,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let r = self.self_attn.forward(&self.layer_norm1.forward(x));
        let h = mlxcel_core::add(x, &r);
        let r = self.mlp.forward(&self.layer_norm2.forward(&h));
        mlxcel_core::add(&h, &r)
    }
}

struct Phi4SigLipVisionEmbeddings {
    patch_embedding: UnifiedLinear,
    position_embedding: UniquePtr<MlxArray>, // [num_positions, hidden]
    grid_size: i32,
}

impl Phi4SigLipVisionEmbeddings {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Phi4SigLipVisionConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let patch_embedding = UnifiedLinear::from_weights(
            weights,
            &format!("{}.patch_embedding", prefix),
            group_size,
            bits,
        )?;
        let position_embedding = weights
            .get(&format!("{}.position_embedding.weight", prefix))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {}.position_embedding.weight", prefix))?;
        Ok(Self {
            patch_embedding,
            position_embedding,
            grid_size: (config.num_patches as f64).sqrt() as i32,
        })
    }

    fn forward(&self, pixel_values: &MlxArray, spatial_shape: (i32, i32)) -> UniquePtr<MlxArray> {
        let patch_embeddings = self.patch_embedding.forward(pixel_values);
        let seq_len = mlxcel_core::array_shape(&patch_embeddings)[1];
        let positional_embeddings = self.resize_positional_embeddings(spatial_shape, seq_len);
        mlxcel_core::add(&patch_embeddings, &positional_embeddings)
    }

    fn resize_positional_embeddings(
        &self,
        spatial_shape: (i32, i32),
        max_length: i32,
    ) -> UniquePtr<MlxArray> {
        let (height, width) = spatial_shape;
        let grid = self.grid_size;
        let hidden_size = mlxcel_core::array_shape(&self.position_embedding)[1];

        let h_step = if height > 1 {
            (grid - 1) as f32 / (height - 1) as f32
        } else {
            0.0
        };
        let w_step = if width > 1 {
            (grid - 1) as f32 / (width - 1) as f32
        } else {
            0.0
        };

        let total_hw = height * width;
        let mut idx_lists: [Vec<i32>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut weight_lists: [Vec<f32>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

        for hi in 0..height {
            let h_idx = hi as f32 * h_step;
            let h_floor = h_idx.floor() as i32;
            let h_ceil = (h_floor + 1).min(grid - 1);
            let dh = h_idx - h_floor as f32;

            for wi in 0..width {
                let w_idx = wi as f32 * w_step;
                let w_floor = w_idx.floor() as i32;
                let w_ceil = (w_floor + 1).min(grid - 1);
                let dw = w_idx - w_floor as f32;

                idx_lists[0].push(h_floor * grid + w_floor);
                idx_lists[1].push(h_floor * grid + w_ceil);
                idx_lists[2].push(h_ceil * grid + w_floor);
                idx_lists[3].push(h_ceil * grid + w_ceil);

                weight_lists[0].push((1.0 - dh) * (1.0 - dw));
                weight_lists[1].push((1.0 - dh) * dw);
                weight_lists[2].push(dh * (1.0 - dw));
                weight_lists[3].push(dh * dw);
            }
        }

        let mut all_idx = Vec::with_capacity((4 * total_hw) as usize);
        let mut all_weights = Vec::with_capacity((4 * total_hw) as usize);
        for i in 0..4 {
            all_idx.extend_from_slice(&idx_lists[i]);
            all_weights.extend_from_slice(&weight_lists[i]);
        }

        let idx = mlxcel_core::from_slice_i32(&all_idx, &[4, total_hw]);
        let weights = mlxcel_core::from_slice_f32(&all_weights, &[4, total_hw]);
        let weights =
            mlxcel_core::astype(&weights, mlxcel_core::array_dtype(&self.position_embedding));
        let idx_flat = mlxcel_core::flatten(&idx);
        let embeds = mlxcel_core::take(&self.position_embedding, &idx_flat, 0);
        let embeds = mlxcel_core::reshape(&embeds, &[4, total_hw, hidden_size]);
        let weights = mlxcel_core::reshape(&weights, &[4, total_hw, 1]);
        let weighted = mlxcel_core::multiply(&embeds, &weights);

        let c0 = mlxcel_core::slice(&weighted, &[0, 0, 0], &[1, total_hw, hidden_size]);
        let c1 = mlxcel_core::slice(&weighted, &[1, 0, 0], &[2, total_hw, hidden_size]);
        let c2 = mlxcel_core::slice(&weighted, &[2, 0, 0], &[3, total_hw, hidden_size]);
        let c3 = mlxcel_core::slice(&weighted, &[3, 0, 0], &[4, total_hw, hidden_size]);
        let c0 = mlxcel_core::squeeze_axis(&c0, 0);
        let c1 = mlxcel_core::squeeze_axis(&c1, 0);
        let c2 = mlxcel_core::squeeze_axis(&c2, 0);
        let c3 = mlxcel_core::squeeze_axis(&c3, 0);
        let sum01 = mlxcel_core::add(&c0, &c1);
        let sum23 = mlxcel_core::add(&c2, &c3);
        let resized = mlxcel_core::add(&sum01, &sum23);

        let resized = if total_hw < max_length {
            let first = mlxcel_core::slice(&resized, &[0, 0], &[1, hidden_size]);
            let padding = mlxcel_core::broadcast_to(&first, &[max_length - total_hw, hidden_size]);
            mlxcel_core::concatenate(&resized, &padding, 0)
        } else {
            resized
        };

        mlxcel_core::reshape(&resized, &[1, max_length, hidden_size])
    }
}

pub struct Phi4SigLipVisionEncoder {
    embeddings: Phi4SigLipVisionEmbeddings,
    layers: Vec<Phi4SigLipEncoderLayer>,
}

impl Phi4SigLipVisionEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        config: &Phi4SigLipVisionConfig,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let embeddings = Phi4SigLipVisionEmbeddings::from_weights(
            weights,
            &format!("{}.embeddings", prefix),
            config,
            group_size,
            bits,
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            layers.push(Phi4SigLipEncoderLayer::from_weights(
                weights,
                &format!("{}.encoder.layers.{}", prefix, layer_idx),
                config,
                group_size,
                bits,
            )?);
        }

        Ok(Self { embeddings, layers })
    }

    pub fn forward_hidden_states(
        &self,
        pixel_values: &MlxArray,
        spatial_shape: (i32, i32),
    ) -> Vec<UniquePtr<MlxArray>> {
        let mut outputs = Vec::with_capacity(self.layers.len() + 1);
        let mut hidden_states = self.embeddings.forward(pixel_values, spatial_shape);
        outputs.push(mlxcel_core::copy(&hidden_states));
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states);
            outputs.push(mlxcel_core::copy(&hidden_states));
        }
        outputs
    }
}

fn load_layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    let weight_key = format!("{}.weight", prefix);
    let bias_key = format!("{}.bias", prefix);
    let weight = weights
        .get(&weight_key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", weight_key))?;
    let bias = weights
        .get(&bias_key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", bias_key))?;
    Ok(LayerNorm::new(weight, Some(bias), eps))
}

pub(crate) fn concat_arrays(arrays: &[UniquePtr<MlxArray>], axis: i32) -> UniquePtr<MlxArray> {
    concat_many(arrays, axis)
}
