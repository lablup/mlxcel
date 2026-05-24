// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! MiniCPM-o vision tower and resampler.
//!
//! The vision stack differs from the generic SigLIP path in two places:
//! - patch position IDs depend on the per-image `tgt_size`
//! - a Perceiver-style resampler turns patch features into fixed LM slots

use mlxcel_core::layers::{LayerNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

use crate::vision::encoders::qwen2_vl::concat_many;

#[derive(Debug, Clone, Deserialize)]
pub struct MiniCPMOVisionConfig {
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
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
}

fn default_num_channels() -> usize {
    3
}

fn default_image_size() -> usize {
    980
}

fn default_patch_size() -> usize {
    14
}

fn default_layer_norm_eps() -> f32 {
    1e-6
}

fn load_layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    let weight = weights
        .get(&format!("{}.weight", prefix))
        .map(|value| mlxcel_core::copy(value))
        .ok_or_else(|| format!("Weight not found: {}.weight", prefix))?;
    let bias = weights
        .get(&format!("{}.bias", prefix))
        .map(|value| mlxcel_core::copy(value))
        .ok_or_else(|| format!("Weight not found: {}.bias", prefix))?;
    Ok(LayerNorm::new(weight, Some(bias), eps))
}

fn current_patch_position_ids(
    num_patches_per_side: usize,
    h_patches: usize,
    w_patches: usize,
) -> Vec<i32> {
    let mut pos_ids = Vec::with_capacity(h_patches * w_patches);
    for h in 0..h_patches {
        let bucket_h = (h * num_patches_per_side) / h_patches.max(1);
        for w in 0..w_patches {
            let bucket_w = (w * num_patches_per_side) / w_patches.max(1);
            pos_ids.push((bucket_h * num_patches_per_side + bucket_w) as i32);
        }
    }
    pos_ids
}

fn get_1d_sincos_components(embed_dim: usize, positions: &[f32]) -> Vec<f32> {
    let half_dim = embed_dim / 2;
    let mut omega = Vec::with_capacity(half_dim);
    for idx in 0..half_dim {
        let normalized = idx as f32 / half_dim as f32;
        omega.push(1.0 / 10000f32.powf(normalized));
    }

    let mut output = Vec::with_capacity(positions.len() * embed_dim);
    for &pos in positions {
        for &freq in &omega {
            output.push((pos * freq).sin());
        }
        for &freq in &omega {
            output.push((pos * freq).cos());
        }
    }
    output
}

fn get_2d_sincos_pos_embed(height: usize, width: usize, embed_dim: usize) -> Vec<f32> {
    let half_dim = embed_dim / 2;
    let mut grid_h = Vec::with_capacity(height * width);
    let mut grid_w = Vec::with_capacity(height * width);
    for row in 0..height {
        for col in 0..width {
            grid_h.push(col as f32);
            grid_w.push(row as f32);
        }
    }

    let emb_h = get_1d_sincos_components(half_dim, &grid_h);
    let emb_w = get_1d_sincos_components(half_dim, &grid_w);
    let mut output = Vec::with_capacity(height * width * embed_dim);
    for idx in 0..(height * width) {
        let start_h = idx * half_dim;
        let start_w = idx * half_dim;
        output.extend_from_slice(&emb_h[start_h..start_h + half_dim]);
        output.extend_from_slice(&emb_w[start_w..start_w + half_dim]);
    }
    output
}

struct MiniCPMOAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    out_proj: UnifiedLinear,
    num_heads: i32,
    scale: f32,
}

impl MiniCPMOAttention {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        num_heads: usize,
        hidden_size: usize,
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

        let head_dim = hidden_size / num_heads;
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

struct MiniCPMOMlp {
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
}

impl MiniCPMOMlp {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            fc1: UnifiedLinear::from_weights(
                weights,
                &format!("{}.fc1", prefix),
                group_size,
                bits,
            )?,
            fc2: UnifiedLinear::from_weights(
                weights,
                &format!("{}.fc2", prefix),
                group_size,
                bits,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let x = self.fc1.forward(x);
        let x = mlxcel_core::gelu_approx(&x);
        self.fc2.forward(&x)
    }
}

struct MiniCPMOEncoderLayer {
    self_attn: MiniCPMOAttention,
    layer_norm1: LayerNorm,
    mlp: MiniCPMOMlp,
    layer_norm2: LayerNorm,
}

impl MiniCPMOEncoderLayer {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &MiniCPMOVisionConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            self_attn: MiniCPMOAttention::from_weights(
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
            mlp: MiniCPMOMlp::from_weights(weights, &format!("{}.mlp", prefix), group_size, bits)?,
            layer_norm2: load_layer_norm(
                weights,
                &format!("{}.layer_norm2", prefix),
                config.layer_norm_eps,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let residual = self.self_attn.forward(&self.layer_norm1.forward(x));
        let hidden = mlxcel_core::add(x, &residual);
        let residual = self.mlp.forward(&self.layer_norm2.forward(&hidden));
        mlxcel_core::add(&hidden, &residual)
    }
}

struct MiniCPMOVisionEmbeddings {
    patch_embedding_weight: UniquePtr<MlxArray>,
    patch_embedding_bias: Option<UniquePtr<MlxArray>>,
    position_embedding: UnifiedEmbedding,
    num_patches_per_side: usize,
    patch_size: usize,
}

impl MiniCPMOVisionEmbeddings {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &MiniCPMOVisionConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let mut patch_embedding_weight = weights
            .get(&format!("{}.patch_embedding.weight", prefix))
            .map(|value| mlxcel_core::copy(value))
            .ok_or_else(|| format!("Weight not found: {}.patch_embedding.weight", prefix))?;
        let patch_embedding_bias = weights
            .get(&format!("{}.patch_embedding.bias", prefix))
            .map(|value| mlxcel_core::copy(value));

        let weight_shape = mlxcel_core::array_shape(&patch_embedding_weight);
        if weight_shape.len() == 4 {
            let (out_ch, dim1, dim2) = (weight_shape[0], weight_shape[1], weight_shape[2]);
            if !(out_ch >= dim1 && out_ch >= dim2 && dim1 == dim2) {
                patch_embedding_weight =
                    mlxcel_core::transpose_axes(&patch_embedding_weight, &[0, 2, 3, 1]);
            }
        }

        Ok(Self {
            patch_embedding_weight,
            patch_embedding_bias,
            position_embedding: UnifiedEmbedding::from_weights(
                weights,
                &format!("{}.position_embedding", prefix),
                group_size,
                bits,
            )?,
            num_patches_per_side: config.image_size / config.patch_size,
            patch_size: config.patch_size,
        })
    }

    fn forward(&self, pixel_values: &MlxArray, spatial_shape: (i32, i32)) -> UniquePtr<MlxArray> {
        let patch_embeddings = if let Some(bias) = &self.patch_embedding_bias {
            let conv = mlxcel_core::conv2d(
                pixel_values,
                &self.patch_embedding_weight,
                self.patch_size as i32,
                self.patch_size as i32,
                0,
                0,
                1,
                1,
                1,
            );
            mlxcel_core::add(&conv, bias)
        } else {
            mlxcel_core::conv2d(
                pixel_values,
                &self.patch_embedding_weight,
                self.patch_size as i32,
                self.patch_size as i32,
                0,
                0,
                1,
                1,
                1,
            )
        };

        let shape = mlxcel_core::array_shape(&patch_embeddings);
        let batch = shape[0];
        let seq_len = shape[1] * shape[2];
        let hidden_size = shape[3];
        let patch_embeddings =
            mlxcel_core::reshape(&patch_embeddings, &[batch, seq_len, hidden_size]);

        let position_ids = current_patch_position_ids(
            self.num_patches_per_side,
            spatial_shape.0 as usize,
            spatial_shape.1 as usize,
        );
        let position_ids = mlxcel_core::from_slice_i32(&position_ids, &[1, seq_len]);
        let position_embeddings = self.position_embedding.forward(&position_ids);
        mlxcel_core::add(&patch_embeddings, &position_embeddings)
    }
}

pub struct MiniCPMOVisionModel {
    embeddings: MiniCPMOVisionEmbeddings,
    layers: Vec<MiniCPMOEncoderLayer>,
    post_layernorm: LayerNorm,
}

impl MiniCPMOVisionModel {
    pub fn from_weights(
        weights: &WeightMap,
        config: &MiniCPMOVisionConfig,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let embeddings = MiniCPMOVisionEmbeddings::from_weights(
            weights,
            &format!("{}.embeddings", prefix),
            config,
            group_size,
            bits,
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            layers.push(MiniCPMOEncoderLayer::from_weights(
                weights,
                &format!("{}.encoder.layers.{}", prefix, layer_idx),
                config,
                group_size,
                bits,
            )?);
        }

        let post_layernorm = load_layer_norm(
            weights,
            &format!("{}.post_layernorm", prefix),
            config.layer_norm_eps,
        )?;

        Ok(Self {
            embeddings,
            layers,
            post_layernorm,
        })
    }

    pub fn forward(
        &self,
        pixel_values: &MlxArray,
        spatial_shape: (i32, i32),
    ) -> UniquePtr<MlxArray> {
        let mut hidden_states = self.embeddings.forward(pixel_values, spatial_shape);
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states);
        }
        self.post_layernorm.forward(&hidden_states)
    }

    // ── Per-layer access used by MiniCPM-V 4.6 ─────────────────
    //
    // MiniCPM-V 4.6 injects a VitMerger after a specific encoder layer, so we
    // expose the patch embedding, individual layer forward, and post_layernorm
    // as separate methods.
    //
    // Used by: MiniCPMO (via full forward above), MiniCPMV46VLM (per-layer)

    /// Run only the patch embedding + positional embedding step.
    /// Returns hidden states with shape `[1, seq_len, hidden_size]`.
    pub fn forward_embeddings(
        &self,
        pixel_values: &MlxArray,
        spatial_shape: (i32, i32),
    ) -> UniquePtr<MlxArray> {
        self.embeddings.forward(pixel_values, spatial_shape)
    }

    /// Run a single encoder layer by index.
    pub fn forward_layer(&self, hidden_states: &MlxArray, layer_idx: usize) -> UniquePtr<MlxArray> {
        self.layers[layer_idx].forward(hidden_states)
    }

    /// Apply the post-layernorm.
    pub fn forward_post_layernorm(&self, hidden_states: &MlxArray) -> UniquePtr<MlxArray> {
        self.post_layernorm.forward(hidden_states)
    }

    /// Total number of encoder layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

struct MiniCPMOCrossAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    out_proj: UnifiedLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl MiniCPMOCrossAttention {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        num_heads: usize,
        hidden_size: usize,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let head_dim = (hidden_size / num_heads) as i32;
        Ok(Self {
            q_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.q_proj", prefix),
                group_size,
                bits,
            )?,
            k_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.k_proj", prefix),
                group_size,
                bits,
            )?,
            v_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.v_proj", prefix),
                group_size,
                bits,
            )?,
            out_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.out_proj", prefix),
                group_size,
                bits,
            )?,
            num_heads: num_heads as i32,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(
        &self,
        queries: &MlxArray,
        keys: &MlxArray,
        values: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let q_shape = mlxcel_core::array_shape(queries);
        let kv_shape = mlxcel_core::array_shape(keys);
        let batch = q_shape[0];
        let q_len = q_shape[1];
        let kv_len = kv_shape[1];
        let hidden_size = q_shape[2];

        let queries = self.q_proj.forward(queries);
        let keys = self.k_proj.forward(keys);
        let values = self.v_proj.forward(values);

        let queries =
            mlxcel_core::reshape(&queries, &[batch, q_len, self.num_heads, self.head_dim]);
        let queries = mlxcel_core::transpose_axes(&queries, &[0, 2, 1, 3]);
        let keys = mlxcel_core::reshape(&keys, &[batch, kv_len, self.num_heads, self.head_dim]);
        let keys = mlxcel_core::transpose_axes(&keys, &[0, 2, 1, 3]);
        let values = mlxcel_core::reshape(&values, &[batch, kv_len, self.num_heads, self.head_dim]);
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
        let output = mlxcel_core::reshape(&output, &[batch, q_len, hidden_size]);
        self.out_proj.forward(&output)
    }
}

pub struct MiniCPMOResampler {
    query: UniquePtr<MlxArray>,
    kv_proj: UnifiedLinear,
    attn: MiniCPMOCrossAttention,
    ln_q: LayerNorm,
    ln_kv: LayerNorm,
    ln_post: LayerNorm,
    proj: UniquePtr<MlxArray>,
    embed_dim: usize,
}

impl MiniCPMOResampler {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        embed_dim: usize,
        num_heads: usize,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            query: weights
                .get(&format!("{}.query", prefix))
                .map(|value| mlxcel_core::copy(value))
                .ok_or_else(|| format!("Weight not found: {}.query", prefix))?,
            kv_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.kv_proj", prefix),
                group_size,
                bits,
            )?,
            attn: MiniCPMOCrossAttention::from_weights(
                weights,
                &format!("{}.attn", prefix),
                num_heads,
                embed_dim,
                group_size,
                bits,
            )?,
            ln_q: load_layer_norm(weights, &format!("{}.ln_q", prefix), 1e-6)?,
            ln_kv: load_layer_norm(weights, &format!("{}.ln_kv", prefix), 1e-6)?,
            ln_post: load_layer_norm(weights, &format!("{}.ln_post", prefix), 1e-6)?,
            proj: weights
                .get(&format!("{}.proj", prefix))
                .map(|value| mlxcel_core::copy(value))
                .ok_or_else(|| format!("Weight not found: {}.proj", prefix))?,
            embed_dim,
        })
    }

    pub fn forward(&self, x: &MlxArray, spatial_shape: (i32, i32)) -> UniquePtr<MlxArray> {
        let batch = mlxcel_core::array_shape(x)[0];
        let query_count = mlxcel_core::array_shape(&self.query)[0];
        let kv = self.kv_proj.forward(x);
        let kv = self.ln_kv.forward(&kv);

        let (height, width) = (
            spatial_shape.0.max(1) as usize,
            spatial_shape.1.max(1) as usize,
        );
        let pos_embed = get_2d_sincos_pos_embed(height, width, self.embed_dim);
        let pos_embed = mlxcel_core::from_slice_f32(
            &pos_embed,
            &[1, (height * width) as i32, self.embed_dim as i32],
        );
        let pos_embed = mlxcel_core::astype(&pos_embed, mlxcel_core::array_dtype(&kv));
        let keys = mlxcel_core::add(&kv, &pos_embed);

        let query = mlxcel_core::reshape(&self.query, &[1, query_count, self.embed_dim as i32]);
        let query = mlxcel_core::broadcast_to(&query, &[batch, query_count, self.embed_dim as i32]);
        let query = self.ln_q.forward(&query);

        let output = self.attn.forward(&query, &keys, &kv);
        let output = self.ln_post.forward(&output);
        mlxcel_core::matmul(&output, &self.proj)
    }
}

pub(crate) fn concat_arrays(arrays: &[UniquePtr<MlxArray>], axis: i32) -> UniquePtr<MlxArray> {
    concat_many(arrays, axis)
}

#[cfg(test)]
#[path = "minicpmo_tests.rs"]
mod tests;
