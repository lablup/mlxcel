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

//! Muse Glimmer vision patch embedder and transformer tower.

use crate::models::muse_glimmer::MuseGlimmerVisionConfig;
use mlxcel_core::layers::{LayerNorm, Linear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::muse_glimmer_layers::{BiasMode, MuseVisionLayer, load_layer_norm, load_linear};
use super::muse_glimmer_layout::{full_cu_seqlens, window_index_plan};
use super::muse_glimmer_pos::{interpolate_position_table, muse_2d_rope};

pub const MUSE_GLIMMER_VISION_TOWER_ROOT: &str = "model.vision_tower";

pub struct MuseGlimmerPatchEmbedder {
    patch_embedding: Linear,
    position_embedding_table: UniquePtr<MlxArray>,
    patch_input_size: usize,
    pos_emb_height: usize,
    pos_emb_width: usize,
}

impl MuseGlimmerPatchEmbedder {
    pub fn from_weights(
        weights: &WeightMap,
        config: &MuseGlimmerVisionConfig,
        root: &str,
    ) -> Result<Self, String> {
        let prefix = format!("{root}.patch_embedder");
        let patch_input_size = patch_input_size(config);
        let patch_embedding = load_linear(
            weights,
            &format!("{prefix}.patch_embedding"),
            BiasMode::Forbidden,
        )?;
        let weight_shape = mlxcel_core::array_shape(&patch_embedding.weight);
        let expected_weight = vec![config.hidden_size as i32, patch_input_size as i32];
        if weight_shape != expected_weight {
            return Err(format!(
                "Muse Glimmer patch embedding weight must be {expected_weight:?}, got {weight_shape:?}"
            ));
        }

        let table_key = format!("{prefix}.position_embedding_table.weight");
        let position_embedding_table = weights
            .get(&table_key)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {table_key}"))?;
        let table_shape = mlxcel_core::array_shape(&position_embedding_table);
        let expected_table = vec![
            (config.pos_emb_height * config.pos_emb_width) as i32,
            config.hidden_size as i32,
        ];
        if table_shape != expected_table {
            return Err(format!(
                "Muse Glimmer position table must be {expected_table:?}, got {table_shape:?}"
            ));
        }

        Ok(Self {
            patch_embedding,
            position_embedding_table,
            patch_input_size,
            pos_emb_height: config.pos_emb_height,
            pos_emb_width: config.pos_emb_width,
        })
    }

    pub fn forward(
        &self,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> Result<UniquePtr<MlxArray>, String> {
        validate_patch_input(pixel_values, grid_thw, self.patch_input_size)?;
        let patch_embeds = self.patch_embedding.forward(pixel_values);
        let pos_embeds = interpolate_position_table(
            &self.position_embedding_table,
            grid_thw,
            self.pos_emb_height,
            self.pos_emb_width,
        )?;
        Ok(mlxcel_core::add(&patch_embeds, &pos_embeds))
    }

    pub fn patch_input_size(&self) -> usize {
        self.patch_input_size
    }
}

pub struct MuseGlimmerVisionTower {
    patch_embedder: MuseGlimmerPatchEmbedder,
    ln_pre: LayerNorm,
    layers: Vec<MuseVisionLayer>,
    ln_post: LayerNorm,
    head_dim: usize,
    rope_theta: f32,
    window_patch_size: i32,
}

impl MuseGlimmerVisionTower {
    pub fn from_weights(
        weights: &WeightMap,
        config: &MuseGlimmerVisionConfig,
    ) -> Result<Self, String> {
        Self::from_weights_with_root(weights, config, MUSE_GLIMMER_VISION_TOWER_ROOT)
    }

    pub fn from_weights_with_root(
        weights: &WeightMap,
        config: &MuseGlimmerVisionConfig,
        root: &str,
    ) -> Result<Self, String> {
        config.validate()?;
        let patch_embedder = MuseGlimmerPatchEmbedder::from_weights(weights, config, root)?;
        let ln_pre = load_layer_norm(weights, &format!("{root}.ln_pre"), config.layer_norm_eps)?;
        let ln_post = load_layer_norm(weights, &format!("{root}.ln_post"), config.layer_norm_eps)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for idx in 0..config.num_hidden_layers {
            layers.push(MuseVisionLayer::from_weights(weights, config, idx, root)?);
        }

        Ok(Self {
            patch_embedder,
            ln_pre,
            layers,
            ln_post,
            head_dim: config.head_dim(),
            rope_theta: config.rope_theta(),
            window_patch_size: config.pos_emb_height as i32,
        })
    }

    pub fn forward(
        &self,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> Result<UniquePtr<MlxArray>, String> {
        self.forward_inner(pixel_values, grid_thw, true)
    }

    fn forward_inner(
        &self,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
        restore_patch_order: bool,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let mut h = self.patch_embedder.forward(pixel_values, grid_thw)?;
        h = self.ln_pre.forward(&h);

        let plan = window_index_plan(grid_thw, self.window_patch_size)?;
        let window_index = mlxcel_core::from_slice_i32(&plan.indices, &[plan.indices.len() as i32]);
        h = mlxcel_core::take(&h, &window_index, 0);

        let rope = muse_2d_rope(grid_thw, self.head_dim, self.rope_theta)?;
        let rope = mlxcel_core::take(&rope, &window_index, 0);
        let full_cu = full_cu_seqlens(grid_thw)?;
        for layer in &self.layers {
            let cu_seqlens = if layer.is_window_attention() {
                &plan.cu_window_seqlens
            } else {
                &full_cu
            };
            h = layer.forward(&h, cu_seqlens, &rope);
        }
        h = self.ln_post.forward(&h);

        if restore_patch_order {
            let inverse_index = mlxcel_core::from_slice_i32(
                &plan.inverse_indices,
                &[plan.inverse_indices.len() as i32],
            );
            h = mlxcel_core::take(&h, &inverse_index, 0);
        }
        Ok(h)
    }

    pub fn layer_kinds(&self) -> Vec<&'static str> {
        self.layers
            .iter()
            .map(|layer| {
                if layer.is_window_attention() {
                    "window_attention"
                } else {
                    "full_attention"
                }
            })
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn forward_window_ordered_for_tests(
        &self,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> Result<UniquePtr<MlxArray>, String> {
        self.forward_inner(pixel_values, grid_thw, false)
    }
}

pub fn patch_input_size(config: &MuseGlimmerVisionConfig) -> usize {
    config.patch_temporal * 3 * config.patch_size * config.patch_size
}

fn validate_patch_input(
    pixel_values: &MlxArray,
    grid_thw: &[(i32, i32, i32)],
    patch_input_size: usize,
) -> Result<(), String> {
    let shape = mlxcel_core::array_shape(pixel_values);
    if shape.len() != 2 || shape[1] as usize != patch_input_size {
        return Err(format!(
            "Muse Glimmer pixel_values must be [tokens, {patch_input_size}], got {shape:?}"
        ));
    }
    let expected_tokens: i32 = grid_thw.iter().map(|(t, h, w)| t * h * w).sum();
    if shape[0] != expected_tokens {
        return Err(format!(
            "Muse Glimmer pixel_values token count {} does not match image_grid_thw tokens {expected_tokens}",
            shape[0]
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "muse_glimmer_tests.rs"]
mod tests;
