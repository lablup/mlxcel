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

//! Moondream3 vision tower.
//!
//! The Rust port mirrors the ViT backbone, crop reconstruction, and projection
//! MLP from the shipped reference code. Local crops are stitched back into a
//! spatial feature map via `reconstruct_from_crops`, then pooled to the encoder
//! grid size before concatenation with global features.

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Moondream3VisionConfig {
    #[serde(default = "default_enc_dim")]
    pub enc_dim: usize,
    #[serde(default = "default_enc_patch_size")]
    pub enc_patch_size: usize,
    #[serde(default = "default_enc_n_layers")]
    pub enc_n_layers: usize,
    #[serde(default = "default_enc_ff_dim")]
    pub enc_ff_dim: usize,
    #[serde(default = "default_enc_n_heads")]
    pub enc_n_heads: usize,
    #[serde(default = "default_proj_out_dim")]
    pub proj_out_dim: usize,
    #[serde(default = "default_crop_size")]
    pub crop_size: usize,
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_max_crops")]
    pub max_crops: usize,
    #[serde(default = "default_overlap_margin")]
    pub overlap_margin: usize,
    #[serde(default = "default_proj_inner_dim")]
    pub proj_inner_dim: usize,
}

fn default_enc_dim() -> usize {
    1152
}

fn default_enc_patch_size() -> usize {
    14
}

fn default_enc_n_layers() -> usize {
    27
}

fn default_enc_ff_dim() -> usize {
    4304
}

fn default_enc_n_heads() -> usize {
    16
}

fn default_proj_out_dim() -> usize {
    2048
}

fn default_crop_size() -> usize {
    378
}

fn default_in_channels() -> usize {
    3
}

fn default_max_crops() -> usize {
    12
}

fn default_overlap_margin() -> usize {
    4
}

fn default_proj_inner_dim() -> usize {
    8192
}

struct Moondream3VisionAttention {
    qkv: UnifiedLinear,
    proj: UnifiedLinear,
    num_heads: i32,
    scale: f32,
}

impl Moondream3VisionAttention {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        dims: usize,
        num_heads: usize,
    ) -> Result<Self, String> {
        let qkv = UnifiedLinear::from_weights(weights, &format!("{}.qkv", prefix), 64, 4)?;
        let proj = UnifiedLinear::from_weights(weights, &format!("{}.proj", prefix), 64, 4)?;
        let head_dim = dims / num_heads;
        Ok(Self {
            qkv,
            proj,
            num_heads: num_heads as i32,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0];
        let seq_len = shape[1];
        let qkv = self.qkv.forward(x);
        let qkv_dim = mlxcel_core::array_shape(&qkv)[2];
        let chunk = qkv_dim / 3;
        let q = mlxcel_core::slice_last_dim(&qkv, 0, chunk);
        let k = mlxcel_core::slice_last_dim(&qkv, chunk, chunk * 2);
        let v = mlxcel_core::slice_last_dim(&qkv, chunk * 2, chunk * 3);
        let head_dim = chunk / self.num_heads;

        let q = mlxcel_core::reshape(&q, &[batch, seq_len, self.num_heads, head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::reshape(&k, &[batch, seq_len, self.num_heads, head_dim]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::reshape(&v, &[batch, seq_len, self.num_heads, head_dim]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let attn = unsafe {
            mlxcel_core::layers::attention_from_ptr(
                &q,
                &k,
                &v,
                self.scale,
                std::ptr::null(),
                0.0,
                0,
            )
        };
        let attn = mlxcel_core::transpose_axes(&attn, &[0, 2, 1, 3]);
        let attn = mlxcel_core::reshape(&attn, &[batch, seq_len, self.num_heads * head_dim]);
        self.proj.forward(&attn)
    }
}

struct Moondream3VisionMlp {
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
}

impl Moondream3VisionMlp {
    fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            fc1: UnifiedLinear::from_weights(weights, &format!("{}.fc1", prefix), 64, 4)?,
            fc2: UnifiedLinear::from_weights(weights, &format!("{}.fc2", prefix), 64, 4)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let hidden = self.fc1.forward(x);
        let hidden = mlxcel_core::gelu_approx(&hidden);
        self.fc2.forward(&hidden)
    }
}

struct Moondream3VisionBlock {
    ln1: LayerNorm,
    attn: Moondream3VisionAttention,
    ln2: LayerNorm,
    mlp: Moondream3VisionMlp,
}

impl Moondream3VisionBlock {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Moondream3VisionConfig,
    ) -> Result<Self, String> {
        Ok(Self {
            ln1: load_layer_norm(weights, &format!("{}.ln1", prefix), 1e-5)?,
            attn: Moondream3VisionAttention::from_weights(
                weights,
                &format!("{}.attn", prefix),
                config.enc_dim,
                config.enc_n_heads,
            )?,
            ln2: load_layer_norm(weights, &format!("{}.ln2", prefix), 1e-5)?,
            mlp: Moondream3VisionMlp::from_weights(weights, &format!("{}.mlp", prefix))?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let attn = self.attn.forward(&self.ln1.forward(x));
        let hidden = mlxcel_core::add(x, &attn);
        let mlp = self.mlp.forward(&self.ln2.forward(&hidden));
        mlxcel_core::add(&hidden, &mlp)
    }
}

pub struct Moondream3VisionModel {
    patch_emb: UnifiedLinear,
    pos_emb: UniquePtr<MlxArray>,
    blocks: Vec<Moondream3VisionBlock>,
    post_ln: LayerNorm,
    proj_fc1: UnifiedLinear,
    proj_fc2: UnifiedLinear,
    config: Moondream3VisionConfig,
}

impl Moondream3VisionModel {
    pub fn from_weights(
        weights: &WeightMap,
        config: &Moondream3VisionConfig,
    ) -> Result<Self, String> {
        let mut blocks = Vec::with_capacity(config.enc_n_layers);
        for idx in 0..config.enc_n_layers {
            blocks.push(Moondream3VisionBlock::from_weights(
                weights,
                &format!("vision.blocks.{}", idx),
                config,
            )?);
        }

        Ok(Self {
            patch_emb: UnifiedLinear::from_weights(weights, "vision.patch_emb", 64, 4)?,
            pos_emb: get_weight_copy(weights, "vision.pos_emb")?,
            blocks,
            post_ln: load_layer_norm(weights, "vision.post_ln", 1e-5)?,
            proj_fc1: UnifiedLinear::from_weights(weights, "vision.proj_mlp.fc1", 64, 4)?,
            proj_fc2: UnifiedLinear::from_weights(weights, "vision.proj_mlp.fc2", 64, 4)?,
            config: config.clone(),
        })
    }

    fn create_patches(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];
        let patch = self.config.enc_patch_size as i32;

        let x = mlxcel_core::reshape(
            x,
            &[batch, channels, height / patch, patch, width / patch, patch],
        );
        let x = mlxcel_core::transpose_axes(&x, &[0, 2, 4, 1, 3, 5]);
        mlxcel_core::reshape(
            &x,
            &[
                batch,
                (height / patch) * (width / patch),
                channels * patch * patch,
            ],
        )
    }

    fn encode_crops(&self, pixel_values: &MlxArray) -> UniquePtr<MlxArray> {
        let mut hidden = self.patch_emb.forward(&self.create_patches(pixel_values));
        hidden = mlxcel_core::add(&hidden, &self.pos_emb);
        for block in &self.blocks {
            hidden = block.forward(&hidden);
        }
        self.post_ln.forward(&hidden)
    }

    pub fn encode_image_embeddings(
        &self,
        pixel_values: &MlxArray,
        tiling: (usize, usize),
    ) -> UniquePtr<MlxArray> {
        let crops = self.encode_crops(pixel_values);
        let crop_count = mlxcel_core::array_shape(&crops)[0];
        let enc_n = self.config.enc_n_layers as i32; // 27
        let dim = self.config.enc_dim as i32; // 1152

        // Global features: [1, 729, dim] → [729, dim]
        let global = mlxcel_core::slice(&crops, &[0, 0, 0], &[1, enc_n * enc_n, dim]);
        let global = mlxcel_core::reshape(&global, &[enc_n * enc_n, dim]);

        // Local features: reconstruct spatial layout then pool
        let reconstructed = if crop_count > 1 {
            let local = mlxcel_core::slice(&crops, &[1, 0, 0], &[crop_count, enc_n * enc_n, dim]);
            let num_local = crop_count - 1;
            let local_spatial = mlxcel_core::reshape(&local, &[num_local, enc_n, enc_n, dim]);
            let stitched =
                reconstruct_from_crops(&local_spatial, tiling, self.config.overlap_margin);
            let pool_target = self.config.enc_n_layers;
            adaptive_avg_pool2d(&stitched, pool_target, pool_target)
        } else {
            mlxcel_core::reshape(&global, &[enc_n, enc_n, dim])
        };

        // Flatten reconstructed: [27, 27, dim] → [729, dim]
        let reconstructed_flat = mlxcel_core::reshape(&reconstructed, &[enc_n * enc_n, dim]);

        // Concatenate global + reconstructed in feature dim → [729, 2304]
        let merged = mlxcel_core::concatenate(&global, &reconstructed_flat, 1);
        let merged = mlxcel_core::reshape(&merged, &[1, enc_n * enc_n, dim * 2]);

        let projected = self.proj_fc1.forward(&merged);
        let projected = mlxcel_core::gelu_approx(&projected);
        self.proj_fc2.forward(&projected)
    }

    pub fn output_token_count(&self) -> usize {
        (self.config.crop_size / self.config.enc_patch_size).pow(2)
    }
}

/// Reconstruct a spatial feature map from overlapping crops.
///
/// Input `crops`: [num_crops, crop_h, crop_w, dim] — local crop feature grids.
/// Returns [output_h, output_w, dim] with overlapping margins removed.
///
/// Port of Python `image_crops.py::reconstruct_from_crops` with patch_size=1.
fn reconstruct_from_crops(
    crops: &MlxArray,
    tiling: (usize, usize),
    overlap_margin: usize,
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(crops);
    let crop_h = shape[1] as usize;
    let crop_w = shape[2] as usize;
    let dim = shape[3];
    let (tiling_h, tiling_w) = tiling;
    let margin = overlap_margin; // In patch units (patch_size=1)

    // Build the result by slicing non-overlapping portions and concatenating
    let mut row_results: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(tiling_h);
    for tile_y in 0..tiling_h {
        let y_start = if tile_y == 0 { 0 } else { margin };
        let y_end = if tile_y == tiling_h - 1 {
            crop_h
        } else {
            crop_h - margin
        };

        let mut col_results: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(tiling_w);
        for tile_x in 0..tiling_w {
            let x_start = if tile_x == 0 { 0 } else { margin };
            let x_end = if tile_x == tiling_w - 1 {
                crop_w
            } else {
                crop_w - margin
            };

            let crop_idx = (tile_y * tiling_w + tile_x) as i32;
            // Slice this crop: [1, crop_h, crop_w, dim] → [y_end-y_start, x_end-x_start, dim]
            let patch = mlxcel_core::slice(
                crops,
                &[crop_idx, y_start as i32, x_start as i32, 0],
                &[crop_idx + 1, y_end as i32, x_end as i32, dim],
            );
            // Remove batch dim: [1, h, w, dim] → [h, w, dim]
            let patch = mlxcel_core::reshape(
                &patch,
                &[(y_end - y_start) as i32, (x_end - x_start) as i32, dim],
            );
            col_results.push(patch);
        }
        // Concatenate columns: [row_h, total_w, dim]
        let row = concat_along_axis(&col_results, 1);
        row_results.push(row);
    }
    // Concatenate rows: [total_h, total_w, dim]
    concat_along_axis(&row_results, 0)
}

/// Adaptive 2D average pooling: [H, W, C] → [target_h, target_w, C].
///
/// Uses separable two-pass approach (row pooling then column pooling).
fn adaptive_avg_pool2d(input: &MlxArray, target_h: usize, target_w: usize) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(input);
    let h = shape[0] as usize;
    let w = shape[1] as usize;
    let c = shape[2];

    // Pass 1: pool rows — [H, W, C] → [target_h, W, C]
    let inter = if h == target_h {
        mlxcel_core::copy(input)
    } else {
        let mut rows: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(target_h);
        for i in 0..target_h {
            let y1 = (i * h / target_h) as i32;
            let y2 = ((i + 1) * h).div_ceil(target_h) as i32;
            let row_slice = mlxcel_core::slice(input, &[y1, 0, 0], &[y2, w as i32, c]);
            let row_mean = mlxcel_core::mean_axis(&row_slice, 0, true);
            rows.push(row_mean);
        }
        concat_along_axis(&rows, 0)
    };

    // Pass 2: pool columns — [target_h, W, C] → [target_h, target_w, C]
    if w == target_w {
        inter
    } else {
        let mut cols: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(target_w);
        for j in 0..target_w {
            let x1 = (j * w / target_w) as i32;
            let x2 = ((j + 1) * w).div_ceil(target_w) as i32;
            let col_slice = mlxcel_core::slice(&inter, &[0, x1, 0], &[target_h as i32, x2, c]);
            let col_mean = mlxcel_core::mean_axis(&col_slice, 1, true);
            cols.push(col_mean);
        }
        concat_along_axis(&cols, 1)
    }
}

/// Concatenate arrays along a given axis.
fn concat_along_axis(arrays: &[UniquePtr<MlxArray>], axis: i32) -> UniquePtr<MlxArray> {
    if arrays.len() == 1 {
        return mlxcel_core::copy(&arrays[0]);
    }
    let mut result = mlxcel_core::copy(&arrays[0]);
    for arr in &arrays[1..] {
        result = mlxcel_core::concatenate(&result, arr, axis);
    }
    result
}

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|weight| mlxcel_core::copy(weight))
        .ok_or_else(|| format!("Missing weight: {}", name))
}

fn load_layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    let weight = get_weight_copy(weights, &format!("{}.weight", prefix))?;
    let bias = weights
        .get(&format!("{}.bias", prefix))
        .map(|value| mlxcel_core::copy(value));
    Ok(LayerNorm::new(weight, bias, eps))
}
