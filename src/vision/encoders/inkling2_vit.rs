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

//! Inkling 2 vision tower: 2x2-merged 20-px patches, bilinear-resized
//! learned positions, a CLS token, 2-D RoPE, layer-scaled pre-norm blocks and
//! a GELU projector.
//!
//! Derived from mlx-vlm `VisionModel` (`mlx_vlm/models/inkling2/vision.py`,
//! MIT).

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use crate::models::inkling2::config::Inkling2VisionConfig;
use crate::vision::dense::interpolate::resize_bilinear_nhwc;
use crate::vision::weights::load_dense_param;

/// A learned parameter, dequantized when an affine `.scales` sidecar is
/// present; a malformed sidecar is a load error naming the tensor.
fn tensor(
    weights: &WeightMap,
    key: &str,
    shape: &[i32],
    quant: (i32, i32),
) -> Result<UniquePtr<MlxArray>, String> {
    load_dense_param(weights, key, shape, Some(quant))
}

fn layer_norm(
    weights: &WeightMap,
    prefix: &str,
    width: i32,
    eps: f32,
) -> Result<LayerNorm, String> {
    let dense = |key: String| load_dense_param(weights, &key, &[width], None);
    Ok(LayerNorm::new(
        dense(format!("{prefix}.weight"))?,
        Some(dense(format!("{prefix}.bias"))?),
        eps,
    ))
}

struct VisionLayer {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    out_proj: UnifiedLinear,
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
    layernorm_before: LayerNorm,
    layernorm_after: LayerNorm,
    lambda_1: UniquePtr<MlxArray>,
    lambda_2: UniquePtr<MlxArray>,
}

pub struct Inkling2VisionTower {
    config: Inkling2VisionConfig,
    patch_embedding: UnifiedLinear,
    /// `[1, grid, grid, hidden]` f32.
    position_embedding: UniquePtr<MlxArray>,
    /// `class_embedding + cls_position_embedding`, `[1, hidden]`.
    cls: UniquePtr<MlxArray>,
    pre_layernorm: LayerNorm,
    layers: Vec<VisionLayer>,
    proj_in: UnifiedLinear,
    proj_out: UnifiedLinear,
}

/// `(x, y) -> (-y, x)` on consecutive pairs of the last axis.
fn rotate_pairs(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let mut paired = shape.clone();
    let last = paired.pop().expect("rank >= 1");
    paired.extend([last / 2, 2]);
    let pairs = mlxcel_core::reshape(x, &paired);
    let rank = paired.len() as i32;
    let first = mlxcel_core::utils::slice_axis(&pairs, rank - 1, 0, 1);
    let second = mlxcel_core::utils::slice_axis(&pairs, rank - 1, 1, 2);
    let rotated = mlxcel_core::concatenate(&mlxcel_core::negative(&second), &first, rank - 1);
    mlxcel_core::reshape(&rotated, &shape)
}

impl Inkling2VisionTower {
    pub fn from_weights(
        weights: &WeightMap,
        config: &Inkling2VisionConfig,
        quant: (i32, i32),
    ) -> Result<Self, String> {
        config.validate()?;
        let hidden = config.hidden_size as i32;
        let eps = config.layer_norm_eps;
        let linear = |name: &str| UnifiedLinear::from_weights(weights, name, quant.0, quant.1);
        let grid = config.posemb_grid_size as i32;
        let position = tensor(
            weights,
            "vision_tower.embeddings.position_embedding",
            &[grid * grid, hidden],
            quant,
        )?;
        let position = mlxcel_core::reshape(
            &mlxcel_core::astype(&position, dtype::FLOAT32),
            &[1, grid, grid, hidden],
        );
        let class = tensor(
            weights,
            "vision_tower.embeddings.class_embedding",
            &[hidden],
            quant,
        )?;
        let class_position = tensor(
            weights,
            "vision_tower.embeddings.cls_position_embedding",
            &[hidden],
            quant,
        )?;
        let cls = mlxcel_core::add(&class, &class_position);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let p = format!("vision_tower.layers.{index}");
            layers.push(VisionLayer {
                q_proj: linear(&format!("{p}.self_attn.q_proj"))?,
                k_proj: linear(&format!("{p}.self_attn.k_proj"))?,
                v_proj: linear(&format!("{p}.self_attn.v_proj"))?,
                out_proj: linear(&format!("{p}.self_attn.out_proj"))?,
                fc1: linear(&format!("{p}.mlp.fc1"))?,
                fc2: linear(&format!("{p}.mlp.fc2"))?,
                layernorm_before: layer_norm(
                    weights,
                    &format!("{p}.layernorm_before"),
                    hidden,
                    eps,
                )?,
                layernorm_after: layer_norm(weights, &format!("{p}.layernorm_after"), hidden, eps)?,
                lambda_1: tensor(weights, &format!("{p}.lambda_1"), &[hidden], quant)?,
                lambda_2: tensor(weights, &format!("{p}.lambda_2"), &[hidden], quant)?,
            });
        }
        Ok(Self {
            config: config.clone(),
            patch_embedding: linear("vision_tower.embeddings.patch_embedding")?,
            position_embedding: position,
            cls: mlxcel_core::reshape(&cls, &[1, hidden]),
            pre_layernorm: layer_norm(weights, "vision_tower.pre_layernorm", hidden, eps)?,
            layers,
            proj_in: linear("vision_tower.proj_in")?,
            proj_out: linear("vision_tower.proj_out")?,
        })
    }

    /// `(cos, sin)` `[1 + 4 * tiles, head_dim]` f32 for a `rows x cols` image.
    fn rope_tables(&self, rows: usize, cols: usize) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let merge = self.config.adapter_merge_size;
        let mut positions = vec![0.0_f32, 0.0];
        for row in 0..rows {
            for col in 0..cols {
                for my in 0..merge {
                    for mx in 0..merge {
                        let y = row * merge + my;
                        let x = col * merge + mx;
                        positions.extend([(x + 1) as f32, (y + 1) as f32]);
                    }
                }
            }
        }
        let count = (positions.len() / 2) as i32;
        let spatial = (self.config.head_dim() / 2) as i32;
        let positions = mlxcel_core::from_slice_f32(&positions, &[count, 2, 1]);
        let steps = mlxcel_core::arange_f32(0.0, spatial as f32, 2.0);
        let exponent = mlxcel_core::divide_scalar(&mlxcel_core::negative(&steps), spatial as f32);
        let theta = mlxcel_core::from_slice_f32(&[self.config.rope_theta], &[1]);
        let inv = mlxcel_core::power(&theta, &exponent);
        let angles =
            mlxcel_core::reshape(&mlxcel_core::multiply(&positions, &inv), &[count, spatial]);
        let angles = mlxcel_core::repeat(&angles, 2, -1);
        (mlxcel_core::cos(&angles), mlxcel_core::sin(&angles))
    }

    fn layer_forward(
        &self,
        layer: &VisionLayer,
        x: &MlxArray,
        cos: &MlxArray,
        sin: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let tokens = mlxcel_core::array_shape(x)[0];
        let heads = self.config.num_attention_heads as i32;
        let head_dim = self.config.head_dim() as i32;
        let normed = layer.layernorm_before.forward(x);
        let split = |proj: &UnifiedLinear| {
            mlxcel_core::reshape(&proj.forward(&normed), &[1, tokens, heads, head_dim])
        };
        let cos = mlxcel_core::reshape(cos, &[1, tokens, 1, head_dim]);
        let sin = mlxcel_core::reshape(sin, &[1, tokens, 1, head_dim]);
        let rope = |a: &MlxArray| {
            let af = mlxcel_core::astype(a, dtype::FLOAT32);
            let out = mlxcel_core::add(
                &mlxcel_core::multiply(&af, &cos),
                &mlxcel_core::multiply(&rotate_pairs(&af), &sin),
            );
            mlxcel_core::astype(&out, mlxcel_core::array_dtype(a))
        };
        let q = rope(&split(&layer.q_proj));
        let k = rope(&split(&layer.k_proj));
        let v = split(&layer.v_proj);
        let [q, k, v] = [q, k, v].map(|a| mlxcel_core::transpose_axes(&a, &[0, 2, 1, 3]));
        // SAFETY: q, k, v are live MLX arrays; a null mask is unmasked.
        let out = unsafe {
            mlxcel_core::fast_scaled_dot_product_attention(
                &q,
                &k,
                &v,
                (1.0 / f64::from(head_dim).sqrt()) as f32,
                std::ptr::null(),
            )
        };
        let out = mlxcel_core::reshape(
            &mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]),
            &[tokens, heads * head_dim],
        );
        let attn = layer.out_proj.forward(&out);
        let x = mlxcel_core::add(x, &mlxcel_core::multiply(&layer.lambda_1, &attn));
        let mlp = layer.fc2.forward(&mlxcel_core::gelu(
            &layer.fc1.forward(&layer.layernorm_after.forward(&x)),
        ));
        mlxcel_core::add(&x, &mlxcel_core::multiply(&layer.lambda_2, &mlp))
    }

    /// Text-width features `[tiles, text_hidden]` for `pixel_values
    /// [tiles, 40, 40, 3]` covering the `(rows, cols)` grids in order.
    pub fn forward(
        &self,
        pixel_values: &MlxArray,
        grids: &[(usize, usize)],
    ) -> Result<UniquePtr<MlxArray>, String> {
        let cfg = &self.config;
        let total: usize = grids.iter().map(|(rows, cols)| rows * cols).sum();
        let shape = mlxcel_core::array_shape(pixel_values);
        let tile = cfg.tile_size() as i32;
        if shape != [total as i32, tile, tile, cfg.num_channels as i32] {
            return Err(format!(
                "Inkling 2 pixel_values must be [{total}, {tile}, {tile}, {}], got {shape:?}",
                cfg.num_channels
            ));
        }
        let pixel_values = mlxcel_core::astype(
            pixel_values,
            mlxcel_core::array_dtype(&self.pre_layernorm.weight),
        );
        let (merge, patch, hidden) = (
            cfg.adapter_merge_size as i32,
            cfg.native_patch_size as i32,
            cfg.hidden_size as i32,
        );
        let mut outputs = Vec::with_capacity(grids.len());
        let mut start = 0_i32;
        for &(rows, cols) in grids {
            let count = (rows * cols) as i32;
            let pixels = mlxcel_core::utils::slice_axis(&pixel_values, 0, start, start + count);
            start += count;
            let patches = mlxcel_core::reshape(
                &pixels,
                &[count, merge, patch, merge, patch, cfg.num_channels as i32],
            );
            let patches = mlxcel_core::transpose_axes(&patches, &[0, 1, 3, 2, 4, 5]);
            let patches = mlxcel_core::reshape(&patches, &[count * merge * merge, -1]);
            let x = self.patch_embedding.forward(&patches);
            let resized = resize_bilinear_nhwc(
                &self.position_embedding,
                (rows * merge as usize, cols * merge as usize),
                false,
                false,
            )
            .map_err(|error| format!("Inkling 2 position resize failed: {error}"))?;
            let pos =
                mlxcel_core::reshape(&resized, &[rows as i32, merge, cols as i32, merge, hidden]);
            let pos = mlxcel_core::transpose_axes(&pos, &[0, 2, 1, 3, 4]);
            let pos = mlxcel_core::reshape(&pos, &[count * merge * merge, hidden]);
            let x_dtype = mlxcel_core::array_dtype(&x);
            let x = mlxcel_core::add(&x, &mlxcel_core::astype(&pos, x_dtype));
            let cls = mlxcel_core::astype(&self.cls, x_dtype);
            let mut x = self
                .pre_layernorm
                .forward(&mlxcel_core::concatenate(&cls, &x, 0));
            let (cos, sin) = self.rope_tables(rows, cols);
            for layer in &self.layers {
                x = self.layer_forward(layer, &x, &cos, &sin);
            }
            let tokens = mlxcel_core::array_shape(&x)[0];
            let x = mlxcel_core::utils::slice_axis(&x, 0, 1, tokens);
            let x = mlxcel_core::reshape(&x, &[count, merge * merge * hidden]);
            outputs.push(
                self.proj_out
                    .forward(&mlxcel_core::gelu(&self.proj_in.forward(&x))),
            );
        }
        let refs: Vec<&MlxArray> = outputs.iter().map(|output| &**output).collect();
        Ok(mlxcel_core::concatenate_many(&refs, 0))
    }
}

#[cfg(test)]
#[path = "inkling2_vit_tests.rs"]
mod tests;
