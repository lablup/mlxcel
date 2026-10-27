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

//! CLIP-style ViT-L body for DeepSeek-OCR (`vision_model.*`).
//!
//! Ingests the SAM tower's compressed grid as its patch embeddings (its own
//! conv patch-embedding is bypassed), prepends a class token, adds a resampled
//! learned position embedding, applies `pre_layrnorm` (checkpoint spelling),
//! then 24 pre-LN encoder layers. Returns `(B, tokens+1, hidden)` including the
//! CLS token; the connector drops CLS.
//!
//! Reference: mlx-vlm `mlx_vlm/models/deepseekocr/vision.py`
//! (<https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/deepseekocr/vision.py>).

use super::deepseekocr_sam::bicubic_resample;
use crate::vision::weights::load_dense_param;
use mlxcel_core::layers::{
    LayerNorm, Linear, QuantizedTensorShapes, UnifiedEmbedding, validate_quantized_packing,
};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

#[derive(Clone)]
pub struct ClipConfig {
    pub hidden_size: i32,
    pub num_heads: i32,
    pub num_layers: usize,
    pub layer_norm_eps: f32,
    /// Grid side of the stored position embedding (`sqrt(num_positions - 1)` = 16).
    pub pos_grid: i32,
}

impl Default for ClipConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            num_heads: 16,
            num_layers: 24,
            layer_norm_eps: 1e-6,
            pos_grid: 16,
        }
    }
}

struct ClipAttention {
    qkv_proj: Linear,
    out_proj: Linear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl ClipAttention {
    /// `x`: `(B, L, D)` -> `(B, L, D)`. Bidirectional (no mask).
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let s = mlxcel_core::array_shape(x);
        let (b, l) = (s[0], s[1]);
        let (heads, hd) = (self.num_heads, self.head_dim);
        let qkv = self.qkv_proj.forward(x); // (B, L, 3D)
        let split = |i: i32| {
            let sl =
                mlxcel_core::slice(&qkv, &[0, 0, i * heads * hd], &[b, l, (i + 1) * heads * hd]);
            let sl = mlxcel_core::reshape(&sl, &[b, l, heads, hd]);
            mlxcel_core::transpose_axes(&sl, &[0, 2, 1, 3]) // (B, heads, L, hd)
        };
        let q = split(0);
        let k = split(1);
        let v = split(2);
        // SAFETY: q/k/v valid; null mask (bidirectional).
        let out = unsafe {
            mlxcel_core::scaled_dot_product_attention(&q, &k, &v, self.scale, std::ptr::null())
        };
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[b, l, heads * hd]);
        self.out_proj.forward(&out)
    }
}

struct ClipLayer {
    layer_norm1: LayerNorm,
    attn: ClipAttention,
    layer_norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl ClipLayer {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let y = self.attn.forward(&self.layer_norm1.forward(x));
        let x = mlxcel_core::add(x, &y);
        let y = self.fc1.forward(&self.layer_norm2.forward(&x));
        let y = mlxcel_core::gelu(&y);
        let y = self.fc2.forward(&y);
        mlxcel_core::add(&x, &y)
    }
}

pub struct ClipEncoder {
    config: ClipConfig,
    class_embedding: UniquePtr<MlxArray>,    // (hidden,)
    position_embedding: UniquePtr<MlxArray>, // (num_positions, hidden)
    pre_layrnorm: LayerNorm,
    layers: Vec<ClipLayer>,
}

impl ClipEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: ClipConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let get = |name: &str| -> Result<UniquePtr<MlxArray>, String> {
            weights
                .get(name)
                .map(|w| mlxcel_core::copy(w))
                .ok_or_else(|| format!("CLIP weight missing: {name}"))
        };
        let lin = |p: &str| -> Result<Linear, String> {
            Ok(Linear::new(
                get(&format!("{p}.weight"))?,
                weights
                    .get(&format!("{p}.bias"))
                    .map(|w| mlxcel_core::copy(w)),
            ))
        };
        let ln = |p: &str, eps: f32| -> Result<LayerNorm, String> {
            Ok(LayerNorm::new(
                get(&format!("{p}.weight"))?,
                weights
                    .get(&format!("{p}.bias"))
                    .map(|w| mlxcel_core::copy(w)),
                eps,
            ))
        };

        let head_dim = config.hidden_size / config.num_heads;
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let lp = format!("{prefix}.transformer.layers.{i}");
            layers.push(ClipLayer {
                layer_norm1: ln(&format!("{lp}.layer_norm1"), config.layer_norm_eps)?,
                attn: ClipAttention {
                    qkv_proj: lin(&format!("{lp}.self_attn.qkv_proj"))?,
                    out_proj: lin(&format!("{lp}.self_attn.out_proj"))?,
                    num_heads: config.num_heads,
                    head_dim,
                    scale: (head_dim as f32).powf(-0.5),
                },
                layer_norm2: ln(&format!("{lp}.layer_norm2"), config.layer_norm_eps)?,
                fc1: lin(&format!("{lp}.mlp.fc1"))?,
                fc2: lin(&format!("{lp}.mlp.fc2"))?,
            });
        }

        let class_key = format!("{prefix}.embeddings.class_embedding");
        if !weights.contains_key(&class_key) {
            return Err(format!("CLIP weight missing: {class_key}"));
        }
        let class_embedding = load_dense_param(
            weights,
            &class_key,
            &[config.hidden_size],
            Some((group_size, bits)),
        )?;
        let position_prefix = format!("{prefix}.embeddings.position_embedding");
        let position_key = format!("{position_prefix}.weight");
        let stored = weights
            .get(&position_key)
            .ok_or_else(|| format!("CLIP weight missing: {position_key}"))?;
        let stored_shape = mlxcel_core::array_shape(stored);
        let expected_rows = config.pos_grid * config.pos_grid + 1;
        if stored_shape.len() != 2 || stored_shape[0] != expected_rows {
            return Err(format!(
                "{position_key} shape mismatch: expected [{expected_rows}, {}], got {stored_shape:?}",
                config.hidden_size
            ));
        }
        let (effective_group_size, effective_bits) =
            crate::vision::weights::resolve_param_quantization(
                weights,
                &position_key,
                &[expected_rows, config.hidden_size],
                Some((group_size, bits)),
            )?
            .unwrap_or((group_size, bits));
        let position = UnifiedEmbedding::from_weights(
            weights,
            &position_prefix,
            effective_group_size,
            effective_bits,
        )?;
        if let Some(quantized) = position.quantized() {
            if mlxcel_core::array_dtype(&quantized.weight) != mlxcel_core::dtype::UINT32 {
                return Err(format!("{position_key} must have uint32 packed weights"));
            }
            let scales_shape = mlxcel_core::array_shape(&quantized.scales);
            let biases_shape = quantized
                .biases
                .as_ref()
                .map(|biases| mlxcel_core::array_shape(biases));
            validate_quantized_packing(
                &position_key,
                &QuantizedTensorShapes {
                    weight: &stored_shape,
                    scales: &scales_shape,
                    biases: biases_shape.as_deref(),
                },
                config.hidden_size as usize,
                quantized.group_size,
                quantized.bits,
                &quantized.mode,
            )?;
        } else if stored_shape[1] != config.hidden_size {
            return Err(format!(
                "{position_key} shape mismatch: expected [{expected_rows}, {}], got {stored_shape:?}",
                config.hidden_size
            ));
        }
        let position_embedding = position.dequantized_weight();
        let dense_shape = mlxcel_core::array_shape(&position_embedding);
        if dense_shape != [expected_rows, config.hidden_size] {
            return Err(format!(
                "{position_key} dequantized to {dense_shape:?}, expected [{expected_rows}, {}]",
                config.hidden_size
            ));
        }

        Ok(Self {
            class_embedding,
            position_embedding,
            // `pre_layrnorm` is `nn.LayerNorm(hidden)` in the reference: MLX
            // default eps 1e-5 (not the 1e-6 of the encoder layers).
            pre_layrnorm: ln(&format!("{prefix}.pre_layrnorm"), 1e-5)?,
            config,
            layers,
        })
    }

    /// `patch_embeds`: the SAM grid `(B, gh, gw, hidden)`. Returns
    /// `(B, gh*gw + 1, hidden)` (CLS included).
    pub fn forward(&self, patch_embeds: &MlxArray) -> UniquePtr<MlxArray> {
        let s = mlxcel_core::array_shape(patch_embeds);
        let (b, gh, gw, hidden) = (s[0], s[1], s[2], s[3]);
        let n_patch = gh * gw;

        // Flatten spatial axes: (B, gh, gw, H) -> (B, gh*gw, H).
        let patches = mlxcel_core::reshape(patch_embeds, &[b, n_patch, hidden]);

        // Prepend the class token: (hidden,) -> (B, 1, hidden).
        let cls = mlxcel_core::reshape(&self.class_embedding, &[1, 1, hidden]);
        let cls = mlxcel_core::broadcast_to(&cls, &[b, 1, hidden]);
        let cls = mlxcel_core::astype(&cls, mlxcel_core::array_dtype(&patches));
        let mut x = mlxcel_core::concatenate(&cls, &patches, 1); // (B, n_patch+1, hidden)

        // Position embedding, resampled (plain cubic, no antialias) for tiles.
        let pos = self.abs_pos(n_patch + 1);
        let pos = mlxcel_core::astype(&pos, mlxcel_core::array_dtype(&x));
        x = mlxcel_core::add(&x, &pos);

        x = self.pre_layrnorm.forward(&x);
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    /// The `(1, n_tokens, hidden)` position embedding for `n_tokens` (CLS + grid).
    fn abs_pos(&self, n_tokens: i32) -> UniquePtr<MlxArray> {
        let hidden = self.config.hidden_size;
        let src = self.config.pos_grid; // 16
        let tgt = ((n_tokens - 1) as f64).sqrt() as i32;
        if tgt == src {
            let np = mlxcel_core::array_shape(&self.position_embedding)[0];
            return mlxcel_core::reshape(&self.position_embedding, &[1, np, hidden]);
        }
        // Rows [0]=CLS, [1..] = src*src grid.
        let cls = mlxcel_core::slice(&self.position_embedding, &[0, 0], &[1, hidden]);
        let total = mlxcel_core::array_shape(&self.position_embedding)[0];
        let grid = mlxcel_core::slice(&self.position_embedding, &[1, 0], &[total, hidden]);
        let grid = mlxcel_core::reshape(&grid, &[src, src, hidden]);
        let grid = bicubic_resample(&grid, src, src, hidden, tgt, tgt, false);
        let grid = mlxcel_core::reshape(&grid, &[tgt * tgt, hidden]);
        let out = mlxcel_core::concatenate(&cls, &grid, 0);
        mlxcel_core::reshape(&out, &[1, tgt * tgt + 1, hidden])
    }
}

#[cfg(test)]
#[path = "deepseekocr_clip_quant_tests.rs"]
mod tests;
