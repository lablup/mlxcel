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

//! Qwen3-VL Vision Encoder
//!
//! Evolution of Qwen2-VL vision encoder with:
//! - LayerNorm + GELU MLP (like Qwen2-VL, not Qwen2.5-VL's RMSNorm + SwiGLU)
//! - Learned position embeddings with bilinear interpolation (new)
//! - DeepStack multi-layer visual injection
//! - No windowed attention (simpler than Qwen2.5-VL)
//! - cu_seqlens = h * w per frame (not multiplied by spatial_merge_unit)
//! - Fused SDPA head-dim padding to match upstream MLX kernel preferences
//!
//! Used by: Qwen3-VL
//! Reference: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen3_vl/vision.py

use super::qwen2_vl::{VisionRotaryEmbedding, apply_rotary_pos_emb_vision, concat_many};
use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

/// Qwen3-VL vision encoder configuration
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3VLVisionConfig {
    #[serde(default = "default_depth")]
    pub depth: usize,
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_out_hidden_size")]
    pub out_hidden_size: usize,
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: usize,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(alias = "in_chans", default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_num_position_embeddings")]
    pub num_position_embeddings: usize,
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<usize>,
    /// Quantization group_size (inherited from top-level config)
    #[serde(default)]
    pub quant_group_size: i32,
    /// Quantization bits (inherited from top-level config)
    #[serde(default)]
    pub quant_bits: i32,
}

fn default_depth() -> usize {
    32
}
fn default_intermediate_size() -> usize {
    3420
}
fn default_out_hidden_size() -> usize {
    1536
}
fn default_num_heads() -> usize {
    16
}
fn default_patch_size() -> usize {
    14
}
fn default_spatial_merge_size() -> usize {
    2
}
fn default_temporal_patch_size() -> usize {
    2
}
fn default_in_channels() -> usize {
    3
}
fn default_num_position_embeddings() -> usize {
    2304
}

// Helper: load LayerNorm from weights.
fn load_layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    let weight_key = format!("{}.weight", prefix);
    let bias_key = format!("{}.bias", prefix);

    let weight = weights
        .get(&weight_key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", weight_key))?;
    let bias = weights.get(&bias_key).map(|b| mlxcel_core::copy(b));

    Ok(LayerNorm::new(weight, bias, eps))
}

// PatchEmbed - Conv3d degenerated to Linear (same as Qwen2-VL/2.5-VL).
struct PatchEmbed {
    proj_weight: UniquePtr<MlxArray>,
    proj_bias: Option<UniquePtr<MlxArray>>,
    in_channels: usize,
    temporal_patch_size: usize,
    patch_size: usize,
}

impl PatchEmbed {
    fn from_weights(
        weights: &WeightMap,
        config: &Qwen3VLVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let weight_key = format!("{}.proj.weight", prefix);
        let w = weights
            .get(&weight_key)
            .ok_or_else(|| format!("Missing {}", weight_key))?;

        let shape = mlxcel_core::array_shape(w);
        let out_features = config.hidden_size as i32;
        let in_features = (config.in_channels
            * config.temporal_patch_size
            * config.patch_size
            * config.patch_size) as i32;

        // Handle Conv3d weight shape -> 2D Linear weight
        let w_reshaped = if shape.len() == 5 {
            // MLX Conv3d weight: [out, kT, kH, kW, in_channels]
            // Reorder weight to [out, T, C, H, W] to match input layout
            let w_reordered = mlxcel_core::transpose_axes(w, &[0, 1, 4, 2, 3]);
            mlxcel_core::reshape(&w_reordered, &[out_features, in_features])
        } else if shape.len() == 2 {
            mlxcel_core::copy(w)
        } else {
            return Err(format!("Unexpected patch_embed weight shape: {:?}", shape));
        };

        let bias_key = format!("{}.proj.bias", prefix);
        let proj_bias = weights.get(&bias_key).map(|b| mlxcel_core::copy(b));

        Ok(Self {
            proj_weight: w_reshaped,
            proj_bias,
            in_channels: config.in_channels,
            temporal_patch_size: config.temporal_patch_size,
            patch_size: config.patch_size,
        })
    }

    fn forward(&self, hidden_states: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(hidden_states);
        let total_elements = shape[0];
        let n = total_elements / self.temporal_patch_size as i32;
        let in_features =
            (self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size)
                as i32;

        let h = mlxcel_core::reshape(
            hidden_states,
            &[n, self.temporal_patch_size as i32, shape[1]],
        );
        let h = mlxcel_core::reshape(&h, &[n, in_features]);

        let wt = mlxcel_core::transpose(&self.proj_weight);
        let result = mlxcel_core::matmul(&h, &wt);
        match &self.proj_bias {
            Some(b) => mlxcel_core::add(&result, b),
            None => result,
        }
    }
}

// Learned Position Embeddings with Bilinear Interpolation.
struct PositionEmbedding {
    weight: UniquePtr<MlxArray>, // [num_position_embeddings, hidden_size]
    num_grid_per_side: i32,      // sqrt(num_position_embeddings)
}

impl PositionEmbedding {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Qwen3VLVisionConfig,
    ) -> Result<Self, String> {
        let weight_key = format!("{}.weight", prefix);
        let weight = weights
            .get(&weight_key)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", weight_key))?;

        let num_grid_per_side = (config.num_position_embeddings as f64).sqrt() as i32;

        Ok(Self {
            weight,
            num_grid_per_side,
        })
    }

    /// Fast bilinear interpolation of position embeddings
    /// Returns: [total_merged_tokens, hidden_size]
    fn fast_pos_embed_interpolate(
        &self,
        grid_thw: &[(i32, i32, i32)],
        spatial_merge_size: usize,
    ) -> UniquePtr<MlxArray> {
        let grid = self.num_grid_per_side;
        let merge = spatial_merge_size as i32;

        // Build index and weight arrays for all 4 bilinear corners
        let mut idx_lists: [Vec<i32>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut weight_lists: [Vec<f32>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

        for &(_, h, w) in grid_thw {
            // Compute interpolation indices/weights for this image
            let h_step = if h > 1 {
                (grid - 1) as f32 / (h - 1) as f32
            } else {
                0.0
            };
            let w_step = if w > 1 {
                (grid - 1) as f32 / (w - 1) as f32
            } else {
                0.0
            };

            for hi in 0..h {
                let h_idx = hi as f32 * h_step;
                let h_floor = h_idx.floor() as i32;
                let h_ceil = (h_floor + 1).min(grid - 1);
                let dh = h_idx - h_floor as f32;

                for wi in 0..w {
                    let w_idx = wi as f32 * w_step;
                    let w_floor = w_idx.floor() as i32;
                    let w_ceil = (w_floor + 1).min(grid - 1);
                    let dw = w_idx - w_floor as f32;

                    // 4 corners: (floor_h, floor_w), (floor_h, ceil_w), (ceil_h, floor_w), (ceil_h, ceil_w)
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
        }

        let total_hw: i32 = idx_lists[0].len() as i32;

        // Build [4, total_hw] index tensor and [4, total_hw] weight tensor
        let mut all_idx = Vec::with_capacity((4 * total_hw) as usize);
        let mut all_wt = Vec::with_capacity((4 * total_hw) as usize);
        for i in 0..4 {
            all_idx.extend_from_slice(&idx_lists[i]);
            all_wt.extend_from_slice(&weight_lists[i]);
        }

        let idx_tensor = mlxcel_core::from_slice_i32(&all_idx, &[4, total_hw]);
        let wt_tensor = mlxcel_core::from_slice_f32(&all_wt, &[4, total_hw]);

        // Cast weight tensor to embedding dtype
        let embed_dtype = mlxcel_core::array_dtype(&self.weight);
        let wt_tensor = mlxcel_core::astype(&wt_tensor, embed_dtype);

        // Look up embeddings: [4, total_hw] -> [4, total_hw, hidden_size]
        let idx_flat = mlxcel_core::flatten(&idx_tensor);
        let embeds = mlxcel_core::take(&self.weight, &idx_flat, 0);
        let hidden_size = mlxcel_core::array_shape(&self.weight)[1];
        let embeds = mlxcel_core::reshape(&embeds, &[4, total_hw, hidden_size]);

        // Multiply by weights: [4, total_hw, 1] * [4, total_hw, hidden_size]
        let wt_expanded = mlxcel_core::reshape(&wt_tensor, &[4, total_hw, 1]);
        let weighted = mlxcel_core::multiply(&embeds, &wt_expanded);

        // Sum 4 corners -> [total_hw, hidden_size]
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
        let patch_pos_embeds = mlxcel_core::add(&sum01, &sum23);

        // Split by images, permute with spatial_merge_size, tile by temporal
        let mut split_start = 0;
        let mut result_parts: Vec<UniquePtr<MlxArray>> = Vec::new();

        for &(t, h, w) in grid_thw {
            let hw = h * w;
            let part = mlxcel_core::slice(
                &patch_pos_embeds,
                &[split_start, 0],
                &[split_start + hw, hidden_size],
            );
            split_start += hw;

            // Tile by temporal dimension
            let tiled = if t > 1 {
                mlxcel_core::tile(&part, &[t, 1])
            } else {
                mlxcel_core::copy(part.as_ref().unwrap())
            };

            // Reshape and permute for spatial merge:
            // [t*h*w, D] -> [t, h, w, D] -> [t, h/m, m, w/m, m, D] -> [t, h/m, w/m, m, m, D] -> flatten
            let feature_dim = hidden_size;
            let h_merged = h / merge;
            let w_merged = w / merge;

            let reshaped = mlxcel_core::reshape(&tiled, &[t, h, w, feature_dim]);
            let reshaped = mlxcel_core::reshape(
                &reshaped,
                &[t, h_merged, merge, w_merged, merge, feature_dim],
            );
            // transpose: [t, h/m, m, w/m, m, D] -> [t, h/m, w/m, m, m, D]
            let permuted = mlxcel_core::transpose_axes(&reshaped, &[0, 1, 3, 2, 4, 5]);
            let flattened = mlxcel_core::reshape(&permuted, &[-1, feature_dim]);

            result_parts.push(flattened);
        }

        if result_parts.len() == 1 {
            result_parts.into_iter().next().unwrap()
        } else {
            concat_many(&result_parts, 0)
        }
    }
}

// Vision Attention - fused QKV, same as Qwen2-VL.
//
// Qwen3-VL uses head dimensions (e.g. 72/96) that do not always map to the
// fused MLX SDPA kernel's preferred widths. Matching upstream `mlx-vlm`, pad
// the head dimension to the next supported fused width, run attention, then
// slice the result back to the original size.
fn fused_sdpa_target_dim(head_dim: i32) -> i32 {
    const SUPPORTED_FUSED_HEAD_DIMS: [i32; 3] = [64, 80, 128];
    SUPPORTED_FUSED_HEAD_DIMS
        .into_iter()
        .find(|&candidate| head_dim <= candidate)
        .unwrap_or(head_dim)
}

fn sdpa_pad_width(ndim: usize, original_dim: i32, target_dim: i32) -> Option<Vec<i32>> {
    if target_dim <= original_dim {
        return None;
    }

    let mut pad_width = vec![0; ndim * 2];
    pad_width[ndim * 2 - 1] = target_dim - original_dim;
    Some(pad_width)
}

fn ensure_fused_sdpa(
    q: &MlxArray,
    k: &MlxArray,
    v: &MlxArray,
    scale: f32,
    mask: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    let head_dim = *mlxcel_core::array_shape(q)
        .last()
        .expect("vision attention queries should have a head dimension");
    let target_dim = fused_sdpa_target_dim(head_dim);
    let mask_ptr = mask.map_or(std::ptr::null(), |mask| mask as *const MlxArray);

    if let Some(pad_width) = sdpa_pad_width(mlxcel_core::array_ndim(q), head_dim, target_dim) {
        let q = mlxcel_core::pad(q, &pad_width, 0.0);
        let k = mlxcel_core::pad(k, &pad_width, 0.0);
        let v = mlxcel_core::pad(v, &pad_width, 0.0);
        let attn =
            unsafe { mlxcel_core::layers::attention_from_ptr(&q, &k, &v, scale, mask_ptr, 0.0, 0) };
        mlxcel_core::slice_last_dim(&attn, 0, head_dim)
    } else {
        unsafe { mlxcel_core::layers::attention_from_ptr(q, k, v, scale, mask_ptr, 0.0, 0) }
    }
}

struct VisionAttention {
    qkv: UnifiedLinear,
    proj: UnifiedLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl VisionAttention {
    fn from_weights(
        weights: &WeightMap,
        config: &Qwen3VLVisionConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let qkv = UnifiedLinear::from_weights(weights, &format!("{}.attn.qkv", prefix), gs, bits)?;
        let proj =
            UnifiedLinear::from_weights(weights, &format!("{}.attn.proj", prefix), gs, bits)?;
        let head_dim = (config.hidden_size / config.num_heads) as i32;

        Ok(Self {
            qkv,
            proj,
            num_heads: config.num_heads as i32,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        cu_seqlens: &[i32],
        rotary_pos_emb: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let seq_length = shape[0];

        let qkv = self.qkv.forward(x);
        let qkv = mlxcel_core::reshape(&qkv, &[seq_length, 3, self.num_heads, self.head_dim]);
        let qkv = mlxcel_core::transpose_axes(&qkv, &[1, 0, 2, 3]);

        let q = mlxcel_core::slice(
            &qkv,
            &[0, 0, 0, 0],
            &[1, seq_length, self.num_heads, self.head_dim],
        );
        let k = mlxcel_core::slice(
            &qkv,
            &[1, 0, 0, 0],
            &[2, seq_length, self.num_heads, self.head_dim],
        );
        let v = mlxcel_core::slice(
            &qkv,
            &[2, 0, 0, 0],
            &[3, seq_length, self.num_heads, self.head_dim],
        );
        let q = mlxcel_core::squeeze_axis(&q, 0);
        let k = mlxcel_core::squeeze_axis(&k, 0);
        let v = mlxcel_core::squeeze_axis(&v, 0);

        let q = apply_rotary_pos_emb_vision(&q, rotary_pos_emb);
        let k = apply_rotary_pos_emb_vision(&k, rotary_pos_emb);

        // [seq, heads, head_dim] -> [1, heads, seq, head_dim]
        let q = mlxcel_core::transpose_axes(&q, &[1, 0, 2]);
        let k = mlxcel_core::transpose_axes(&k, &[1, 0, 2]);
        let v = mlxcel_core::transpose_axes(&v, &[1, 0, 2]);
        let q = mlxcel_core::expand_dims(&q, 0);
        let k = mlxcel_core::expand_dims(&k, 0);
        let v = mlxcel_core::expand_dims(&v, 0);

        // Per-segment attention
        let num_segments = cu_seqlens.len() - 1;
        let mut attn_outputs = Vec::with_capacity(num_segments);

        for seg in 0..num_segments {
            let start = cu_seqlens[seg];
            let end = cu_seqlens[seg + 1];

            let q_seg = mlxcel_core::slice(
                &q,
                &[0, 0, start, 0],
                &[1, self.num_heads, end, self.head_dim],
            );
            let k_seg = mlxcel_core::slice(
                &k,
                &[0, 0, start, 0],
                &[1, self.num_heads, end, self.head_dim],
            );
            let v_seg = mlxcel_core::slice(
                &v,
                &[0, 0, start, 0],
                &[1, self.num_heads, end, self.head_dim],
            );

            let attn = ensure_fused_sdpa(&q_seg, &k_seg, &v_seg, self.scale, None);
            attn_outputs.push(attn);
        }

        let output = if attn_outputs.len() == 1 {
            attn_outputs.into_iter().next().unwrap()
        } else {
            concat_many(&attn_outputs, 2)
        };

        let output = mlxcel_core::squeeze_axis(&output, 0);
        let output = mlxcel_core::transpose_axes(&output, &[1, 0, 2]);
        let output = mlxcel_core::reshape(&output, &[seq_length, -1]);

        self.proj.forward(&output)
    }
}

#[cfg(test)]
#[path = "qwen3_vl_tests.rs"]
mod tests;

// Vision MLP - GELU (like Qwen2-VL, NOT SwiGLU like Qwen2.5-VL).
// Weight keys: linear_fc1, linear_fc2 (not fc1/fc2 or gate_proj/up_proj/down_proj).
struct VisionMLP {
    linear_fc1: UnifiedLinear,
    linear_fc2: UnifiedLinear,
}

impl VisionMLP {
    fn from_weights(weights: &WeightMap, prefix: &str, gs: i32, bits: i32) -> Result<Self, String> {
        Ok(Self {
            linear_fc1: UnifiedLinear::from_weights(
                weights,
                &format!("{}.mlp.linear_fc1", prefix),
                gs,
                bits,
            )?,
            linear_fc2: UnifiedLinear::from_weights(
                weights,
                &format!("{}.mlp.linear_fc2", prefix),
                gs,
                bits,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.linear_fc1.forward(x);
        let h = mlxcel_core::gelu_approx(&h);
        self.linear_fc2.forward(&h)
    }
}

// VisionBlock - LayerNorm + GELU MLP.
struct VisionBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attn: VisionAttention,
    mlp: VisionMLP,
}

impl VisionBlock {
    fn from_weights(
        weights: &WeightMap,
        config: &Qwen3VLVisionConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            norm1: load_layer_norm(weights, &format!("{}.norm1", prefix), 1e-6)?,
            norm2: load_layer_norm(weights, &format!("{}.norm2", prefix), 1e-6)?,
            attn: VisionAttention::from_weights(weights, config, prefix, gs, bits)?,
            mlp: VisionMLP::from_weights(weights, prefix, gs, bits)?,
        })
    }

    fn forward(
        &self,
        hidden_states: &MlxArray,
        cu_seqlens: &[i32],
        rotary_pos_emb: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let normed = self.norm1.forward(hidden_states);
        let attn_out = self.attn.forward(&normed, cu_seqlens, rotary_pos_emb);
        let h = mlxcel_core::add(hidden_states, &attn_out);
        let normed = self.norm2.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }
}

// PatchMerger - LayerNorm + GELU MLP (projection to text hidden size).
// Weight keys: norm, linear_fc1, linear_fc2 (not ln_q/mlp.0/mlp.2).
// Has use_postshuffle_norm parameter:
// - Main merger: use_postshuffle_norm=false → norm on hidden_size, then reshape.
// - DeepStack mergers: use_postshuffle_norm=true → reshape first, then norm.
struct PatchMerger {
    norm: LayerNorm,
    linear_fc1: UnifiedLinear,
    linear_fc2: UnifiedLinear,
    hidden_size: usize,
    use_postshuffle_norm: bool,
}

impl PatchMerger {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        context_dim: usize,
        spatial_merge_size: usize,
        use_postshuffle_norm: bool,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let hidden_size = context_dim * spatial_merge_size * spatial_merge_size;
        let norm_dim = if use_postshuffle_norm {
            hidden_size
        } else {
            context_dim
        };
        // Load LayerNorm with the appropriate dimension
        let _ = norm_dim; // norm dimension is determined by weight shape
        let norm = load_layer_norm(weights, &format!("{}.norm", prefix), 1e-6)?;
        let linear_fc1 =
            UnifiedLinear::from_weights(weights, &format!("{}.linear_fc1", prefix), gs, bits)?;
        let linear_fc2 =
            UnifiedLinear::from_weights(weights, &format!("{}.linear_fc2", prefix), gs, bits)?;

        Ok(Self {
            norm,
            linear_fc1,
            linear_fc2,
            hidden_size,
            use_postshuffle_norm,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = if self.use_postshuffle_norm {
            // DeepStack: reshape first, then norm
            let reshaped = mlxcel_core::reshape(x, &[-1, self.hidden_size as i32]);
            self.norm.forward(&reshaped)
        } else {
            // Main merger: norm first, then reshape
            let normed = self.norm.forward(x);
            mlxcel_core::reshape(&normed, &[-1, self.hidden_size as i32])
        };
        let h = self.linear_fc1.forward(&h);
        let h = mlxcel_core::gelu(&h);
        self.linear_fc2.forward(&h)
    }
}

// Qwen3-VL Vision Encoder Output (includes deepstack features).
/// Output from Qwen3-VL vision encoder including DeepStack features
pub struct Qwen3VLVisionEncoderOutput {
    pub hidden_states: UniquePtr<MlxArray>,
    pub deepstack_features: Vec<UniquePtr<MlxArray>>,
}

// Qwen3-VL Vision Encoder.
/// Qwen3-VL Vision Model with learned position embeddings and DeepStack
///
/// Used by: Qwen3-VL
pub struct Qwen3VLVisionEncoder {
    patch_embed: PatchEmbed,
    pos_embed: PositionEmbedding,
    rotary_pos_emb: VisionRotaryEmbedding,
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
    deepstack_merger_list: Vec<PatchMerger>,
    deepstack_visual_indexes: Vec<usize>,
    spatial_merge_size: usize,
}

impl Qwen3VLVisionEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        config: &Qwen3VLVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.quant_group_size;
        let bits = config.quant_bits;

        let patch_embed =
            PatchEmbed::from_weights(weights, config, &format!("{}.patch_embed", prefix))?;
        let pos_embed =
            PositionEmbedding::from_weights(weights, &format!("{}.pos_embed", prefix), config)?;

        let head_dim = config.hidden_size / config.num_heads;
        let rotary_pos_emb = VisionRotaryEmbedding::new(head_dim / 2);

        let mut blocks = Vec::with_capacity(config.depth);
        for i in 0..config.depth {
            blocks.push(VisionBlock::from_weights(
                weights,
                config,
                &format!("{}.blocks.{}", prefix, i),
                gs,
                bits,
            )?);
        }

        // Main merger: use_postshuffle_norm=false
        let merger = PatchMerger::from_weights(
            weights,
            &format!("{}.merger", prefix),
            config.hidden_size,
            config.spatial_merge_size,
            false,
            gs,
            bits,
        )?;

        // DeepStack mergers: use_postshuffle_norm=true
        let mut deepstack_merger_list = Vec::with_capacity(config.deepstack_visual_indexes.len());
        for i in 0..config.deepstack_visual_indexes.len() {
            deepstack_merger_list.push(PatchMerger::from_weights(
                weights,
                &format!("{}.deepstack_merger_list.{}", prefix, i),
                config.hidden_size,
                config.spatial_merge_size,
                true,
                gs,
                bits,
            )?);
        }

        Ok(Self {
            patch_embed,
            pos_embed,
            rotary_pos_emb,
            blocks,
            merger,
            deepstack_merger_list,
            deepstack_visual_indexes: config.deepstack_visual_indexes.clone(),
            spatial_merge_size: config.spatial_merge_size,
        })
    }

    /// Compute 2D rotary position embeddings from grid_thw
    fn rot_pos_emb(&self, grid_thw: &[(i32, i32, i32)]) -> UniquePtr<MlxArray> {
        let mut all_pos_ids: Vec<UniquePtr<MlxArray>> = Vec::new();
        let mut max_grid_dim: i32 = 0;
        let merge = self.spatial_merge_size as i32;

        for &(t, h, w) in grid_thw {
            if h > max_grid_dim {
                max_grid_dim = h;
            }
            if w > max_grid_dim {
                max_grid_dim = w;
            }

            // Create block indices and intra-block indices
            let merged_h = h / merge;
            let merged_w = w / merge;

            // Build position IDs with merge-size grouping
            // block_rows * merge + intra_row, block_cols * merge + intra_col
            let total = merged_h * merged_w * merge * merge;
            let mut h_coords = Vec::with_capacity(total as usize);
            let mut w_coords = Vec::with_capacity(total as usize);

            for bh in 0..merged_h {
                for bw in 0..merged_w {
                    for ih in 0..merge {
                        for iw in 0..merge {
                            h_coords.push(bh * merge + ih);
                            w_coords.push(bw * merge + iw);
                        }
                    }
                }
            }

            // Stack [total, 2] and tile by temporal
            let h_arr = mlxcel_core::from_slice_i32(&h_coords, &[total, 1]);
            let w_arr = mlxcel_core::from_slice_i32(&w_coords, &[total, 1]);
            let stacked = mlxcel_core::concatenate(&h_arr, &w_arr, 1);

            let tiled = if t > 1 {
                mlxcel_core::tile(&stacked, &[t, 1])
            } else {
                stacked
            };
            all_pos_ids.push(tiled);
        }

        let pos_ids = if all_pos_ids.len() == 1 {
            all_pos_ids.into_iter().next().unwrap()
        } else {
            concat_many(&all_pos_ids, 0)
        };

        let rotary_table = self.rotary_pos_emb.forward(max_grid_dim);
        let pos_ids_flat = mlxcel_core::flatten(&pos_ids);
        let all_freqs = mlxcel_core::take(&rotary_table, &pos_ids_flat, 0);
        let total_shape = mlxcel_core::array_shape(&pos_ids);
        let total_tokens = total_shape[0];
        let freq_shape = mlxcel_core::array_shape(&all_freqs);
        let half_dim = freq_shape[1];
        let all_freqs = mlxcel_core::reshape(&all_freqs, &[total_tokens, 2, half_dim]);
        mlxcel_core::reshape(&all_freqs, &[total_tokens, 2 * half_dim])
    }

    /// Compute cu_seqlens from grid_thw
    /// Qwen3-VL: cu_seqlens = h * w per frame (NOT multiplied by spatial_merge_unit)
    fn compute_cu_seqlens(grid_thw: &[(i32, i32, i32)]) -> Vec<i32> {
        let mut cu_seqlens = vec![0i32];
        let mut cumulative = 0i32;
        for &(t, h, w) in grid_thw {
            let seq_per_frame = h * w;
            for _ in 0..t {
                cumulative += seq_per_frame;
                cu_seqlens.push(cumulative);
            }
        }
        cu_seqlens
    }

    /// Forward pass returning hidden_states and deepstack features
    pub fn forward_with_grid(
        &self,
        hidden_states: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> Qwen3VLVisionEncoderOutput {
        // 1. Patch embedding
        let mut h = self.patch_embed.forward(hidden_states);

        // 2. Add learned position embeddings with bilinear interpolation
        let pos_embeds = self
            .pos_embed
            .fast_pos_embed_interpolate(grid_thw, self.spatial_merge_size);
        h = mlxcel_core::add(&h, &pos_embeds);

        // 3. Compute rotary position embeddings
        let rotary_pos_emb = self.rot_pos_emb(grid_thw);

        // Ensure shapes match
        let h_shape = mlxcel_core::array_shape(&h);
        let seq_len = h_shape[0];
        h = mlxcel_core::reshape(&h, &[seq_len, -1]);
        let rope_shape = mlxcel_core::array_shape(&rotary_pos_emb);
        let rotary_pos_emb = mlxcel_core::reshape(&rotary_pos_emb, &[rope_shape[0], -1]);

        // 4. Compute cu_seqlens (h * w per frame)
        let cu_seqlens = Self::compute_cu_seqlens(grid_thw);

        // 5. Run vision blocks and extract deepstack features
        let mut deepstack_features: Vec<UniquePtr<MlxArray>> = Vec::new();

        for (layer_num, block) in self.blocks.iter().enumerate() {
            h = block.forward(&h, &cu_seqlens, &rotary_pos_emb);

            if let Some(ds_idx) = self
                .deepstack_visual_indexes
                .iter()
                .position(|&idx| idx == layer_num)
            {
                let ds_feature = self.deepstack_merger_list[ds_idx].forward(&h);
                deepstack_features.push(ds_feature);
            }
        }

        // 6. Apply main merger
        h = self.merger.forward(&h);

        Qwen3VLVisionEncoderOutput {
            hidden_states: h,
            deepstack_features,
        }
    }
}
