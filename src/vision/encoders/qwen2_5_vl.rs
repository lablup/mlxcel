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

//! Qwen2.5-VL Vision Encoder
//!
//! Evolution of Qwen2-VL vision encoder with:
//! - RMSNorm instead of LayerNorm
//! - SwiGLU MLP (gate_proj/up_proj/down_proj with SiLU) instead of GELU fc1/fc2
//! - Windowed attention with fullatt_block_indexes for selective full attention
//! - PatchMerger with RMSNorm
//!
//! Used by: Qwen2.5-VL
//! Reference: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen2_5_vl/vision.py

use super::VisionEncoderOutput;
use super::qwen2_vl::{VisionRotaryEmbedding, apply_rotary_pos_emb_vision, concat_many};
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

/// Qwen2.5-VL vision encoder configuration
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen25VLVisionConfig {
    #[serde(default = "default_depth")]
    pub depth: usize,
    /// Vision hidden size (replaces embed_dim in Qwen2-VL)
    pub hidden_size: usize,
    /// Explicit MLP intermediate size (replaces mlp_ratio)
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    /// Output hidden size (projection to text space)
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
    /// Window size for windowed attention
    #[serde(default = "default_window_size")]
    pub window_size: usize,
    /// Block indices that use full attention (rest use windowed)
    #[serde(default = "default_fullatt_block_indexes")]
    pub fullatt_block_indexes: Vec<usize>,
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
fn default_window_size() -> usize {
    112
}
fn default_fullatt_block_indexes() -> Vec<usize> {
    vec![7, 15, 23, 31]
}

// RMSNorm for vision encoder.
struct VisionRMSNorm {
    weight: UniquePtr<MlxArray>,
    eps: f32,
}

impl VisionRMSNorm {
    fn from_weights(weights: &WeightMap, prefix: &str, eps: f32) -> Result<Self, String> {
        let weight_key = format!("{}.weight", prefix);
        let weight = weights
            .get(&weight_key)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {}", weight_key))?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::rms_norm(x, &self.weight, self.eps)
    }
}

// PatchEmbed - Conv3d degenerated to Linear (same as Qwen2-VL).
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
        config: &Qwen25VLVisionConfig,
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
            // Input data is in TCHW order (temporal, channel, height, width)
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

// Vision Attention - same as Qwen2-VL.
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
        config: &Qwen25VLVisionConfig,
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

            let attn = unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q_seg,
                    &k_seg,
                    &v_seg,
                    self.scale,
                    std::ptr::null(),
                    0.0,
                    0,
                )
            };
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

// Vision MLP - SwiGLU (gate_proj/up_proj/down_proj with SiLU).
struct VisionMLP {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl VisionMLP {
    fn from_weights(weights: &WeightMap, prefix: &str, gs: i32, bits: i32) -> Result<Self, String> {
        Ok(Self {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.mlp.gate_proj", prefix),
                gs,
                bits,
            )?,
            up_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.mlp.up_proj", prefix),
                gs,
                bits,
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.mlp.down_proj", prefix),
                gs,
                bits,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.gate_proj.forward(x);
        let gate = mlxcel_core::silu(&gate);
        let up = self.up_proj.forward(x);
        let h = mlxcel_core::multiply(&gate, &up);
        self.down_proj.forward(&h)
    }
}

// VisionBlock - RMSNorm + SwiGLU MLP.
struct VisionBlock {
    norm1: VisionRMSNorm,
    norm2: VisionRMSNorm,
    attn: VisionAttention,
    mlp: VisionMLP,
}

impl VisionBlock {
    fn from_weights(
        weights: &WeightMap,
        config: &Qwen25VLVisionConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            norm1: VisionRMSNorm::from_weights(weights, &format!("{}.norm1", prefix), 1e-6)?,
            norm2: VisionRMSNorm::from_weights(weights, &format!("{}.norm2", prefix), 1e-6)?,
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

// PatchMerger - RMSNorm + GELU MLP (projection to text hidden size).
struct PatchMerger {
    ln_q: VisionRMSNorm,
    mlp_0: UnifiedLinear,
    mlp_2: UnifiedLinear,
    hidden_size: usize,
}

impl PatchMerger {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        context_dim: usize,
        spatial_merge_size: usize,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let hidden_size = context_dim * spatial_merge_size * spatial_merge_size;
        Ok(Self {
            ln_q: VisionRMSNorm::from_weights(weights, &format!("{}.ln_q", prefix), 1e-6)?,
            mlp_0: UnifiedLinear::from_weights(weights, &format!("{}.mlp.0", prefix), gs, bits)?,
            mlp_2: UnifiedLinear::from_weights(weights, &format!("{}.mlp.2", prefix), gs, bits)?,
            hidden_size,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.ln_q.forward(x);
        let h = mlxcel_core::reshape(&h, &[-1, self.hidden_size as i32]);
        let h = self.mlp_0.forward(&h);
        let h = mlxcel_core::gelu(&h);
        self.mlp_2.forward(&h)
    }
}

// Qwen2.5-VL Vision Encoder.
/// Qwen2.5-VL Vision Model with windowed attention
///
/// Used by: Qwen2.5-VL
pub struct Qwen25VLVisionEncoder {
    patch_embed: PatchEmbed,
    rotary_pos_emb: VisionRotaryEmbedding,
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
    spatial_merge_size: usize,
    window_size: usize,
    patch_size: usize,
    fullatt_block_indexes: Vec<usize>,
}

impl Qwen25VLVisionEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        config: &Qwen25VLVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.quant_group_size;
        let bits = config.quant_bits;

        let patch_embed =
            PatchEmbed::from_weights(weights, config, &format!("{}.patch_embed", prefix))?;
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

        let merger = PatchMerger::from_weights(
            weights,
            &format!("{}.merger", prefix),
            config.hidden_size,
            config.spatial_merge_size,
            gs,
            bits,
        )?;

        Ok(Self {
            patch_embed,
            rotary_pos_emb,
            blocks,
            merger,
            spatial_merge_size: config.spatial_merge_size,
            window_size: config.window_size,
            patch_size: config.patch_size,
            fullatt_block_indexes: config.fullatt_block_indexes.clone(),
        })
    }

    /// Compute 2D rotary position embeddings from grid_thw (same as Qwen2-VL)
    fn rot_pos_emb(&self, grid_thw: &[(i32, i32, i32)]) -> UniquePtr<MlxArray> {
        let mut all_pos_ids: Vec<UniquePtr<MlxArray>> = Vec::new();
        let mut max_grid_dim: i32 = 0;

        for &(t, h, w) in grid_thw {
            if h > max_grid_dim {
                max_grid_dim = h;
            }
            if w > max_grid_dim {
                max_grid_dim = w;
            }
            let merge = self.spatial_merge_size as i32;

            let h_arange = mlxcel_core::arange_i32(0, h, 1);
            let h_col = mlxcel_core::reshape(&h_arange, &[h, 1]);
            let hpos = mlxcel_core::repeat(&h_col, w, 1);
            let hpos = mlxcel_core::reshape(&hpos, &[h / merge, merge, w / merge, merge]);
            let hpos = mlxcel_core::transpose_axes(&hpos, &[0, 2, 1, 3]);
            let hpos = mlxcel_core::flatten(&hpos);

            let w_arange = mlxcel_core::arange_i32(0, w, 1);
            let w_row = mlxcel_core::reshape(&w_arange, &[1, w]);
            let wpos = mlxcel_core::repeat(&w_row, h, 0);
            let wpos = mlxcel_core::reshape(&wpos, &[h / merge, merge, w / merge, merge]);
            let wpos = mlxcel_core::transpose_axes(&wpos, &[0, 2, 1, 3]);
            let wpos = mlxcel_core::flatten(&wpos);

            let stacked = mlxcel_core::stack_owned(&[hpos, wpos], -1);
            let tiled = mlxcel_core::tile(&stacked, &[t, 1]);
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

    /// Compute windowed attention indices and cu_seqlens
    ///
    /// Returns (window_index, cu_window_seqlens) where:
    /// - window_index: reordering of merged patches into window groups
    /// - cu_window_seqlens: cumulative sequence lengths for windowed attention
    fn get_window_index(&self, grid_thw: &[(i32, i32, i32)]) -> (Vec<i32>, Vec<i32>) {
        let spatial_merge_unit = (self.spatial_merge_size * self.spatial_merge_size) as i32;
        let vit_merger_window_size =
            (self.window_size / self.spatial_merge_size / self.patch_size) as i32;

        let mut window_index: Vec<i32> = Vec::new();
        let mut cu_window_seqlens: Vec<i32> = vec![0];
        let mut window_index_id: i32 = 0;

        for &(grid_t, grid_h, grid_w) in grid_thw {
            let llm_grid_h = grid_h / self.spatial_merge_size as i32;
            let llm_grid_w = grid_w / self.spatial_merge_size as i32;

            let total = grid_t * llm_grid_h * llm_grid_w;

            // Create index array [0..total)
            let index_3d: Vec<i32> = (0..total).collect();

            // Compute padding
            let pad_h = if llm_grid_h % vit_merger_window_size == 0 {
                0
            } else {
                vit_merger_window_size - llm_grid_h % vit_merger_window_size
            };
            let pad_w = if llm_grid_w % vit_merger_window_size == 0 {
                0
            } else {
                vit_merger_window_size - llm_grid_w % vit_merger_window_size
            };
            let num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
            let num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;
            let padded_h = llm_grid_h + pad_h;
            let padded_w = llm_grid_w + pad_w;

            // Pad index to [grid_t, padded_h, padded_w] with -100
            let mut index_padded = vec![-100i32; (grid_t * padded_h * padded_w) as usize];
            for ti in 0..grid_t {
                for hi in 0..llm_grid_h {
                    for wi in 0..llm_grid_w {
                        let src_idx =
                            (ti * llm_grid_h * llm_grid_w + hi * llm_grid_w + wi) as usize;
                        let dst_idx = (ti * padded_h * padded_w + hi * padded_w + wi) as usize;
                        index_padded[dst_idx] = index_3d[src_idx];
                    }
                }
            }

            // Reshape to [grid_t, num_windows_h, ws, num_windows_w, ws]
            // Then transpose to [grid_t, num_windows_h, num_windows_w, ws, ws]
            // Then reshape to [grid_t, num_windows_h*num_windows_w, ws, ws]
            let ws = vit_merger_window_size;
            let mut reordered =
                vec![-100i32; (grid_t * num_windows_h * num_windows_w * ws * ws) as usize];

            for ti in 0..grid_t {
                for wh in 0..num_windows_h {
                    for ww in 0..num_windows_w {
                        for sh in 0..ws {
                            for sw in 0..ws {
                                let src_h = wh * ws + sh;
                                let src_w = ww * ws + sw;
                                let src =
                                    (ti * padded_h * padded_w + src_h * padded_w + src_w) as usize;
                                let win_idx = wh * num_windows_w + ww;
                                let dst = (ti * num_windows_h * num_windows_w * ws * ws
                                    + win_idx * ws * ws
                                    + sh * ws
                                    + sw) as usize;
                                reordered[dst] = index_padded[src];
                            }
                        }
                    }
                }
            }

            // Compute seqlens per window (count non-padding entries)
            let num_windows = (grid_t * num_windows_h * num_windows_w) as usize;
            let ws2 = (ws * ws) as usize;
            let mut seqlens: Vec<i32> = Vec::with_capacity(num_windows);
            for win in 0..num_windows {
                let mut count = 0i32;
                for j in 0..ws2 {
                    if reordered[win * ws2 + j] != -100 {
                        count += 1;
                    }
                }
                seqlens.push(count);
            }

            // Extract non-padding indices in order
            let mut valid_indices: Vec<i32> = Vec::new();
            for &val in &reordered {
                if val != -100 {
                    valid_indices.push(val + window_index_id);
                }
            }
            window_index.extend_from_slice(&valid_indices);

            // Compute cu_window_seqlens
            let last_cum = *cu_window_seqlens.last().unwrap();
            let mut cum = last_cum;
            for &sl in &seqlens {
                cum += sl * spatial_merge_unit;
                cu_window_seqlens.push(cum);
            }

            window_index_id += total;
        }

        // Deduplicate cu_window_seqlens (remove consecutive duplicates)
        let mut deduped: Vec<i32> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for &val in &cu_window_seqlens {
            if seen.insert(val) {
                deduped.push(val);
            }
        }

        (window_index, deduped)
    }

    /// Compute cu_seqlens for full attention from grid_thw
    /// Returns cumulative counts in pre-merge token space (multiplied by spatial_merge_unit)
    fn compute_cu_seqlens(grid_thw: &[(i32, i32, i32)], spatial_merge_size: i32) -> Vec<i32> {
        let spatial_merge_unit = spatial_merge_size * spatial_merge_size;
        let mut cu_seqlens = vec![0i32];
        let mut cumulative = 0i32;
        for &(t, h, w) in grid_thw {
            let merged_h = h / spatial_merge_size;
            let merged_w = w / spatial_merge_size;
            // Each merged patch corresponds to spatial_merge_unit pre-merge tokens
            let tokens_per_frame = merged_h * merged_w * spatial_merge_unit;
            for _ in 0..t {
                cumulative += tokens_per_frame;
                cu_seqlens.push(cumulative);
            }
        }
        cu_seqlens
    }

    /// Forward pass with windowed attention
    pub fn forward_with_grid(
        &self,
        hidden_states: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> VisionEncoderOutput {
        let mut h = self.patch_embed.forward(hidden_states);
        let rotary_pos_emb = self.rot_pos_emb(grid_thw);

        let (window_index, cu_window_seqlens) = self.get_window_index(grid_thw);

        let spatial_merge_unit = (self.spatial_merge_size * self.spatial_merge_size) as i32;

        let shape = mlxcel_core::array_shape(&h);
        let seq_len = shape[0];
        let dim = shape[1];
        // Reorder hidden states by window index
        // [seq, dim] -> [seq/merge_unit, merge_unit, dim]
        let h_grouped =
            mlxcel_core::reshape(&h, &[seq_len / spatial_merge_unit, spatial_merge_unit, dim]);
        let window_idx_arr =
            mlxcel_core::from_slice_i32(&window_index, &[window_index.len() as i32]);
        let h_reordered = mlxcel_core::take(&h_grouped, &window_idx_arr, 0);
        h = mlxcel_core::reshape(&h_reordered, &[-1, dim]);

        // Reorder rotary_pos_emb similarly
        let rope_shape = mlxcel_core::array_shape(&rotary_pos_emb);
        let rope_dim = rope_shape[1];
        let rope_grouped = mlxcel_core::reshape(
            &rotary_pos_emb,
            &[seq_len / spatial_merge_unit, spatial_merge_unit, rope_dim],
        );
        let rope_reordered = mlxcel_core::take(&rope_grouped, &window_idx_arr, 0);
        let rotary_pos_emb = mlxcel_core::reshape(&rope_reordered, &[-1, rope_dim]);

        // Compute full-attention cu_seqlens
        let cu_seqlens = Self::compute_cu_seqlens(grid_thw, self.spatial_merge_size as i32);

        // Run blocks with windowed or full attention
        for (layer_num, block) in self.blocks.iter().enumerate() {
            let cu_seqlens_now = if self.fullatt_block_indexes.contains(&layer_num) {
                &cu_seqlens
            } else {
                &cu_window_seqlens
            };
            h = block.forward(&h, cu_seqlens_now, &rotary_pos_emb);
        }

        // Merge patches
        h = self.merger.forward(&h);

        // Un-reorder: apply reverse indices from argsort(window_index)
        let mut reverse_indices = vec![0i32; window_index.len()];
        let mut indexed: Vec<(i32, usize)> = window_index
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i))
            .collect();
        indexed.sort_by_key(|&(v, _)| v);
        for (rank, &(_, orig_idx)) in indexed.iter().enumerate() {
            reverse_indices[orig_idx] = rank as i32;
        }

        // After merger: h has shape [total_merged, out_hidden_size]
        // reverse_indices maps from window-ordered to original order
        let reverse_arr =
            mlxcel_core::from_slice_i32(&reverse_indices, &[reverse_indices.len() as i32]);
        h = mlxcel_core::take(&h, &reverse_arr, 0);

        VisionEncoderOutput { hidden_states: h }
    }
}

/// VisionEncoder trait - panics since grid_thw is required
impl super::VisionEncoder for Qwen25VLVisionEncoder {
    fn forward(&self, _pixel_values: &MlxArray) -> VisionEncoderOutput {
        panic!("Qwen2.5-VL vision encoder requires grid_thw; use forward_with_grid() instead");
    }
}
