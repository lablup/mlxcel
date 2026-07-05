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

//! DFNRope Vision Transformer for ERNIE-4.5-VL (`vision_tower.*`).
//!
//! Nearly the Qwen2-VL ViT skeleton: packed variable-length sequences over
//! `cu_seqlens`, fused-QKV attention, 2D vision RoPE with merge-window patch
//! order and the concat-half rotation. Differences: the patch embedding is a
//! plain Linear over 588-wide rows (`3 * 14 * 14`, no conv-kernel reorder and
//! no bias), the MLP activation is quick_gelu, and there is no in-encoder
//! merger; the tower ends with a plain LayerNorm (`ln`) and the external
//! resampler does all downsampling.
//!
//! Reuses [`super::qwen2_vl::VisionRotaryEmbedding`],
//! [`super::qwen2_vl::apply_rotary_pos_emb_vision`], and
//! [`super::qwen2_vl::concat_many`].
//!
//! Reference: mlx-vlm
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ernie4_5_moe_vl/vision.py>.

use super::qwen2_vl::{VisionRotaryEmbedding, apply_rotary_pos_emb_vision, concat_many};
use super::{VisionEncoder, VisionEncoderOutput};
use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

fn default_depth() -> usize {
    32
}
fn default_embed_dim() -> usize {
    1280
}
fn default_num_heads() -> usize {
    16
}
fn default_mlp_ratio() -> f32 {
    4.0
}
fn default_patch_size() -> usize {
    14
}
fn default_spatial_merge_size() -> usize {
    2
}
fn default_layer_norm_eps() -> f32 {
    1e-6
}

#[derive(Debug, Clone, Deserialize)]
pub struct Ernie45VlVisionConfig {
    #[serde(default = "default_depth")]
    pub depth: usize,
    #[serde(default = "default_embed_dim")]
    pub embed_dim: usize,
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f32,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default)]
    pub quant_group_size: i32,
    #[serde(default)]
    pub quant_bits: i32,
}

fn load_layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    let weight_key = format!("{prefix}.weight");
    let weight = weights
        .get(&weight_key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {weight_key}"))?;
    let bias = weights
        .get(&format!("{prefix}.bias"))
        .map(|b| mlxcel_core::copy(b));
    Ok(LayerNorm::new(weight, bias, eps))
}

// Attention: fused QKV over packed sequences (per-image full attention).
struct VisionAttention {
    qkv: UnifiedLinear,
    proj: UnifiedLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl VisionAttention {
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

        let split = |i: i32| {
            let sl = mlxcel_core::slice(
                &qkv,
                &[i, 0, 0, 0],
                &[i + 1, seq_length, self.num_heads, self.head_dim],
            );
            mlxcel_core::squeeze_axis(&sl, 0)
        };
        let q = split(0);
        let k = split(1);
        let v = split(2);

        let q = apply_rotary_pos_emb_vision(&q, rotary_pos_emb);
        let k = apply_rotary_pos_emb_vision(&k, rotary_pos_emb);

        // [seq, heads, head_dim] -> [1, heads, seq, head_dim]
        let to_bhsd = |t: &MlxArray| {
            let t = mlxcel_core::transpose_axes(t, &[1, 0, 2]);
            mlxcel_core::expand_dims(&t, 0)
        };
        let q = to_bhsd(&q);
        let k = to_bhsd(&k);
        let v = to_bhsd(&v);

        let num_segments = cu_seqlens.len() - 1;
        let mut attn_outputs = Vec::with_capacity(num_segments);
        for seg in 0..num_segments {
            let (start, end) = (cu_seqlens[seg], cu_seqlens[seg + 1]);
            let seg_of = |t: &MlxArray| {
                mlxcel_core::slice(
                    t,
                    &[0, 0, start, 0],
                    &[1, self.num_heads, end, self.head_dim],
                )
            };
            let (q_seg, k_seg, v_seg) = (seg_of(&q), seg_of(&k), seg_of(&v));
            // SAFETY: segment slices are valid arrays; null mask (full attention).
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

struct VisionBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attn: VisionAttention,
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
}

impl VisionBlock {
    fn forward(
        &self,
        x: &MlxArray,
        cu_seqlens: &[i32],
        rotary_pos_emb: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let attn_out = self
            .attn
            .forward(&self.norm1.forward(x), cu_seqlens, rotary_pos_emb);
        let h = mlxcel_core::add(x, &attn_out);
        let m = self.fc1.forward(&self.norm2.forward(&h));
        let m = mlxcel_core::utils::gelu_sigmoid(&m); // quick_gelu
        let m = self.fc2.forward(&m);
        mlxcel_core::add(&h, &m)
    }
}

pub struct Ernie45VlVisionEncoder {
    patch_embed: UnifiedLinear, // Linear (embed_dim, 3 * p * p), no bias
    rotary_pos_emb: VisionRotaryEmbedding,
    blocks: Vec<VisionBlock>,
    ln: LayerNorm,
    spatial_merge_size: usize,
}

impl Ernie45VlVisionEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        config: &Ernie45VlVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.quant_group_size;
        let bits = config.quant_bits;
        let head_dim = config.embed_dim / config.num_heads;

        let patch_embed =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.patch_embed.proj"), gs, bits)?;
        let rotary_pos_emb = VisionRotaryEmbedding::new(head_dim / 2);

        let mut blocks = Vec::with_capacity(config.depth);
        for i in 0..config.depth {
            let bp = format!("{prefix}.blocks.{i}");
            blocks.push(VisionBlock {
                norm1: load_layer_norm(weights, &format!("{bp}.norm1"), config.layer_norm_eps)?,
                norm2: load_layer_norm(weights, &format!("{bp}.norm2"), config.layer_norm_eps)?,
                attn: VisionAttention {
                    qkv: UnifiedLinear::from_weights(weights, &format!("{bp}.attn.qkv"), gs, bits)?,
                    proj: UnifiedLinear::from_weights(
                        weights,
                        &format!("{bp}.attn.proj"),
                        gs,
                        bits,
                    )?,
                    num_heads: config.num_heads as i32,
                    head_dim: head_dim as i32,
                    scale: (head_dim as f32).powf(-0.5),
                },
                fc1: UnifiedLinear::from_weights(weights, &format!("{bp}.mlp.fc1"), gs, bits)?,
                fc2: UnifiedLinear::from_weights(weights, &format!("{bp}.mlp.fc2"), gs, bits)?,
            });
        }

        let ln = load_layer_norm(weights, &format!("{prefix}.ln"), config.layer_norm_eps)?;

        Ok(Self {
            patch_embed,
            rotary_pos_emb,
            blocks,
            ln,
            spatial_merge_size: config.spatial_merge_size,
        })
    }

    /// 2D rotary table lookup with merge-window patch order (same recipe as
    /// Qwen2-VL: position grids reshaped `(h/m, m, w/m, m)` and transposed to
    /// window-major order before flattening).
    fn rot_pos_emb(&self, grid_thw: &[(i32, i32, i32)]) -> UniquePtr<MlxArray> {
        let merge = self.spatial_merge_size as i32;
        let mut all_pos_ids: Vec<UniquePtr<MlxArray>> = Vec::new();
        let mut max_grid_dim: i32 = 0;

        for &(t, h, w) in grid_thw {
            max_grid_dim = max_grid_dim.max(h).max(w);

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
        let total_tokens = mlxcel_core::array_shape(&pos_ids)[0];
        let half_dim = mlxcel_core::array_shape(&all_freqs)[1];
        let all_freqs = mlxcel_core::reshape(&all_freqs, &[total_tokens, 2, half_dim]);
        mlxcel_core::reshape(&all_freqs, &[total_tokens, 2 * half_dim])
    }

    fn compute_cu_seqlens(grid_thw: &[(i32, i32, i32)]) -> Vec<i32> {
        let mut cu_seqlens = vec![0i32];
        let mut cumulative = 0i32;
        for &(t, h, w) in grid_thw {
            let tokens_per_frame = h * w;
            for _ in 0..t {
                cumulative += tokens_per_frame;
                cu_seqlens.push(cumulative);
            }
        }
        cu_seqlens
    }

    /// `hidden_states`: `[total_patches, 3 * p * p]` merge-window-ordered rows.
    /// Returns `[total_patches, embed_dim]` (no in-encoder downsampling).
    pub fn forward_with_grid(
        &self,
        hidden_states: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> VisionEncoderOutput {
        let mut h = self.patch_embed.forward(hidden_states);
        let rotary_pos_emb = self.rot_pos_emb(grid_thw);
        let cu_seqlens = Self::compute_cu_seqlens(grid_thw);

        for block in &self.blocks {
            h = block.forward(&h, &cu_seqlens, &rotary_pos_emb);
        }

        let h = self.ln.forward(&h);
        VisionEncoderOutput { hidden_states: h }
    }
}

impl VisionEncoder for Ernie45VlVisionEncoder {
    fn forward(&self, _pixel_values: &MlxArray) -> VisionEncoderOutput {
        panic!("ERNIE-4.5-VL vision encoder requires grid_thw; use forward_with_grid() instead");
    }
}
