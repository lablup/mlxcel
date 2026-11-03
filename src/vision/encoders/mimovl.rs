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

//! MiMo-V2 `mimovl` vision tower.
//!
//! Ported from `mlx_vlm/models/mimo_v2/vision.py`. The tower is a
//! Qwen2.5-VL-shaped ViT with four deliberate differences:
//!
//! * **GQA with an explicit Q/K width.** `qk_channels` (64) is not
//!   `hidden_size / num_heads` (40), so the fused QKV output is 3072 wide
//!   (`(32 + 2 * 8) * 64`) and the attention output is 2048
//!   (`32 * 64`), then projected back to 1280.
//! * **Learned per-head attention sinks** on windowed blocks. Upstream adds
//!   each head's sink logit to the attention logit of key position 0, so the
//!   sink participates in the softmax without consuming a key slot. This is
//!   not a standard denominator-only sink and the port reproduces it exactly.
//! * **Banded window attention** (`|i - j| > visual_token_window_size`) inside
//!   each frame's `cu_seqlens` segment rather than Qwen's reordered windows.
//! * **Column-major token order** on `vit_window_attn_types == 1` blocks. The
//!   permutation is applied to whole spatial-merge blocks and restored before
//!   any non-`1` block, with column rotary frequencies while it is active.
//!
//! The `patch_embed` is a Conv3d whose kernel equals its stride. MLX stores
//! the converted weight either 2-D (`[out, C*T*P*P]`, upstream's sanitized
//! form) or 5-D; both are reordered into the temporal-major input order
//! `Qwen2VLProcessor` emits (`T, C, P, P` per patch).

use super::VisionEncoderOutput;
use super::qwen2_vl::{VisionRotaryEmbedding, concat_many};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

fn d_model_type() -> String {
    "mimovl".to_string()
}
fn d_depth() -> usize {
    28
}
fn d_hidden_size() -> usize {
    1280
}
fn d_intermediate_size() -> usize {
    4608
}
fn d_num_heads() -> usize {
    32
}
fn d_num_kv_heads() -> usize {
    8
}
fn d_in_channels() -> usize {
    3
}
fn d_patch_size() -> usize {
    16
}
fn d_spatial_merge_size() -> usize {
    2
}
fn d_temporal_patch_size() -> usize {
    2
}
fn d_qk_channels() -> usize {
    64
}
fn d_rms_norm_eps() -> f32 {
    1e-6
}
fn d_true() -> bool {
    true
}
fn d_window_size() -> usize {
    128
}
fn d_visual_token_window_size() -> usize {
    64
}
fn d_fullatt_block_indexes() -> Vec<usize> {
    vec![0, 9, 18, 27]
}
fn d_silu() -> String {
    "silu".to_string()
}

/// `vision_config` from a MiMo-V2 `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct MimovlVisionConfig {
    #[serde(default = "d_model_type")]
    pub model_type: String,
    #[serde(default = "d_depth")]
    pub depth: usize,
    #[serde(default = "d_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "d_intermediate_size")]
    pub intermediate_size: usize,
    /// Width of the merger output, which is the text hidden size (4096).
    #[serde(default)]
    pub out_hidden_size: usize,
    #[serde(default = "d_num_heads")]
    pub num_heads: usize,
    #[serde(default = "d_num_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub num_query_groups: usize,
    /// The released configs spell this `in_chans`.
    #[serde(alias = "in_chans", default = "d_in_channels")]
    pub in_channels: usize,
    #[serde(default = "d_patch_size")]
    pub patch_size: usize,
    #[serde(default = "d_patch_size")]
    pub spatial_patch_size: usize,
    #[serde(default = "d_spatial_merge_size")]
    pub spatial_merge_size: usize,
    #[serde(default = "d_temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(default)]
    pub tokens_per_second: usize,
    #[serde(default = "d_silu")]
    pub hidden_act: String,
    /// Q/K/V head width. The released configs omit it and 64 is the value
    /// both the checkpoint shape and the reference default use.
    #[serde(default = "d_qk_channels")]
    pub qk_channels: usize,
    #[serde(default = "d_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "d_true")]
    pub use_sink: bool,
    /// Present in the config; the banded mask uses
    /// [`Self::visual_token_window_size`] instead.
    #[serde(default = "d_window_size")]
    pub window_size: usize,
    #[serde(default = "d_visual_token_window_size")]
    pub visual_token_window_size: usize,
    #[serde(default = "d_fullatt_block_indexes")]
    pub fullatt_block_indexes: Vec<usize>,
    /// `1` = column-major token order, `0` = row-major, `-1` = a full-attention
    /// block. Empty means every block is row-major.
    #[serde(default)]
    pub vit_window_attn_types: Vec<i32>,
    #[serde(default)]
    pub quant_group_size: i32,
    #[serde(default)]
    pub quant_bits: i32,
}

impl MimovlVisionConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.depth == 0 || self.hidden_size == 0 || self.num_heads == 0 {
            return Err(
                "MiMo-VL vision_config needs a non-zero depth, hidden_size and num_heads"
                    .to_string(),
            );
        }
        if self.num_key_value_heads == 0 || !self.num_heads.is_multiple_of(self.num_key_value_heads)
        {
            return Err(format!(
                "MiMo-VL vision_config num_heads ({}) must be divisible by \
                 num_key_value_heads ({})",
                self.num_heads, self.num_key_value_heads
            ));
        }
        if self.qk_channels == 0 || self.patch_size == 0 || self.temporal_patch_size == 0 {
            return Err("MiMo-VL vision_config has a zero qk_channels or patch extent".to_string());
        }
        if self.spatial_merge_size == 0 {
            return Err("MiMo-VL vision_config spatial_merge_size must be positive".to_string());
        }
        if self.out_hidden_size == 0 {
            return Err("MiMo-VL vision_config out_hidden_size must be positive".to_string());
        }
        if self.hidden_act != "silu" {
            return Err(format!(
                "MiMo-VL vision_config hidden_act {:?} is not implemented; the tower is SiLU",
                self.hidden_act
            ));
        }
        for &index in &self.fullatt_block_indexes {
            if index >= self.depth {
                return Err(format!(
                    "MiMo-VL vision_config fullatt_block_indexes contains {index}, but the tower \
                     has {} blocks",
                    self.depth
                ));
            }
        }
        if !self.vit_window_attn_types.is_empty() && self.vit_window_attn_types.len() != self.depth
        {
            return Err(format!(
                "MiMo-VL vision_config vit_window_attn_types has {} entries for {} blocks; it is \
                 the per-block token-ordering schedule",
                self.vit_window_attn_types.len(),
                self.depth
            ));
        }
        Ok(())
    }

    fn attn_types(&self) -> Vec<i32> {
        if self.vit_window_attn_types.is_empty() {
            vec![-1; self.depth]
        } else {
            self.vit_window_attn_types.clone()
        }
    }
}

#[path = "mimovl_layers.rs"]
mod layers;

use layers::{PatchEmbed, PatchMerger, VisionBlock};

/// MiMo-V2 `mimovl` vision tower.
pub struct MimovlVisionEncoder {
    config: MimovlVisionConfig,
    patch_embed: PatchEmbed,
    rotary_pos_emb: VisionRotaryEmbedding,
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
    attn_types: Vec<i32>,
}

impl MimovlVisionEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        config: &MimovlVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        config.validate()?;
        let gs = config.quant_group_size;
        let bits = config.quant_bits;
        let patch_embed = PatchEmbed::from_weights(weights, config, prefix)?;
        let rotary_pos_emb = VisionRotaryEmbedding::new(config.qk_channels / 2);
        let mut blocks = Vec::with_capacity(config.depth);
        for index in 0..config.depth {
            let full = config.fullatt_block_indexes.contains(&index);
            blocks.push(VisionBlock::from_weights(
                weights,
                config,
                &format!("{prefix}.blocks.{index}"),
                config.use_sink && !full,
                gs,
                bits,
            )?);
        }
        let merger =
            PatchMerger::from_weights(weights, config, &format!("{prefix}.merger"), gs, bits)?;
        Ok(Self {
            config: config.clone(),
            patch_embed,
            rotary_pos_emb,
            blocks,
            merger,
            attn_types: config.attn_types(),
        })
    }

    pub fn spatial_merge_size(&self) -> usize {
        self.config.spatial_merge_size
    }

    /// Output rows for a batch of media, so prompt expansion can check the
    /// placeholder count before running the tower.
    pub fn merged_token_count(&self, grid_thw: &[(i32, i32, i32)]) -> usize {
        let unit = self.config.spatial_merge_size * self.config.spatial_merge_size;
        grid_thw
            .iter()
            .map(|&(t, h, w)| ((t as usize) * (h as usize) * (w as usize)) / unit.max(1))
            .sum()
    }

    /// Column-major (window) ordering of the spatial-merge blocks.
    ///
    /// `arange(t*gh*gw).reshape(t, gh, gw).transpose(0, 2, 1)` in upstream;
    /// computed on the host because the value is a permutation, not a tensor.
    fn window_index(&self, grid_thw: &[(i32, i32, i32)]) -> Result<Vec<i32>, String> {
        let merge = self.config.spatial_merge_size as i32;
        let mut out = Vec::new();
        let mut offset = 0i32;
        for &(t, h, w) in grid_thw {
            if h <= 0 || w <= 0 || h % merge != 0 || w % merge != 0 {
                return Err(format!(
                    "MiMo-VL grid ({t}, {h}, {w}) is not divisible by spatial_merge_size {merge}"
                ));
            }
            let gh = h / merge;
            let gw = w / merge;
            for temporal in 0..t {
                for x in 0..gw {
                    for y in 0..gh {
                        out.push(offset + temporal * gh * gw + y * gw + x);
                    }
                }
            }
            offset += t * gh * gw;
        }
        Ok(out)
    }

    /// 2-D rotary frequencies for every patch token, in row-major order.
    fn rot_pos_emb(&self, grid_thw: &[(i32, i32, i32)]) -> Result<UniquePtr<MlxArray>, String> {
        let merge = self.config.spatial_merge_size as i32;
        let mut all_pos_ids = Vec::new();
        let mut max_grid = 0i32;
        for &(t, h, w) in grid_thw {
            if h <= 0 || w <= 0 || h % merge != 0 || w % merge != 0 {
                return Err(format!(
                    "MiMo-VL grid ({t}, {h}, {w}) is not divisible by spatial_merge_size {merge}"
                ));
            }
            max_grid = max_grid.max(h).max(w);

            let h_arange = mlxcel_core::arange_i32(0, h, 1);
            let h_col = mlxcel_core::reshape(&h_arange, &[h, 1]);
            let hpos = mlxcel_core::broadcast_to(&h_col, &[h, w]);
            let hpos = mlxcel_core::reshape(&hpos, &[h / merge, merge, w / merge, merge]);
            let hpos = mlxcel_core::flatten(&mlxcel_core::transpose_axes(&hpos, &[0, 2, 1, 3]));

            let w_arange = mlxcel_core::arange_i32(0, w, 1);
            let w_row = mlxcel_core::reshape(&w_arange, &[1, w]);
            let wpos = mlxcel_core::broadcast_to(&w_row, &[h, w]);
            let wpos = mlxcel_core::reshape(&wpos, &[h / merge, merge, w / merge, merge]);
            let wpos = mlxcel_core::flatten(&mlxcel_core::transpose_axes(&wpos, &[0, 2, 1, 3]));

            let hpos = mlxcel_core::astype(&hpos, mlxcel_core::dtype::FLOAT32);
            let wpos = mlxcel_core::astype(&wpos, mlxcel_core::dtype::FLOAT32);
            let stacked = mlxcel_core::stack_owned(&[hpos, wpos], -1);
            all_pos_ids.push(mlxcel_core::tile(&stacked, &[t, 1]));
        }
        let pos_ids = if all_pos_ids.len() == 1 {
            all_pos_ids.pop().expect("one grid")
        } else {
            concat_many(&all_pos_ids, 0)
        };
        let table = self.rotary_pos_emb.forward(max_grid);
        let flat = mlxcel_core::flatten(&pos_ids);
        let flat = mlxcel_core::astype(&flat, mlxcel_core::dtype::INT32);
        let freqs = mlxcel_core::take(&table, &flat, 0);
        let tokens = mlxcel_core::array_shape(&pos_ids)[0];
        let half = mlxcel_core::array_shape(&freqs)[1];
        let freqs = mlxcel_core::reshape(&freqs, &[tokens, 2, half]);
        Ok(mlxcel_core::reshape(&freqs, &[tokens, 2 * half]))
    }

    fn cu_seqlens(grid_thw: &[(i32, i32, i32)]) -> Vec<i32> {
        let mut out = vec![0i32];
        let mut total = 0i32;
        for &(t, h, w) in grid_thw {
            for _ in 0..t {
                total += h * w;
                out.push(total);
            }
        }
        out
    }

    fn apply_index(&self, x: &MlxArray, index: &MlxArray, rows: i32) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let unit = (self.config.spatial_merge_size * self.config.spatial_merge_size) as i32;
        let grouped = mlxcel_core::reshape(x, &[rows / unit, unit, shape[1]]);
        let taken = mlxcel_core::take(&grouped, index, 0);
        mlxcel_core::reshape(&taken, &[-1, shape[1]])
    }

    pub fn forward_with_grid(
        &self,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> Result<VisionEncoderOutput, String> {
        if grid_thw.is_empty() {
            return Err("MiMo-VL vision forward needs at least one grid entry".to_string());
        }
        let mut h = self.patch_embed.forward(pixel_values)?;
        let rows = mlxcel_core::array_shape(&h)[0];
        let expected: i32 = grid_thw.iter().map(|&(t, hh, ww)| t * hh * ww).sum();
        if rows != expected {
            return Err(format!(
                "MiMo-VL patch embedding produced {rows} tokens for grids {grid_thw:?} which \
                 contain {expected}"
            ));
        }

        let rotary = self.rot_pos_emb(grid_thw)?;
        let col_index = self.window_index(grid_thw)?;
        let col_index_arr = mlxcel_core::from_slice_i32(&col_index, &[col_index.len() as i32]);
        let mut reverse = vec![0i32; col_index.len()];
        for (position, &original) in col_index.iter().enumerate() {
            let original = usize::try_from(original)
                .map_err(|_| "MiMo-VL window index is negative".to_string())?;
            if original >= reverse.len() {
                return Err("MiMo-VL window index is out of range".to_string());
            }
            reverse[original] = position as i32;
        }
        let reverse_arr = mlxcel_core::from_slice_i32(&reverse, &[reverse.len() as i32]);
        let col_rotary = self.apply_index(&rotary, &col_index_arr, rows);
        let cu_seqlens = Self::cu_seqlens(grid_thw);

        for (index, block) in self.blocks.iter().enumerate() {
            let kind = self.attn_types[index];
            if kind == 1 && (index == 0 || self.attn_types[index - 1] != 1) {
                h = self.apply_index(&h, &col_index_arr, rows);
            } else if index > 0 && kind != 1 && self.attn_types[index - 1] == 1 {
                h = self.apply_index(&h, &reverse_arr, rows);
            }
            let freqs = if kind == 1 { &col_rotary } else { &rotary };
            h = block.forward(
                &h,
                &cu_seqlens,
                freqs,
                self.config.fullatt_block_indexes.contains(&index),
            );
        }

        let hidden_states = self.merger.forward(&h);
        Ok(VisionEncoderOutput { hidden_states })
    }
}

#[cfg(test)]
#[path = "mimovl_tests.rs"]
mod tests;
