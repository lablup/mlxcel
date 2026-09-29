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

//! Mage-ViT, the vision tower of Mage-VL (`mage_vl`).
//!
//! Port of `MageVLVisionPretrainedModel` in the hub's `modeling_mage_vl.py`
//! (cross-checked against mlx-vlm's `mage_vl/vision.py`) for still images:
//!
//! 1. Patch embedding: a bias-free 16x16 stride-16 conv, applied as a Linear
//!    over each `[C*P*P]` processor row.
//! 2. `layernorm_pre`.
//! 3. 24 pre-norm layers: fused `qkv` (contiguous q|k|v split), interleaved
//!    4:6:6 3D rotary in f32 (see [`super::mage_vl_rope`]), unmasked SDPA
//!    within one attention window, `proj`; then `fc1 -> exact GELU -> fc2`.
//! 4. No post-layernorm (`use_head = false`); the merger reads the LAST hidden
//!    state, whatever upstream's "second-to-last" comment says.
//! 5. Merger: `LayerNorm(ln_q)`, reshape each 4-row merge cell into one
//!    `4 * hidden` vector, `Linear -> exact GELU -> Linear` to the decoder
//!    width.
//!
//! Every attention window (one per image; a clip longer than
//! `frame_windows_size` frames would split further) goes through the tower on
//! its own, which is exactly upstream's block-diagonal `cu_seqlens` attention
//! because every other op is row-wise.
//!
//! Used by: `vision::mage_vl::MageVlModel`, `loading::load_mage_vl`.

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::mage_vl_rope::{
    apply_rotary, attention_windows, inv_freqs, positions_from_grid, rotary_angles,
};
use crate::vision::mage_vl_config::MageVlVisionConfig;

fn load_layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {prefix}.weight"))?;
    let bias = weights
        .get(&format!("{prefix}.bias"))
        .map(|b| mlxcel_core::copy(b));
    Ok(LayerNorm::new(weight, bias, eps))
}

/// Bring the patch-embedding kernel to the `[out, C*P*P]` Linear layout that
/// matches the processor's `(c, dy, dx)` row order.
///
/// Accepts the hub's torch layout `[out, C, P, P]` and the channels-last
/// `[out, P, P, C]` an MLX conversion stores (the published 8-bit checkpoint
/// ships `[1024, 16, 16, 3]`); the channel axis is the one equal to
/// `num_channels`. A kernel already flattened to `[out, C*P*P]` passes through.
pub fn patch_kernel_as_linear(
    weight: &MlxArray,
    num_channels: usize,
    patch_size: usize,
) -> Result<UniquePtr<MlxArray>, String> {
    let shape = mlxcel_core::array_shape(weight);
    let c = num_channels as i32;
    let p = patch_size as i32;
    let flat = c * p * p;
    match shape.as_slice() {
        [out, a, b, d] if *a == c && *b == p && *d == p => {
            Ok(mlxcel_core::reshape(weight, &[*out, flat]))
        }
        [out, a, b, d] if *a == p && *b == p && *d == c => {
            let torch = mlxcel_core::transpose_axes(weight, &[0, 3, 1, 2]);
            Ok(mlxcel_core::reshape(&torch, &[*out, flat]))
        }
        [_, width] if *width == flat => Ok(mlxcel_core::copy(weight)),
        other => Err(format!(
            "Unexpected Mage-ViT patch_embedding.weight shape {other:?} (expected \
             [out, {c}, {p}, {p}] or [out, {p}, {p}, {c}])"
        )),
    }
}

struct MageVlAttention {
    qkv: UnifiedLinear,
    proj: UnifiedLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl MageVlAttention {
    /// `x`: `[1, L, hidden]`; `cos` / `sin`: f32 `[1, 1, L, head_dim]`.
    fn forward(&self, x: &MlxArray, cos: &MlxArray, sin: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);
        let (h, d) = (self.num_heads, self.head_dim);

        // [B, L, 3*H*D] -> [B, L, 3, H, D] -> [3, B, H, L, D]
        let qkv = self.qkv.forward(x);
        let qkv = mlxcel_core::reshape(&qkv, &[b, l, 3, h, d]);
        let qkv = mlxcel_core::transpose_axes(&qkv, &[2, 0, 3, 1, 4]);
        let take = |i: i32| {
            let part = mlxcel_core::slice(&qkv, &[i, 0, 0, 0, 0], &[i + 1, b, h, l, d]);
            mlxcel_core::squeeze_axis(&part, 0)
        };
        let (q, k, v) = (take(0), take(1), take(2));

        let q = apply_rotary(&q, cos, sin);
        let k = apply_rotary(&k, cos, sin);

        // SAFETY: the mask pointer is null, which `attention_from_ptr` accepts.
        let out = unsafe {
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
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[b, l, h * d]);
        self.proj.forward(&out)
    }
}

struct MageVlLayer {
    layer_norm1: LayerNorm,
    attn: MageVlAttention,
    layer_norm2: LayerNorm,
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
}

impl MageVlLayer {
    fn forward(&self, x: &MlxArray, cos: &MlxArray, sin: &MlxArray) -> UniquePtr<MlxArray> {
        let attn = self.attn.forward(&self.layer_norm1.forward(x), cos, sin);
        let h = mlxcel_core::add(x, &attn);
        let mlp = self.fc1.forward(&self.layer_norm2.forward(&h));
        let mlp = self.fc2.forward(&mlxcel_core::gelu(&mlp));
        mlxcel_core::add(&h, &mlp)
    }
}

struct MageVlPatchMerger {
    ln_q: LayerNorm,
    mlp_0: UnifiedLinear,
    mlp_2: UnifiedLinear,
    merged_width: i32,
}

impl MageVlPatchMerger {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.ln_q.forward(x);
        let h = mlxcel_core::reshape(&h, &[-1, self.merged_width]);
        let h = self.mlp_0.forward(&h);
        self.mlp_2.forward(&mlxcel_core::gelu(&h))
    }
}

/// Mage-ViT tower plus its 2x2 patch merger.
pub struct MageVlVisionEncoder {
    /// `[hidden, C*P*P]`, the conv kernel in Linear form.
    patch_weight: UniquePtr<MlxArray>,
    layernorm_pre: LayerNorm,
    layers: Vec<MageVlLayer>,
    merger: MageVlPatchMerger,
    inv_freq: [Vec<f32>; 3],
    spatial_merge_size: usize,
    frame_windows_size: usize,
    hidden_size: usize,
}

impl MageVlVisionEncoder {
    /// Load the tower from `weights` under `prefix` (the loader remaps both
    /// the hub `model.visual.*` and the converted `vision_tower.*` spellings
    /// to `vision_tower`).
    pub fn from_weights(
        weights: &WeightMap,
        config: &MageVlVisionConfig,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        config.validate()?;
        let eps = config.layer_norm_eps;
        let linear = |name: String| UnifiedLinear::from_weights(weights, &name, group_size, bits);

        let patch_key = format!("{prefix}.embeddings.patch_embedding.weight");
        let patch_raw = weights
            .get(&patch_key)
            .ok_or_else(|| format!("Weight not found: {patch_key}"))?;
        let patch_weight =
            patch_kernel_as_linear(patch_raw, config.num_channels, config.patch_size)?;

        let head_dim = config.head_dim();
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("{prefix}.encoder.layers.{i}");
            layers.push(MageVlLayer {
                layer_norm1: load_layer_norm(weights, &format!("{p}.layer_norm1"), eps)?,
                attn: MageVlAttention {
                    qkv: linear(format!("{p}.self_attn.qkv"))?,
                    proj: linear(format!("{p}.self_attn.proj"))?,
                    num_heads: config.num_attention_heads as i32,
                    head_dim: head_dim as i32,
                    scale: (head_dim as f32).powf(-0.5),
                },
                layer_norm2: load_layer_norm(weights, &format!("{p}.layer_norm2"), eps)?,
                fc1: linear(format!("{p}.mlp.fc1"))?,
                fc2: linear(format!("{p}.mlp.fc2"))?,
            });
        }

        let merge_cells = config.spatial_merge_size * config.spatial_merge_size;
        let merger = MageVlPatchMerger {
            ln_q: load_layer_norm(weights, &format!("{prefix}.merger.ln_q"), eps)?,
            mlp_0: linear(format!("{prefix}.merger.mlp.0"))?,
            mlp_2: linear(format!("{prefix}.merger.mlp.2"))?,
            merged_width: (config.hidden_size * merge_cells) as i32,
        };

        Ok(Self {
            patch_weight,
            layernorm_pre: load_layer_norm(weights, &format!("{prefix}.layernorm_pre"), eps)?,
            layers,
            merger,
            inv_freq: inv_freqs(head_dim, config.rope_theta),
            spatial_merge_size: config.spatial_merge_size,
            frame_windows_size: config.frame_windows_size,
            hidden_size: config.hidden_size,
        })
    }

    /// Dtype the tower's weights are stored in; patch rows are cast to it
    /// before the embedding, as upstream does.
    #[must_use]
    pub fn weight_dtype(&self) -> i32 {
        mlxcel_core::array_dtype(&self.patch_weight)
    }

    /// Encode processor rows into merged decoder-width features.
    ///
    /// `patches`: `[sum(t*h*w), C*P*P]` in merge-block order (the output of
    /// `Qwen2VLProcessor::preprocess_with_grid` with `temporal_patch_size = 1`).
    /// Returns `[sum(t*h*w) / merge^2, out_hidden_size]`, one row per
    /// `<|image_pad|>` in media order.
    pub fn forward(
        &self,
        patches: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> Result<UniquePtr<MlxArray>, String> {
        let merge = self.spatial_merge_size as i32;
        for &(t, h, w) in grid_thw {
            if t <= 0 || h <= 0 || w <= 0 || h % merge != 0 || w % merge != 0 {
                return Err(format!(
                    "Mage-ViT grid ({t}, {h}, {w}) is not a positive multiple of the \
                     {merge}x{merge} merge cell"
                ));
            }
        }
        let positions = positions_from_grid(grid_thw, self.spatial_merge_size);
        let rows = mlxcel_core::array_shape(patches)[0] as usize;
        if rows != positions.len() {
            return Err(format!(
                "Mage-ViT got {rows} patch rows for grids {grid_thw:?}, which need {}",
                positions.len()
            ));
        }

        let windows = attention_windows(grid_thw, self.frame_windows_size);
        let mut outputs: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(windows.len());
        let mut start = 0usize;
        for len in windows {
            let end = start + len;
            let width = mlxcel_core::array_shape(patches)[1];
            let rows = mlxcel_core::slice(patches, &[start as i32, 0], &[end as i32, width]);
            outputs.push(self.forward_window(&rows, &positions[start..end]));
            start = end;
        }
        let mut merged = outputs.remove(0);
        for next in &outputs {
            merged = mlxcel_core::concatenate(&merged, next, 0);
        }
        Ok(merged)
    }

    /// One attention window: `[L, C*P*P]` rows and their positions.
    fn forward_window(&self, rows: &MlxArray, positions: &[[i32; 3]]) -> UniquePtr<MlxArray> {
        let l = positions.len() as i32;
        let head_dim = self.inv_freq.iter().map(Vec::len).sum::<usize>() as i32 * 2;

        let rows = mlxcel_core::astype(rows, self.weight_dtype());
        let x = mlxcel_core::matmul(&rows, &mlxcel_core::transpose(&self.patch_weight));
        let x = mlxcel_core::reshape(&x, &[1, l, self.hidden_size as i32]);
        let mut x = self.layernorm_pre.forward(&x);

        let angles = rotary_angles(positions, &self.inv_freq);
        let angles = mlxcel_core::from_slice_f32(&angles, &[1, 1, l, head_dim]);
        let cos = mlxcel_core::cos(&angles);
        let sin = mlxcel_core::sin(&angles);

        for layer in &self.layers {
            x = layer.forward(&x, &cos, &sin);
        }
        let x = mlxcel_core::reshape(&x, &[l, self.hidden_size as i32]);
        self.merger.forward(&x)
    }
}

#[cfg(test)]
#[path = "mage_vl_tests.rs"]
mod tests;
