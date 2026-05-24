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

//! MiniCPM-V 4.6 vision encoder.
//!
//! Architecture differences vs MiniCPM-O:
//! - Language backbone: Qwen3.5 (standard 1D RoPE, not Qwen3-VL / MRoPE)
//! - Vision: SigLIP ViT with mid-tower VitMerger inserted after layer 6,
//!   then a pixel-shuffle Merger (not a Perceiver-style resampler).
//!
//! The vision pipeline for one image:
//! 1. SiglipEmbeddings — patch projection + positional embeddings
//! 2. SigLIP encoder — N = 27 layers. After layer 6 (insert_layer_id) the
//!    VitMerger reduces (H,W) by merge_group_size (default 2x2) via
//!    cross-attention pooling.
//! 3. post_layernorm
//! 4. Merger — pixel-shuffle MLP downsampling (default 2x2)
//!
//! Used by: MiniCPMV46VLM loader in vlm_special.rs

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

use crate::vision::encoders::minicpmo::MiniCPMOVisionConfig;

// ── Config ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct MiniCPMV46VisionConfig {
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
    /// Merge group size for VitMerger. Default (2, 2) matches upstream.
    #[serde(default = "default_window_kernel_size")]
    pub window_kernel_size: [usize; 2],
}

fn default_num_channels() -> usize {
    3
}
fn default_image_size() -> usize {
    448
}
fn default_patch_size() -> usize {
    14
}
fn default_layer_norm_eps() -> f32 {
    1e-6
}
fn default_window_kernel_size() -> [usize; 2] {
    [2, 2]
}

/// Convert `MiniCPMV46VisionConfig` into the shared `MiniCPMOVisionConfig`
/// so we can reuse the identical SigLIP encoder blocks.
pub(crate) fn to_minicpmo_vision_config(cfg: &MiniCPMV46VisionConfig) -> MiniCPMOVisionConfig {
    MiniCPMOVisionConfig {
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        num_channels: cfg.num_channels,
        image_size: cfg.image_size,
        patch_size: cfg.patch_size,
        layer_norm_eps: cfg.layer_norm_eps,
    }
}

/// Whole-model config loaded from the MiniCPM-V 4.6 `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct MiniCPMV46Config {
    #[serde(default = "default_insert_layer_id")]
    pub insert_layer_id: usize,
    #[serde(default = "default_merge_kernel_size")]
    pub merge_kernel_size: [usize; 2],
    #[serde(default = "default_merger_times")]
    pub merger_times: usize,
}

fn default_insert_layer_id() -> usize {
    6
}
fn default_merge_kernel_size() -> [usize; 2] {
    [2, 2]
}
fn default_merger_times() -> usize {
    1
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn load_layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    let weight = weights
        .get(&format!("{}.weight", prefix))
        .map(|v| mlxcel_core::copy(v))
        .ok_or_else(|| format!("Weight not found: {}.weight", prefix))?;
    let bias = weights
        .get(&format!("{}.bias", prefix))
        .map(|v| mlxcel_core::copy(v));
    Ok(LayerNorm::new(weight, bias, eps))
}

// ── CrossAttention (shared by VitMerger) ────────────────────────────────────

struct CrossAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    out_proj: UnifiedLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl CrossAttention {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        dim: usize,
        num_heads: usize,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let head_dim = (dim / num_heads) as i32;
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
        let q_dim = q_shape[2];

        let q = self.q_proj.forward(queries);
        let k = self.k_proj.forward(keys);
        let v = self.v_proj.forward(values);

        let q = mlxcel_core::reshape(&q, &[batch, q_len, self.num_heads, self.head_dim]);
        let q = mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]);
        let k = mlxcel_core::reshape(&k, &[batch, kv_len, self.num_heads, self.head_dim]);
        let k = mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]);
        let v = mlxcel_core::reshape(&v, &[batch, kv_len, self.num_heads, self.head_dim]);
        let v = mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]);

        let output = unsafe {
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
        let output = mlxcel_core::transpose_axes(&output, &[0, 2, 1, 3]);
        let output = mlxcel_core::reshape(&output, &[batch, q_len, q_dim]);
        self.out_proj.forward(&output)
    }
}

// ── VitMerger ────────────────────────────────────────────────────────────────
//
// Inserted after layer `insert_layer_id` (default 6) in the ViT encoder.
// Reduces spatial resolution by `merge_group_size` (default 2×2) via:
//   1. Window-mean residual + self-attention refinement per group.
//   2. GELU MLP projecting group_hidden → hidden.
//
// Python reference: minicpmv4_6.py → class VitMerger

struct VitMerger {
    pre_norm: LayerNorm,
    self_attn: CrossAttention,
    layer_norm1: LayerNorm,
    linear_1: UnifiedLinear,
    linear_2: UnifiedLinear,
    group_h: usize,
    group_w: usize,
    group_tokens: i32,
    group_hidden_size: i32,
    vision_hidden_size: i32,
}

impl VitMerger {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        vision_hidden_size: usize,
        num_heads: usize,
        merge_group_size: [usize; 2],
        eps: f32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let group_h = merge_group_size[0];
        let group_w = merge_group_size[1];
        if group_h == 0 || group_w == 0 {
            return Err(
                "MiniCPM-V 4.6 VitMerger window_kernel_size entries must be positive".to_string(),
            );
        }
        let group_tokens = (group_h * group_w) as i32;
        let group_hidden_size = (vision_hidden_size * group_h * group_w) as i32;

        // The VitMerger intermediate size is `intermediate_size * group_h * group_w`.
        // We load it as UnifiedLinear and let the weight shape determine dims.
        let pre_norm = load_layer_norm(weights, &format!("{}.pre_norm", prefix), eps)?;
        let self_attn = CrossAttention::from_weights(
            weights,
            &format!("{}.self_attn", prefix),
            vision_hidden_size,
            num_heads,
            group_size,
            bits,
        )?;
        let layer_norm1 = load_layer_norm(weights, &format!("{}.layer_norm1", prefix), eps)?;
        let linear_1 = UnifiedLinear::from_weights(
            weights,
            &format!("{}.linear_1", prefix),
            group_size,
            bits,
        )?;
        let linear_2 = UnifiedLinear::from_weights(
            weights,
            &format!("{}.linear_2", prefix),
            group_size,
            bits,
        )?;

        Ok(Self {
            pre_norm,
            self_attn,
            layer_norm1,
            linear_1,
            linear_2,
            group_h,
            group_w,
            group_tokens,
            group_hidden_size,
            vision_hidden_size: vision_hidden_size as i32,
        })
    }

    /// Forward pass.
    ///
    /// `x` has shape `[seq_len, hidden_size]` (single image, no batch dim yet).
    /// `grid_h` × `grid_w` = seq_len.
    /// Returns `(merged, new_h, new_w)`.
    fn forward(
        &self,
        x: &MlxArray,
        grid_h: usize,
        grid_w: usize,
    ) -> (UniquePtr<MlxArray>, usize, usize) {
        let gh = self.group_h;
        let gw = self.group_w;
        let merged_h = grid_h / gh;
        let merged_w = grid_w / gw;
        let num_windows = (merged_h * merged_w) as i32;
        let hs = self.vision_hidden_size;

        // Reshape into [merged_h, group_h, merged_w, group_w, hidden]
        // then transpose to [merged_h, merged_w, group_h, group_w, hidden]
        // then reshape to [num_windows, group_tokens, hidden]
        let windows = mlxcel_core::reshape(
            x,
            &[merged_h as i32, gh as i32, merged_w as i32, gw as i32, hs],
        );
        let windows = mlxcel_core::transpose_axes(&windows, &[0, 2, 1, 3, 4]);
        let windows = mlxcel_core::reshape(&windows, &[num_windows, self.group_tokens, hs]);

        // Self-attention refinement within each window.
        let normed = self.layer_norm1.forward(&windows);
        let attn = self.self_attn.forward(&normed, &normed, &normed);
        let windows = mlxcel_core::add(&windows, &attn);

        // Window-mean residual.
        let residual = mlxcel_core::mean_axis(&windows, 1, false);

        // Flatten each window to [num_windows, group_tokens * hidden].
        let merged = mlxcel_core::reshape(&windows, &[num_windows, self.group_hidden_size]);

        // pre_norm → linear_1 → GELU → linear_2 → add residual.
        let merged = self.pre_norm.forward(&merged);
        let merged = self.linear_1.forward(&merged);
        let merged = mlxcel_core::gelu_approx(&merged);
        let merged = self.linear_2.forward(&merged);
        let merged = mlxcel_core::add(&merged, &residual);

        (merged, merged_h, merged_w)
    }
}

// ── MergerBlock / Merger ─────────────────────────────────────────────────────
//
// Pixel-shuffle style spatial downsampling after the ViT encoder.
// Each round: reshape spatial (h, w) into groups of merge_kernel_size,
//   flatten each group and project with a MLP.
//
// Python reference: minicpmv4_6.py → class Merger / MergerBlock

struct MergerBlock {
    pre_norm: LayerNorm,
    linear_1: UnifiedLinear,
    linear_2: UnifiedLinear,
}

impl MergerBlock {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        eps: f32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            pre_norm: load_layer_norm(weights, &format!("{}.pre_norm", prefix), eps)?,
            linear_1: UnifiedLinear::from_weights(
                weights,
                &format!("{}.linear_1", prefix),
                group_size,
                bits,
            )?,
            linear_2: UnifiedLinear::from_weights(
                weights,
                &format!("{}.linear_2", prefix),
                group_size,
                bits,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let x = self.pre_norm.forward(x);
        let x = self.linear_1.forward(&x);
        let x = mlxcel_core::gelu_approx(&x);
        self.linear_2.forward(&x)
    }
}

pub struct MiniCPMV46Merger {
    blocks: Vec<MergerBlock>,
    merge_h: usize,
    merge_w: usize,
}

impl MiniCPMV46Merger {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        merger_times: usize,
        merge_kernel_size: [usize; 2],
        eps: f32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        if merge_kernel_size[0] == 0 || merge_kernel_size[1] == 0 {
            return Err(
                "MiniCPM-V 4.6 merger merge_kernel_size entries must be positive".to_string(),
            );
        }

        let mut blocks = Vec::with_capacity(merger_times);
        for i in 0..merger_times {
            blocks.push(MergerBlock::from_weights(
                weights,
                &format!("{}.mlp.{}", prefix, i),
                eps,
                group_size,
                bits,
            )?);
        }
        // Upstream `Merger` reads `merge_kernel_size` from config (default 2×2);
        // see minicpmv4_6.py:165,186.
        Ok(Self {
            blocks,
            merge_h: merge_kernel_size[0],
            merge_w: merge_kernel_size[1],
        })
    }

    /// Validate and compute the output spatial grid for the configured
    /// post-ViT Merger blocks without running any MLX kernels.
    pub fn output_grid_size(
        &self,
        mut grid_h: usize,
        mut grid_w: usize,
    ) -> Result<(usize, usize), String> {
        for _ in &self.blocks {
            if !grid_h.is_multiple_of(self.merge_h) || !grid_w.is_multiple_of(self.merge_w) {
                return Err(format!(
                    "MiniCPM-V 4.6 patch grid {}x{} is not divisible by merger kernel {}x{}",
                    grid_h, grid_w, self.merge_h, self.merge_w
                ));
            }
            grid_h /= self.merge_h;
            grid_w /= self.merge_w;
        }

        Ok((grid_h, grid_w))
    }

    /// `x` shape: `[seq_len, hidden_size]`  (no batch dim).
    /// Returns `(merged_tokens, new_h, new_w)`.
    pub fn forward(
        &self,
        x: &MlxArray,
        grid_h: usize,
        grid_w: usize,
    ) -> (UniquePtr<MlxArray>, usize, usize) {
        let mut cur_h = grid_h;
        let mut cur_w = grid_w;
        let mut hidden = mlxcel_core::copy(x);

        for block in &self.blocks {
            let mh = self.merge_h as i32;
            let mw = self.merge_w as i32;
            let merged_h = cur_h / self.merge_h;
            let merged_w = cur_w / self.merge_w;
            let num_out = (merged_h * merged_w) as i32;

            let inner_dim = mlxcel_core::array_shape(&hidden)[1];
            let merge_tokens = mh * mw;

            // Reshape to [merged_h, merge_h, merged_w, merge_w, hidden]
            let h = mlxcel_core::reshape(&hidden, &[cur_h as i32, cur_w as i32, inner_dim]);
            let h =
                mlxcel_core::reshape(&h, &[merged_h as i32, mh, merged_w as i32, mw, inner_dim]);
            // Transpose → [merged_h, merged_w, merge_h, merge_w, hidden]
            let h = mlxcel_core::transpose_axes(&h, &[0, 2, 1, 3, 4]);
            // Flatten → [num_out, merge_tokens * hidden]
            let h = mlxcel_core::reshape(&h, &[num_out, merge_tokens * inner_dim]);

            hidden = block.forward(&h);
            cur_h = merged_h;
            cur_w = merged_w;
        }

        (hidden, cur_h, cur_w)
    }
}

// ── SigLIP ViT components (reusing minicpmo) ─────────────────────────────────
//
// The SigLIP encoder layers are identical to MiniCPM-O, so we import and
// reuse the existing types.  We only add VitMerger insertion logic here.

use crate::vision::encoders::minicpmo::MiniCPMOVisionModel;

// ── MiniCPM-V 4.6 full vision model ──────────────────────────────────────────

pub struct MiniCPMV46VisionModel {
    /// Shared SigLIP-style ViT.  We keep it as a whole model but only invoke
    /// its layers individually so we can inject VitMerger mid-stack.
    encoder: MiniCPMOVisionModel,
    vit_merger: VitMerger,
    pub merger: MiniCPMV46Merger,
    /// Which encoder layer to insert VitMerger after (0-indexed).
    insert_layer_id: usize,
    num_layers: usize,
}

impl MiniCPMV46VisionModel {
    pub fn from_weights(
        weights: &WeightMap,
        vision_cfg: &MiniCPMV46VisionConfig,
        model_cfg: &MiniCPMV46Config,
        prefix: &str,
        vit_merger_prefix: &str,
        merger_prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let base_cfg = to_minicpmo_vision_config(vision_cfg);
        let encoder =
            MiniCPMOVisionModel::from_weights(weights, &base_cfg, prefix, group_size, bits)?;

        let vit_merger = VitMerger::from_weights(
            weights,
            vit_merger_prefix,
            vision_cfg.hidden_size,
            vision_cfg.num_attention_heads,
            vision_cfg.window_kernel_size,
            vision_cfg.layer_norm_eps,
            group_size,
            bits,
        )?;

        let merger = MiniCPMV46Merger::from_weights(
            weights,
            merger_prefix,
            model_cfg.merger_times,
            model_cfg.merge_kernel_size,
            vision_cfg.layer_norm_eps,
            group_size,
            bits,
        )?;

        Ok(Self {
            encoder,
            vit_merger,
            merger,
            insert_layer_id: model_cfg.insert_layer_id,
            num_layers: vision_cfg.num_hidden_layers,
        })
    }

    /// Compute how many vision tokens the full MiniCPM-V 4.6 vision pipeline
    /// emits for a preprocessed patch grid.
    ///
    /// Used by request-time prompt preparation to reserve exactly one
    /// `<unk>` placeholder per emitted vision token before text tokenization.
    pub fn output_token_count_for_spatial_shape(
        &self,
        spatial_shape: (i32, i32),
    ) -> Result<usize, String> {
        if spatial_shape.0 <= 0 || spatial_shape.1 <= 0 {
            return Err(format!(
                "MiniCPM-V 4.6 spatial shape must be positive, got {:?}",
                spatial_shape
            ));
        }

        let grid_h = spatial_shape.0 as usize;
        let grid_w = spatial_shape.1 as usize;
        if !grid_h.is_multiple_of(self.vit_merger.group_h)
            || !grid_w.is_multiple_of(self.vit_merger.group_w)
        {
            return Err(format!(
                "MiniCPM-V 4.6 patch grid {}x{} is not divisible by VitMerger group {}x{}",
                grid_h, grid_w, self.vit_merger.group_h, self.vit_merger.group_w
            ));
        }

        let grid_h = grid_h / self.vit_merger.group_h;
        let grid_w = grid_w / self.vit_merger.group_w;
        let (grid_h, grid_w) = self.merger.output_grid_size(grid_h, grid_w)?;
        grid_h
            .checked_mul(grid_w)
            .ok_or_else(|| "MiniCPM-V 4.6 output token count overflowed usize".to_string())
    }

    /// Run the full vision forward pass on a single image.
    ///
    /// `pixel_values`: `[1, H, W, 3]` (HWC, 1-batch)
    /// `spatial_shape`: `(h_patches, w_patches)`
    /// Returns `[num_tokens, hidden_size]` vision tokens after merger.
    pub fn forward(
        &self,
        pixel_values: &MlxArray,
        spatial_shape: (i32, i32),
    ) -> UniquePtr<MlxArray> {
        // Run embeddings (positional + patch projection).
        let mut hidden = self.encoder.forward_embeddings(pixel_values, spatial_shape);
        let mut grid_h = spatial_shape.0 as usize;
        let mut grid_w = spatial_shape.1 as usize;

        // Run encoder layers, injecting VitMerger after insert_layer_id.
        for layer_idx in 0..self.num_layers {
            hidden = self.encoder.forward_layer(&hidden, layer_idx);

            if layer_idx == self.insert_layer_id {
                // hidden shape is [1, seq_len, hidden_size]; strip batch dim.
                let h_shape = mlxcel_core::array_shape(&hidden);
                let seq_len = h_shape[1];
                let hidden_sz = h_shape[2];
                let flat = mlxcel_core::reshape(&hidden, &[seq_len, hidden_sz]);

                let (merged, mh, mw) = self.vit_merger.forward(&flat, grid_h, grid_w);
                grid_h = mh;
                grid_w = mw;
                // Restore batch dim.
                let new_seq = (mh * mw) as i32;
                hidden = mlxcel_core::reshape(&merged, &[1, new_seq, hidden_sz]);
            }
        }

        // post_layernorm.
        let hidden = self.encoder.forward_post_layernorm(&hidden);

        // Strip batch dim → [seq_len, hidden_size].
        let h_shape = mlxcel_core::array_shape(&hidden);
        let seq_len = h_shape[1];
        let hidden_sz = h_shape[2];
        let flat = mlxcel_core::reshape(&hidden, &[seq_len, hidden_sz]);

        // Merger (pixel-shuffle downsampling).
        let (tokens, _, _) = self.merger.forward(&flat, grid_h, grid_w);
        tokens
    }
}

#[cfg(test)]
#[path = "minicpmv4_6_tests.rs"]
mod tests;
