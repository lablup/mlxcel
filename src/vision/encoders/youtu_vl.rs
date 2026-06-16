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

//! Youtu-VL vision encoder.
//!
//! Faithful port of https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/youtu_vl/vision.py.
//!
//! Architecture:
//! - `embeddings.patch_embedding`: Linear projection over flattened patches
//!   (`patch_size * patch_size * num_channels` → `hidden_size`). The encoder
//!   does NOT carry learned 2D position embeddings — RoPE is applied per
//!   layer instead.
//! - `encoder.layers[i]`: SigLIP-2 style block with `LayerNorm → MHA → resid →
//!   LayerNorm → GELU(tanh-approx) MLP → resid`.
//! - Vision RoPE: 2D positional encoding split half-and-half across H and W.
//! - Windowed attention: blocks NOT in `fullatt_block_indexes` attend within
//!   `window_size`-sized windows; full-attention blocks attend across the full
//!   patch sequence.
//! - `merger`: RMSNorm + 2-layer GELU MLP that projects merged patches into
//!   the language model hidden size (`out_hidden_size`).
//!
//! This module owns its own attention/MLP/embedding implementations because
//! the existing `siglip.rs` encoder uses Conv2d patch embedding, learned
//! positional embeddings, and supports neither RoPE nor windowed attention —
//! reusing it would require pervasive conditional branches that obscure both
//! code paths. The shared `mlxcel_core::layers::{LayerNorm, UnifiedLinear}`
//! primitives are reused throughout.
//!
//! Used by: `vision::youtu_vl::YoutuVLModel`.

use mlxcel_core::layers::LayerNorm;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

use super::{VisionEncoder, VisionEncoderOutput};

#[path = "youtu_vl_layers.rs"]
mod layers;
#[path = "youtu_vl_merger.rs"]
mod merger;
#[path = "youtu_vl_rope.rs"]
mod rope;
#[path = "youtu_vl_window.rs"]
mod window;

use layers::{EncoderLayer, VisionEmbeddings, load_layer_norm};
use merger::PatchMerger;
use rope::VisionRoPE;

/// Youtu-VL vision encoder configuration.
///
/// Mirrors https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/youtu_vl/config.py (VisionConfig).
#[derive(Debug, Clone, Deserialize)]
pub struct YoutuVisionConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_out_hidden_size")]
    pub out_hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    #[serde(default = "default_num_patches")]
    pub num_patches: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: usize,
    #[serde(default = "default_window_size")]
    pub window_size: usize,
    #[serde(default = "default_fullatt_block_indexes")]
    pub fullatt_block_indexes: Vec<usize>,
    /// Quantization group_size inherited from the top-level config.
    #[serde(default)]
    pub quant_group_size: i32,
    /// Quantization bits inherited from the top-level config.
    #[serde(default)]
    pub quant_bits: i32,
}

fn default_model_type() -> String {
    "siglip2_vision_model".to_string()
}
fn default_hidden_size() -> usize {
    1152
}
fn default_out_hidden_size() -> usize {
    2560
}
fn default_intermediate_size() -> usize {
    4304
}
fn default_num_hidden_layers() -> usize {
    27
}
fn default_num_attention_heads() -> usize {
    16
}
fn default_num_channels() -> usize {
    3
}
fn default_num_patches() -> usize {
    4096
}
fn default_patch_size() -> usize {
    16
}
fn default_layer_norm_eps() -> f32 {
    1e-6
}
fn default_spatial_merge_size() -> usize {
    2
}
fn default_window_size() -> usize {
    256
}
fn default_fullatt_block_indexes() -> Vec<usize> {
    vec![7, 15, 23, 26]
}

impl YoutuVisionConfig {
    fn group_size(&self) -> i32 {
        if self.quant_group_size > 0 {
            self.quant_group_size
        } else {
            64
        }
    }

    fn bits(&self) -> i32 {
        if self.quant_bits > 0 {
            self.quant_bits
        } else {
            4
        }
    }
}

// Vision RoPE primitives live in `rope`; per-block helpers live in `layers`;
// the patch merger lives in `merger`. This file owns the top-level encoder
// orchestration only.

// Top-level Youtu-VL vision encoder.
pub struct YoutuVLVisionEncoder {
    embeddings: VisionEmbeddings,
    layers: Vec<EncoderLayer>,
    post_layernorm: LayerNorm,
    merger: PatchMerger,
    rotary_pos_emb: VisionRoPE,
    spatial_merge_size: usize,
    spatial_merge_unit: i32,
    patch_size: usize,
    window_size: usize,
    fullatt_block_indexes: Vec<usize>,
    /// Cached for diagnostics — every consumer of `hidden_size` accesses it
    /// indirectly through embedded modules (LayerNorm dim, attention head
    /// dim) so there is no direct read on the encoder struct itself.
    #[allow(dead_code)]
    hidden_size: i32,
}

impl YoutuVLVisionEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        config: &YoutuVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        // M3: validate merger_window_size divisors at construction time so that
        // `get_window_index` is never called with a zero divisor.  A
        // spatial_merge_size or patch_size of zero, or a window_size that is
        // not an exact multiple of their product, all produce integer
        // division-by-zero or silently wrong window counts at runtime.
        let sms = config.spatial_merge_size;
        let ps = config.patch_size;
        let ws = config.window_size;
        if sms == 0 {
            return Err(format!(
                "invalid vision config: spatial_merge_size must be > 0 (got {sms})"
            ));
        }
        if ps == 0 {
            return Err(format!(
                "invalid vision config: patch_size must be > 0 (got {ps})"
            ));
        }
        let divisor = sms * ps;
        if !ws.is_multiple_of(divisor) {
            return Err(format!(
                "invalid vision config: window_size ({ws}) must be divisible by \
                 spatial_merge_size * patch_size ({sms} * {ps} = {divisor})"
            ));
        }

        let gs = config.group_size();
        let bits = config.bits();

        let embeddings = VisionEmbeddings::from_weights(
            weights,
            config,
            &format!("{}.embeddings", prefix),
            gs,
            bits,
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer_prefix = format!("{}.encoder.layers.{}", prefix, i);
            let layer = EncoderLayer::from_weights(weights, config, &layer_prefix, gs, bits)?;
            layers.push(layer);
        }

        let post_layernorm = load_layer_norm(
            weights,
            &format!("{}.post_layernorm", prefix),
            config.layer_norm_eps,
        )?;

        // After `Model.sanitize`, both the encoder body and the patch merger
        // sit directly under `prefix` (= `"vision_tower"`), so the merger key
        // is just `<prefix>.merger.*`. We keep the lookup explicit instead of
        // hiding it inside `PatchMerger` so the loader can be inspected end
        // to end without cross-referencing the upstream sanitize rules.
        let merger_prefix = format!("{}.merger", prefix);

        let merger = PatchMerger::from_weights(
            weights,
            &merger_prefix,
            config.hidden_size,
            config.spatial_merge_size,
            gs,
            bits,
        )?;

        let head_dim = (config.hidden_size / config.num_attention_heads) as i32;
        let rotary_pos_emb = VisionRoPE::new(head_dim / 2);

        Ok(Self {
            embeddings,
            layers,
            post_layernorm,
            merger,
            rotary_pos_emb,
            spatial_merge_size: config.spatial_merge_size,
            spatial_merge_unit: (config.spatial_merge_size * config.spatial_merge_size) as i32,
            patch_size: config.patch_size,
            window_size: config.window_size,
            fullatt_block_indexes: config.fullatt_block_indexes.clone(),
            hidden_size: config.hidden_size as i32,
        })
    }

    /// Compute 2D rotary position embeddings indexed by (h, w) for each token.
    /// Returns a flat tensor of shape `[total_tokens, head_dim/2]` (NOT yet
    /// concatenated to full head_dim).
    fn rot_pos_emb(&self, spatial_shapes: &[(i32, i32)]) -> UniquePtr<MlxArray> {
        // M4: guard against an empty spatial_shapes slice — the fold below would
        // panic on an empty iterator.  The forward pass always provides at least
        // one image, so this branch indicates a programming error at the call site.
        assert!(
            !spatial_shapes.is_empty(),
            "rot_pos_emb: spatial_shapes must not be empty (no images provided)"
        );

        let merge = self.spatial_merge_size as i32;
        let mut all_pos_ids: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(spatial_shapes.len());
        let mut max_grid: i32 = 0;

        for &(h, w) in spatial_shapes {
            if h > max_grid {
                max_grid = h;
            }
            if w > max_grid {
                max_grid = w;
            }

            // hpos_ids: arange(h)[:, None].repeat(w, axis=1)
            let h_arange = mlxcel_core::arange_i32(0, h, 1);
            let h_col = mlxcel_core::reshape(&h_arange, &[h, 1]);
            let hpos = mlxcel_core::repeat(&h_col, w, 1);
            let hpos = mlxcel_core::reshape(&hpos, &[h / merge, merge, w / merge, merge]);
            let hpos = mlxcel_core::transpose_axes(&hpos, &[0, 2, 1, 3]);
            let hpos = mlxcel_core::flatten(&hpos);

            // wpos_ids: arange(w)[None, :].repeat(h, axis=0)
            let w_arange = mlxcel_core::arange_i32(0, w, 1);
            let w_row = mlxcel_core::reshape(&w_arange, &[1, w]);
            let wpos = mlxcel_core::repeat(&w_row, h, 0);
            let wpos = mlxcel_core::reshape(&wpos, &[h / merge, merge, w / merge, merge]);
            let wpos = mlxcel_core::transpose_axes(&wpos, &[0, 2, 1, 3]);
            let wpos = mlxcel_core::flatten(&wpos);

            // stack along last → [tokens, 2]
            let stacked = mlxcel_core::stack_owned(&[hpos, wpos], -1);
            all_pos_ids.push(stacked);
        }

        // all_pos_ids is non-empty because spatial_shapes is non-empty (guarded above).
        let pos_ids = if all_pos_ids.len() == 1 {
            all_pos_ids.into_iter().next().unwrap()
        } else {
            let mut iter = all_pos_ids.into_iter();
            let first = iter.next().unwrap();
            iter.fold(first, |acc, next| mlxcel_core::concatenate(&acc, &next, 0))
        };

        // pos_ids: [total_tokens, 2]
        let rotary_table = self.rotary_pos_emb.freqs(max_grid);
        let pos_ids_flat = mlxcel_core::flatten(&pos_ids);
        let all_freqs = mlxcel_core::take(&rotary_table, &pos_ids_flat, 0);

        let pos_shape = mlxcel_core::array_shape(&pos_ids);
        let total_tokens = pos_shape[0];
        let freq_shape = mlxcel_core::array_shape(&all_freqs);
        let half_dim = freq_shape[1];
        let all_freqs = mlxcel_core::reshape(&all_freqs, &[total_tokens, 2, half_dim]);
        // Flatten last two: [total_tokens, 2 * half_dim]
        mlxcel_core::reshape(&all_freqs, &[total_tokens, 2 * half_dim])
    }

    /// Compute window indices and cu_window_seqlens for windowed attention.
    /// Delegates to the pure-CPU bookkeeping helper in [`window`] so the
    /// encoder body stays focused on tensor work.
    fn get_window_index(&self, spatial_shapes: &[(i32, i32)]) -> (Vec<i32>, Vec<i32>) {
        window::get_window_index(
            spatial_shapes,
            self.spatial_merge_size as i32,
            self.window_size as i32,
            self.patch_size as i32,
            self.spatial_merge_unit,
        )
    }

    /// Forward pass with explicit `spatial_shapes`. Each entry is `(h, w)` —
    /// number of patches along the height and width dimensions for one image
    /// (this corresponds to upstream's `spatial_shapes[:, 0]`, `spatial_shapes[:, 1]`).
    pub fn forward_with_spatial(
        &self,
        pixel_values: &MlxArray,
        spatial_shapes: &[(i32, i32)],
    ) -> VisionEncoderOutput {
        // Embed → [total_tokens, hidden]
        let mut h = self.embeddings.forward(pixel_values);

        // 2D rotary frequencies and window indices.
        let rotary_pos_emb = self.rot_pos_emb(spatial_shapes);
        let (window_index, cu_window_seqlens) = self.get_window_index(spatial_shapes);

        let shape = mlxcel_core::array_shape(&h);
        let seq_len = shape[0];

        // Reorder hidden states by window index. The reordering happens at the
        // merged-patch granularity: reshape to [n_groups, merge_unit, dim],
        // gather along the group axis, then flatten back.
        let dim = shape[1];
        let n_groups = seq_len / self.spatial_merge_unit;
        let h_grouped = mlxcel_core::reshape(&h, &[n_groups, self.spatial_merge_unit, dim]);
        let win_idx_arr = mlxcel_core::from_slice_i32(&window_index, &[window_index.len() as i32]);
        let h_reordered = mlxcel_core::take(&h_grouped, &win_idx_arr, 0);
        h = mlxcel_core::reshape(&h_reordered, &[-1, dim]);

        // Reorder rotary frequencies the same way.
        let rope_shape = mlxcel_core::array_shape(&rotary_pos_emb);
        let rope_dim = rope_shape[1];
        let rope_grouped = mlxcel_core::reshape(
            &rotary_pos_emb,
            &[n_groups, self.spatial_merge_unit, rope_dim],
        );
        let rope_reordered = mlxcel_core::take(&rope_grouped, &win_idx_arr, 0);
        let rotary_pos_emb = mlxcel_core::reshape(&rope_reordered, &[-1, rope_dim]);

        // Build (cos, sin) position-embedding tables of shape [seq, head_dim]
        // by concatenating the half-dim freq table to itself along the last
        // axis (matches the upstream `mx.concatenate([rotary_pos_emb,
        // rotary_pos_emb], axis=-1)` step) and taking element-wise cos/sin.
        let emb = mlxcel_core::concatenate(&rotary_pos_emb, &rotary_pos_emb, -1);
        let cos = mlxcel_core::cos(&emb);
        let sin = mlxcel_core::sin(&emb);

        // Full-attention cu_seqlens (cumsum over h*w per image, padded with 0).
        let mut full_cu = vec![0i32];
        let mut acc = 0i32;
        for &(hh, ww) in spatial_shapes {
            acc += hh * ww;
            full_cu.push(acc);
        }

        // Run encoder blocks, swapping cu_seqlens for the windowed/full layers.
        for (layer_num, layer) in self.layers.iter().enumerate() {
            let cu_seqlens_now = if self.fullatt_block_indexes.contains(&layer_num) {
                full_cu.as_slice()
            } else {
                cu_window_seqlens.as_slice()
            };
            h = layer.forward(&h, cu_seqlens_now, &cos, &sin);
        }

        // Post-LN over the encoder output, then run the patch merger.
        h = self.post_layernorm.forward(&h);
        h = self.merger.forward(&h);

        // Reverse the window reordering. Equivalent to `argsort(window_index)`
        // (Python upstream); we compute the inverse permutation directly so we
        // can feed it into a flat `mx::take` rather than triggering an extra
        // 3D reshape inside the encoder body.
        let reverse_indices = window::reverse_window_indices(&window_index);
        let reverse_arr =
            mlxcel_core::from_slice_i32(&reverse_indices, &[reverse_indices.len() as i32]);
        h = mlxcel_core::take(&h, &reverse_arr, 0);

        VisionEncoderOutput { hidden_states: h }
    }
}

impl VisionEncoder for YoutuVLVisionEncoder {
    fn forward(&self, _pixel_values: &MlxArray) -> VisionEncoderOutput {
        // Youtu-VL's vision tower requires `spatial_shapes`; the multimodal
        // runtime always calls `forward_with_spatial` directly. The
        // generic-trait shim should not be reached.
        panic!("Youtu-VL vision encoder requires spatial_shapes; call forward_with_spatial()");
    }
}

#[cfg(test)]
#[path = "youtu_vl_tests.rs"]
mod tests;
