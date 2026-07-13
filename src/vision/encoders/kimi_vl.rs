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

//! MoonViT vision encoder (Kimi-VL / Kimi-VL 2.5).
//!
//! Faithful port of the image path of upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/kimi_vl/vision.py.
//!
//! MoonViT is a native-resolution ViT: each image is pre-patchified by the
//! processor into `[num_patches, C, p, p]`, flattened per patch through a
//! `Conv2d(kernel=stride=patch)` patch embedding (equivalent to a linear over
//! the flattened patch), given a learned + bicubically-interpolated 2D position
//! embedding, then run through `depth` Qwen2-VL-style blocks that share a 2D
//! rotary embedding. Attention is block-diagonal across images: each image's
//! patches attend only within that image. A final layer norm precedes a
//! `spatial_merge_size × spatial_merge_size` patch merge that groups
//! neighbouring patches for the language connector.
//!
//! Scope: image and video. The Kimi-VL 2.5 3D MoonViT video path extends the
//! image path with a computed temporal position embedding, per-frame tiling of
//! the 2D rotary tables, one attention segment spanning the whole clip, and a
//! temporal mean-pool in the patch merger (issue #551). Media items are
//! described by [`KimiMediaGrid`] (`Image { h, w }` or `Video { t, h, w }`).
//!
//! Weight layout (post-`Model.sanitize`, rooted at `vision_tower.`):
//! - `patch_embed.proj.{weight,bias}` — Conv2d (`weight` MLX-transposed to
//!   `[embed_dim, p, p, C]` at load).
//! - `patch_embed.pos_emb.weight` — learned `[init_h, init_w, embed_dim]` grid.
//! - `blocks.{i}.{norm0,norm1}.{weight,bias}` — LayerNorm.
//! - `blocks.{i}.attn.{wqkv,wo}.{weight,bias}` — fused QKV + output projection.
//! - `blocks.{i}.mlp.{fc0,fc1}.{weight,bias}` — GELU MLP.
//! - `final_layernorm.{weight,bias}` — LayerNorm.
//!
//! Used by: `vision::kimi_vl::KimiVLModel`.

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

use super::{VisionEncoder, VisionEncoderOutput};

#[path = "kimi_vl_pos_emb.rs"]
pub(crate) mod pos_emb;
#[path = "kimi_vl_rope.rs"]
mod rope;

use pos_emb::Learnable2DInterpPosEmb;
use rope::Rope2DPosEmb;

/// A media item's patch grid handed to the MoonViT tower.
///
/// Images carry a 2D grid `(h, w)`; videos (Kimi-VL 2.5, `kimi_k25`) carry a 3D
/// grid `(t, h, w)` where `t` is the number of sampled frames and `(h, w)` is
/// the per-frame patch grid shared by every frame of the clip. The two are kept
/// distinct on purpose: a video adds a computed temporal position embedding that
/// an image does not, so a `t = 1` clip is not bit-identical to the same frame
/// processed as an image (see `pos_emb::temporal_sinusoid`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KimiMediaGrid {
    /// A still image with a `(h, w)` patch grid.
    Image { h: i32, w: i32 },
    /// A video clip with `t` frames, each a `(h, w)` patch grid.
    Video { t: i32, h: i32, w: i32 },
}

impl KimiMediaGrid {
    /// The spatial `(h, w)` patch grid shared by every frame of the item.
    #[inline]
    pub fn spatial(&self) -> (i32, i32) {
        match *self {
            KimiMediaGrid::Image { h, w } => (h, w),
            KimiMediaGrid::Video { h, w, .. } => (h, w),
        }
    }

    /// Number of frames: `1` for an image, `t` for a video.
    #[inline]
    pub fn frames(&self) -> i32 {
        match *self {
            KimiMediaGrid::Image { .. } => 1,
            KimiMediaGrid::Video { t, .. } => t,
        }
    }

    /// Number of encoder tokens (pre-merge): `h*w` for an image, `t*h*w` for a
    /// video. This is also the item's `cu_seqlens` attention-segment length.
    #[inline]
    pub fn token_count(&self) -> i32 {
        let (h, w) = self.spatial();
        self.frames() * h * w
    }

    /// Number of merged tokens the tower emits for this item:
    /// `(h/merge)*(w/merge)`, independent of `t` (the temporal mean-pool
    /// collapses all frames to one spatial map before the spatial merge).
    #[inline]
    pub fn merged_count(&self, merge: i32) -> i32 {
        let (h, w) = self.spatial();
        (h / merge) * (w / merge)
    }
}

/// MoonViT vision configuration.
///
/// Mirrors `VisionConfig` in upstream
/// https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/kimi_vl/config.py.
#[derive(Debug, Clone, Deserialize)]
pub struct KimiVLVisionConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_depth")]
    pub depth: usize,
    #[serde(default = "default_embed_dim")]
    pub embed_dim: usize,
    #[serde(default = "default_embed_dim")]
    pub hidden_size: usize,
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_init_pos_emb")]
    pub init_pos_emb_height: usize,
    #[serde(default = "default_init_pos_emb")]
    pub init_pos_emb_width: usize,
    #[serde(default = "default_merge_size")]
    pub spatial_merge_size: usize,
    /// Frame-sampling granularity for the video path (Kimi-VL 2.5). This is
    /// **not** a convolution kernel depth: the checkpoint has no temporal conv
    /// axis. Its only observable effect is that sampled frame counts are
    /// multiples of this value (which coincides with `video::FRAME_FACTOR = 2`).
    /// Present in `kimi_k25` configs and previously ignored.
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    /// Quantization group_size inherited from the top-level config (0 = unset).
    #[serde(default)]
    pub quant_group_size: i32,
    /// Quantization bits inherited from the top-level config (0 = unset).
    #[serde(default)]
    pub quant_bits: i32,
}

fn default_model_type() -> String {
    "moonvit".to_string()
}
fn default_depth() -> usize {
    27
}
fn default_embed_dim() -> usize {
    1152
}
fn default_num_heads() -> usize {
    16
}
fn default_patch_size() -> usize {
    14
}
fn default_num_channels() -> usize {
    3
}
fn default_intermediate_size() -> usize {
    4304
}
fn default_init_pos_emb() -> usize {
    64
}
fn default_merge_size() -> usize {
    2
}
fn default_temporal_patch_size() -> usize {
    2
}
fn default_layer_norm_eps() -> f32 {
    1e-6
}

impl KimiVLVisionConfig {
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

    fn head_dim(&self) -> i32 {
        (self.embed_dim / self.num_heads) as i32
    }
}

/// Conv2d patch embedding + learned 2D position embedding.
struct PatchEmbed {
    proj_weight: UnifiedLinearConv,
    pos_emb: Learnable2DInterpPosEmb,
    patch_size: i32,
    embed_dim: i32,
}

/// The patch-embed convolution weight/bias, applied as a stride-`p` Conv2d over
/// each `p × p` patch (each patch collapses to a single embedding vector).
struct UnifiedLinearConv {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
}

impl PatchEmbed {
    fn from_weights(
        weights: &WeightMap,
        config: &KimiVLVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let wkey = format!("{prefix}.proj.weight");
        let weight = weights
            .get(&wkey)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {wkey}"))?;
        let bias = weights
            .get(&format!("{prefix}.proj.bias"))
            .map(|b| mlxcel_core::copy(b));

        let pos_emb = Learnable2DInterpPosEmb::from_weights(
            weights,
            &format!("{prefix}.pos_emb"),
            config.init_pos_emb_height as i32,
            config.init_pos_emb_width as i32,
            config.embed_dim as i32,
        )?;

        Ok(Self {
            proj_weight: UnifiedLinearConv { weight, bias },
            pos_emb,
            patch_size: config.patch_size as i32,
            embed_dim: config.embed_dim as i32,
        })
    }

    /// `pixel_values`: `[num_patches, p, p, C]` (channels-last). Returns
    /// `[num_patches, embed_dim]` with the spatial (and, for video items, the
    /// temporal) position embedding added.
    fn forward(
        &self,
        pixel_values: &MlxArray,
        media_grids: &[KimiMediaGrid],
    ) -> UniquePtr<MlxArray> {
        // Stride-p convolution over each p×p patch -> [N, 1, 1, embed_dim].
        let conv = mlxcel_core::conv2d(
            pixel_values,
            &self.proj_weight.weight,
            self.patch_size,
            self.patch_size,
            0,
            0,
            1,
            1,
            1,
        );
        let n = mlxcel_core::array_shape(pixel_values)[0];
        let mut h = mlxcel_core::reshape(&conv, &[n, self.embed_dim]);
        if let Some(ref bias) = self.proj_weight.bias {
            h = mlxcel_core::add(&h, bias);
        }
        self.pos_emb.add_to_media(&h, media_grids)
    }
}

/// Block attention with a shared 2D rotary embedding. Attention is
/// block-diagonal across images: each `(h, w)` grid attends only within itself.
struct VisionAttention {
    wqkv: UnifiedLinear,
    wo: UnifiedLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl VisionAttention {
    fn from_weights(
        weights: &WeightMap,
        config: &KimiVLVisionConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let wqkv = UnifiedLinear::from_weights(weights, &format!("{prefix}.wqkv"), gs, bits)?;
        let wo = UnifiedLinear::from_weights(weights, &format!("{prefix}.wo"), gs, bits)?;
        let head_dim = config.head_dim();
        Ok(Self {
            wqkv,
            wo,
            num_heads: config.num_heads as i32,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        cu_seqlens: &[i32],
        cos: &MlxArray,
        sin: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let seq = mlxcel_core::array_shape(x)[0];

        // wqkv -> [seq, 3, num_heads, head_dim]; slice out q/k/v.
        let qkv = self.wqkv.forward(x);
        let qkv = mlxcel_core::reshape(&qkv, &[seq, 3, self.num_heads, self.head_dim]);
        let pick = |i: i32| {
            let s = mlxcel_core::slice(
                &qkv,
                &[0, i, 0, 0],
                &[seq, i + 1, self.num_heads, self.head_dim],
            );
            mlxcel_core::reshape(&s, &[seq, self.num_heads, self.head_dim])
        };
        let q = pick(0);
        let k = pick(1);
        let v = pick(2);

        let (q, k) = rope::apply_rope(&q, &k, cos, sin);

        // [num_heads, seq, head_dim] for windowed SDPA.
        let q = mlxcel_core::transpose_axes(&q, &[1, 0, 2]);
        let k = mlxcel_core::transpose_axes(&k, &[1, 0, 2]);
        let v = mlxcel_core::transpose_axes(&v, &[1, 0, 2]);

        assert!(
            cu_seqlens.len() >= 2,
            "VisionAttention: cu_seqlens must have >= 2 entries; got {}",
            cu_seqlens.len()
        );
        let num_segments = cu_seqlens.len() - 1;
        let mut outputs: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(num_segments);
        for seg in 0..num_segments {
            let start = cu_seqlens[seg];
            let end = cu_seqlens[seg + 1];
            let q_seg =
                mlxcel_core::slice(&q, &[0, start, 0], &[self.num_heads, end, self.head_dim]);
            let k_seg =
                mlxcel_core::slice(&k, &[0, start, 0], &[self.num_heads, end, self.head_dim]);
            let v_seg =
                mlxcel_core::slice(&v, &[0, start, 0], &[self.num_heads, end, self.head_dim]);
            let q_seg = mlxcel_core::expand_dims(&q_seg, 0);
            let k_seg = mlxcel_core::expand_dims(&k_seg, 0);
            let v_seg = mlxcel_core::expand_dims(&v_seg, 0);
            // Full (bidirectional) attention within the image segment.
            let attn = unsafe {
                mlxcel_core::scaled_dot_product_attention(
                    &q_seg,
                    &k_seg,
                    &v_seg,
                    self.scale,
                    std::ptr::null(),
                )
            };
            let attn = mlxcel_core::reshape(&attn, &[self.num_heads, end - start, self.head_dim]);
            outputs.push(attn);
        }

        let concatenated = if outputs.len() == 1 {
            outputs.into_iter().next().unwrap()
        } else {
            let mut iter = outputs.into_iter();
            let first = iter.next().unwrap();
            iter.fold(first, |acc, next| mlxcel_core::concatenate(&acc, &next, 1))
        };

        // [num_heads, seq, head_dim] -> [seq, num_heads*head_dim].
        let out = mlxcel_core::transpose_axes(&concatenated, &[1, 0, 2]);
        let out = mlxcel_core::reshape(&out, &[seq, self.num_heads * self.head_dim]);
        self.wo.forward(&out)
    }
}

/// GELU MLP (`fc0 -> GELU -> fc1`). MoonViT uses the exact (erf) GELU.
struct VisionMLP {
    fc0: UnifiedLinear,
    fc1: UnifiedLinear,
}

impl VisionMLP {
    fn from_weights(weights: &WeightMap, prefix: &str, gs: i32, bits: i32) -> Result<Self, String> {
        let fc0 = UnifiedLinear::from_weights(weights, &format!("{prefix}.fc0"), gs, bits)?;
        let fc1 = UnifiedLinear::from_weights(weights, &format!("{prefix}.fc1"), gs, bits)?;
        Ok(Self { fc0, fc1 })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let x = self.fc0.forward(x);
        let x = mlxcel_core::gelu(&x);
        self.fc1.forward(&x)
    }
}

/// Qwen2-VL-style vision block: `norm0 -> attn -> +resid -> norm1 -> mlp -> +resid`.
struct VisionBlock {
    norm0: LayerNorm,
    norm1: LayerNorm,
    attn: VisionAttention,
    mlp: VisionMLP,
}

impl VisionBlock {
    fn from_weights(
        weights: &WeightMap,
        config: &KimiVLVisionConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let norm0 = load_layer_norm(weights, &format!("{prefix}.norm0"), config.layer_norm_eps)?;
        let norm1 = load_layer_norm(weights, &format!("{prefix}.norm1"), config.layer_norm_eps)?;
        let attn =
            VisionAttention::from_weights(weights, config, &format!("{prefix}.attn"), gs, bits)?;
        let mlp = VisionMLP::from_weights(weights, &format!("{prefix}.mlp"), gs, bits)?;
        Ok(Self {
            norm0,
            norm1,
            attn,
            mlp,
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        cu_seqlens: &[i32],
        cos: &MlxArray,
        sin: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let attn_out = self
            .attn
            .forward(&self.norm0.forward(x), cu_seqlens, cos, sin);
        let h = mlxcel_core::add(x, &attn_out);
        let mlp_out = self.mlp.forward(&self.norm1.forward(&h));
        mlxcel_core::add(&h, &mlp_out)
    }
}

/// Block-diagonal attention `cu_seqlens` for the media items: one segment per
/// item spanning the whole clip (`t*h*w` for a video, `h*w` for an image),
/// prefixed with `0`. All frames of one clip form a single bidirectional
/// segment; different media items never attend to each other.
fn cu_seqlens(media_grids: &[KimiMediaGrid]) -> Vec<i32> {
    let mut cu = Vec::with_capacity(media_grids.len() + 1);
    cu.push(0i32);
    let mut acc = 0i32;
    for grid in media_grids {
        acc += grid.token_count();
        cu.push(acc);
    }
    cu
}

/// `spatial_merge_size × spatial_merge_size` patch merge, with a temporal
/// mean-pool for video items.
///
/// For an image item the item's `(h*w, dim)` slice is grouped directly. For a
/// video item the `(t*h*w, dim)` slice is first reshaped to `(t, h*w, dim)` and
/// mean-pooled over the frame axis, collapsing the clip to one `(h*w, dim)`
/// spatial map before the identical spatial merge. Each item therefore
/// contributes `(h/merge)*(w/merge)` merged tokens regardless of `t`.
///
/// Groups each `(kh, kw)` block of neighbouring patches into one merged token
/// carrying `kh * kw` channel-stacked vectors. Returns the per-item merged
/// features concatenated along the token axis: `[total_merged, kh*kw, dim]`.
fn patch_merger(x: &MlxArray, media_grids: &[KimiMediaGrid], merge: i32) -> UniquePtr<MlxArray> {
    let dim = mlxcel_core::array_shape(x)[1];
    let mut offset = 0i32;
    let mut outs: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(media_grids.len());
    for grid in media_grids {
        let (h, w) = grid.spatial();
        let n = grid.token_count();
        let seq = mlxcel_core::slice(x, &[offset, 0], &[offset + n, dim]);
        offset += n;

        // Temporal mean-pool: collapse the clip's frames to one spatial map.
        let seq = if let KimiMediaGrid::Video { t, .. } = *grid {
            let frames = mlxcel_core::reshape(&seq, &[t, h * w, dim]);
            mlxcel_core::mean_axis(&frames, 0, false)
        } else {
            seq
        };

        let (new_h, new_w) = (h / merge, w / merge);
        let seq = mlxcel_core::reshape(&seq, &[new_h, merge, new_w, merge, dim]);
        let seq = mlxcel_core::transpose_axes(&seq, &[0, 2, 1, 3, 4]);
        let seq = mlxcel_core::reshape(&seq, &[new_h * new_w, merge * merge, dim]);
        outs.push(seq);
    }
    if outs.len() == 1 {
        outs.into_iter().next().unwrap()
    } else {
        let mut iter = outs.into_iter();
        let first = iter.next().unwrap();
        iter.fold(first, |acc, next| mlxcel_core::concatenate(&acc, &next, 0))
    }
}

/// Top-level MoonViT vision encoder.
pub struct KimiVLVisionModel {
    patch_embed: PatchEmbed,
    blocks: Vec<VisionBlock>,
    final_layernorm: LayerNorm,
    rope: Rope2DPosEmb,
    spatial_merge_size: i32,
}

impl KimiVLVisionModel {
    pub fn from_weights(
        weights: &WeightMap,
        config: &KimiVLVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        if config.num_heads == 0 || !config.embed_dim.is_multiple_of(config.num_heads) {
            return Err(format!(
                "invalid MoonViT config: embed_dim ({}) must be divisible by num_heads ({})",
                config.embed_dim, config.num_heads
            ));
        }
        let gs = config.group_size();
        let bits = config.bits();

        let patch_embed =
            PatchEmbed::from_weights(weights, config, &format!("{prefix}.patch_embed"))?;

        let mut blocks = Vec::with_capacity(config.depth);
        for i in 0..config.depth {
            blocks.push(VisionBlock::from_weights(
                weights,
                config,
                &format!("{prefix}.blocks.{i}"),
                gs,
                bits,
            )?);
        }

        let final_layernorm = load_layer_norm(
            weights,
            &format!("{prefix}.final_layernorm"),
            config.layer_norm_eps,
        )?;

        Ok(Self {
            patch_embed,
            blocks,
            final_layernorm,
            rope: Rope2DPosEmb::new(config.head_dim()),
            spatial_merge_size: config.spatial_merge_size as i32,
        })
    }

    /// Forward pass over one or more media items (images and/or video clips).
    ///
    /// `pixel_values`: `[total_patches, p, p, C]` (channels-last), packed in
    /// media order (frame-major within each video). `media_grids`: one
    /// [`KimiMediaGrid`] per item, in the same order the patches are
    /// concatenated. Returns the merged features `[total_merged, kh*kw, dim]`.
    pub fn forward_with_grid(
        &self,
        pixel_values: &MlxArray,
        media_grids: &[KimiMediaGrid],
    ) -> UniquePtr<MlxArray> {
        assert!(
            !media_grids.is_empty(),
            "MoonViT forward: media_grids must not be empty"
        );

        let mut h = self.patch_embed.forward(pixel_values, media_grids);
        let (cos, sin) = self.rope.cos_sin(media_grids);

        let cu = cu_seqlens(media_grids);

        for block in &self.blocks {
            h = block.forward(&h, &cu, &cos, &sin);
        }
        h = self.final_layernorm.forward(&h);
        patch_merger(&h, media_grids, self.spatial_merge_size)
    }
}

impl VisionEncoder for KimiVLVisionModel {
    fn forward(&self, _pixel_values: &MlxArray) -> VisionEncoderOutput {
        panic!("MoonViT requires per-image grid shapes; call forward_with_grid()");
    }
}

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

#[cfg(test)]
#[path = "kimi_vl_tests.rs"]
mod tests;
