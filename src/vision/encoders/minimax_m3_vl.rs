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

//! MiniMax-M3-VL vision tower (`model_type: "minimax_m3_vl"`, nested
//! `vision_config.model_type: "clip_vision_model"`).
//!
//! A CLIP-style ViT with native-resolution packing, structurally the Qwen2-VL
//! vision tower (hidden 1280, 16 heads, head_dim 80, 32 layers, intermediate
//! 5120, patch 14) but with the differences that live in the real checkpoint:
//! - CLIP tensor naming (`vision_tower.vision_model.encoder.layers.N.*`) with a
//!   leading `pre_layrnorm` (the checkpoint's spelling) before the encoder,
//! - separate `self_attn.{q,k,v,out}_proj` (each with bias) instead of a fused
//!   `qkv`,
//! - LayerNorm + exact-GELU MLP (`layer_norm1/2`, `mlp.fc1/fc2`),
//! - a two-stage projector: a per-patch `multi_modal_projector`
//!   (`linear_1` -> GELU -> `linear_2`, into `projection_dim` 6144) followed by
//!   a `patch_merge_mlp` that folds `spatial_merge_size^2 = 4` adjacent patches
//!   (`linear_1` [6144, 24576] -> GELU -> `linear_2`) into the text hidden size.
//!
//! The tower reuses the shared Qwen2-VL `cu_seqlens` variable-length attention
//! and the frequency-table / `apply_rotary_pos_emb_vision` helpers, but drives
//! them with a genuine 3D (t, h, w) vision RoPE that matches the reference
//! `MiniMaxVLVisionTransformer`. The head dimension is split into three equal
//! axis sections (t, h, w) plus a trailing unrotated tail: `axis_dim =
//! 2 * ((head_dim / 2) / 3 / 2)` head dims per axis, `rot_dim = 3 * axis_dim`
//! head dims rotated, and the remaining `head_dim - rot_dim` trailing dims pass
//! through untouched (2 dims for head_dim 80, 4 for the reduced test head_dim
//! 64). For images (`grid_t == 1`) the temporal section is inert (all-zero t
//! ids), but it still reserves its slice of the head dimension, so the split
//! cannot collapse to the 2D (h, w) form. Video (`grid_t > 1`) is out of scope
//! for this port. The `image_grid_thw` packing emitted by the processor drives
//! both the position ids and the per-image `cu_seqlens`.
//!
//! The whole tower runs in f32: the checkpoint stores the vision weights as
//! f32/bf16 non-quantized, and running the tower uniformly in f32 avoids
//! mixed-dtype matmuls while the projector output is cast back to the text
//! embedding dtype by the LLaVA-style merge. The 427B checkpoint cannot be
//! loaded on the development machine, so the validated surface is the synthetic
//! reduced-config unit tests plus the real-config parse test.

use super::VisionEncoderOutput;
use super::qwen2_vl::{VisionRotaryEmbedding, apply_rotary_pos_emb_vision, concat_many};
use mlxcel_core::layers::LayerNorm;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

/// `img_token_compression_config` block: how the projector folds patches.
#[derive(Debug, Clone, Deserialize)]
pub struct ImgTokenCompressionConfig {
    #[serde(default = "default_compression_method")]
    pub image_token_compression_method: String,
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: usize,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
}

impl Default for ImgTokenCompressionConfig {
    fn default() -> Self {
        Self {
            image_token_compression_method: default_compression_method(),
            spatial_merge_size: default_spatial_merge_size(),
            temporal_patch_size: default_temporal_patch_size(),
        }
    }
}

/// MiniMax-M3-VL vision encoder configuration (nested `vision_config`).
///
/// The real checkpoint also ships LLaVA-style keys (`image_grid_pinpoints`,
/// `vision_feature_layer`, `vision_feature_select_strategy`, `image_seq_length`)
/// that are vestigial for image processing; they are ignored here (serde drops
/// unknown fields) so the config parses permissively without building logic on
/// them.
#[derive(Debug, Clone, Deserialize)]
pub struct MiniMaxM3VisionConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_projection_dim")]
    pub projection_dim: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(alias = "num_channels", default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default)]
    pub img_token_compression_config: ImgTokenCompressionConfig,
}

fn default_hidden_size() -> usize {
    1280
}
fn default_num_attention_heads() -> usize {
    16
}
fn default_num_hidden_layers() -> usize {
    32
}
fn default_intermediate_size() -> usize {
    5120
}
fn default_patch_size() -> usize {
    14
}
fn default_projection_dim() -> usize {
    6144
}
fn default_rope_theta() -> f32 {
    10000.0
}
fn default_layer_norm_eps() -> f32 {
    1e-5
}
fn default_in_channels() -> usize {
    3
}
fn default_compression_method() -> String {
    "patch_merge".to_string()
}
fn default_spatial_merge_size() -> usize {
    2
}
fn default_temporal_patch_size() -> usize {
    2
}

impl MiniMaxM3VisionConfig {
    pub fn spatial_merge_size(&self) -> usize {
        self.img_token_compression_config.spatial_merge_size
    }

    pub fn temporal_patch_size(&self) -> usize {
        self.img_token_compression_config.temporal_patch_size
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Per-axis rotary width of the 3D vision RoPE, shared by the t/h/w sections.
///
/// Matches the reference `MiniMaxVLVisionTransformer`: `rope_dims` rounds the
/// head dim down to an even width, then each of the three axis sections gets an
/// even slice `axis_dim = 2 * ((rope_dims / 3) / 2)` (integer division). The
/// frequency table therefore holds `axis_dim / 2` entries per axis. head_dim 80
/// -> axis_dim 26; head_dim 64 -> axis_dim 20.
pub(crate) fn rope_axis_dim(head_dim: i32) -> i32 {
    let rope_dims = 2 * (head_dim / 2);
    2 * ((rope_dims / 3) / 2)
}

/// Number of head dims actually rotated by the 3D vision RoPE (`3 * axis_dim`).
/// The remaining `head_dim - rot_dim` trailing dims pass through unrotated.
/// head_dim 80 -> rot_dim 78 (2 pass-through); head_dim 64 -> rot_dim 60 (4
/// pass-through).
pub(crate) fn rope_rot_dim(head_dim: i32) -> i32 {
    3 * rope_axis_dim(head_dim)
}

// ============================================================================
// Plain f32 linear / layernorm helpers
// ============================================================================

fn load_f32(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|w| mlxcel_core::astype(w, mlxcel_core::dtype::FLOAT32))
        .ok_or_else(|| format!("Weight not found: {}", key))
}

fn load_f32_opt(weights: &WeightMap, key: &str) -> Option<UniquePtr<MlxArray>> {
    weights
        .get(key)
        .map(|w| mlxcel_core::astype(w, mlxcel_core::dtype::FLOAT32))
}

/// Non-quantized f32 linear (`y = x @ W^T + b`). The vision tower is not
/// quantized in the checkpoint, so a plain matmul keeps the whole tower in a
/// single dtype and avoids the quantized-linear machinery.
struct VisionLinear {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
}

impl VisionLinear {
    fn load(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            weight: load_f32(weights, &format!("{}.weight", prefix))?,
            bias: load_f32_opt(weights, &format!("{}.bias", prefix)),
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let wt = mlxcel_core::transpose(&self.weight);
        let y = mlxcel_core::matmul(x, &wt);
        match &self.bias {
            Some(b) => mlxcel_core::add(&y, b),
            None => y,
        }
    }
}

fn load_layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    let weight = load_f32(weights, &format!("{}.weight", prefix))?;
    let bias = load_f32_opt(weights, &format!("{}.bias", prefix));
    Ok(LayerNorm::new(weight, bias, eps))
}

// ============================================================================
// Patch embedding
// ============================================================================

/// Temporal 3D patch conv degenerated to a linear.
///
/// The checkpoint weight is `[out, in_channels, temporal, patch_h, patch_w]`
/// (PyTorch Conv3d layout). The processor flattens each patch row in the exact
/// same `[channel, temporal, patch_h, patch_w]` order, so a row-major reshape
/// to `[out, in_features]` aligns the two without any axis permutation.
struct PatchEmbed {
    proj_weight: UniquePtr<MlxArray>,
    proj_bias: Option<UniquePtr<MlxArray>>,
}

impl PatchEmbed {
    fn from_weights(
        weights: &WeightMap,
        config: &MiniMaxM3VisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let key = format!("{}.weight", prefix);
        let w = load_f32(weights, &key)?;
        let out_features = config.hidden_size as i32;
        let in_features = (config.in_channels
            * config.temporal_patch_size()
            * config.patch_size
            * config.patch_size) as i32;
        let shape = mlxcel_core::array_shape(&w);
        let proj_weight = if shape.len() == 2 {
            w
        } else {
            mlxcel_core::reshape(&w, &[out_features, in_features])
        };
        Ok(Self {
            proj_weight,
            proj_bias: load_f32_opt(weights, &format!("{}.bias", prefix)),
        })
    }

    /// `hidden_states`: `[num_patches, in_features]` -> `[num_patches, hidden]`.
    fn forward(&self, hidden_states: &MlxArray) -> UniquePtr<MlxArray> {
        let wt = mlxcel_core::transpose(&self.proj_weight);
        let y = mlxcel_core::matmul(hidden_states, &wt);
        match &self.proj_bias {
            Some(b) => mlxcel_core::add(&y, b),
            None => y,
        }
    }
}

// ============================================================================
// Encoder layer (CLIP style)
// ============================================================================

struct VisionAttention {
    q_proj: VisionLinear,
    k_proj: VisionLinear,
    v_proj: VisionLinear,
    out_proj: VisionLinear,
    num_heads: i32,
    head_dim: i32,
    /// Leading head dims rotated by the 3D vision RoPE (`3 * axis_dim`); the
    /// trailing `head_dim - rot_dim` dims pass through unrotated.
    rot_dim: i32,
    scale: f32,
}

impl VisionAttention {
    fn from_weights(
        weights: &WeightMap,
        config: &MiniMaxM3VisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let head_dim = config.head_dim() as i32;
        Ok(Self {
            q_proj: VisionLinear::load(weights, &format!("{}.q_proj", prefix))?,
            k_proj: VisionLinear::load(weights, &format!("{}.k_proj", prefix))?,
            v_proj: VisionLinear::load(weights, &format!("{}.v_proj", prefix))?,
            out_proj: VisionLinear::load(weights, &format!("{}.out_proj", prefix))?,
            num_heads: config.num_attention_heads as i32,
            head_dim,
            rot_dim: rope_rot_dim(head_dim),
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// Packed variable-length attention: `x` is `[total_tokens, hidden]`,
    /// `cu_seqlens` marks the per-image segment boundaries, and full attention
    /// runs independently within each segment.
    fn forward(
        &self,
        x: &MlxArray,
        cu_seqlens: &[i32],
        rotary_pos_emb: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let seq_length = shape[0];

        let reshape_heads = |proj: UniquePtr<MlxArray>| {
            mlxcel_core::reshape(&proj, &[seq_length, self.num_heads, self.head_dim])
        };
        let q = reshape_heads(self.q_proj.forward(x));
        let k = reshape_heads(self.k_proj.forward(x));
        let v = reshape_heads(self.v_proj.forward(x));

        // Partial 3D vision RoPE: rotate only the leading `rot_dim` head dims
        // (the concatenated t/h/w axis sections) and pass the trailing
        // `head_dim - rot_dim` dims through unrotated. v is not rotated.
        let apply_rope = |t: &MlxArray| -> UniquePtr<MlxArray> {
            if self.rot_dim >= self.head_dim {
                return apply_rotary_pos_emb_vision(t, rotary_pos_emb);
            }
            let rot =
                mlxcel_core::slice(t, &[0, 0, 0], &[seq_length, self.num_heads, self.rot_dim]);
            let pass = mlxcel_core::slice(
                t,
                &[0, 0, self.rot_dim],
                &[seq_length, self.num_heads, self.head_dim],
            );
            let rot = apply_rotary_pos_emb_vision(&rot, rotary_pos_emb);
            mlxcel_core::concatenate(&rot, &pass, 2)
        };
        let q = apply_rope(&q);
        let k = apply_rope(&k);

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
            let start = cu_seqlens[seg];
            let end = cu_seqlens[seg + 1];
            let take = |t: &MlxArray| {
                mlxcel_core::slice(
                    t,
                    &[0, 0, start, 0],
                    &[1, self.num_heads, end, self.head_dim],
                )
            };
            let attn = unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &take(&q),
                    &take(&k),
                    &take(&v),
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
        self.out_proj.forward(&output)
    }
}

struct VisionMlp {
    fc1: VisionLinear,
    fc2: VisionLinear,
}

impl VisionMlp {
    fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            fc1: VisionLinear::load(weights, &format!("{}.fc1", prefix))?,
            fc2: VisionLinear::load(weights, &format!("{}.fc2", prefix))?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.fc1.forward(x);
        let h = mlxcel_core::gelu(&h);
        self.fc2.forward(&h)
    }
}

struct VisionLayer {
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
    attn: VisionAttention,
    mlp: VisionMlp,
}

impl VisionLayer {
    fn from_weights(
        weights: &WeightMap,
        config: &MiniMaxM3VisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let eps = config.layer_norm_eps;
        Ok(Self {
            layer_norm1: load_layer_norm(weights, &format!("{}.layer_norm1", prefix), eps)?,
            layer_norm2: load_layer_norm(weights, &format!("{}.layer_norm2", prefix), eps)?,
            attn: VisionAttention::from_weights(weights, config, &format!("{}.self_attn", prefix))?,
            mlp: VisionMlp::from_weights(weights, &format!("{}.mlp", prefix))?,
        })
    }

    fn forward(
        &self,
        hidden_states: &MlxArray,
        cu_seqlens: &[i32],
        rotary_pos_emb: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let normed = self.layer_norm1.forward(hidden_states);
        let attn_out = self.attn.forward(&normed, cu_seqlens, rotary_pos_emb);
        let h = mlxcel_core::add(hidden_states, &attn_out);
        let normed = self.layer_norm2.forward(&h);
        let mlp_out = self.mlp.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }
}

// ============================================================================
// Two-stage projector
// ============================================================================

/// Per-patch projector into `projection_dim` (`linear_1` -> GELU -> `linear_2`).
struct MultiModalProjector {
    linear_1: VisionLinear,
    linear_2: VisionLinear,
}

impl MultiModalProjector {
    fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            linear_1: VisionLinear::load(weights, &format!("{}.linear_1", prefix))?,
            linear_2: VisionLinear::load(weights, &format!("{}.linear_2", prefix))?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.linear_1.forward(x);
        let h = mlxcel_core::gelu(&h);
        self.linear_2.forward(&h)
    }
}

/// Patch-merge MLP: folds `spatial_merge_size^2` adjacent patches
/// (`linear_1` [projection_dim, merge^2 * projection_dim] -> GELU ->
/// `linear_2`) into the text hidden size.
struct PatchMergeMlp {
    linear_1: VisionLinear,
    linear_2: VisionLinear,
    fold: i32,
    projection_dim: i32,
}

impl PatchMergeMlp {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        merge: usize,
        projection_dim: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            linear_1: VisionLinear::load(weights, &format!("{}.linear_1", prefix))?,
            linear_2: VisionLinear::load(weights, &format!("{}.linear_2", prefix))?,
            fold: (merge * merge) as i32,
            projection_dim: projection_dim as i32,
        })
    }

    /// `x`: `[num_patches, projection_dim]`. Adjacent groups of `fold` patches
    /// are the same 2x2 spatial-merge cell (the processor emits them
    /// contiguously), so a row-major reshape concatenates their features.
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let merged = mlxcel_core::reshape(x, &[-1, self.fold * self.projection_dim]);
        let h = self.linear_1.forward(&merged);
        let h = mlxcel_core::gelu(&h);
        self.linear_2.forward(&h)
    }
}

// ============================================================================
// Encoder
// ============================================================================

pub struct MiniMaxM3VisionEncoder {
    patch_embed: PatchEmbed,
    pre_layrnorm: LayerNorm,
    rotary_pos_emb: VisionRotaryEmbedding,
    layers: Vec<VisionLayer>,
    projector: MultiModalProjector,
    patch_merge: PatchMergeMlp,
    spatial_merge_size: usize,
}

impl MiniMaxM3VisionEncoder {
    /// Build the tower from the full (raw) weight map. The vision layers live
    /// under `vision_tower.vision_model.*` while the two-stage projector lives
    /// at the top level (`multi_modal_projector.*`, `patch_merge_mlp.*`).
    pub fn from_weights(
        weights: &WeightMap,
        config: &MiniMaxM3VisionConfig,
    ) -> Result<Self, String> {
        let vt = "vision_tower.vision_model";
        let patch_embed = PatchEmbed::from_weights(
            weights,
            config,
            &format!("{}.embeddings.patch_embedding", vt),
        )?;
        let pre_layrnorm = load_layer_norm(
            weights,
            &format!("{}.pre_layrnorm", vt),
            config.layer_norm_eps,
        )?;

        // The 3D vision RoPE splits the head dim into three equal (t, h, w) axis
        // sections plus an unrotated tail. All three axes share one frequency
        // table of `axis_dim / 2` entries (their widths are identical), so a
        // single `VisionRotaryEmbedding::new(axis_dim)` covers t, h, and w.
        let axis_dim = rope_axis_dim(config.head_dim() as i32);
        let rotary_pos_emb = VisionRotaryEmbedding::new(axis_dim as usize);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(VisionLayer::from_weights(
                weights,
                config,
                &format!("{}.encoder.layers.{}", vt, i),
            )?);
        }

        let projector = MultiModalProjector::from_weights(weights, "multi_modal_projector")?;
        let patch_merge = PatchMergeMlp::from_weights(
            weights,
            "patch_merge_mlp",
            config.spatial_merge_size(),
            config.projection_dim,
        )?;

        Ok(Self {
            patch_embed,
            pre_layrnorm,
            rotary_pos_emb,
            layers,
            projector,
            patch_merge,
            spatial_merge_size: config.spatial_merge_size(),
        })
    }

    /// 3D (t, h, w) rotary position embeddings, merge-grouped to match the
    /// processor patch order. The (h, w) grouping is identical to the Qwen2-VL
    /// tower; the temporal ids repeat each frame index over its `h * w` tokens
    /// and are all-zero for images (`grid_t == 1`). Emits
    /// `[total_tokens, 3 * (axis_dim / 2)]` (the t, h, w frequency sections
    /// concatenated) for the partial rotation in `VisionAttention::forward`.
    ///
    /// `pub(crate)` (rather than private) so the unit tests in
    /// `vision::minimax_m3_vl_tests` can pin the emitted shape and the
    /// all-zero temporal section for `grid_t == 1` directly, without
    /// duplicating this method's logic.
    pub(crate) fn rot_pos_emb(&self, grid_thw: &[(i32, i32, i32)]) -> UniquePtr<MlxArray> {
        let mut all_pos_ids: Vec<UniquePtr<MlxArray>> = Vec::new();
        let mut max_grid_dim: i32 = 0;
        let merge = self.spatial_merge_size as i32;

        for &(t, h, w) in grid_thw {
            max_grid_dim = max_grid_dim.max(t).max(h).max(w);

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

            // Stack the spatial (h, w) ids and tile them across the t frames.
            let hw = mlxcel_core::stack_owned(&[hpos, wpos], -1);
            let hw = mlxcel_core::tile(&hw, &[t, 1]);

            // Temporal ids: each frame index repeated over its h * w tokens
            // (all-zero for images, where grid_t == 1).
            let t_arange = mlxcel_core::arange_i32(0, t, 1);
            let t_col = mlxcel_core::reshape(&t_arange, &[t, 1]);
            let tpos = mlxcel_core::repeat(&t_col, h * w, 1);
            let tpos = mlxcel_core::reshape(&tpos, &[t * h * w, 1]);

            // Concatenate the (t, h, w) columns in that order -> [t*h*w, 3].
            let stacked = mlxcel_core::concatenate(&tpos, &hw, 1);
            all_pos_ids.push(stacked);
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
        // [total*3, axis_dim/2] -> [total, 3, axis_dim/2] -> [total, 3*axis_dim/2],
        // concatenating the t, h, w frequency sections in that order.
        let all_freqs = mlxcel_core::reshape(&all_freqs, &[total_tokens, 3, half_dim]);
        mlxcel_core::reshape(&all_freqs, &[total_tokens, 3 * half_dim])
    }

    /// Per-image `cu_seqlens` (full attention over each image's patches):
    /// `h * w` tokens per temporal frame.
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

    /// `hidden_states`: `[num_patches, in_features]` (channels-last patch rows),
    /// `grid_thw`: per-image `(t, h, w)`. Returns `[num_merged_tokens,
    /// text_hidden]` where `num_merged_tokens = sum(t * h * w) / merge^2`.
    pub fn forward_with_grid(
        &self,
        hidden_states: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> VisionEncoderOutput {
        let hidden_states = mlxcel_core::astype(hidden_states, mlxcel_core::dtype::FLOAT32);
        let mut h = self.patch_embed.forward(&hidden_states);
        h = self.pre_layrnorm.forward(&h);

        let rotary_pos_emb = self.rot_pos_emb(grid_thw);
        let cu_seqlens = Self::compute_cu_seqlens(grid_thw);

        for layer in &self.layers {
            h = layer.forward(&h, &cu_seqlens, &rotary_pos_emb);
        }

        // Per-patch projection, then fold spatial_merge_size^2 patches.
        h = self.projector.forward(&h);
        h = self.patch_merge.forward(&h);

        VisionEncoderOutput { hidden_states: h }
    }
}

/// VisionEncoder trait - panics since grid_thw is required.
impl super::VisionEncoder for MiniMaxM3VisionEncoder {
    fn forward(&self, _pixel_values: &MlxArray) -> VisionEncoderOutput {
        panic!("MiniMax-M3-VL vision encoder requires grid_thw; use forward_with_grid() instead");
    }
}
