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

//! GLM-OCR Vision Encoder
//!
//! A close variant of the in-tree GLM-4V tower ([`super::glm4v`]): a 24-block
//! ViT with a 3D patch embedding (Conv3d evaluated as a linear), 2D vision RoPE,
//! packed variable-length attention via `cu_seqlens`, a Conv2d spatial
//! downsample, and a SwiGLU patch merger that projects into the text hidden
//! size. The GLM-OCR-specific differences from the GLM-4V tower are:
//!
//! - **Per-head q/k RMSNorm** over `head_dim` (64) applied before the vision
//!   rotary (GLM-4V has no q/k norm).
//! - **No learned position embedding and no `post_conv_layernorm`**: the
//!   checkpoint ships neither tensor, so patch embeddings feed block 0 directly
//!   and all spatial information comes from the 2D rotary.
//! - **Block norms use `rms_norm_eps` (1e-5)** rather than the hardcoded 1e-6.
//! - **Bias on qkv / proj / MLP** (auto-loaded by [`UnifiedLinear`]).
//!
//! Patch-order fix (OCR-critical): the shared `Qwen2VLProcessor` emits patches
//! in plain raster order (`(py, px)` row-major), but this tower's `host_pos_ids`
//! (rotary), the downsample's consecutive-4 window grouping, and the text-side
//! merged-token grid all assume spatial-merge-window order (the reference HF
//! layout: `(h/2, w/2, 2, 2)`). We therefore permute the patch rows from raster
//! into merge-window order right after the patch embedding so every downstream
//! stage agrees. GLM-4V leaves this latent misalignment in place (tolerable for
//! coarse VQA), but for OCR a spatial scramble destroys recognition, so the
//! GLM-OCR path fixes it.
//!
//! Used by: GLM-OCR (`glm_ocr`).
//! Reference: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/glm4v/vision.py

use super::VisionEncoderOutput;
use super::glm4v::Glm4vVisionConfig;
use super::qwen2_vl::{VisionRotaryEmbedding, apply_rotary_pos_emb_vision, concat_many};
use mlxcel_core::layers::{LayerNorm, RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// Raster -> merge-window permutation over all patches. `perm[k]` is the raster
/// row index (in the patch-embed output) of the k-th patch in spatial-merge-window
/// order, so gathering with it makes the downstream rotary / downsample /
/// merged-token grid consistent. Free function so the OCR-critical ordering can
/// be unit-tested without building a full encoder.
pub(super) fn merge_window_perm(grid_thw: &[(i32, i32, i32)], merge: i32) -> Vec<i32> {
    let mut perm = Vec::new();
    let mut base = 0i32;
    for &(t, h, w) in grid_thw {
        for _ in 0..t {
            for hb in 0..(h / merge) {
                for wb in 0..(w / merge) {
                    for hi in 0..merge {
                        for wi in 0..merge {
                            let py = hb * merge + hi;
                            let px = wb * merge + wi;
                            perm.push(base + py * w + px);
                        }
                    }
                }
            }
            base += h * w;
        }
    }
    perm
}

fn load_rms_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<RMSNorm, String> {
    let key = format!("{}.weight", prefix);
    let weight = weights
        .get(&key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", key))?;
    Ok(RMSNorm::new(weight, eps))
}

fn load_layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    let weight = weights
        .get(&format!("{}.weight", prefix))
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}.weight", prefix))?;
    let bias = weights
        .get(&format!("{}.bias", prefix))
        .map(|w| mlxcel_core::copy(w));
    Ok(LayerNorm::new(weight, bias, eps))
}

// PatchEmbed - Conv3d degenerated to Linear (kernel == stride). Supports both
// the pre-permuted channels-last export layout `[out, kT, kH, kW, in]` and the
// raw channels-second checkpoint layout `[out, in, kT, kH, kW]`.
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
        config: &Glm4vVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let weight_key = format!("{}.proj.weight", prefix);
        let w = weights
            .get(&weight_key)
            .ok_or_else(|| format!("Missing {}", weight_key))?;

        let shape = mlxcel_core::array_shape(w);
        let out_features = config.hidden_size as i32;
        let in_ch = config.in_channels as i32;
        let temporal = config.temporal_patch_size as i32;
        let in_features = (config.in_channels
            * config.temporal_patch_size
            * config.patch_size
            * config.patch_size) as i32;

        // Flatten to `[out, in_features]` in (kT, in, kH, kW) order to match the
        // per-patch input flatten (temporal, channel, patch_h, patch_w).
        let w_reshaped = if shape.len() == 5 {
            if shape[4] == in_ch {
                // Channels-last export `[out, kT, kH, kW, in]` -> [0, 1, 4, 2, 3].
                let w_reordered = mlxcel_core::transpose_axes(w, &[0, 1, 4, 2, 3]);
                mlxcel_core::reshape(&w_reordered, &[out_features, in_features])
            } else if shape[1] == in_ch && shape[2] == temporal {
                // Raw checkpoint `[out, in, kT, kH, kW]` -> [0, 2, 1, 3, 4].
                let w_reordered = mlxcel_core::transpose_axes(w, &[0, 2, 1, 3, 4]);
                mlxcel_core::reshape(&w_reordered, &[out_features, in_features])
            } else {
                return Err(format!(
                    "Unexpected GLM-OCR patch_embed weight shape: {:?}",
                    shape
                ));
            }
        } else if shape.len() == 2 {
            mlxcel_core::copy(w)
        } else {
            return Err(format!(
                "Unexpected GLM-OCR patch_embed weight shape: {:?}",
                shape
            ));
        };

        let proj_bias = weights
            .get(&format!("{}.proj.bias", prefix))
            .map(|w| mlxcel_core::copy(w));

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

// Vision attention - fused QKV with per-head q/k RMSNorm, packed sequences.
struct VisionAttention {
    qkv: UnifiedLinear,
    proj: UnifiedLinear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl VisionAttention {
    fn from_weights(
        weights: &WeightMap,
        config: &Glm4vVisionConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let qkv = UnifiedLinear::from_weights(weights, &format!("{}.attn.qkv", prefix), gs, bits)?;
        let proj =
            UnifiedLinear::from_weights(weights, &format!("{}.attn.proj", prefix), gs, bits)?;
        let head_dim = (config.hidden_size / config.num_heads) as i32;
        let q_norm = load_rms_norm(
            weights,
            &format!("{}.attn.q_norm", prefix),
            config.rms_norm_eps,
        )?;
        let k_norm = load_rms_norm(
            weights,
            &format!("{}.attn.k_norm", prefix),
            config.rms_norm_eps,
        )?;
        Ok(Self {
            qkv,
            proj,
            q_norm,
            k_norm,
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

        // Per-head RMSNorm over head_dim before rotary (GLM-OCR-specific).
        let q = self.q_norm.forward(&q);
        let k = self.k_norm.forward(&k);

        let q = apply_rotary_pos_emb_vision(&q, rotary_pos_emb);
        let k = apply_rotary_pos_emb_vision(&k, rotary_pos_emb);

        let q = mlxcel_core::transpose_axes(&q, &[1, 0, 2]);
        let k = mlxcel_core::transpose_axes(&k, &[1, 0, 2]);
        let v = mlxcel_core::transpose_axes(&v, &[1, 0, 2]);
        let q = mlxcel_core::expand_dims(&q, 0);
        let k = mlxcel_core::expand_dims(&k, 0);
        let v = mlxcel_core::expand_dims(&v, 0);

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

// Vision MLP - SwiGLU (gate/up -> silu(gate)*up -> down), all with bias.
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
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.down_proj.forward(&activated)
    }
}

// Transformer block - RMSNorm pre-norm (eps from config) + attention + SwiGLU.
struct VisionBlock {
    norm1: RMSNorm,
    norm2: RMSNorm,
    attn: VisionAttention,
    mlp: VisionMLP,
}

impl VisionBlock {
    fn from_weights(
        weights: &WeightMap,
        config: &Glm4vVisionConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            norm1: load_rms_norm(weights, &format!("{}.norm1", prefix), config.rms_norm_eps)?,
            norm2: load_rms_norm(weights, &format!("{}.norm2", prefix), config.rms_norm_eps)?,
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

// Conv2d spatial downsample (kernel == stride == merge). Supports both the
// channels-last export layout `[out, kH, kW, in]` and the raw channels-second
// checkpoint layout `[out, in, kH, kW]`.
struct Downsample {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
    block_features: i32,
    out_hidden_size: i32,
}

impl Downsample {
    fn from_weights(
        weights: &WeightMap,
        config: &Glm4vVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let key = format!("{}.weight", prefix);
        let w = weights
            .get(&key)
            .ok_or_else(|| format!("Missing {}", key))?;
        let shape = mlxcel_core::array_shape(w);
        let out_hidden_size = config.out_hidden_size as i32;
        let merge = config.spatial_merge_size as i32;
        let hidden = config.hidden_size as i32;
        let block_features = merge * merge * hidden;

        // Flatten to `[out, block_features]` in (kH, kW, channels) order to match
        // the `[N, merge, merge, hidden]` input-block flatten.
        let weight = if shape.len() == 4 {
            if shape[3] == hidden {
                // Channels-last `[out, kH, kW, in]` -> reshape directly.
                mlxcel_core::reshape(w, &[out_hidden_size, block_features])
            } else if shape[1] == hidden {
                // Raw `[out, in, kH, kW]` -> [0, 2, 3, 1] then reshape.
                let w_reordered = mlxcel_core::transpose_axes(w, &[0, 2, 3, 1]);
                mlxcel_core::reshape(&w_reordered, &[out_hidden_size, block_features])
            } else {
                return Err(format!(
                    "Unexpected GLM-OCR downsample weight shape: {:?}",
                    shape
                ));
            }
        } else if shape.len() == 2 {
            mlxcel_core::copy(w)
        } else {
            return Err(format!(
                "Unexpected GLM-OCR downsample weight shape: {:?}",
                shape
            ));
        };

        let bias = weights
            .get(&format!("{}.bias", prefix))
            .map(|w| mlxcel_core::copy(w));
        Ok(Self {
            weight,
            bias,
            block_features,
            out_hidden_size,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // x: [total, hidden] (merge-window order) -> [total/4, merge*merge*hidden].
        let h = mlxcel_core::reshape(x, &[-1, self.block_features]);
        let wt = mlxcel_core::transpose(&self.weight);
        let out = mlxcel_core::matmul(&h, &wt);
        let out = match &self.bias {
            Some(b) => mlxcel_core::add(&out, b),
            None => out,
        };
        mlxcel_core::reshape(&out, &[-1, self.out_hidden_size])
    }
}

// Patch merger - proj + LayerNorm + gelu + SwiGLU projecting to text hidden.
struct PatchMerger {
    proj: UnifiedLinear,
    post_projection_norm: LayerNorm,
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl PatchMerger {
    fn from_weights(weights: &WeightMap, prefix: &str, gs: i32, bits: i32) -> Result<Self, String> {
        Ok(Self {
            proj: UnifiedLinear::from_weights(weights, &format!("{}.proj", prefix), gs, bits)?,
            post_projection_norm: load_layer_norm(
                weights,
                &format!("{}.post_projection_norm", prefix),
                1e-5,
            )?,
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.gate_proj", prefix),
                gs,
                bits,
            )?,
            up_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.up_proj", prefix),
                gs,
                bits,
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{}.down_proj", prefix),
                gs,
                bits,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let x = self.proj.forward(x);
        let x = self.post_projection_norm.forward(&x);
        let x = mlxcel_core::gelu(&x);
        let gate = self.gate_proj.forward(&x);
        let up = self.up_proj.forward(&x);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.down_proj.forward(&activated)
    }
}

/// GLM-OCR vision model.
pub struct GlmOcrVisionEncoder {
    patch_embed: PatchEmbed,
    rotary_pos_emb: VisionRotaryEmbedding,
    blocks: Vec<VisionBlock>,
    post_layernorm: RMSNorm,
    downsample: Downsample,
    merger: PatchMerger,
    spatial_merge_size: usize,
}

impl GlmOcrVisionEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        config: &Glm4vVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        // The checkpoint must not carry the GLM-4V-only position embedding or
        // post-conv norm; if it does, we are loading the wrong tower.
        if weights.contains_key(&format!("{}.embeddings.position_embedding.weight", prefix)) {
            return Err(
                "GLM-OCR tower unexpectedly has embeddings.position_embedding (GLM-4V layout)"
                    .to_string(),
            );
        }
        if weights.contains_key(&format!("{}.post_conv_layernorm.weight", prefix)) {
            return Err(
                "GLM-OCR tower unexpectedly has post_conv_layernorm (GLM-4V layout)".to_string(),
            );
        }

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

        let post_layernorm = load_rms_norm(
            weights,
            &format!("{}.post_layernorm", prefix),
            config.rms_norm_eps,
        )?;
        let downsample =
            Downsample::from_weights(weights, config, &format!("{}.downsample", prefix))?;
        let merger = PatchMerger::from_weights(weights, &format!("{}.merger", prefix), gs, bits)?;

        Ok(Self {
            patch_embed,
            rotary_pos_emb,
            blocks,
            post_layernorm,
            downsample,
            merger,
            spatial_merge_size: config.spatial_merge_size,
        })
    }

    /// Per-patch (h, w) grid indices in spatial-merge order plus the max grid
    /// dimension for the rotary table.
    fn host_pos_ids(&self, grid_thw: &[(i32, i32, i32)]) -> (Vec<i32>, Vec<i32>, i32) {
        let merge = self.spatial_merge_size as i32;
        let mut h_ids = Vec::new();
        let mut w_ids = Vec::new();
        let mut max_grid = 0i32;
        for &(t, h, w) in grid_thw {
            max_grid = max_grid.max(h).max(w);
            let mut hpos = vec![0i32; (h * w) as usize];
            let mut wpos = vec![0i32; (h * w) as usize];
            let mut out = 0usize;
            for hb in 0..(h / merge) {
                for wb in 0..(w / merge) {
                    for hi in 0..merge {
                        for wi in 0..merge {
                            hpos[out] = hb * merge + hi;
                            wpos[out] = wb * merge + wi;
                            out += 1;
                        }
                    }
                }
            }
            for _ in 0..t {
                h_ids.extend_from_slice(&hpos);
                w_ids.extend_from_slice(&wpos);
            }
        }
        (h_ids, w_ids, max_grid)
    }

    fn rotary_table(&self, h_ids: &[i32], w_ids: &[i32], max_grid: i32) -> UniquePtr<MlxArray> {
        let total = h_ids.len() as i32;
        let mut pair_ids = Vec::with_capacity(h_ids.len() * 2);
        for i in 0..h_ids.len() {
            pair_ids.push(h_ids[i]);
            pair_ids.push(w_ids[i]);
        }
        let table = self.rotary_pos_emb.forward(max_grid);
        let idx = mlxcel_core::from_slice_i32(&pair_ids, &[total * 2]);
        let all_freqs = mlxcel_core::take(&table, &idx, 0);
        let freq_shape = mlxcel_core::array_shape(&all_freqs);
        let half_dim = freq_shape[1];
        let all_freqs = mlxcel_core::reshape(&all_freqs, &[total, 2, half_dim]);
        mlxcel_core::reshape(&all_freqs, &[total, 2 * half_dim])
    }

    fn compute_cu_seqlens(grid_thw: &[(i32, i32, i32)]) -> Vec<i32> {
        let mut cu = vec![0i32];
        let mut cumulative = 0i32;
        for &(t, h, w) in grid_thw {
            let per_frame = h * w;
            for _ in 0..t {
                cumulative += per_frame;
                cu.push(cumulative);
            }
        }
        cu
    }

    /// Forward pass.
    ///
    /// `hidden_states`: `[total * temporal_patch_size, C*P*P]` in the raster
    /// order emitted by `Qwen2VLProcessor`. `grid_thw`: `(temporal, height,
    /// width)` per image (patch units).
    pub fn forward_with_grid(
        &self,
        hidden_states: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> VisionEncoderOutput {
        let mut h = self.patch_embed.forward(hidden_states);

        // Raster -> merge-window reorder (OCR spatial-alignment fix).
        let perm = merge_window_perm(grid_thw, self.spatial_merge_size as i32);
        let perm_arr = mlxcel_core::from_slice_i32(&perm, &[perm.len() as i32]);
        h = mlxcel_core::take(&h, &perm_arr, 0);

        let (h_ids, w_ids, max_grid) = self.host_pos_ids(grid_thw);
        let rotary_pos_emb = self.rotary_table(&h_ids, &w_ids, max_grid);
        let cu_seqlens = Self::compute_cu_seqlens(grid_thw);

        for block in &self.blocks {
            h = block.forward(&h, &cu_seqlens, &rotary_pos_emb);
        }

        h = self.post_layernorm.forward(&h);
        h = self.downsample.forward(&h);
        h = self.merger.forward(&h);

        VisionEncoderOutput { hidden_states: h }
    }
}

impl super::VisionEncoder for GlmOcrVisionEncoder {
    fn forward(&self, _pixel_values: &MlxArray) -> VisionEncoderOutput {
        panic!("GLM-OCR vision encoder requires grid_thw; use forward_with_grid() instead");
    }
}

#[cfg(test)]
#[path = "glm_ocr_tests.rs"]
mod tests;
