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

//! GLM-4V Vision Encoder
//!
//! A ViT close to the Qwen2-VL tower (3D patch embedding as a linear, 2D vision
//! RoPE with half-split rotation, packed variable-length attention via
//! `cu_seqlens`) plus the GLM-4V specific pieces:
//! - a post-conv RMSNorm right after patch embedding,
//! - adaptive learned position embeddings resampled with bilinear
//!   `grid_sample` (coordinates and weights computed host-side, gather in MLX),
//! - RMSNorm + SwiGLU transformer blocks,
//! - a Conv2d spatial downsample (kernel == stride == merge) applied as a
//!   linear over `merge x merge x hidden` blocks,
//! - a SwiGLU patch merger that projects to the text hidden size.
//!
//! Used by: GLM-4V, GLM-4V MoE (shared encoder)
//! Reference: references/mlx-vlm/mlx_vlm/models/glm4v/vision.py

use super::VisionEncoderOutput;
use super::qwen2_vl::{VisionRotaryEmbedding, apply_rotary_pos_emb_vision, concat_many};
use mlxcel_core::layers::{LayerNorm, RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

/// GLM-4V vision encoder configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Glm4vVisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub patch_size: usize,
    #[serde(default = "default_out_hidden_size")]
    pub out_hidden_size: usize,
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: usize,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    #[serde(alias = "in_chans", default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// Quantization group_size (inherited from the top-level config).
    #[serde(default)]
    pub quant_group_size: i32,
    /// Quantization bits (inherited from the top-level config).
    #[serde(default)]
    pub quant_bits: i32,
}

fn default_out_hidden_size() -> usize {
    4096
}
fn default_spatial_merge_size() -> usize {
    2
}
fn default_temporal_patch_size() -> usize {
    2
}
fn default_image_size() -> usize {
    336
}
fn default_in_channels() -> usize {
    3
}
fn default_rms_norm_eps() -> f32 {
    1e-5
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

// PatchEmbed - Conv3d degenerated to Linear (kernel == stride), matching the
// Qwen2-VL tower. Consumes the [total * temporal_patch_size, C*P*P] patch
// layout emitted by the shared Qwen-VL processor.
struct PatchEmbed {
    proj_weight: UniquePtr<MlxArray>,
    proj_bias: Option<UniquePtr<MlxArray>>,
    in_channels: usize,
    temporal_patch_size: usize,
    patch_size: usize,
    embed_dim: usize,
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
        let in_features = (config.in_channels
            * config.temporal_patch_size
            * config.patch_size
            * config.patch_size) as i32;

        // MLX Conv3d weight [out, kT, kH, kW, in] -> reorder to [out, kT, in,
        // kH, kW] so the flattened linear matches the (T, C, H, W) patch order.
        let w_reshaped = if shape.len() == 5 {
            let w_reordered = mlxcel_core::transpose_axes(w, &[0, 1, 4, 2, 3]);
            mlxcel_core::reshape(&w_reordered, &[out_features, in_features])
        } else if shape.len() == 2 {
            mlxcel_core::copy(w)
        } else {
            return Err(format!("Unexpected patch_embed weight shape: {:?}", shape));
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
            embed_dim: config.hidden_size,
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
        let _ = self.embed_dim;
        match &self.proj_bias {
            Some(b) => mlxcel_core::add(&result, b),
            None => result,
        }
    }
}

// Adaptive learned position embeddings resampled with bilinear grid_sample.
struct VisionEmbeddings {
    // [orig * orig, hidden] learned grid, cast to f32 for interpolation.
    position_embedding: UniquePtr<MlxArray>,
    orig_size: i32,
    hidden_size: usize,
}

impl VisionEmbeddings {
    fn from_weights(
        weights: &WeightMap,
        config: &Glm4vVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let key = format!("{}.position_embedding.weight", prefix);
        let w = weights
            .get(&key)
            .ok_or_else(|| format!("Missing {}", key))?;
        let shape = mlxcel_core::array_shape(w);
        let num_positions = shape[0];
        let orig_size = (num_positions as f64).sqrt().round() as i32;
        let position_embedding = mlxcel_core::astype(w, mlxcel_core::dtype::FLOAT32);
        Ok(Self {
            position_embedding,
            orig_size,
            hidden_size: config.hidden_size,
        })
    }

    /// Add resampled position embeddings to `embeddings` `[total, hidden]`.
    ///
    /// `h_coords`/`w_coords` are per-patch grid indices, `target_h`/`target_w`
    /// the full grid dims of each patch's image. Bilinear coordinates and
    /// corner weights are computed on host, then gathered in MLX.
    fn forward(
        &self,
        embeddings: &MlxArray,
        h_coords: &[i32],
        w_coords: &[i32],
        target_h: &[i32],
        target_w: &[i32],
    ) -> UniquePtr<MlxArray> {
        let total = h_coords.len();
        if total == 0 {
            return mlxcel_core::copy(embeddings);
        }
        let orig = self.orig_size;
        let orig_f = orig as f32;

        let mut idx_a = Vec::with_capacity(total);
        let mut idx_b = Vec::with_capacity(total);
        let mut idx_c = Vec::with_capacity(total);
        let mut idx_d = Vec::with_capacity(total);
        let mut wa = Vec::with_capacity(total);
        let mut wb = Vec::with_capacity(total);
        let mut wc = Vec::with_capacity(total);
        let mut wd = Vec::with_capacity(total);

        for i in 0..total {
            let norm_w = ((w_coords[i] as f32 + 0.5) / target_w[i] as f32) * 2.0 - 1.0;
            let norm_h = ((h_coords[i] as f32 + 0.5) / target_h[i] as f32) * 2.0 - 1.0;
            let ix = ((norm_w + 1.0) * orig_f - 1.0) / 2.0;
            let iy = ((norm_h + 1.0) * orig_f - 1.0) / 2.0;
            let ix0 = ix.floor();
            let iy0 = iy.floor();
            let ix0i = ix0 as i32;
            let iy0i = iy0 as i32;
            let ix1i = ix0i + 1;
            let iy1i = iy0i + 1;
            let fx = ix - ix0;
            let fy = iy - iy0;

            let corner = |yy: i32, xx: i32| -> (i32, f32) {
                let valid = yy >= 0 && yy < orig && xx >= 0 && xx < orig;
                let yc = yy.clamp(0, orig - 1);
                let xc = xx.clamp(0, orig - 1);
                (yc * orig + xc, if valid { 1.0 } else { 0.0 })
            };

            let (ia, va) = corner(iy0i, ix0i);
            let (ib, vb) = corner(iy0i, ix1i);
            let (ic, vc) = corner(iy1i, ix0i);
            let (id, vd) = corner(iy1i, ix1i);

            idx_a.push(ia);
            idx_b.push(ib);
            idx_c.push(ic);
            idx_d.push(id);
            wa.push((1.0 - fx) * (1.0 - fy) * va);
            wb.push(fx * (1.0 - fy) * vb);
            wc.push((1.0 - fx) * fy * vc);
            wd.push(fx * fy * vd);
        }

        let total_i = total as i32;
        let gather = |idx: &[i32], w: &[f32]| -> UniquePtr<MlxArray> {
            let idx_arr = mlxcel_core::from_slice_i32(idx, &[total_i]);
            let g = mlxcel_core::take(&self.position_embedding, &idx_arr, 0);
            let w_arr = mlxcel_core::from_slice_f32(w, &[total_i, 1]);
            mlxcel_core::multiply(&g, &w_arr)
        };

        let mut adapted = gather(&idx_a, &wa);
        let gb = gather(&idx_b, &wb);
        adapted = mlxcel_core::add(&adapted, &gb);
        let gc = gather(&idx_c, &wc);
        adapted = mlxcel_core::add(&adapted, &gc);
        let gd = gather(&idx_d, &wd);
        adapted = mlxcel_core::add(&adapted, &gd);

        let embed_dtype = mlxcel_core::array_dtype(embeddings);
        let adapted = mlxcel_core::astype(&adapted, embed_dtype);
        let _ = self.hidden_size;
        mlxcel_core::add(embeddings, &adapted)
    }
}

// Vision attention - fused QKV, packed sequences (no bias).
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
        config: &Glm4vVisionConfig,
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

// Vision MLP - SwiGLU (gate/up -> silu(gate)*up -> down).
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

// Transformer block - RMSNorm pre-norm + attention + SwiGLU MLP.
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
            norm1: load_rms_norm(weights, &format!("{}.norm1", prefix), 1e-6)?,
            norm2: load_rms_norm(weights, &format!("{}.norm2", prefix), 1e-6)?,
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

// Conv2d spatial downsample (kernel == stride == merge) applied as a linear
// over each merge x merge x hidden block.
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
        let out_hidden_size = config.out_hidden_size as i32;
        let merge = config.spatial_merge_size as i32;
        let block_features = merge * merge * config.hidden_size as i32;
        // MLX Conv2d weight [out, kH, kW, in] flattens directly in (h, w, c)
        // order to match the [N, merge, merge, hidden] input block flatten.
        let weight = mlxcel_core::reshape(w, &[out_hidden_size, block_features]);
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
        // x: [total, hidden] -> [N, merge*merge*hidden].
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

/// GLM-4V vision model.
pub struct Glm4vVisionEncoder {
    patch_embed: PatchEmbed,
    post_conv_layernorm: RMSNorm,
    embeddings: VisionEmbeddings,
    rotary_pos_emb: VisionRotaryEmbedding,
    blocks: Vec<VisionBlock>,
    post_layernorm: RMSNorm,
    downsample: Downsample,
    merger: PatchMerger,
    spatial_merge_size: usize,
}

impl Glm4vVisionEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        config: &Glm4vVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.quant_group_size;
        let bits = config.quant_bits;

        let patch_embed =
            PatchEmbed::from_weights(weights, config, &format!("{}.patch_embed", prefix))?;
        let post_conv_layernorm = load_rms_norm(
            weights,
            &format!("{}.post_conv_layernorm", prefix),
            config.rms_norm_eps,
        )?;
        let embeddings =
            VisionEmbeddings::from_weights(weights, config, &format!("{}.embeddings", prefix))?;

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
            post_conv_layernorm,
            embeddings,
            rotary_pos_emb,
            blocks,
            post_layernorm,
            downsample,
            merger,
            spatial_merge_size: config.spatial_merge_size,
        })
    }

    /// Compute per-patch (h, w) grid indices in spatial-merge order plus the
    /// max grid dimension for the rotary table.
    fn host_pos_ids(&self, grid_thw: &[(i32, i32, i32)]) -> (Vec<i32>, Vec<i32>, i32) {
        let merge = self.spatial_merge_size as i32;
        let mut h_ids = Vec::new();
        let mut w_ids = Vec::new();
        let mut max_grid = 0i32;
        for &(t, h, w) in grid_thw {
            max_grid = max_grid.max(h).max(w);
            // hpos: arange(h) repeated across w, regrouped into merge blocks.
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

    /// Build the [total, head_dim] rotary frequency table from host pos ids.
    fn rotary_table(&self, h_ids: &[i32], w_ids: &[i32], max_grid: i32) -> UniquePtr<MlxArray> {
        let total = h_ids.len() as i32;
        // Interleave [h, w] pair indices: [h0, w0, h1, w1, ...].
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
    /// `hidden_states`: `[total * temporal_patch_size, C*P*P]`.
    /// `grid_thw`: `(temporal, height, width)` per image (patch units).
    pub fn forward_with_grid(
        &self,
        hidden_states: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> VisionEncoderOutput {
        let mut h = self.patch_embed.forward(hidden_states);
        h = self.post_conv_layernorm.forward(&h);

        let (h_ids, w_ids, max_grid) = self.host_pos_ids(grid_thw);
        let rotary_pos_emb = self.rotary_table(&h_ids, &w_ids, max_grid);

        // Per-patch target grid dims (repeat each image's h/w over its tokens).
        let mut target_h = Vec::with_capacity(h_ids.len());
        let mut target_w = Vec::with_capacity(w_ids.len());
        for &(t, gh, gw) in grid_thw {
            for _ in 0..(t * gh * gw) {
                target_h.push(gh);
                target_w.push(gw);
            }
        }
        h = self
            .embeddings
            .forward(&h, &h_ids, &w_ids, &target_h, &target_w);

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

impl super::VisionEncoder for Glm4vVisionEncoder {
    fn forward(&self, _pixel_values: &MlxArray) -> VisionEncoderOutput {
        panic!("GLM-4V vision encoder requires grid_thw; use forward_with_grid() instead");
    }
}
