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

//! Hunyuan-VL vision tower (`vision_tower.*`).
//!
//! A ViT over raster-ordered 768-wide patch rows (`3 * 16 * 16`): a per-patch
//! conv embedding folded into a linear, a learned position-embedding table
//! (`(max_image_size / patch)^2 + 1` entries, CLS row skipped) bilinearly
//! interpolated to each image's grid, pre-LN blocks with full attention over
//! the whole packed sequence and exact-GELU MLPs, and a `perceive` PatchMerger
//! per image: RMSNorm, a stride-2 conv pair over the raster grid, a learned
//! `image_newline` column, a linear to the text width, and learned
//! `image_begin` / `image_end` rows, RMSNorm-ed. Per image the output is
//! `mh * (mw + 1) + 2` rows (`mh = grid_h / merge`, `mw = grid_w / merge`),
//! which is exactly the prompt placeholder count.
//!
//! Reference: mlx-vlm
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/hunyuan_vl/vision.py>.

use crate::vision::encoders::gemma3n::Conv2dLayer;
use mlxcel_core::layers::{LayerNorm, Linear, RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

fn default_hidden_size() -> usize {
    1152
}
fn default_out_hidden_size() -> usize {
    1024
}
fn default_layers() -> usize {
    27
}
fn default_heads() -> usize {
    16
}
fn default_intermediate() -> usize {
    4304
}
fn default_patch_size() -> usize {
    16
}
fn default_merge() -> usize {
    2
}
fn default_eps() -> f32 {
    1e-5
}
fn default_max_image_size() -> usize {
    2048
}

#[derive(Debug, Clone, Deserialize)]
pub struct HunyuanVlVisionConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_out_hidden_size")]
    pub out_hidden_size: usize,
    #[serde(default = "default_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_intermediate")]
    pub intermediate_size: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_merge")]
    pub spatial_merge_size: usize,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_max_image_size")]
    pub max_image_size: usize,
    #[serde(default)]
    pub quant_group_size: i32,
    #[serde(default)]
    pub quant_bits: i32,
}

fn get(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Hunyuan-VL vision weight missing: {name}"))
}

fn load_layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    let weight = get(weights, &format!("{prefix}.weight"))?;
    let bias = weights
        .get(&format!("{prefix}.bias"))
        .map(|b| mlxcel_core::copy(b));
    Ok(LayerNorm::new(weight, bias, eps))
}

// Full-attention block (packed sequence, no mask).
struct VisionBlock {
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    fc1: UnifiedLinear, // dense_h_to_4h
    fc2: UnifiedLinear, // dense_4h_to_h
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl VisionBlock {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);

        let normed = self.input_layernorm.forward(x);
        let q = self.q_proj.forward(&normed);
        let k = self.k_proj.forward(&normed);
        let v = self.v_proj.forward(&normed);
        let to_bhsd = |t: &MlxArray| {
            let t = mlxcel_core::reshape(t, &[b, l, self.num_heads, self.head_dim]);
            mlxcel_core::transpose_axes(&t, &[0, 2, 1, 3])
        };
        let q = to_bhsd(&q);
        let k = to_bhsd(&k);
        let v = to_bhsd(&v);
        // SAFETY: q/k/v valid; null mask (full bidirectional attention).
        let attn = unsafe {
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
        let attn = mlxcel_core::transpose_axes(&attn, &[0, 2, 1, 3]);
        let attn = mlxcel_core::reshape(&attn, &[b, l, -1]);
        let attn = self.o_proj.forward(&attn);
        let h = mlxcel_core::add(x, &attn);

        let normed = self.post_attention_layernorm.forward(&h);
        let m = self.fc1.forward(&normed);
        let m = mlxcel_core::gelu(&m); // exact erf GELU
        let m = self.fc2.forward(&m);
        mlxcel_core::add(&h, &m)
    }
}

// PatchMerger ("perceive"): per-image spatial downsample + framing embeddings.
struct PatchMerger {
    before_rms: RMSNorm,
    after_rms: RMSNorm,
    conv0: Conv2dLayer,                 // hidden -> 2*hidden, k = merge, s = merge
    conv2: Conv2dLayer,                 // 2*hidden -> 4*hidden, k = 1
    mlp: UnifiedLinear,                 // 4*hidden -> out_hidden
    image_newline: UniquePtr<MlxArray>, // (4*hidden,)
    image_begin: UniquePtr<MlxArray>,   // (out_hidden,)
    image_end: UniquePtr<MlxArray>,     // (out_hidden,)
    merge: i32,
    hidden: i32,
    final_hidden: i32,
    out_hidden: i32,
}

impl PatchMerger {
    /// `x`: `(1, grid_h * grid_w, hidden)` raster-ordered tokens for one image.
    /// Returns `(1, mh * (mw + 1) + 2, out_hidden)`.
    fn forward(&self, x: &MlxArray, grid_h: i32, grid_w: i32) -> UniquePtr<MlxArray> {
        let x = self.before_rms.forward(x);
        let x = mlxcel_core::reshape(&x, &[1, grid_h, grid_w, self.hidden]);
        let x = self.conv0.forward(&x);
        let x = mlxcel_core::gelu(&x);
        let x = self.conv2.forward(&x); // (1, mh, mw, final_hidden)

        let mh = grid_h / self.merge;
        let mw = grid_w / self.merge;
        let nl = mlxcel_core::reshape(&self.image_newline, &[1, 1, 1, self.final_hidden]);
        let nl = mlxcel_core::astype(&nl, mlxcel_core::array_dtype(&x));
        let nl = mlxcel_core::broadcast_to(&nl, &[1, mh, 1, self.final_hidden]);
        let x = mlxcel_core::concatenate(&x, &nl, 2); // (1, mh, mw+1, F)
        let x = mlxcel_core::reshape(&x, &[1, mh * (mw + 1), self.final_hidden]);

        let x = self.mlp.forward(&x); // (1, rows, out_hidden)

        let frame = |v: &MlxArray| {
            let f = mlxcel_core::reshape(v, &[1, 1, self.out_hidden]);
            mlxcel_core::astype(&f, mlxcel_core::array_dtype(&x))
        };
        let begin = frame(&self.image_begin);
        let end = frame(&self.image_end);
        let x = mlxcel_core::concatenate(&begin, &x, 1);
        let x = mlxcel_core::concatenate(&x, &end, 1);
        self.after_rms.forward(&x)
    }
}

pub struct HunyuanVlVisionEncoder {
    patch_embed: Linear,            // (hidden, 3 * p * p) with bias
    pos_table: UniquePtr<MlxArray>, // (edge, edge, hidden), CLS row dropped
    pos_edge: i32,
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
    merge: i32,
}

impl HunyuanVlVisionEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        config: &HunyuanVlVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.quant_group_size;
        let bits = config.quant_bits;
        let hidden = config.hidden_size as i32;
        let head_dim = (config.hidden_size / config.num_attention_heads) as i32;
        let p = config.patch_size as i32;

        // Patch conv folded into a Linear over (c, py, px)-ordered rows. The
        // MLX conv weight is channels-last (out, kH, kW, C); the processor rows
        // flatten (C, py, px), so permute to (out, C, kH, kW) before reshape.
        let conv_w = get(
            weights,
            &format!("{prefix}.embeddings.patch_embedding.weight"),
        )?;
        let shape = mlxcel_core::array_shape(&conv_w);
        let patch_w = if shape.len() == 4 {
            let t = mlxcel_core::transpose_axes(&conv_w, &[0, 3, 1, 2]);
            mlxcel_core::reshape(&t, &[hidden, 3 * p * p])
        } else {
            conv_w
        };
        let patch_b = get(
            weights,
            &format!("{prefix}.embeddings.patch_embedding.bias"),
        )?;
        let patch_embed = Linear::new(patch_w, Some(patch_b));

        // Learned positions: table row 0 is the (unused) CLS slot.
        let table = get(
            weights,
            &format!("{prefix}.embeddings.position_embedding.weight"),
        )?;
        let table_shape = mlxcel_core::array_shape(&table);
        let n = table_shape[0] - 1;
        let edge = (n as f64).sqrt().round() as i32;
        let pos_table = mlxcel_core::slice(&table, &[1, 0], &[table_shape[0], hidden]);
        let pos_table = mlxcel_core::reshape(&pos_table, &[edge, edge, hidden]);

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let bp = format!("{prefix}.layers.{i}");
            blocks.push(VisionBlock {
                input_layernorm: load_layer_norm(
                    weights,
                    &format!("{bp}.input_layernorm"),
                    config.rms_norm_eps,
                )?,
                post_attention_layernorm: load_layer_norm(
                    weights,
                    &format!("{bp}.post_attention_layernorm"),
                    config.rms_norm_eps,
                )?,
                q_proj: UnifiedLinear::from_weights(
                    weights,
                    &format!("{bp}.self_attn.q_proj"),
                    gs,
                    bits,
                )?,
                k_proj: UnifiedLinear::from_weights(
                    weights,
                    &format!("{bp}.self_attn.k_proj"),
                    gs,
                    bits,
                )?,
                v_proj: UnifiedLinear::from_weights(
                    weights,
                    &format!("{bp}.self_attn.v_proj"),
                    gs,
                    bits,
                )?,
                o_proj: UnifiedLinear::from_weights(
                    weights,
                    &format!("{bp}.self_attn.o_proj"),
                    gs,
                    bits,
                )?,
                fc1: UnifiedLinear::from_weights(
                    weights,
                    &format!("{bp}.mlp.dense_h_to_4h"),
                    gs,
                    bits,
                )?,
                fc2: UnifiedLinear::from_weights(
                    weights,
                    &format!("{bp}.mlp.dense_4h_to_h"),
                    gs,
                    bits,
                )?,
                num_heads: config.num_attention_heads as i32,
                head_dim,
                scale: (head_dim as f32).powf(-0.5),
            });
        }

        let mp = format!("{prefix}.perceive");
        let merge = config.spatial_merge_size as i32;
        let final_hidden = hidden * 4;
        let merger = PatchMerger {
            before_rms: RMSNorm::new(
                get(weights, &format!("{mp}.before_rms.weight"))?,
                config.rms_norm_eps,
            ),
            after_rms: RMSNorm::new(
                get(weights, &format!("{mp}.after_rms.weight"))?,
                config.rms_norm_eps,
            ),
            conv0: Conv2dLayer {
                weight: get(weights, &format!("{mp}.proj.0.weight"))?,
                bias: Some(get(weights, &format!("{mp}.proj.0.bias"))?),
                stride_h: merge,
                stride_w: merge,
                padding_h: 0,
                padding_w: 0,
                dilation_h: 1,
                dilation_w: 1,
                groups: 1,
            },
            conv2: Conv2dLayer {
                weight: get(weights, &format!("{mp}.proj.2.weight"))?,
                bias: Some(get(weights, &format!("{mp}.proj.2.bias"))?),
                stride_h: 1,
                stride_w: 1,
                padding_h: 0,
                padding_w: 0,
                dilation_h: 1,
                dilation_w: 1,
                groups: 1,
            },
            mlp: UnifiedLinear::from_weights(weights, &format!("{mp}.mlp"), gs, bits)?,
            image_newline: get(weights, &format!("{mp}.image_newline"))?,
            image_begin: get(weights, &format!("{mp}.image_begin"))?,
            image_end: get(weights, &format!("{mp}.image_end"))?,
            merge,
            hidden,
            final_hidden,
            out_hidden: config.out_hidden_size as i32,
        };

        Ok(Self {
            patch_embed,
            pos_table,
            pos_edge: edge,
            blocks,
            merger,
            merge,
        })
    }

    /// Bilinearly interpolate the learned position table to `(gh, gw)` and
    /// flatten to `(1, gh * gw, hidden)` (raster order, matching the rows).
    fn interpolated_pos_embed(&self, gh: i32, gw: i32) -> UniquePtr<MlxArray> {
        let src = self.pos_edge;
        let hidden = mlxcel_core::array_shape(&self.pos_table)[2];
        if gh == src && gw == src {
            return mlxcel_core::reshape(&self.pos_table, &[1, src * src, hidden]);
        }

        // Host-computed corner indices and lerp weights (upstream formula:
        // scale = src / (target + 0.1), coord = (i + 0.5) * scale - 0.5,
        // truncation toward zero for the low corner).
        let build = |target: i32| {
            let scale = src as f32 / (target as f32 + 0.1);
            let mut i0 = Vec::with_capacity(target as usize);
            let mut i1 = Vec::with_capacity(target as usize);
            let mut d = Vec::with_capacity(target as usize);
            for i in 0..target {
                let c = (i as f32 + 0.5) * scale - 0.5;
                let lo = c as i32; // trunc toward zero, matches astype(int32)
                i0.push(lo);
                i1.push((lo + 1).min(src - 1));
                d.push(c - lo as f32);
            }
            (i0, i1, d)
        };
        let (h0, h1, dh) = build(gh);
        let (w0, w1, dw) = build(gw);

        let take_rows = |idx: &[i32]| {
            let idx_arr = mlxcel_core::from_slice_i32(idx, &[idx.len() as i32]);
            mlxcel_core::take(&self.pos_table, &idx_arr, 0) // (t, src, hidden)
        };
        let rows0 = take_rows(&h0);
        let rows1 = take_rows(&h1);
        let take_cols = |rows: &MlxArray, idx: &[i32]| {
            let idx_arr = mlxcel_core::from_slice_i32(idx, &[idx.len() as i32]);
            mlxcel_core::take(rows, &idx_arr, 1) // (t_h, t_w, hidden)
        };
        let p00 = take_cols(&rows0, &w0);
        let p01 = take_cols(&rows0, &w1);
        let p10 = take_cols(&rows1, &w0);
        let p11 = take_cols(&rows1, &w1);

        let dh_arr = mlxcel_core::from_slice_f32(&dh, &[gh, 1, 1]);
        let dw_arr = mlxcel_core::from_slice_f32(&dw, &[1, gw, 1]);
        let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::dtype::FLOAT32);
        let inv_dh = mlxcel_core::subtract(&one, &dh_arr);
        let inv_dw = mlxcel_core::subtract(&one, &dw_arr);

        let dtype = mlxcel_core::array_dtype(&p00);
        let f32p = |x: &MlxArray| mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32);
        let t00 = mlxcel_core::multiply(&mlxcel_core::multiply(&inv_dh, &inv_dw), &f32p(&p00));
        let t01 = mlxcel_core::multiply(&mlxcel_core::multiply(&inv_dh, &dw_arr), &f32p(&p01));
        let t10 = mlxcel_core::multiply(&mlxcel_core::multiply(&dh_arr, &inv_dw), &f32p(&p10));
        let t11 = mlxcel_core::multiply(&mlxcel_core::multiply(&dh_arr, &dw_arr), &f32p(&p11));
        let sum = mlxcel_core::add(&mlxcel_core::add(&t00, &t01), &mlxcel_core::add(&t10, &t11));
        let sum = mlxcel_core::astype(&sum, dtype);
        mlxcel_core::reshape(&sum, &[1, gh * gw, hidden])
    }

    /// `pixel_values`: `(total_patches, 3 * p * p)` raster-ordered rows.
    /// Returns `(1, sum_i(mh_i * (mw_i + 1) + 2), out_hidden)`.
    pub fn forward_with_grid(
        &self,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> UniquePtr<MlxArray> {
        let embeds = self.patch_embed.forward(pixel_values); // (N, hidden)
        let n = mlxcel_core::array_shape(&embeds)[0];
        let hidden = mlxcel_core::array_shape(&embeds)[1];
        let mut h = mlxcel_core::reshape(&embeds, &[1, n, hidden]);

        // Per-image interpolated position embeddings, concatenated in order.
        let mut pos: Option<UniquePtr<MlxArray>> = None;
        for &(_t, gh, gw) in grid_thw {
            let p = self.interpolated_pos_embed(gh, gw);
            pos = Some(match pos {
                None => p,
                Some(acc) => mlxcel_core::concatenate(&acc, &p, 1),
            });
        }
        if let Some(p) = pos {
            let p = mlxcel_core::astype(&p, mlxcel_core::array_dtype(&h));
            h = mlxcel_core::add(&h, &p);
        }

        // Full attention over the packed sequence (matches upstream, which
        // attends across all images in the batch).
        for block in &self.blocks {
            h = block.forward(&h);
        }

        // Per-image merge + framing.
        let mut outputs: Option<UniquePtr<MlxArray>> = None;
        let mut offset = 0i32;
        for &(_t, gh, gw) in grid_thw {
            let len = gh * gw;
            let item = mlxcel_core::slice(&h, &[0, offset, 0], &[1, offset + len, hidden]);
            offset += len;
            let merged = self.merger.forward(&item, gh, gw);
            outputs = Some(match outputs {
                None => merged,
                Some(acc) => mlxcel_core::concatenate(&acc, &merged, 1),
            });
        }
        let _ = self.merge;
        outputs.expect("at least one image grid required")
    }
}
