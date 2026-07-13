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

//! Step-3.7 (`perception_encoder`) vision tower.
//!
//! A ViT patch-embed tower used by the StepFun Step-3 multimodal model:
//! - `conv1` patchify (kernel == stride == patch_size) evaluated as an MLX
//!   channels-last Conv2d producing a `52x52` (base) or `36x36` (patch) grid.
//! - Learned absolute position embeddings, bilinearly resized for non-`52x52`
//!   grids (the `504` px patch pass grid `36x36`).
//! - Optional `ln_pre` before the transformer stack and optional `ln_post`
//!   after it (`use_ln_pre` default true, `use_ln_post` default false).
//! - 47 blocks: `ln_1 -> fused-qkv attention with 2D rope -> LayerScale ls_1`,
//!   then `ln_2 -> quick_gelu MLP (c_fc/c_proj) -> LayerScale ls_2`.
//!
//! The two stride-2 downsampler convs and the linear projector live in the
//! connector (`vision::connectors::step3p7`), matching the reference split.
//!
//! Used by: Step-3.7 (step3p7).

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

/// Step-3.7 vision (`perception_encoder`) configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Step3p7VisionConfig {
    #[serde(default = "default_width")]
    pub width: usize,
    #[serde(default = "default_layers")]
    pub layers: usize,
    #[serde(default = "default_heads")]
    pub heads: usize,
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f64,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    /// Checkpoints may spell this `ues_cls_token` (upstream typo); honored.
    #[serde(default, alias = "ues_cls_token")]
    pub use_cls_token: bool,
    #[serde(default = "default_true")]
    pub use_ln_pre: bool,
    #[serde(default)]
    pub use_ln_post: bool,
    #[serde(default = "default_true")]
    pub use_abs_posemb: bool,
    #[serde(default = "default_true")]
    pub use_rope2d: bool,
    #[serde(default = "default_ls_init_value")]
    pub ls_init_value: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Quantization group_size (inherited from the top-level config).
    #[serde(default)]
    pub quant_group_size: i32,
    /// Quantization bits (inherited from the top-level config).
    #[serde(default)]
    pub quant_bits: i32,
}

fn default_width() -> usize {
    1536
}
fn default_layers() -> usize {
    47
}
fn default_heads() -> usize {
    16
}
fn default_num_channels() -> usize {
    3
}
fn default_image_size() -> usize {
    728
}
fn default_patch_size() -> usize {
    14
}
fn default_mlp_ratio() -> f64 {
    8960.0 / 1536.0
}
fn default_layer_norm_eps() -> f32 {
    1e-5
}
fn default_true() -> bool {
    true
}
fn default_ls_init_value() -> f64 {
    0.1
}
fn default_rope_theta() -> f32 {
    10000.0
}

impl Step3p7VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.width / self.heads
    }
}

/// Permute a checkpoint conv kernel `(out, in, kH, kW)` to MLX channels-last
/// `(out, kH, kW, in)`. Idempotent: when the trailing axis already equals the
/// expected `in_ch`, the weight passes through unchanged, so a re-sanitized or
/// pre-converted checkpoint is not double-permuted (same guard convention as
/// `src/vision/encoders/paddleocr_vl.rs`).
pub(crate) fn permute_conv_weight_to_channels_last(
    w: &MlxArray,
    in_ch: i32,
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(w);
    if shape.len() == 4 && shape[3] == in_ch {
        return mlxcel_core::copy(w);
    }
    mlxcel_core::transpose_axes(w, &[0, 2, 3, 1])
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

fn load_vector(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {name}"))
}

/// Host-computed 2D rotary cos/sin tables of shape `(1, 1, tokens, head_dim)`.
///
/// Tokens enumerate the grid row-major (`token = row * grid_w + col`). The
/// quarter-dim inverse frequencies are `rope_theta^(-2j/(head_dim/2))` for
/// `j in 0..head_dim/4`. Per token the width component (`col`) fills the first
/// `head_dim/2` entries and the height component (`row`) the second; each
/// frequency is repeated element-wise (`f0 f0 f1 f1 ...`) so it pairs with the
/// `rotate_half` pair layout. A leading class token (when present) gets an
/// identity rotation (`cos = 1`, `sin = 0`).
fn build_rope_tables(
    grid_h: i32,
    grid_w: i32,
    head_dim: i32,
    theta: f32,
    has_cls: bool,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let quarter = (head_dim / 4) as usize;
    let half_dim = (head_dim as f32) / 2.0;
    let inv_freq: Vec<f32> = (0..quarter)
        .map(|j| theta.powf(-(2.0 * j as f32) / half_dim))
        .collect();

    let cls_off = if has_cls { 1usize } else { 0usize };
    let n_grid = (grid_h * grid_w) as usize;
    let n_tokens = n_grid + cls_off;
    let hd = head_dim as usize;

    let mut cos_data = vec![0f32; n_tokens * hd];
    let mut sin_data = vec![0f32; n_tokens * hd];

    if has_cls {
        for d in 0..hd {
            cos_data[d] = 1.0;
            sin_data[d] = 0.0;
        }
    }

    for row in 0..grid_h {
        for col in 0..grid_w {
            let t = cls_off + (row * grid_w + col) as usize;
            let base = t * hd;
            for (j, &freq) in inv_freq.iter().enumerate() {
                let fc = col as f32 * freq;
                let fr = row as f32 * freq;
                let cpos = base + 2 * j;
                cos_data[cpos] = fc.cos();
                cos_data[cpos + 1] = fc.cos();
                sin_data[cpos] = fc.sin();
                sin_data[cpos + 1] = fc.sin();
                let rpos = base + 2 * quarter + 2 * j;
                cos_data[rpos] = fr.cos();
                cos_data[rpos + 1] = fr.cos();
                sin_data[rpos] = fr.sin();
                sin_data[rpos + 1] = fr.sin();
            }
        }
    }

    let shape = [1, 1, n_tokens as i32, head_dim];
    (
        mlxcel_core::from_slice_f32(&cos_data, &shape),
        mlxcel_core::from_slice_f32(&sin_data, &shape),
    )
}

/// `rotate_half` for the paired 2D-rope layout: reshape the last axis
/// `(..., head_dim)` to `(..., head_dim/2, 2)`, map each pair `(a, b)` to
/// `(-b, a)`, and reshape back.
fn rotate_half(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let ndim = shape.len();
    let head_dim = shape[ndim - 1];
    let half = head_dim / 2;

    let mut pair_shape: Vec<i32> = shape[..ndim - 1].to_vec();
    pair_shape.push(half);
    pair_shape.push(2);
    let pair_ndim = pair_shape.len();
    let paired = mlxcel_core::reshape(x, &pair_shape);

    let mut starts = vec![0i32; pair_ndim];
    let mut stops = pair_shape.clone();
    stops[pair_ndim - 1] = 1;
    let even = mlxcel_core::slice(&paired, &starts, &stops);
    starts[pair_ndim - 1] = 1;
    stops[pair_ndim - 1] = 2;
    let odd = mlxcel_core::slice(&paired, &starts, &stops);

    let neg_odd = mlxcel_core::negative(&odd);
    let rotated = mlxcel_core::concatenate(
        neg_odd.as_ref().unwrap(),
        even.as_ref().unwrap(),
        (pair_ndim - 1) as i32,
    );
    mlxcel_core::reshape(&rotated, &shape)
}

/// `x' = x * cos + rotate_half(x) * sin`, broadcasting cos/sin over batch/head.
fn apply_rope_2d(x: &MlxArray, cos: &MlxArray, sin: &MlxArray) -> UniquePtr<MlxArray> {
    let x_cos = mlxcel_core::multiply(x, cos);
    let rot = rotate_half(x);
    let rot_sin = mlxcel_core::multiply(&rot, sin);
    mlxcel_core::add(&x_cos, &rot_sin)
}

/// Bilinear resize of a `(h_in*w_in, embed)` row-major grid to
/// `(h_out*w_out, embed)` with `align_corners = false` (the PyTorch
/// `F.interpolate` default). Indices/weights are host-derived and gathered with
/// `take`, matching `src/vision/encoders/paddleocr_vl.rs`.
fn bilinear_resize(
    table: &MlxArray,
    h_in: i32,
    w_in: i32,
    h_out: i32,
    w_out: i32,
    embed: i32,
) -> UniquePtr<MlxArray> {
    let row_pos: Vec<f64> = (0..h_out)
        .map(|i| (i as f64 + 0.5) * h_in as f64 / h_out as f64 - 0.5)
        .collect();
    let col_pos: Vec<f64> = (0..w_out)
        .map(|j| (j as f64 + 0.5) * w_in as f64 / w_out as f64 - 0.5)
        .collect();

    let clamp = |v: i32, hi: i32| v.max(0).min(hi - 1);
    let row_floor: Vec<i32> = row_pos
        .iter()
        .map(|&p| clamp(p.floor() as i32, h_in))
        .collect();
    let row_ceil: Vec<i32> = row_pos
        .iter()
        .map(|&p| clamp(p.floor() as i32 + 1, h_in))
        .collect();
    let col_floor: Vec<i32> = col_pos
        .iter()
        .map(|&p| clamp(p.floor() as i32, w_in))
        .collect();
    let col_ceil: Vec<i32> = col_pos
        .iter()
        .map(|&p| clamp(p.floor() as i32 + 1, w_in))
        .collect();
    let row_w: Vec<f32> = row_pos
        .iter()
        .zip(&row_floor)
        .map(|(&p, &f)| (p - f as f64) as f32)
        .collect();
    let col_w: Vec<f32> = col_pos
        .iter()
        .zip(&col_floor)
        .map(|(&p, &f)| (p - f as f64) as f32)
        .collect();

    let n = (h_out * w_out) as usize;
    let mut idx_tl = Vec::with_capacity(n);
    let mut idx_tr = Vec::with_capacity(n);
    let mut idx_bl = Vec::with_capacity(n);
    let mut idx_br = Vec::with_capacity(n);
    let mut w_tl = Vec::with_capacity(n);
    let mut w_tr = Vec::with_capacity(n);
    let mut w_bl = Vec::with_capacity(n);
    let mut w_br = Vec::with_capacity(n);
    for i in 0..h_out as usize {
        for j in 0..w_out as usize {
            let rf = row_floor[i];
            let rc = row_ceil[i];
            let cf = col_floor[j];
            let cc = col_ceil[j];
            idx_tl.push(rf * w_in + cf);
            idx_tr.push(rf * w_in + cc);
            idx_bl.push(rc * w_in + cf);
            idx_br.push(rc * w_in + cc);
            let rw = row_w[i];
            let cw = col_w[j];
            w_tl.push((1.0 - rw) * (1.0 - cw));
            w_tr.push((1.0 - rw) * cw);
            w_bl.push(rw * (1.0 - cw));
            w_br.push(rw * cw);
        }
    }

    let gather = |idx: &[i32], wts: &[f32]| -> UniquePtr<MlxArray> {
        let idx_arr = mlxcel_core::from_slice_i32(idx, &[n as i32]);
        let g = mlxcel_core::take(table, &idx_arr, 0);
        let wa = mlxcel_core::from_slice_f32(wts, &[n as i32, 1]);
        mlxcel_core::multiply(&g, &wa)
    };

    let tl = gather(&idx_tl, &w_tl);
    let tr = gather(&idx_tr, &w_tr);
    let bl = gather(&idx_bl, &w_bl);
    let br = gather(&idx_br, &w_br);
    let a = mlxcel_core::add(&tl, &tr);
    let b = mlxcel_core::add(&bl, &br);
    let _ = embed;
    mlxcel_core::add(&a, &b)
}

// Fused-QKV vision attention with 2D rope, full (bidirectional) softmax.
struct Attention {
    in_proj: UnifiedLinear,
    out_proj: UnifiedLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl Attention {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        num_heads: i32,
        head_dim: i32,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let in_proj = UnifiedLinear::from_weights(weights, &format!("{prefix}.in_proj"), gs, bits)?;
        let out_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.out_proj"), gs, bits)?;
        Ok(Self {
            in_proj,
            out_proj,
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        cos: Option<&MlxArray>,
        sin: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let tokens = shape[1];

        let qkv = self.in_proj.forward(x);
        let qkv = mlxcel_core::reshape(&qkv, &[b, tokens, 3, self.num_heads, self.head_dim]);

        let extract = |idx: i32| -> UniquePtr<MlxArray> {
            let starts = [0, 0, idx, 0, 0];
            let stops = [b, tokens, idx + 1, self.num_heads, self.head_dim];
            let s = mlxcel_core::slice(&qkv, &starts, &stops);
            let s = mlxcel_core::squeeze_axis(&s, 2);
            mlxcel_core::transpose_axes(&s, &[0, 2, 1, 3])
        };

        let mut q = extract(0);
        let mut k = extract(1);
        let v = extract(2);

        if let (Some(c), Some(s)) = (cos, sin) {
            q = apply_rope_2d(&q, c, s);
            k = apply_rope_2d(&k, c, s);
        }

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

        let out = mlxcel_core::transpose_axes(&attn, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[b, tokens, -1]);
        self.out_proj.forward(&out)
    }
}

// quick_gelu MLP: c_proj(quick_gelu(c_fc(x))).
struct Mlp {
    c_fc: UnifiedLinear,
    c_proj: UnifiedLinear,
}

impl Mlp {
    fn from_weights(weights: &WeightMap, prefix: &str, gs: i32, bits: i32) -> Result<Self, String> {
        Ok(Self {
            c_fc: UnifiedLinear::from_weights(weights, &format!("{prefix}.c_fc"), gs, bits)?,
            c_proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.c_proj"), gs, bits)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.c_fc.forward(x);
        // quick_gelu(x) = x * sigmoid(1.702 * x).
        let h = mlxcel_core::utils::gelu_sigmoid(&h);
        self.c_proj.forward(&h)
    }
}

struct Block {
    ln_1: LayerNorm,
    ln_2: LayerNorm,
    attn: Attention,
    mlp: Mlp,
    ls_1: UniquePtr<MlxArray>,
    ls_2: UniquePtr<MlxArray>,
}

impl Block {
    fn from_weights(
        weights: &WeightMap,
        config: &Step3p7VisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.quant_group_size;
        let bits = config.quant_bits;
        let head_dim = config.head_dim() as i32;
        Ok(Self {
            ln_1: load_layer_norm(weights, &format!("{prefix}.ln_1"), config.layer_norm_eps)?,
            ln_2: load_layer_norm(weights, &format!("{prefix}.ln_2"), config.layer_norm_eps)?,
            attn: Attention::from_weights(
                weights,
                &format!("{prefix}.attn"),
                config.heads as i32,
                head_dim,
                gs,
                bits,
            )?,
            mlp: Mlp::from_weights(weights, &format!("{prefix}.mlp"), gs, bits)?,
            ls_1: load_vector(weights, &format!("{prefix}.ls_1.gamma"))?,
            ls_2: load_vector(weights, &format!("{prefix}.ls_2.gamma"))?,
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        cos: Option<&MlxArray>,
        sin: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let attn_out = self.attn.forward(&self.ln_1.forward(x), cos, sin);
        let attn_out = mlxcel_core::multiply(&attn_out, &self.ls_1);
        let h = mlxcel_core::add(x, &attn_out);
        let mlp_out = self.mlp.forward(&self.ln_2.forward(&h));
        let mlp_out = mlxcel_core::multiply(&mlp_out, &self.ls_2);
        mlxcel_core::add(&h, &mlp_out)
    }
}

/// Step-3.7 vision tower (patchify conv + transformer stack).
pub struct Step3p7VisionEncoder {
    conv1_weight: UniquePtr<MlxArray>,
    class_embedding: Option<UniquePtr<MlxArray>>,
    positional_embedding: Option<UniquePtr<MlxArray>>,
    ln_pre: Option<LayerNorm>,
    ln_post: Option<LayerNorm>,
    blocks: Vec<Block>,
    patch_size: i32,
    num_channels: i32,
    width: i32,
    head_dim: i32,
    rope_theta: f32,
    use_cls_token: bool,
    use_rope2d: bool,
}

impl Step3p7VisionEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        config: &Step3p7VisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let conv1_weight = load_vector(weights, &format!("{prefix}.conv1.weight"))?;

        let class_embedding = if config.use_cls_token {
            Some(load_vector(weights, &format!("{prefix}.class_embedding"))?)
        } else {
            None
        };

        let positional_embedding = if config.use_abs_posemb {
            Some(load_vector(
                weights,
                &format!("{prefix}.positional_embedding"),
            )?)
        } else {
            None
        };

        let ln_pre = if config.use_ln_pre {
            Some(load_layer_norm(
                weights,
                &format!("{prefix}.ln_pre"),
                config.layer_norm_eps,
            )?)
        } else {
            None
        };

        let ln_post = if config.use_ln_post {
            Some(load_layer_norm(
                weights,
                &format!("{prefix}.ln_post"),
                config.layer_norm_eps,
            )?)
        } else {
            None
        };

        let mut blocks = Vec::with_capacity(config.layers);
        for i in 0..config.layers {
            blocks.push(Block::from_weights(
                weights,
                config,
                &format!("{prefix}.transformer.{i}"),
            )?);
        }

        Ok(Self {
            conv1_weight,
            class_embedding,
            positional_embedding,
            ln_pre,
            ln_post,
            blocks,
            patch_size: config.patch_size as i32,
            num_channels: config.num_channels as i32,
            width: config.width as i32,
            head_dim: config.head_dim() as i32,
            rope_theta: config.rope_theta,
            use_cls_token: config.use_cls_token,
            use_rope2d: config.use_rope2d,
        })
    }

    /// Interpolate the learned position table to the `grid_h x grid_w` grid.
    /// Returns `(tokens_incl_cls, width)`; the `52x52` base grid is a
    /// pass-through, the `36x36` patch grid is bilinearly resized.
    fn interpolate_posemb(
        &self,
        table: &MlxArray,
        grid_h: i32,
        grid_w: i32,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(table);
        let total_rows = shape[0];
        let embed = shape[1];

        let (cls_row, grid_rows) = if self.use_cls_token {
            let cls = mlxcel_core::slice(table, &[0, 0], &[1, embed]);
            let grid = mlxcel_core::slice(table, &[1, 0], &[total_rows, embed]);
            (Some(cls), grid)
        } else {
            (None, mlxcel_core::copy(table))
        };

        let grid_count = mlxcel_core::array_shape(&grid_rows)[0];
        let side = (grid_count as f64).sqrt().round() as i32;

        let grid_pos = if grid_h == side && grid_w == side {
            grid_rows
        } else {
            bilinear_resize(&grid_rows, side, side, grid_h, grid_w, embed)
        };

        match cls_row {
            Some(cls) => {
                mlxcel_core::concatenate(cls.as_ref().unwrap(), grid_pos.as_ref().unwrap(), 0)
            }
            None => grid_pos,
        }
    }

    /// Run the tower on a channels-first `(batch, C, H, W)` pixel batch.
    /// Returns per-image grid tokens `(batch, grid_h*grid_w, width)` (any class
    /// token is stripped so the connector sees a clean spatial grid).
    pub fn forward(&self, pixel_values: &MlxArray) -> UniquePtr<MlxArray> {
        // Conv2d is channels-last in MLX; match the tower weight dtype.
        let pv = mlxcel_core::astype(pixel_values, mlxcel_core::array_dtype(&self.conv1_weight));
        let pv = mlxcel_core::transpose_axes(&pv, &[0, 2, 3, 1]);
        let _ = self.num_channels;

        let conv = mlxcel_core::conv2d(
            &pv,
            &self.conv1_weight,
            self.patch_size,
            self.patch_size,
            0,
            0,
            1,
            1,
            1,
        );
        let cshape = mlxcel_core::array_shape(&conv);
        let (b, grid_h, grid_w) = (cshape[0], cshape[1], cshape[2]);
        let tokens = grid_h * grid_w;
        let mut h = mlxcel_core::reshape(&conv, &[b, tokens, self.width]);

        if let Some(cls) = &self.class_embedding {
            let cls_tok = mlxcel_core::reshape(cls, &[1, 1, self.width]);
            let cls_tok = mlxcel_core::broadcast_to(&cls_tok, &[b, 1, self.width]);
            h = mlxcel_core::concatenate(cls_tok.as_ref().unwrap(), h.as_ref().unwrap(), 1);
        }

        if let Some(table) = &self.positional_embedding {
            let pos = self.interpolate_posemb(table, grid_h, grid_w);
            let pshape = mlxcel_core::array_shape(&pos);
            let pos = mlxcel_core::reshape(&pos, &[1, pshape[0], pshape[1]]);
            h = mlxcel_core::add(&h, &pos);
        }

        if let Some(ln) = &self.ln_pre {
            h = ln.forward(&h);
        }

        let rope = if self.use_rope2d {
            Some(build_rope_tables(
                grid_h,
                grid_w,
                self.head_dim,
                self.rope_theta,
                self.use_cls_token,
            ))
        } else {
            None
        };
        let cos = rope.as_ref().map(|(c, _)| c.as_ref().unwrap());
        let sin = rope.as_ref().map(|(_, s)| s.as_ref().unwrap());

        for block in &self.blocks {
            h = block.forward(&h, cos, sin);
        }

        if let Some(ln) = &self.ln_post {
            h = ln.forward(&h);
        }

        if self.use_cls_token {
            let sh = mlxcel_core::array_shape(&h);
            h = mlxcel_core::slice(&h, &[0, 1, 0], &[sh[0], sh[1], sh[2]]);
        }

        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_permutation_is_idempotent() {
        // Checkpoint layout (out, in, kH, kW) = (4, 3, 2, 2).
        let w = mlxcel_core::from_slice_f32(
            &(0..48).map(|i| i as f32).collect::<Vec<_>>(),
            &[4, 3, 2, 2],
        );
        let once = permute_conv_weight_to_channels_last(w.as_ref().unwrap(), 3);
        assert_eq!(
            mlxcel_core::array_shape(once.as_ref().unwrap()),
            vec![4, 2, 2, 3],
            "first permute goes channels-last"
        );
        let twice = permute_conv_weight_to_channels_last(once.as_ref().unwrap(), 3);
        assert_eq!(
            mlxcel_core::array_shape(twice.as_ref().unwrap()),
            vec![4, 2, 2, 3],
            "second permute is a no-op (idempotent guard)"
        );
        let close =
            mlxcel_core::allclose(once.as_ref().unwrap(), twice.as_ref().unwrap(), 1e-6, 1e-6);
        assert!(
            mlxcel_core::item_bool(&close),
            "idempotent values unchanged"
        );
    }

    #[test]
    fn rope_tables_have_expected_shape_and_cls_identity() {
        // head_dim 8, grid 2x2, with a class token -> 5 tokens.
        let (cos, sin) = build_rope_tables(2, 2, 8, 10000.0, true);
        assert_eq!(
            mlxcel_core::array_shape(cos.as_ref().unwrap()),
            vec![1, 1, 5, 8]
        );
        assert_eq!(
            mlxcel_core::array_shape(sin.as_ref().unwrap()),
            vec![1, 1, 5, 8]
        );

        // Class token (index 0) is an identity rotation: cos=1, sin=0.
        let cls_cos = mlxcel_core::slice(cos.as_ref().unwrap(), &[0, 0, 0, 0], &[1, 1, 1, 8]);
        let cls_sin = mlxcel_core::slice(sin.as_ref().unwrap(), &[0, 0, 0, 0], &[1, 1, 1, 8]);
        let ones = mlxcel_core::from_slice_f32(&[1.0; 8], &[1, 1, 1, 8]);
        let zeros = mlxcel_core::from_slice_f32(&[0.0; 8], &[1, 1, 1, 8]);
        assert!(mlxcel_core::item_bool(&mlxcel_core::allclose(
            cls_cos.as_ref().unwrap(),
            ones.as_ref().unwrap(),
            1e-6,
            1e-6
        )));
        assert!(mlxcel_core::item_bool(&mlxcel_core::allclose(
            cls_sin.as_ref().unwrap(),
            zeros.as_ref().unwrap(),
            1e-6,
            1e-6
        )));
    }

    #[test]
    fn identity_rope_leaves_input_unchanged() {
        // cos=1, sin=0 over (1,1,3,4) leaves q unchanged.
        let q = mlxcel_core::from_slice_f32(
            &(0..12).map(|i| (i as f32) * 0.1).collect::<Vec<_>>(),
            &[1, 1, 3, 4],
        );
        let cos = mlxcel_core::from_slice_f32(&[1.0; 12], &[1, 1, 3, 4]);
        let sin = mlxcel_core::from_slice_f32(&[0.0; 12], &[1, 1, 3, 4]);
        let out = apply_rope_2d(
            q.as_ref().unwrap(),
            cos.as_ref().unwrap(),
            sin.as_ref().unwrap(),
        );
        let close = mlxcel_core::allclose(out.as_ref().unwrap(), q.as_ref().unwrap(), 1e-6, 1e-6);
        assert!(mlxcel_core::item_bool(&close));
    }

    #[test]
    fn rotate_half_maps_pairs_a_b_to_neg_b_a() {
        // (a0,b0,a1,b1) -> (-b0,a0,-b1,a1).
        let x = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4]);
        let out = rotate_half(x.as_ref().unwrap());
        let expected = mlxcel_core::from_slice_f32(&[-2.0, 1.0, -4.0, 3.0], &[1, 1, 1, 4]);
        let close = mlxcel_core::allclose(
            out.as_ref().unwrap(),
            expected.as_ref().unwrap(),
            1e-6,
            1e-6,
        );
        assert!(mlxcel_core::item_bool(&close));
    }

    #[test]
    fn posemb_interpolation_produces_target_grid_shape() {
        // 4x4=16-position table interpolated to a 2x3 grid -> 6 rows.
        let table = mlxcel_core::from_slice_f32(
            &(0..16 * 5).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            &[16, 5],
        );
        let out = bilinear_resize(table.as_ref().unwrap(), 4, 4, 2, 3, 5);
        assert_eq!(mlxcel_core::array_shape(out.as_ref().unwrap()), vec![6, 5]);
    }
}
