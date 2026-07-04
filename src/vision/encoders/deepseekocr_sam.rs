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

//! SAM-style ViT-B encoder for DeepSeek-OCR (`sam_model.*`).
//!
//! Windowed / global attention with a decomposed relative-position bias, a
//! two-conv neck, and a two-stage conv compressor (`net_2`, `net_3`). Consumes
//! a channels-last image tensor `(B, img, img, 3)` and emits the compressed
//! grid `(B, img/64, img/64, final_out_chans)` (`final_out_chans` = 1024 for
//! DeepSeek-OCR, 896 for DeepSeek-OCR 2, so it is a constructor parameter).
//!
//! Reference: mlx-vlm `mlx_vlm/models/deepseekocr/sam.py`
//! (<https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/deepseekocr/sam.py>).
//! Layout convention: activations are channels-last `(B, H, W, C)`; MLX
//! `conv2d` takes input `(B, H, W, C_in)` with weight `(C_out, kH, kW, C_in)`.

use mlxcel_core::layers::{LayerNorm, Linear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// Static geometry / width configuration for the SAM tower.
#[derive(Clone)]
pub struct SamConfig {
    pub embed_dim: i32,
    pub num_heads: i32,
    pub depth: usize,
    pub window_size: i32,
    pub global_attn_indexes: Vec<usize>,
    pub out_chans: i32,
    /// `net_3` output channels (1024 OCR, 896 OCR-2).
    pub final_out_chans: i32,
    /// Pretraining grid side (`img_size / patch_size` = 1024/16 = 64).
    pub grid: i32,
}

impl Default for SamConfig {
    fn default() -> Self {
        Self {
            embed_dim: 768,
            num_heads: 12,
            depth: 12,
            window_size: 14,
            global_attn_indexes: vec![2, 5, 8, 11],
            out_chans: 256,
            final_out_chans: 1024,
            grid: 64,
        }
    }
}

// ── bicubic resample (PyTorch align_corners=False, antialias) ────────────────

/// Keys cubic-convolution kernel, PyTorch ATen coefficient `a = -0.75`.
fn cubic(x: f32) -> f32 {
    const A: f32 = -0.75;
    let x = x.abs();
    if x <= 1.0 {
        (A + 2.0) * x * x * x - (A + 3.0) * x * x + 1.0
    } else if x < 2.0 {
        A * x * x * x - 5.0 * A * x * x + 8.0 * A * x - 4.0 * A
    } else {
        0.0
    }
}

/// `[out_size, in_size]` cubic interpolation matrix for one axis, matching
/// PyTorch `interpolate(mode="bicubic", align_corners=False)`. With `antialias`
/// and downsampling (`scale > 1`) the filter support is stretched by `scale`
/// and the taps renormalized (SAM `pos_embed`); without it the plain 4-tap
/// kernel is used (CLIP `position_embedding`).
fn interp_matrix(in_size: i32, out_size: i32, antialias: bool) -> Vec<f32> {
    let scale = in_size as f32 / out_size as f32;
    let filterscale = if antialias { scale.max(1.0) } else { 1.0 };
    let support = 2.0 * filterscale; // cubic base support is 2
    let inv = 1.0 / filterscale;
    let mut m = vec![0.0f32; (out_size * in_size) as usize];
    for o in 0..out_size {
        let center = (o as f32 + 0.5) * scale;
        let xmin = ((center - support + 0.5).floor() as i32).max(0);
        let xmax = ((center + support + 0.5).floor() as i32).min(in_size);
        let mut taps: Vec<(i32, f32)> = Vec::new();
        let mut total = 0.0f32;
        for idx in xmin..xmax {
            let w = cubic((idx as f32 + 0.5 - center) * inv);
            taps.push((idx, w));
            total += w;
        }
        if total == 0.0 {
            // Degenerate (out of range): clamp to nearest source pixel.
            let idx = center.floor().clamp(0.0, (in_size - 1) as f32) as i32;
            m[(o * in_size + idx) as usize] += 1.0;
            continue;
        }
        for (idx, w) in taps {
            m[(o * in_size + idx) as usize] += w / total;
        }
    }
    m
}

/// Separable bicubic resample of a `[in_h, in_w, dim]` grid to `[out_h, out_w, dim]`.
pub(crate) fn bicubic_resample(
    grid: &MlxArray,
    in_h: i32,
    in_w: i32,
    dim: i32,
    out_h: i32,
    out_w: i32,
    antialias: bool,
) -> UniquePtr<MlxArray> {
    let w_h = mlxcel_core::from_slice_f32(&interp_matrix(in_h, out_h, antialias), &[out_h, in_h]);
    let w_w = mlxcel_core::from_slice_f32(&interp_matrix(in_w, out_w, antialias), &[out_w, in_w]);
    let grid2 = mlxcel_core::reshape(grid, &[in_h, in_w * dim]);
    let tmp = mlxcel_core::matmul(&w_h, &grid2);
    let tmp = mlxcel_core::reshape(&tmp, &[out_h, in_w, dim]);
    let tmp = mlxcel_core::transpose_axes(&tmp, &[1, 0, 2]);
    let tmp = mlxcel_core::reshape(&tmp, &[in_w, out_h * dim]);
    let out = mlxcel_core::matmul(&w_w, &tmp);
    let out = mlxcel_core::reshape(&out, &[out_w, out_h, dim]);
    mlxcel_core::transpose_axes(&out, &[1, 0, 2])
}

// ── window partition / unpartition ───────────────────────────────────────────

/// Partition `(B, H, W, C)` into `(B*nWin, win, win, C)` windows, padding H/W up
/// to a multiple of `win` first. Returns the windows and the padded `(Hp, Wp)`.
pub(crate) fn window_partition(x: &MlxArray, win: i32) -> (UniquePtr<MlxArray>, (i32, i32)) {
    let s = mlxcel_core::array_shape(x);
    let (b, h, w, c) = (s[0], s[1], s[2], s[3]);
    let pad_h = (win - h % win) % win;
    let pad_w = (win - w % win) % win;
    let x = if pad_h > 0 || pad_w > 0 {
        mlxcel_core::pad(x, &[0, 0, 0, pad_h, 0, pad_w, 0, 0], 0.0)
    } else {
        mlxcel_core::copy(x)
    };
    let (hp, wp) = (h + pad_h, w + pad_w);
    let x = mlxcel_core::reshape(&x, &[b, hp / win, win, wp / win, win, c]);
    let x = mlxcel_core::transpose_axes(&x, &[0, 1, 3, 2, 4, 5]);
    let windows = mlxcel_core::reshape(&x, &[-1, win, win, c]);
    (windows, (hp, wp))
}

/// Inverse of [`window_partition`]: `(B*nWin, win, win, C)` -> `(B, H, W, C)`,
/// cropping the padding.
pub(crate) fn window_unpartition(
    windows: &MlxArray,
    win: i32,
    pad_hw: (i32, i32),
    hw: (i32, i32),
) -> UniquePtr<MlxArray> {
    let (hp, wp) = pad_hw;
    let (h, w) = hw;
    let s = mlxcel_core::array_shape(windows);
    let c = *s.last().unwrap();
    let b = s[0] / (hp * wp / win / win);
    let x = mlxcel_core::reshape(windows, &[b, hp / win, wp / win, win, win, c]);
    let x = mlxcel_core::transpose_axes(&x, &[0, 1, 3, 2, 4, 5]);
    let x = mlxcel_core::reshape(&x, &[b, hp, wp, c]);
    if hp > h || wp > w {
        mlxcel_core::slice(&x, &[0, 0, 0, 0], &[b, h, w, c])
    } else {
        x
    }
}

// ── decomposed relative-position bias ────────────────────────────────────────

/// Extract per-relative-position embeddings, interpolating the stored table
/// along axis 0 when the required `2*max(q,k)-1` length differs from it.
/// Returns `(q_size, k_size, head_dim)`.
fn get_rel_pos(q_size: i32, k_size: i32, rel_pos: &MlxArray) -> UniquePtr<MlxArray> {
    let max_rel_dist = 2 * q_size.max(k_size) - 1;
    let l = mlxcel_core::array_shape(rel_pos)[0];
    let resized: UniquePtr<MlxArray> = if l != max_rel_dist {
        // Linear interpolation of the `(L, C)` table to `(max_rel_dist, C)`.
        let rp = mlxcel_core::astype(rel_pos, mlxcel_core::dtype::FLOAT32);
        let scale = l as f32 / max_rel_dist as f32;
        let idx = mlxcel_core::arange_f32(0.0, max_rel_dist as f32, 1.0);
        let idx = mlxcel_core::multiply_scalar(&idx, scale);
        let floor = mlxcel_core::floor(&idx);
        let floor_i = mlxcel_core::astype(&floor, mlxcel_core::dtype::INT32);
        let ones = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::dtype::FLOAT32);
        let ceil_i = mlxcel_core::astype(
            &mlxcel_core::minimum(
                &mlxcel_core::add(&floor, &ones),
                &mlxcel_core::full_f32(&[1], (l - 1) as f32, mlxcel_core::dtype::FLOAT32),
            ),
            mlxcel_core::dtype::INT32,
        );
        let weight = mlxcel_core::subtract(&idx, &floor);
        let weight = mlxcel_core::expand_dims(&weight, 1); // (max_rel_dist, 1)
        let lo = mlxcel_core::take(&rp, &floor_i, 0);
        let hi = mlxcel_core::take(&rp, &ceil_i, 0);
        let one_minus = mlxcel_core::subtract(
            &mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::dtype::FLOAT32),
            &weight,
        );
        mlxcel_core::add(
            &mlxcel_core::multiply(&lo, &one_minus),
            &mlxcel_core::multiply(&hi, &weight),
        )
    } else {
        mlxcel_core::astype(rel_pos, mlxcel_core::dtype::FLOAT32)
    };

    // Relative coords -> gather. q/k grids are equal here so the ratio is 1.
    let ratio_qk = (k_size as f32 / q_size as f32).max(1.0);
    let ratio_kq = (q_size as f32 / k_size as f32).max(1.0);
    let mut coords = vec![0i32; (q_size * k_size) as usize];
    for i in 0..q_size {
        for j in 0..k_size {
            let v = (i as f32 * ratio_qk) - (j as f32 * ratio_kq) + (k_size - 1) as f32 * ratio_qk;
            coords[(i * k_size + j) as usize] = v as i32;
        }
    }
    let coords = mlxcel_core::from_slice_i32(&coords, &[q_size * k_size]);
    let gathered = mlxcel_core::take(&resized, &coords, 0); // (q*k, C)
    let c = mlxcel_core::array_shape(&resized)[1];
    mlxcel_core::reshape(&gathered, &[q_size, k_size, c])
}

/// Additive attention bias `(B, q_h*q_w, k_h*k_w)` from the decomposed tables.
/// `q` is `(B, q_h*q_w, head_dim)`.
fn decomposed_rel_pos_bias(
    q: &MlxArray,
    rel_pos_h: &MlxArray,
    rel_pos_w: &MlxArray,
    q_hw: (i32, i32),
    k_hw: (i32, i32),
) -> UniquePtr<MlxArray> {
    let (q_h, q_w) = q_hw;
    let (k_h, k_w) = k_hw;
    let rh = get_rel_pos(q_h, k_h, rel_pos_h); // (q_h, k_h, C)
    let rw = get_rel_pos(q_w, k_w, rel_pos_w); // (q_w, k_w, C)
    let s = mlxcel_core::array_shape(q);
    let (b, dim) = (s[0], s[2]);
    let r_q = mlxcel_core::reshape(
        &mlxcel_core::astype(q, mlxcel_core::dtype::FLOAT32),
        &[b, q_h, q_w, dim],
    );
    // rel_h[b,i,j,k_h] = sum_c r_q[b,i,j,c] * rh[i,k_h,c]
    let rel_h = unsafe {
        mlxcel_core::einsum(
            "bhwc,hkc->bhwk",
            &[&*r_q as *const MlxArray, &*rh as *const MlxArray],
        )
    };
    let rel_w = unsafe {
        mlxcel_core::einsum(
            "bhwc,wkc->bhwk",
            &[&*r_q as *const MlxArray, &*rw as *const MlxArray],
        )
    };
    let rel_h = mlxcel_core::reshape(&rel_h, &[b, q_h * q_w, k_h, 1]);
    let rel_w = mlxcel_core::reshape(&rel_w, &[b, q_h * q_w, 1, k_w]);
    let bias = mlxcel_core::add(&rel_h, &rel_w); // broadcast -> (b, q_h*q_w, k_h, k_w)
    mlxcel_core::reshape(&bias, &[b, q_h * q_w, k_h * k_w])
}

// ── layers ───────────────────────────────────────────────────────────────────

struct SamAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
    rel_pos_h: UniquePtr<MlxArray>,
    rel_pos_w: UniquePtr<MlxArray>,
}

impl SamAttention {
    /// `x`: `(B, H, W, C)` -> `(B, H, W, C)`.
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let s = mlxcel_core::array_shape(x);
        let (b, h, w) = (s[0], s[1], s[2]);
        let hw = h * w;
        let (heads, hd) = (self.num_heads, self.head_dim);

        let qkv = self.qkv.forward(x); // (B, H, W, 3C)
        let qkv = mlxcel_core::reshape(&qkv, &[b, hw, 3, heads, hd]);
        let qkv = mlxcel_core::transpose_axes(&qkv, &[2, 0, 3, 1, 4]); // (3, B, heads, HW, hd)
        let qkv = mlxcel_core::reshape(&qkv, &[3, b * heads, hw, hd]);
        let pick = |i: i32| {
            let sl = mlxcel_core::slice(&qkv, &[i, 0, 0, 0], &[i + 1, b * heads, hw, hd]);
            mlxcel_core::squeeze_axis(&sl, 0) // (B*heads, HW, hd)
        };
        let q = pick(0);
        let k = pick(1);
        let v = pick(2);

        let bias = decomposed_rel_pos_bias(&q, &self.rel_pos_h, &self.rel_pos_w, (h, w), (h, w));
        let mask = mlxcel_core::reshape(&bias, &[b, heads, hw, hw]);

        let q = mlxcel_core::reshape(&q, &[b, heads, hw, hd]);
        let k = mlxcel_core::reshape(&k, &[b, heads, hw, hd]);
        let v = mlxcel_core::reshape(&v, &[b, heads, hw, hd]);
        // SAFETY: q/k/v/mask are valid arrays live for the call.
        let out = unsafe {
            mlxcel_core::scaled_dot_product_attention(&q, &k, &v, self.scale, &*mask as *const _)
        };
        // (B, heads, H, W, hd) -> (B, H, W, heads*hd)
        let out = mlxcel_core::reshape(&out, &[b, heads, h, w, hd]);
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 3, 1, 4]);
        let out = mlxcel_core::reshape(&out, &[b, h, w, heads * hd]);
        self.proj.forward(&out)
    }
}

struct SamBlock {
    norm1: LayerNorm,
    attn: SamAttention,
    norm2: LayerNorm,
    lin1: Linear,
    lin2: Linear,
    window_size: i32,
}

impl SamBlock {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let normed = self.norm1.forward(x);
        let attn_out = if self.window_size > 0 {
            let s = mlxcel_core::array_shape(&normed);
            let hw = (s[1], s[2]);
            let (win, pad_hw) = window_partition(&normed, self.window_size);
            let a = self.attn.forward(&win);
            window_unpartition(&a, self.window_size, pad_hw, hw)
        } else {
            self.attn.forward(&normed)
        };
        let x = mlxcel_core::add(x, &attn_out);
        // MLP: lin2(gelu(lin1(norm2(x))))
        let m = self.lin1.forward(&self.norm2.forward(&x));
        let m = mlxcel_core::gelu(&m);
        let m = self.lin2.forward(&m);
        mlxcel_core::add(&x, &m)
    }
}

/// SAM ViT-B encoder + neck + `net_2`/`net_3` compressor.
pub struct SamEncoder {
    config: SamConfig,
    patch_w: UniquePtr<MlxArray>,   // (embed_dim, 16, 16, 3)
    patch_b: UniquePtr<MlxArray>,   // (embed_dim,)
    pos_embed: UniquePtr<MlxArray>, // (1, 64, 64, embed_dim)
    blocks: Vec<SamBlock>,
    neck0_w: UniquePtr<MlxArray>, // conv 1x1 (out_chans, 1, 1, embed_dim)
    neck1: LayerNorm,
    neck2_w: UniquePtr<MlxArray>, // conv 3x3 (out_chans, 3, 3, out_chans)
    neck3: LayerNorm,
    net2_w: UniquePtr<MlxArray>, // conv 3x3 s2 (512, 3, 3, 256)
    net3_w: UniquePtr<MlxArray>, // conv 3x3 s2 (final, 3, 3, 512)
}

/// Transpose an OIHW (layout B) conv weight to OHWI (MLX target) only when it is
/// not already channels-last. Mirrors the shape gate in `internvl.rs`.
fn conv_channels_last(w: UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(&w);
    let (out, d1, d2) = (s[0], s[1], s[2]);
    // Already OHWI when the two middle dims are the equal kernel sides and the
    // out-channel count dominates; otherwise it is OIHW and needs [0,2,3,1].
    if out >= d1 && out >= d2 && d1 == d2 {
        w
    } else {
        mlxcel_core::transpose_axes(&w, &[0, 2, 3, 1])
    }
}

impl SamEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: SamConfig,
    ) -> Result<Self, String> {
        let get = |name: &str| -> Result<UniquePtr<MlxArray>, String> {
            weights
                .get(name)
                .map(|w| mlxcel_core::copy(w))
                .ok_or_else(|| format!("SAM weight missing: {name}"))
        };
        let lin = |p: &str| -> Result<Linear, String> {
            Ok(Linear::new(
                get(&format!("{p}.weight"))?,
                weights
                    .get(&format!("{p}.bias"))
                    .map(|w| mlxcel_core::copy(w)),
            ))
        };
        let ln = |p: &str| -> Result<LayerNorm, String> {
            Ok(LayerNorm::new(
                get(&format!("{p}.weight"))?,
                weights
                    .get(&format!("{p}.bias"))
                    .map(|w| mlxcel_core::copy(w)),
                1e-6,
            ))
        };

        let patch_w = conv_channels_last(get(&format!("{prefix}.patch_embed.proj.weight"))?);
        let patch_b = get(&format!("{prefix}.patch_embed.proj.bias"))?;
        let pos_embed = get(&format!("{prefix}.pos_embed"))?;

        let head_dim = config.embed_dim / config.num_heads;
        let mut blocks = Vec::with_capacity(config.depth);
        for i in 0..config.depth {
            let bp = format!("{prefix}.blocks.{i}");
            let is_global = config.global_attn_indexes.contains(&i);
            let attn = SamAttention {
                qkv: lin(&format!("{bp}.attn.qkv"))?,
                proj: lin(&format!("{bp}.attn.proj"))?,
                num_heads: config.num_heads,
                head_dim,
                scale: (head_dim as f32).powf(-0.5),
                rel_pos_h: get(&format!("{bp}.attn.rel_pos_h"))?,
                rel_pos_w: get(&format!("{bp}.attn.rel_pos_w"))?,
            };
            blocks.push(SamBlock {
                norm1: ln(&format!("{bp}.norm1"))?,
                attn,
                norm2: ln(&format!("{bp}.norm2"))?,
                lin1: lin(&format!("{bp}.mlp.lin1"))?,
                lin2: lin(&format!("{bp}.mlp.lin2"))?,
                window_size: if is_global { 0 } else { config.window_size },
            });
        }

        Ok(Self {
            neck0_w: conv_channels_last(get(&format!("{prefix}.neck.0.weight"))?),
            neck1: ln(&format!("{prefix}.neck.1"))?,
            neck2_w: conv_channels_last(get(&format!("{prefix}.neck.2.weight"))?),
            neck3: ln(&format!("{prefix}.neck.3"))?,
            net2_w: conv_channels_last(get(&format!("{prefix}.net_2.weight"))?),
            net3_w: conv_channels_last(get(&format!("{prefix}.net_3.weight"))?),
            config,
            patch_w,
            patch_b,
            pos_embed,
            blocks,
        })
    }

    /// `pixel_values`: `(B, img, img, 3)` -> `(B, img/64, img/64, final_out_chans)`.
    pub fn forward(&self, pixel_values: &MlxArray) -> UniquePtr<MlxArray> {
        // Patch embed: conv 16x16 stride 16 + bias.
        let mut x = mlxcel_core::conv2d(pixel_values, &self.patch_w, 16, 16, 0, 0, 1, 1, 1);
        x = mlxcel_core::add(&x, &self.patch_b); // broadcast over last axis

        // Absolute position embedding, resampled when the grid is not 64x64.
        let s = mlxcel_core::array_shape(&x);
        let g = s[1];
        let pos = if g == self.config.grid {
            mlxcel_core::copy(&self.pos_embed)
        } else {
            let ed = self.config.embed_dim;
            let grid = self.config.grid;
            let flat = mlxcel_core::reshape(&self.pos_embed, &[grid, grid, ed]);
            let r = bicubic_resample(&flat, grid, grid, ed, g, g, true);
            mlxcel_core::reshape(&r, &[1, g, g, ed])
        };
        x = mlxcel_core::add(&x, &pos);

        for blk in &self.blocks {
            x = blk.forward(&x);
        }

        // Neck: conv1x1 -> LN -> conv3x3(pad1) -> LN.
        x = mlxcel_core::conv2d(&x, &self.neck0_w, 1, 1, 0, 0, 1, 1, 1);
        x = self.neck1.forward(&x);
        x = mlxcel_core::conv2d(&x, &self.neck2_w, 1, 1, 1, 1, 1, 1, 1);
        x = self.neck3.forward(&x);

        // Compressor: two stride-2 conv3x3.
        x = mlxcel_core::conv2d(&x, &self.net2_w, 2, 2, 1, 1, 1, 1, 1);
        x = mlxcel_core::conv2d(&x, &self.net3_w, 2, 2, 1, 1, 1, 1, 1);
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `window_partition` then `window_unpartition` recovers the input, including
    /// the padded case (40 is not a multiple of 14, so it pads to 42 and crops).
    #[test]
    fn window_partition_round_trip() {
        let (h, w, c) = (40i32, 40i32, 2i32);
        let n = (h * w * c) as usize;
        let v: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let x = mlxcel_core::from_slice_f32(&v, &[1, h, w, c]);

        let (win, pad_hw) = window_partition(&x, 14);
        let ws = mlxcel_core::array_shape(&win);
        assert_eq!(ws, vec![9, 14, 14, c]); // ceil(40/14)=3 -> 3*3=9 windows of 14x14
        assert_eq!(pad_hw, (42, 42));

        let back = window_unpartition(&win, 14, pad_hw, (h, w));
        assert_eq!(mlxcel_core::array_shape(&back), vec![1, h, w, c]);
        let diff = mlxcel_core::subtract(&back, &x);
        let m = mlxcel_core::mean_axis(
            &mlxcel_core::abs(&mlxcel_core::reshape(&diff, &[-1])),
            0,
            false,
        );
        mlxcel_core::eval(&m);
        assert!(mlxcel_core::item_f32(&m) < 1e-5, "round trip must be exact");
    }

    /// Each row of the interpolation matrix (a set of blend weights) sums to 1.
    #[test]
    fn interp_matrix_rows_sum_to_one() {
        for (a, b, anti) in [(64, 40, true), (16, 10, false), (16, 16, false)] {
            let m = interp_matrix(a, b, anti);
            for row in 0..b {
                let s: f32 = (0..a).map(|k| m[(row * a + k) as usize]).sum();
                assert!((s - 1.0).abs() < 1e-4, "row {row} sums to {s}");
            }
        }
    }

    /// The relative-position gather returns `(q, k, head_dim)` and, for equal
    /// q/k sizes with no interpolation, indexes `rel_pos[i - j + (S-1)]`.
    #[test]
    fn get_rel_pos_shape_and_index() {
        let (s, dim) = (14i32, 3i32);
        let len = 2 * s - 1; // 27
        let v: Vec<f32> = (0..(len * dim)).map(|i| i as f32).collect();
        let rp = mlxcel_core::from_slice_f32(&v, &[len, dim]);
        let r = get_rel_pos(s, s, &rp);
        assert_eq!(mlxcel_core::array_shape(&r), vec![s, s, dim]);
        // r[0, 0, 0] indexes rel_pos[0 - 0 + 13] channel 0 = 13*dim = 39.
        let cell = mlxcel_core::slice(&r, &[0, 0, 0], &[1, 1, 1]);
        mlxcel_core::eval(&cell);
        assert_eq!(mlxcel_core::item_f32(&cell), (13 * dim) as f32);
    }
}
