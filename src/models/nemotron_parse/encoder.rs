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

//! C-RADIO ViT tower for Nemotron-Parse.
//!
//! A timm ViT (ViT-H/16 for C-RADIOv2-H) wrapped in RADIO's CPE patch
//! generator: a bias-free linear patch embedding, a learned 128x128
//! positional grid that is resampled to the input's patch grid, and 8 prefix
//! tokens (one CLS per distillation teacher plus registers). The blocks are
//! plain pre-norm transformer blocks (no layer scale, no final norm).
//!
//! Output is split the way `RADIOModel._extract_final` splits it: the
//! summary is the concatenation of the CLS rows listed in `summary_idxs`, and
//! the features are the patch rows after all prefix tokens.
//!
//! Reference: `vit_patch_generator.py`, `radio_model.py`, and
//! `enable_cpe_support.py` from `nvidia/C-RADIOv2-H`.

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::NemotronParseVisionConfig;

/// LayerNorm epsilon of the timm ViT blocks and of the neck (`eps=1e-6` in
/// both references); the decoder uses the torch default 1e-5 instead.
pub(crate) const VISION_LN_EPS: f32 = 1e-6;

pub(crate) fn layer_norm_eps(
    weights: &WeightMap,
    prefix: &str,
    eps: f32,
) -> Result<LayerNorm, String> {
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Nemotron-Parse weight not found: {prefix}.weight"))?;
    let bias = weights
        .get(&format!("{prefix}.bias"))
        .map(|w| mlxcel_core::copy(w));
    Ok(LayerNorm::new(weight, bias, eps))
}

pub(crate) fn dense_or_quantized(
    weights: &WeightMap,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<UnifiedLinear, String> {
    UnifiedLinear::from_weights(weights, prefix, group_size, bits)
        .map_err(|e| format!("Nemotron-Parse {e}"))
}

fn require(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Nemotron-Parse weight not found: {key}"))
}

/// Row-interpolation matrix `[out, in]` for 1-D bilinear resampling with
/// `align_corners = true` (PyTorch `F.interpolate(mode="bilinear",
/// align_corners=True)` along one axis).
pub(crate) fn align_corners_weights(input: usize, output: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; output * input];
    for i in 0..output {
        let src = if output > 1 && input > 1 {
            i as f64 * (input - 1) as f64 / (output - 1) as f64
        } else {
            0.0
        };
        let i0 = (src.floor() as usize).min(input - 1);
        let i1 = (i0 + 1).min(input - 1);
        let w1 = (src - i0 as f64) as f32;
        m[i * input + i0] += 1.0 - w1;
        m[i * input + i1] += w1;
    }
    m
}

/// Bilinearly resample a `[h, w, c]` grid to `[out_h, out_w, c]` with
/// `align_corners = true`, computed in f32 (the reference upcasts before
/// interpolating) and returned in f32.
pub(crate) fn bilinear_align_corners(
    grid: &MlxArray,
    out_h: i32,
    out_w: i32,
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(grid);
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    let grid = mlxcel_core::astype(grid, mlxcel_core::dtype::FLOAT32);
    let ry = align_corners_weights(h as usize, out_h as usize);
    let rx = align_corners_weights(w as usize, out_w as usize);
    let ry = mlxcel_core::from_slice_f32(&ry, &[out_h, h]);
    let rx = mlxcel_core::from_slice_f32(&rx, &[out_w, w]);
    // Rows: [out_h, h] @ [h, w * c] -> [out_h, w, c].
    let flat = mlxcel_core::reshape(&grid, &[h, w * c]);
    let rows = mlxcel_core::reshape(&mlxcel_core::matmul(&ry, &flat), &[out_h, w, c]);
    // Columns: [out_w, w] broadcast against [out_h, w, c] -> [out_h, out_w, c].
    mlxcel_core::matmul(&rx, &rows)
}

/// Positional rows for an `hp x wp` patch grid, `[hp * wp, c]`, following
/// RADIO's CPE eval path: identity at the native grid; otherwise resample the
/// square grid to `max(hp, wp)` and crop the top-left `hp x wp` window.
pub(crate) fn pos_embed_for_grid(pos_grid: &MlxArray, hp: i32, wp: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(pos_grid);
    let (g, c) = (shape[0], shape[2]);
    let dtype = mlxcel_core::array_dtype(pos_grid);
    if (hp, wp) == (g, g) {
        return mlxcel_core::reshape(pos_grid, &[hp * wp, c]);
    }
    let m = hp.max(wp);
    let resampled = if m == g {
        mlxcel_core::copy(pos_grid)
    } else {
        let r = bilinear_align_corners(pos_grid, m, m);
        mlxcel_core::astype(&r, dtype)
    };
    let cropped = mlxcel_core::slice(&resampled, &[0, 0, 0], &[hp, wp, c]);
    mlxcel_core::reshape(&cropped, &[hp * wp, c])
}

/// `[B, 3, H, W]` pixels to `[B, hp * wp, 3 * p * p]` patches in RADIO's
/// `(c, yy, xx)` inner order and row-major patch order.
pub(crate) fn patchify(pixels: &MlxArray, patch: i32) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(pixels);
    let (b, c, h, w) = (s[0], s[1], s[2], s[3]);
    let (hp, wp) = (h / patch, w / patch);
    let x = mlxcel_core::reshape(pixels, &[b, c, hp, patch, wp, patch]);
    let x = mlxcel_core::transpose_axes(&x, &[0, 2, 4, 1, 3, 5]);
    mlxcel_core::reshape(&x, &[b, hp * wp, c * patch * patch])
}

struct RadioBlock {
    norm1: LayerNorm,
    qkv: UnifiedLinear,
    proj: UnifiedLinear,
    norm2: LayerNorm,
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
    num_heads: i32,
    scale: f32,
}

impl RadioBlock {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &NemotronParseVisionConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let head_dim = cfg.hidden_size / cfg.num_heads;
        Ok(Self {
            norm1: layer_norm_eps(weights, &format!("{prefix}.norm1"), VISION_LN_EPS)?,
            qkv: dense_or_quantized(weights, &format!("{prefix}.attn.qkv"), group_size, bits)?,
            proj: dense_or_quantized(weights, &format!("{prefix}.attn.proj"), group_size, bits)?,
            norm2: layer_norm_eps(weights, &format!("{prefix}.norm2"), VISION_LN_EPS)?,
            fc1: dense_or_quantized(weights, &format!("{prefix}.mlp.fc1"), group_size, bits)?,
            fc2: dense_or_quantized(weights, &format!("{prefix}.mlp.fc2"), group_size, bits)?,
            num_heads: cfg.num_heads,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn attention(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let s = mlxcel_core::array_shape(x);
        let (b, n, d) = (s[0], s[1], s[2]);
        let heads = self.num_heads;
        let hd = d / heads;
        let qkv = self.qkv.forward(x);
        // [B, N, 3, H, hd] -> [3, B, H, N, hd]
        let qkv = mlxcel_core::reshape(&qkv, &[b, n, 3, heads, hd]);
        let qkv = mlxcel_core::transpose_axes(&qkv, &[2, 0, 3, 1, 4]);
        let part = |i: i32| {
            let p = mlxcel_core::slice(&qkv, &[i, 0, 0, 0, 0], &[i + 1, b, heads, n, hd]);
            mlxcel_core::reshape(&p, &[b, heads, n, hd])
        };
        let (q, k, v) = (part(0), part(1), part(2));
        // Full bidirectional attention over ~13k tokens: the fused kernel
        // never materializes the [H, N, N] score matrix.
        // SAFETY: q, k, v are live arrays for the duration of the call and
        // the null mask pointer selects the unmasked kernel.
        let out = unsafe {
            mlxcel_core::scaled_dot_product_attention(&q, &k, &v, self.scale, std::ptr::null())
        };
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[b, n, d]);
        self.proj.forward(&out)
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let y = self.attention(&self.norm1.forward(x));
        let x = mlxcel_core::add(x, &y);
        let h = self.fc1.forward(&self.norm2.forward(&x));
        let h = mlxcel_core::gelu(&h);
        let h = self.fc2.forward(&h);
        mlxcel_core::add(&x, &h)
    }
}

/// Output of the tower: summary `[B, n_summary * C]`, patch features
/// `[B, hp * wp, C]`, and the patch grid.
pub(crate) struct RadioOutput {
    pub summary: UniquePtr<MlxArray>,
    pub features: UniquePtr<MlxArray>,
    pub grid: (i32, i32),
}

pub(crate) struct RadioVisionTower {
    config: NemotronParseVisionConfig,
    patch_embed: UnifiedLinear,
    /// `[pos_grid, pos_grid, C]`.
    pos_embed: UniquePtr<MlxArray>,
    /// `[num_prefix_tokens, C]`.
    cls_token: UniquePtr<MlxArray>,
    blocks: Vec<RadioBlock>,
    dtype: i32,
}

impl RadioVisionTower {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &NemotronParseVisionConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let c = config.hidden_size;
        let patch_dim = 3 * config.patch_size * config.patch_size;
        let patch_embed =
            dense_or_quantized(weights, &format!("{prefix}.patch_embed"), group_size, bits)?;
        let pe_key = format!("{prefix}.patch_embed.weight");
        if !patch_embed.is_quantized() {
            let pe = require(weights, &pe_key)?;
            let shape = mlxcel_core::array_shape(&pe);
            if shape != [c, patch_dim] {
                return Err(format!(
                    "Nemotron-Parse {pe_key} has shape {shape:?}, expected [{c}, {patch_dim}]"
                ));
            }
        }

        let pos_key = format!("{prefix}.pos_embed");
        let pos = require(weights, &pos_key)?;
        let g = config.pos_grid;
        let numel: i64 = mlxcel_core::array_shape(&pos)
            .iter()
            .map(|d| *d as i64)
            .product();
        if numel != g as i64 * g as i64 * c as i64 {
            return Err(format!(
                "Nemotron-Parse {pos_key} has {numel} elements, expected {g}x{g}x{c}"
            ));
        }
        let pos_embed = mlxcel_core::reshape(&pos, &[g, g, c]);

        let cls_key = format!("{prefix}.cls_token");
        let cls_token = require(weights, &cls_key)?;
        let prefix_tokens = config.num_prefix_tokens();
        if mlxcel_core::array_shape(&cls_token) != [prefix_tokens, c] {
            return Err(format!(
                "Nemotron-Parse {cls_key} has shape {:?}, expected [{prefix_tokens}, {c}]",
                mlxcel_core::array_shape(&cls_token)
            ));
        }
        if let Some(bad) = config
            .summary_idxs
            .iter()
            .find(|i| **i < 0 || **i >= config.num_cls_tokens)
        {
            return Err(format!(
                "Nemotron-Parse summary index {bad} is outside the {} CLS tokens",
                config.num_cls_tokens
            ));
        }

        let dtype = mlxcel_core::array_dtype(&cls_token);
        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            blocks.push(RadioBlock::from_weights(
                weights,
                &format!("{prefix}.blocks.{i}"),
                config,
                group_size,
                bits,
            )?);
        }
        Ok(Self {
            config: config.clone(),
            patch_embed,
            pos_embed,
            cls_token,
            blocks,
            dtype,
        })
    }

    /// Run the tower over normalized `[B, 3, H, W]` pixels. `H` and `W` must
    /// be multiples of the patch size (the processor pads to one).
    pub(crate) fn forward(&self, pixels: &MlxArray) -> Result<RadioOutput, String> {
        let s = mlxcel_core::array_shape(pixels);
        let [b, 3, h, w] = s.as_slice() else {
            return Err(format!(
                "Nemotron-Parse pixel_values must be [B, 3, H, W], got {s:?}"
            ));
        };
        let (b, h, w) = (*b, *h, *w);
        let p = self.config.patch_size;
        if h % p != 0 || w % p != 0 {
            return Err(format!(
                "Nemotron-Parse input {h}x{w} is not a multiple of the patch size {p}"
            ));
        }
        let (hp, wp) = (h / p, w / p);
        let c = self.config.hidden_size;

        let pixels = mlxcel_core::astype(pixels, self.dtype);
        let x = self.patch_embed.forward(&patchify(&pixels, p));
        let pos = pos_embed_for_grid(&self.pos_embed, hp, wp);
        let x = mlxcel_core::add(&x, &mlxcel_core::expand_dims(&pos, 0));

        let prefix = self.config.num_prefix_tokens();
        let cls = mlxcel_core::expand_dims(&self.cls_token, 0);
        let cls = if b > 1 {
            mlxcel_core::concatenate_many(&vec![&*cls; b as usize], 0)
        } else {
            cls
        };
        let mut x = mlxcel_core::concatenate(&cls, &x, 1);
        for block in &self.blocks {
            x = block.forward(&x);
        }

        let n = prefix + hp * wp;
        let features = mlxcel_core::slice(&x, &[0, prefix, 0], &[b, n, c]);
        let rows: Vec<UniquePtr<MlxArray>> = self
            .config
            .summary_idxs
            .iter()
            .map(|&i| {
                let r = mlxcel_core::slice(&x, &[0, i, 0], &[b, i + 1, c]);
                mlxcel_core::reshape(&r, &[b, c])
            })
            .collect();
        let refs: Vec<&MlxArray> = rows.iter().map(|r| &**r).collect();
        let summary = mlxcel_core::concatenate_many(&refs, 1);
        Ok(RadioOutput {
            summary,
            features,
            grid: (hp, wp),
        })
    }
}
