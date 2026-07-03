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

//! Granite 4 Vision window-QFormer projector.
//!
//! Each of the eight projectors maps one SigLIP tap `(num_tiles, 576, 1152)` to
//! `(num_tiles, 144, 2560)`. With key window `w = 8`, query window `q = 4`, tile
//! side `s = 24`, window grid `n = s / w = 3`, output side `s' = n * q = 12`:
//!
//! 1. `norm` (LayerNorm, eps 1e-6) over the tap.
//! 2. Key/value windowing: `(num_tiles, 24, 24, 1152)` -> 9 windows of `8x8`,
//!    `(num_tiles*9, 64, 1152)`, plus the learned `image_positions` `(1, 64, 1152)`.
//! 3. Query construction: downsample the normed tap to `(num_tiles, 12, 12,
//!    1152)` (deepstack: 2x2 block mean; spatial: strided `(row,col)` offset),
//!    window into `(num_tiles*9, 16, 1152)`, plus the learned `query` `(1, 16,
//!    1152)`.
//! 4. One post-norm QFormer layer (18 heads, head_dim 64): `layernorm` on the
//!    queries, self-attention, cross-attention to the windowed keys, GELU FFN.
//! 5. Un-window to `(num_tiles, 144, 1152)`.
//! 6. `out_linear` (1152 -> 2560).
//!
//! Used by: Granite 4 Vision (`granite4_vision`).
//! Reference: https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/granite4_vision

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

const QFORMER_EPS: f32 = 1e-5; // MLX nn.LayerNorm default (matches the reference).
const NORM_EPS: f32 = 1e-6; // Projector `norm` uses an explicit 1e-6.

fn load_layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {prefix}.weight"))?;
    let bias = weights
        .get(&format!("{prefix}.bias"))
        .map(|w| mlxcel_core::copy(w));
    Ok(LayerNorm::new(weight, bias, eps))
}

fn load_array(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {key}"))
}

/// Window `(num_tiles, s*s, C)` into `(num_tiles*n*n, win*win, C)` with `n = s /
/// win`: each `win x win` block becomes one window row-group.
fn window_tokens(x_flat: &MlxArray, s: i32, win: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x_flat);
    let (nt, c) = (shape[0], shape[2]);
    let n = s / win;
    let x = mlxcel_core::reshape(x_flat, &[nt, n, win, n, win, c]);
    let x = mlxcel_core::transpose_axes(&x, &[0, 1, 3, 2, 4, 5]);
    mlxcel_core::reshape(&x, &[nt * n * n, win * win, c])
}

/// Inverse of [`window_tokens`]: `(num_tiles*n*n, win*win, C)` -> `(num_tiles,
/// s*s, C)`.
fn unwindow_tokens(x: &MlxArray, nt: i32, s: i32, win: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let c = shape[2];
    let n = s / win;
    let x = mlxcel_core::reshape(x, &[nt, n, n, win, win, c]);
    let x = mlxcel_core::transpose_axes(&x, &[0, 1, 3, 2, 4, 5]);
    mlxcel_core::reshape(&x, &[nt, s * s, c])
}

/// Query downsampler kind: deepstack projectors mean-pool `2x2` blocks; spatial
/// projectors select a fixed `(row, col)` offset within each `2x2` block.
#[derive(Debug, Clone, Copy)]
pub enum Downsampler {
    MeanPool,
    Strided { row_off: i32, col_off: i32 },
}

impl Downsampler {
    /// `normed`: `(num_tiles, 576, 1152)` -> `(num_tiles, 144, 1152)`.
    fn apply(&self, normed: &MlxArray, stride: i32) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(normed);
        let (nt, c) = (shape[0], shape[2]);
        let side = 24; // 384 / patch 16
        let out_side = side / stride; // 12
        // (num_tiles, 12, 2, 12, 2, 1152)
        let x = mlxcel_core::reshape(normed, &[nt, out_side, stride, out_side, stride, c]);
        let pooled = match self {
            Downsampler::MeanPool => {
                let m = mlxcel_core::mean_axis(&x, 4, false); // remove col-stride
                mlxcel_core::mean_axis(&m, 2, false) // remove row-stride
            }
            Downsampler::Strided { row_off, col_off } => {
                let sel_row = mlxcel_core::slice(
                    &x,
                    &[0, 0, *row_off, 0, 0, 0],
                    &[nt, out_side, row_off + 1, out_side, stride, c],
                );
                let sel_row = mlxcel_core::squeeze_axis(&sel_row, 2); // (nt, 12, 12, 2, c)
                let sel_col = mlxcel_core::slice(
                    &sel_row,
                    &[0, 0, 0, *col_off, 0],
                    &[nt, out_side, out_side, col_off + 1, c],
                );
                mlxcel_core::squeeze_axis(&sel_col, 3) // (nt, 12, 12, c)
            }
        };
        mlxcel_core::reshape(&pooled, &[nt, out_side * out_side, c])
    }
}

// Multi-head attention block (self or cross), post-norm residual.
struct QFormerAttention {
    query: UnifiedLinear,
    key: UnifiedLinear,
    value: UnifiedLinear,
    out_dense: UnifiedLinear,
    out_norm: LayerNorm,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl QFormerAttention {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        num_heads: i32,
        head_dim: i32,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            query: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.attention.query"),
                gs,
                bits,
            )?,
            key: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.attention.key"),
                gs,
                bits,
            )?,
            value: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.attention.value"),
                gs,
                bits,
            )?,
            out_dense: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.output.dense"),
                gs,
                bits,
            )?,
            out_norm: load_layer_norm(weights, &format!("{prefix}.output.LayerNorm"), QFORMER_EPS)?,
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn split_heads(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let s = mlxcel_core::array_shape(x);
        let x = mlxcel_core::reshape(
            &mlxcel_core::copy(x),
            &[s[0], s[1], self.num_heads, self.head_dim],
        );
        mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3])
    }

    /// `hidden`: query source `(B, Lq, D)`; `context`: key/value source `(B, Lk, D)`.
    fn forward(&self, hidden: &MlxArray, context: &MlxArray) -> UniquePtr<MlxArray> {
        let q = self.query.forward(hidden);
        let k = self.key.forward(context);
        let v = self.value.forward(context);
        let q = self.split_heads(&q);
        let k = self.split_heads(&k);
        let v = self.split_heads(&v);
        let out = unsafe {
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
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let hs = mlxcel_core::array_shape(hidden);
        let out = mlxcel_core::reshape(&out, &[hs[0], hs[1], self.num_heads * self.head_dim]);
        let dense = self.out_dense.forward(&out);
        self.out_norm.forward(&mlxcel_core::add(&dense, hidden))
    }
}

// One QFormer layer: self-attention, cross-attention, GELU FFN (all post-norm).
struct WindowQFormerLayer {
    self_attn: QFormerAttention,
    cross_attn: QFormerAttention,
    inter_dense: UnifiedLinear,
    output_dense: UnifiedLinear,
    output_norm: LayerNorm,
}

impl WindowQFormerLayer {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        num_heads: i32,
        head_dim: i32,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            self_attn: QFormerAttention::from_weights(
                weights,
                &format!("{prefix}.attention"),
                num_heads,
                head_dim,
                gs,
                bits,
            )?,
            cross_attn: QFormerAttention::from_weights(
                weights,
                &format!("{prefix}.crossattention"),
                num_heads,
                head_dim,
                gs,
                bits,
            )?,
            inter_dense: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.intermediate_query.dense"),
                gs,
                bits,
            )?,
            output_dense: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.output_query.dense"),
                gs,
                bits,
            )?,
            output_norm: load_layer_norm(
                weights,
                &format!("{prefix}.output_query.LayerNorm"),
                QFORMER_EPS,
            )?,
        })
    }

    /// `h0`: layernormed windowed queries `(B, 16, D)`; `keys`: windowed keys
    /// `(B, 64, D)`.
    fn forward(&self, h0: &MlxArray, keys: &MlxArray) -> UniquePtr<MlxArray> {
        let h1 = self.self_attn.forward(h0, h0);
        let h2 = self.cross_attn.forward(&h1, keys);
        let inter = mlxcel_core::gelu(&self.inter_dense.forward(&h2));
        let ffn = self.output_dense.forward(&inter);
        self.output_norm.forward(&mlxcel_core::add(&ffn, &h2))
    }
}

/// One Granite 4 Vision window-QFormer projector.
pub struct WindowQFormerProjector {
    norm: LayerNorm,
    query: UniquePtr<MlxArray>,
    image_positions: UniquePtr<MlxArray>,
    qformer_layernorm: LayerNorm,
    layer: WindowQFormerLayer,
    out_linear: UnifiedLinear,
    downsampler: Downsampler,
    /// Key window side (`w`, 8) and query window side (`q`, 4).
    key_win: i32,
    query_win: i32,
    /// Downsample stride (`w / q`, 2).
    stride: i32,
}

impl WindowQFormerProjector {
    /// `q`/`w` from `config.json` `downsample_rate` ("4/8" -> q=4, w=8); the
    /// `query`/`image_positions` weight shapes are validated against `q^2`/`w^2`.
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        downsampler: Downsampler,
        q: i32,
        w: i32,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let num_heads = 18;
        let head_dim = 64;
        let query = load_array(weights, &format!("{prefix}.query"))?;
        let image_positions = load_array(weights, &format!("{prefix}.image_positions"))?;

        // Validate projector geometry against the weight shapes.
        let q_shape = mlxcel_core::array_shape(&query);
        if q_shape.len() != 3 || q_shape[1] != q * q {
            return Err(format!(
                "granite4 projector query length {} != q^2 ({}) for {prefix}",
                q_shape.get(1).copied().unwrap_or(-1),
                q * q
            ));
        }
        let kv_shape = mlxcel_core::array_shape(&image_positions);
        if kv_shape.len() != 3 || kv_shape[1] != w * w {
            return Err(format!(
                "granite4 projector image_positions length {} != w^2 ({}) for {prefix}",
                kv_shape.get(1).copied().unwrap_or(-1),
                w * w
            ));
        }

        Ok(Self {
            norm: load_layer_norm(weights, &format!("{prefix}.norm"), NORM_EPS)?,
            query,
            image_positions,
            qformer_layernorm: load_layer_norm(
                weights,
                &format!("{prefix}.qformer.layernorm"),
                QFORMER_EPS,
            )?,
            layer: WindowQFormerLayer::from_weights(
                weights,
                &format!("{prefix}.qformer.encoder.layer.0"),
                num_heads,
                head_dim,
                gs,
                bits,
            )?,
            out_linear: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.out_linear"),
                gs,
                bits,
            )?,
            downsampler,
            key_win: w,
            query_win: q,
            stride: w / q,
        })
    }

    /// `tap`: `(num_tiles, 576, 1152)` -> `(num_tiles, 144, 2560)`.
    pub fn forward(&self, tap: &MlxArray) -> UniquePtr<MlxArray> {
        let normed = self.norm.forward(tap);
        let dtype = mlxcel_core::array_dtype(&normed);
        let nt = mlxcel_core::array_shape(&normed)[0];
        let side = 24;
        let out_side = side / self.stride; // 12

        // Keys: window the 24x24 grid into 8x8 windows, add image_positions.
        let keys = window_tokens(&normed, side, self.key_win);
        let img_pos = mlxcel_core::astype(&self.image_positions, dtype);
        let keys = mlxcel_core::add(&keys, &img_pos);

        // Queries: downsample to 12x12, window into 4x4 windows, add query.
        let grid = self.downsampler.apply(&normed, self.stride);
        let q_win = window_tokens(&grid, out_side, self.query_win);
        let query = mlxcel_core::astype(&self.query, dtype);
        let q_win = mlxcel_core::add(&q_win, &query);

        // QFormer layer (layernorm queries first, then self+cross+FFN).
        let h0 = self.qformer_layernorm.forward(&q_win);
        let h = self.layer.forward(&h0, &keys);

        // Un-window and project to text hidden size.
        let unwin = unwindow_tokens(&h, nt, out_side, self.query_win);
        self.out_linear.forward(&unwin)
    }
}

#[cfg(test)]
#[path = "granite4_vision_tests.rs"]
mod tests;
