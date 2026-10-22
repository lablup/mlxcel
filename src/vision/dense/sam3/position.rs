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

//! Sinusoidal and rotary position encodings used by SAM 3.

use std::f32::consts::TAU;

use mlxcel_core::{MlxArray, UniquePtr};

/// Two-dimensional DETR sine position encoding over an NHWC feature map.
#[derive(Debug, Clone)]
pub struct PositionEmbeddingSine {
    num_pos_feats: usize,
    temperature: f32,
    normalize: bool,
    scale: f32,
}

impl PositionEmbeddingSine {
    #[must_use]
    pub fn new(num_pos_feats: usize) -> Self {
        Self::with_options(num_pos_feats, 10_000.0, true, TAU)
    }

    #[must_use]
    pub fn with_options(
        num_pos_feats: usize,
        temperature: f32,
        normalize: bool,
        scale: f32,
    ) -> Self {
        Self {
            num_pos_feats,
            temperature,
            normalize,
            scale,
        }
    }

    /// Return `(B, H, W, 2 * num_pos_feats)` float32 positions.
    #[must_use]
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        debug_assert_eq!(shape.len(), 4, "SAM3 positions expect NHWC input");
        let (batch, height, width) = (shape[0], shape[1], shape[2]);
        let channels = self.num_pos_feats * 2;
        let mut table = Vec::with_capacity(height as usize * width as usize * channels);

        for y in 0..height {
            let mut py = (y + 1) as f32;
            if self.normalize {
                py = py / (height as f32 + 1e-6) * self.scale;
            }
            for x_pos in 0..width {
                let mut px = (x_pos + 1) as f32;
                if self.normalize {
                    px = px / (width as f32 + 1e-6) * self.scale;
                }
                encode_axis(&mut table, py, self.num_pos_feats, self.temperature);
                encode_axis(&mut table, px, self.num_pos_feats, self.temperature);
            }
        }

        let table = mlxcel_core::from_slice_f32(&table, &[1, height, width, channels as i32]);
        mlxcel_core::broadcast_to(&table, &[batch, height, width, channels as i32])
    }
}

fn encode_axis(out: &mut Vec<f32>, position: f32, dim: usize, temperature: f32) {
    for lane in 0..dim {
        let exponent = (2 * (lane / 2)) as f32 / dim as f32;
        let angle = position / temperature.powf(exponent);
        out.push(if lane % 2 == 0 {
            angle.sin()
        } else {
            angle.cos()
        });
    }
}

/// Compute axial 2D RoPE tables in row-major `(y, x)` grid order.
///
/// Frequencies for x precede frequencies for y, and every frequency is
/// repeated for the adjacent rotary pair, exactly matching
/// `Sam3ViTRotaryEmbedding`.
#[must_use]
pub fn compute_axial_cis(
    dim: usize,
    end_x: usize,
    end_y: usize,
    theta: f32,
    scale: f32,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    debug_assert_eq!(dim % 4, 0, "axial RoPE dimension must be divisible by four");
    let count = end_x * end_y;
    let mut cos = Vec::with_capacity(count * dim);
    let mut sin = Vec::with_capacity(count * dim);
    for y in 0..end_y {
        for x in 0..end_x {
            for axis_position in [x as f32 * scale, y as f32 * scale] {
                for lane in 0..dim / 4 {
                    let frequency = theta.powf(-((4 * lane) as f32) / dim as f32);
                    let angle = axis_position * frequency;
                    cos.extend_from_slice(&[angle.cos(), angle.cos()]);
                    sin.extend_from_slice(&[angle.sin(), angle.sin()]);
                }
            }
        }
    }
    (
        mlxcel_core::from_slice_f32(&cos, &[count as i32, dim as i32]),
        mlxcel_core::from_slice_f32(&sin, &[count as i32, dim as i32]),
    )
}

/// Interleaved pair rotation `(x0,x1,...) -> (-x1,x0,...)`.
#[must_use]
pub fn rotate_pairwise(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let last_axis = shape.len() - 1;
    let width = shape[last_axis];
    debug_assert_eq!(width % 2, 0, "rotary dimension must be even");

    let mut pair_shape = shape[..last_axis].to_vec();
    pair_shape.extend_from_slice(&[width / 2, 2]);
    let pairs = mlxcel_core::reshape(x, &pair_shape);
    let pair_rank = pair_shape.len();
    let mut starts = vec![0; pair_rank];
    let mut stops = pair_shape.clone();
    stops[pair_rank - 1] = 1;
    let even = mlxcel_core::slice(&pairs, &starts, &stops);
    starts[pair_rank - 1] = 1;
    stops[pair_rank - 1] = 2;
    let odd = mlxcel_core::slice(&pairs, &starts, &stops);
    let rotated =
        mlxcel_core::concatenate(&mlxcel_core::negative(&odd), &even, pair_rank as i32 - 1);
    mlxcel_core::reshape(&rotated, &shape)
}

/// Apply a 2D axial rotary table to `[B, heads, tokens, head_dim]` Q/K.
#[must_use]
pub fn apply_rotary_enc(
    query: &MlxArray,
    key: &MlxArray,
    cos: &MlxArray,
    sin: &MlxArray,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    (apply_rotary(query, cos, sin), apply_rotary(key, cos, sin))
}

fn apply_rotary(x: &MlxArray, cos: &MlxArray, sin: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::add(
        &mlxcel_core::multiply(x, cos),
        &mlxcel_core::multiply(&rotate_pairwise(x), sin),
    )
}

/// Apply one-dimensional RoPE to tracker Q/K in `[B, tokens, heads, dim]`.
#[must_use]
pub fn apply_rotary_enc_1d(
    query: &MlxArray,
    key: &MlxArray,
    freqs_cos: &MlxArray,
    freqs_sin: &MlxArray,
    repeat_freqs_k: bool,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let q_shape = mlxcel_core::array_shape(query);
    let k_shape = mlxcel_core::array_shape(key);
    let freq_shape = mlxcel_core::array_shape(freqs_cos);
    let q_len = q_shape[1];
    let k_len = k_shape[1];

    let cos_q = frequency_view(freqs_cos, q_len, false);
    let sin_q = frequency_view(freqs_sin, q_len, false);
    let cos_k = frequency_view(freqs_cos, k_len, repeat_freqs_k);
    let sin_k = frequency_view(freqs_sin, k_len, repeat_freqs_k);
    debug_assert_eq!(freq_shape, mlxcel_core::array_shape(freqs_sin));

    (
        apply_complex_pairs(query, &cos_q, &sin_q),
        apply_complex_pairs(key, &cos_k, &sin_k),
    )
}

fn frequency_view(freqs: &MlxArray, length: i32, repeat: bool) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(freqs);
    let rows = if repeat && length > shape[0] {
        let repeats = (length + shape[0] - 1) / shape[0];
        mlxcel_core::tile(freqs, &[repeats, 1])
    } else {
        mlxcel_core::copy(freqs)
    };
    let rows = mlxcel_core::slice(&rows, &[0, 0], &[length, shape[1]]);
    mlxcel_core::expand_dims_multi(&rows, &[0, 2])
}

fn apply_complex_pairs(x: &MlxArray, cos: &MlxArray, sin: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let rank = shape.len();
    let width = shape[rank - 1];
    let mut paired_shape = shape[..rank - 1].to_vec();
    paired_shape.extend_from_slice(&[width / 2, 2]);
    let pairs = mlxcel_core::reshape(x, &paired_shape);
    let mut starts = vec![0; rank + 1];
    let mut stops = paired_shape.clone();
    stops[rank] = 1;
    let real = mlxcel_core::reshape(
        &mlxcel_core::slice(&pairs, &starts, &stops),
        &half_shape(&shape),
    );
    starts[rank] = 1;
    stops[rank] = 2;
    let imag = mlxcel_core::reshape(
        &mlxcel_core::slice(&pairs, &starts, &stops),
        &half_shape(&shape),
    );
    let out_real = mlxcel_core::subtract(
        &mlxcel_core::multiply(&real, cos),
        &mlxcel_core::multiply(&imag, sin),
    );
    let out_imag = mlxcel_core::add(
        &mlxcel_core::multiply(&real, sin),
        &mlxcel_core::multiply(&imag, cos),
    );
    let out_real = mlxcel_core::expand_dims(&out_real, rank as i32);
    let out_imag = mlxcel_core::expand_dims(&out_imag, rank as i32);
    let interleaved = mlxcel_core::concatenate(&out_real, &out_imag, rank as i32);
    mlxcel_core::reshape(&interleaved, &shape)
}

fn half_shape(shape: &[i32]) -> Vec<i32> {
    let mut half = shape.to_vec();
    let last = half.len() - 1;
    half[last] /= 2;
    half
}

/// Initialize tracker 2D RoPE frequencies, with x lanes before y lanes.
#[must_use]
pub fn init_2d_freqs(
    dim: usize,
    feat_h: usize,
    feat_w: usize,
    theta: f32,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    debug_assert_eq!(dim % 4, 0, "2D RoPE dimension must be divisible by four");
    let width = dim / 2;
    let count = feat_h * feat_w;
    let mut cos = Vec::with_capacity(count * width);
    let mut sin = Vec::with_capacity(count * width);
    for y in 0..feat_h {
        for x in 0..feat_w {
            for position in [x as f32, y as f32] {
                for lane in 0..dim / 4 {
                    let frequency = theta.powf(-((2 * lane) as f32) / width as f32);
                    let angle = position * frequency;
                    cos.push(angle.cos());
                    sin.push(angle.sin());
                }
            }
        }
    }
    (
        mlxcel_core::from_slice_f32(&cos, &[count as i32, width as i32]),
        mlxcel_core::from_slice_f32(&sin, &[count as i32, width as i32]),
    )
}
