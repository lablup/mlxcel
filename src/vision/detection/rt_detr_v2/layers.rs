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

//! Shared low-level building blocks for the RT-DETRv2 vision tower.
//!
//! These primitives (Conv2d, inference BatchNorm, max/avg pool, nearest
//! upsample, `grid_sample`) are not all exposed by `mlxcel-core`, so they are
//! composed here from the available array ops. All tensors are NHWC.
//!
//! `grid_sample` follows PyTorch `mode="bilinear", padding_mode="zeros",
//! align_corners=False`, the exact semantics the upstream Metal kernel and its
//! pure-MLX fallback implement (see
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/kernels.py (grid_sample)).

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::common::copy_weight;

/// Resolve an activation name to a closure. `None` -> identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Relu,
    Silu,
    Gelu,
    None,
}

impl Activation {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "relu" => Ok(Activation::Relu),
            "silu" => Ok(Activation::Silu),
            "gelu" => Ok(Activation::Gelu),
            other => Err(format!("Unsupported activation: {other}")),
        }
    }

    pub fn apply(self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Activation::Relu => mlxcel_core::relu(x),
            Activation::Silu => mlxcel_core::silu(x),
            // RT-DETRv2 uses exact (erf) GELU, matching nn.GELU() default.
            Activation::Gelu => mlxcel_core::gelu(x),
            Activation::None => mlxcel_core::copy(x),
        }
    }
}

/// 2D convolution (no bias) loaded from a single `*.weight` key. The weight is
/// stored in MLX NHWC layout `(out, kH, kW, in)`.
pub struct Conv2d {
    weight: UniquePtr<MlxArray>,
    stride: i32,
    padding: i32,
}

impl Conv2d {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        stride: i32,
        padding: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: copy_weight(weights, &format!("{prefix}.weight"))?,
            stride,
            padding,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::conv2d(
            x,
            &self.weight,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            1,
            1,
            1,
        )
    }
}

/// Inference-time BatchNorm over the channel (last) axis of an NHWC tensor:
/// `y = (x - running_mean) / sqrt(running_var + eps) * weight + bias`.
///
/// Used by every conv block in the ResNet backbone and the hybrid encoder.
pub struct BatchNorm {
    weight: UniquePtr<MlxArray>,
    bias: UniquePtr<MlxArray>,
    running_mean: UniquePtr<MlxArray>,
    running_var: UniquePtr<MlxArray>,
    eps: f32,
}

impl BatchNorm {
    pub fn from_weights(weights: &WeightMap, prefix: &str, eps: f32) -> Result<Self, String> {
        Ok(Self {
            weight: copy_weight(weights, &format!("{prefix}.weight"))?,
            bias: copy_weight(weights, &format!("{prefix}.bias"))?,
            running_mean: copy_weight(weights, &format!("{prefix}.running_mean"))?,
            running_var: copy_weight(weights, &format!("{prefix}.running_var"))?,
            eps,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let centered = mlxcel_core::subtract(x, &self.running_mean);
        let eps_arr = mlxcel_core::full_f32(&[1], self.eps, mlxcel_core::array_dtype(x));
        let var_eps = mlxcel_core::add(&self.running_var, &eps_arr);
        let inv_std = mlxcel_core::rsqrt(&var_eps);
        let scaled = mlxcel_core::multiply(&centered, &inv_std);
        let with_weight = mlxcel_core::multiply(&scaled, &self.weight);
        mlxcel_core::add(&with_weight, &self.bias)
    }
}

/// Conv2d (no bias) + BatchNorm + optional activation. The most common block in
/// the model; the backbone `RTDetrResNetConvLayer` and the hybrid-encoder
/// `RTDetrV2ConvNormLayer` both reduce to this once weights are renamed (both
/// surface `.conv.` / `.bn.` sub-keys).
pub struct ConvNorm {
    conv: Conv2d,
    bn: BatchNorm,
    act: Activation,
}

impl ConvNorm {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        stride: i32,
        padding: i32,
        act: Activation,
        eps: f32,
    ) -> Result<Self, String> {
        Ok(Self {
            conv: Conv2d::from_weights(weights, &format!("{prefix}.conv"), stride, padding)?,
            bn: BatchNorm::from_weights(weights, &format!("{prefix}.bn"), eps)?,
            act,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let y = self.conv.forward(x);
        let y = self.bn.forward(&y);
        self.act.apply(&y)
    }
}

/// 1x1 Conv (no bias) + BatchNorm, no activation. Used by `ShortCut`,
/// `EncoderInputProj`, and `DecoderInputProj`.
pub struct ConvBn {
    conv: Conv2d,
    bn: BatchNorm,
}

impl ConvBn {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        stride: i32,
        eps: f32,
    ) -> Result<Self, String> {
        Ok(Self {
            // kernel 1x1 -> padding 0.
            conv: Conv2d::from_weights(weights, &format!("{prefix}.conv"), stride, 0)?,
            bn: BatchNorm::from_weights(weights, &format!("{prefix}.bn"), eps)?,
        })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.bn.forward(&self.conv.forward(x))
    }
}

/// Average pool 2x2 stride 2 (vd downsampling shortcut), via the `mlxcel-core`
/// depthwise-conv-backed `avg_pool2d`.
pub fn avg_pool_2x2(x: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::avg_pool2d(x, 2, 2, 2, 2, 0, 0)
}

/// Max pool 3x3 stride 2 padding 1 over an NHWC tensor (ResNet-vd stem
/// `pooler`).
///
/// `mlxcel-core` exposes `avg_pool2d` but not `max_pool2d`, so this composes
/// the op from `pad` + gather + element-wise `maximum`. The padded border is
/// filled with a large negative sentinel so it never wins the max (PyTorch
/// MaxPool2d uses `-inf` padding). Output positions are
/// `out = floor((H + 2*pad - kernel) / stride) + 1`, matching PyTorch.
pub fn max_pool_3x3_s2_p1(x: &MlxArray) -> UniquePtr<MlxArray> {
    max_pool2d(x, 3, 2, 1)
}

/// General square max pool over NHWC. `kernel`/`stride`/`pad` are scalar
/// (square). See [`max_pool_3x3_s2_p1`] for the algorithm note.
pub fn max_pool2d(x: &MlxArray, kernel: i32, stride: i32, pad: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    debug_assert_eq!(shape.len(), 4, "max_pool2d expects NHWC");
    let (h, w) = (shape[1], shape[2]);

    // Pad H and W with a large-negative sentinel; channels/batch untouched.
    // NEG sentinel: avoid true -inf to keep arithmetic well-defined under f16.
    const NEG: f32 = -1.0e30;
    let padded = mlxcel_core::pad(x, &[0, 0, pad, pad, pad, pad, 0, 0], NEG);

    let out_h = (h + 2 * pad - kernel) / stride + 1;
    let out_w = (w + 2 * pad - kernel) / stride + 1;

    // For each kernel offset (ky, kx), gather the strided window slice
    // [B, out_h, out_w, C] and fold it into the running max.
    let mut acc: Option<UniquePtr<MlxArray>> = None;
    for ky in 0..kernel {
        // Row indices into the padded tensor: ky + stride * [0..out_h).
        let h_idx: Vec<i32> = (0..out_h).map(|o| ky + stride * o).collect();
        let h_idx_arr = mlxcel_core::from_slice_i32(&h_idx, &[out_h]);
        // Gather along H (axis 1).
        let rows = mlxcel_core::take(&padded, &h_idx_arr, 1);
        for kx in 0..kernel {
            let w_idx: Vec<i32> = (0..out_w).map(|o| kx + stride * o).collect();
            let w_idx_arr = mlxcel_core::from_slice_i32(&w_idx, &[out_w]);
            // Gather along W (axis 2) of the row-gathered tensor.
            let cell = mlxcel_core::take(&rows, &w_idx_arr, 2);
            acc = Some(match acc {
                None => cell,
                Some(prev) => mlxcel_core::maximum(&prev, &cell),
            });
        }
    }
    acc.expect("max_pool2d kernel must be >= 1")
}

/// Nearest-neighbour 2x upsample of an NHWC tensor along H and W.
///
/// Matches https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/rt_detr_v2/vision.py (_upsample_nearest_2x):
/// `broadcast x[:, :, None, :, None, :] -> (B, H, 2, W, 2, C)` then reshape.
pub fn upsample_nearest_2x(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let (b, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
    // Insert singleton axes at positions 2 and 4 -> (B, H, 1, W, 1, C).
    let expanded = mlxcel_core::expand_dims_multi(x, &[2, 4]);
    let broadcast = mlxcel_core::broadcast_to(&expanded, &[b, h, 2, w, 2, c]);
    mlxcel_core::reshape(&broadcast, &[b, h * 2, w * 2, c])
}

/// Bilinear `grid_sample` over an NHWC tensor with `padding_mode="zeros"` and
/// `align_corners=False`.
///
/// Args:
/// - `x`: `(B, H, W, C)`.
/// - `grid`: `(B, gN, gM, 2)`, normalized to `[-1, 1]` (last dim is `(gx, gy)`).
///
/// Returns `(B, gN, gM, C)`. This is a faithful Rust port of the pure-MLX
/// fallback `_grid_sample_mlx` in
/// https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/kernels.py, which is numerically
/// identical to the upstream Metal kernel.
pub fn grid_sample(x: &MlxArray, grid: &MlxArray) -> UniquePtr<MlxArray> {
    let xs = mlxcel_core::array_shape(x);
    let gs = mlxcel_core::array_shape(grid);
    let (b, h, w, c) = (xs[0], xs[1], xs[2], xs[3]);
    let (g_n, g_m) = (gs[1], gs[2]);

    let dtype = mlxcel_core::array_dtype(x);
    let f = |v: f32| mlxcel_core::full_f32(&[1], v, dtype);

    // Split grid into x / y components, each (B, gN, gM). Slice the last axis
    // then drop the size-1 trailing dim via reshape (not `squeeze`, which would
    // collapse *all* unit dims).
    let grid_x = slice_last(grid, 0, 1, &gs);
    let grid_y = slice_last(grid, 1, 2, &gs);
    let grid_x = mlxcel_core::reshape(&grid_x, &[b, g_n, g_m]);
    let grid_y = mlxcel_core::reshape(&grid_y, &[b, g_n, g_m]);

    // Unnormalize to pixel coords: ix = ((gx + 1) * W - 1) / 2.
    let ix = {
        let t = mlxcel_core::add(&grid_x, &f(1.0));
        let t = mlxcel_core::multiply_scalar(&t, w as f32);
        let t = mlxcel_core::subtract(&t, &f(1.0));
        mlxcel_core::divide_scalar(&t, 2.0)
    };
    let iy = {
        let t = mlxcel_core::add(&grid_y, &f(1.0));
        let t = mlxcel_core::multiply_scalar(&t, h as f32);
        let t = mlxcel_core::subtract(&t, &f(1.0));
        mlxcel_core::divide_scalar(&t, 2.0)
    };

    let ix0 = mlxcel_core::floor(&ix);
    let iy0 = mlxcel_core::floor(&iy);
    let one = f(1.0);
    let ix1 = mlxcel_core::add(&ix0, &one);
    let iy1 = mlxcel_core::add(&iy0, &one);

    // Bilinear weights (each (B, gN, gM)).
    let wa = mlxcel_core::multiply(
        &mlxcel_core::subtract(&ix1, &ix),
        &mlxcel_core::subtract(&iy1, &iy),
    );
    let wb = mlxcel_core::multiply(
        &mlxcel_core::subtract(&ix, &ix0),
        &mlxcel_core::subtract(&iy1, &iy),
    );
    let wc = mlxcel_core::multiply(
        &mlxcel_core::subtract(&ix1, &ix),
        &mlxcel_core::subtract(&iy, &iy0),
    );
    let wd = mlxcel_core::multiply(
        &mlxcel_core::subtract(&ix, &ix0),
        &mlxcel_core::subtract(&iy, &iy0),
    );

    let x_flat = mlxcel_core::reshape(x, &[b, h * w, c]);

    // Gather one bilinear corner: zero out-of-bounds, clip indices, gather.
    let gather = |yy: &MlxArray, xx: &MlxArray| -> UniquePtr<MlxArray> {
        let h_arr = f(h as f32);
        let w_arr = f(w as f32);
        let zero = mlxcel_core::full_f32(&[1], 0.0, dtype);
        let hm1 = f((h - 1) as f32);
        let wm1 = f((w - 1) as f32);

        // valid = (yy>=0)&(yy<H)&(xx>=0)&(xx<W) as 0/1 float mask.
        let vy_lo = ge(yy, &zero, dtype);
        let vy_hi = lt(yy, &h_arr, dtype);
        let vx_lo = ge(xx, &zero, dtype);
        let vx_hi = lt(xx, &w_arr, dtype);
        let valid = mlxcel_core::multiply(
            &mlxcel_core::multiply(&vy_lo, &vy_hi),
            &mlxcel_core::multiply(&vx_lo, &vx_hi),
        ); // (B, gN, gM)

        let yy_c = mlxcel_core::clip(yy, &zero, &hm1);
        let xx_c = mlxcel_core::clip(xx, &zero, &wm1);
        // flat index = yy_c * W + xx_c, as int32, shape (B, gN*gM).
        let idx = mlxcel_core::add(&mlxcel_core::multiply(&yy_c, &w_arr), &xx_c);
        let idx = mlxcel_core::astype(&idx, mlxcel_core::dtype::INT32);
        let idx = mlxcel_core::reshape(&idx, &[b, g_n * g_m]);
        // Expand index to (B, gN*gM, C) and gather along axis 1.
        let idx = mlxcel_core::expand_dims(&idx, 2);
        let idx = mlxcel_core::broadcast_to(&idx, &[b, g_n * g_m, c]);
        let vals = mlxcel_core::take_along_axis(&x_flat, &idx, 1); // (B, gN*gM, C)
        let vals = mlxcel_core::reshape(&vals, &[b, g_n, g_m, c]);
        // Multiply by validity mask (broadcast over channel).
        let valid = mlxcel_core::expand_dims(&valid, 3); // (B, gN, gM, 1)
        mlxcel_core::multiply(&vals, &valid)
    };

    // Weighted sum of the four corners; weights need a trailing channel axis.
    let wch = |wt: &MlxArray| mlxcel_core::expand_dims(wt, 3);
    let term_a = mlxcel_core::multiply(&wch(&wa), &gather(&iy0, &ix0));
    let term_b = mlxcel_core::multiply(&wch(&wb), &gather(&iy0, &ix1));
    let term_c = mlxcel_core::multiply(&wch(&wc), &gather(&iy1, &ix0));
    let term_d = mlxcel_core::multiply(&wch(&wd), &gather(&iy1, &ix1));

    let s1 = mlxcel_core::add(&term_a, &term_b);
    let s2 = mlxcel_core::add(&term_c, &term_d);
    mlxcel_core::add(&s1, &s2)
}

/// Slice `[start, stop)` along the last axis, preserving rank.
fn slice_last(a: &MlxArray, start: i32, stop: i32, shape: &[i32]) -> UniquePtr<MlxArray> {
    let last = shape.len() - 1;
    let mut starts: Vec<i32> = vec![0; shape.len()];
    let mut stops: Vec<i32> = shape.to_vec();
    starts[last] = start;
    stops[last] = stop;
    mlxcel_core::slice(a, &starts, &stops)
}

/// Element-wise `a >= b` as a 0/1 float mask in `dtype`.
fn ge(a: &MlxArray, b: &MlxArray, dtype: i32) -> UniquePtr<MlxArray> {
    // a >= b  <=>  NOT (a < b). Use maximum trick: (a >= b) = (maximum(a,b)==a).
    // Simpler: compute via where_cond on a boolean from subtraction sign.
    // diff = a - b; mask = diff >= 0. We synthesize the comparison with
    // clip-free arithmetic: sign-based using `maximum`.
    let one = mlxcel_core::full_f32(&[1], 1.0, dtype);
    let zero = mlxcel_core::full_f32(&[1], 0.0, dtype);
    let cond = mlxcel_core::greater_equal(a, b);
    mlxcel_core::where_cond(&cond, &one, &zero)
}

/// Element-wise `a < b` as a 0/1 float mask in `dtype`.
fn lt(a: &MlxArray, b: &MlxArray, dtype: i32) -> UniquePtr<MlxArray> {
    let one = mlxcel_core::full_f32(&[1], 1.0, dtype);
    let zero = mlxcel_core::full_f32(&[1], 0.0, dtype);
    let cond = mlxcel_core::less(a, b);
    mlxcel_core::where_cond(&cond, &one, &zero)
}
