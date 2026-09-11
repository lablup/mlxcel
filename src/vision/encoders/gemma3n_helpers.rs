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

//! Helper routines for Gemma 3n's MobileNet-style vision encoder.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

pub(super) fn make_divisible(v: f32, divisor: i32) -> i32 {
    let min_value = divisor;
    let new_v = ((v + divisor as f32 / 2.0) as i32 / divisor) * divisor;
    let new_v = new_v.max(min_value);
    if (new_v as f32) < 0.9 * v {
        new_v + divisor
    } else {
        new_v
    }
}

pub(super) fn num_groups(group_size: i32, channels: i32) -> i32 {
    if group_size == 0 {
        1
    } else {
        channels / group_size
    }
}

pub(super) fn get_same_padding(
    input_size: i32,
    kernel_size: i32,
    stride: i32,
    dilation: i32,
) -> i32 {
    let eff_k = dilation * (kernel_size - 1) + 1;
    let out_size = (input_size + stride - 1) / stride;
    (0i32).max((out_size - 1) * stride + eff_k - input_size)
}

pub(super) fn is_static_pad(stride: i32) -> bool {
    stride == 1
}

pub(super) fn get_static_padding(kernel_size: i32, dilation: i32) -> i32 {
    let eff_k = dilation * (kernel_size - 1) + 1;
    (eff_k - 1) / 2
}

pub(super) fn split_symmetric_padding(total_pad: i32) -> (i32, i32) {
    (total_pad / 2, total_pad - total_pad / 2)
}

pub(super) fn get_weight(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|weight| mlxcel_core::copy(weight))
        .ok_or_else(|| format!("Weight not found: {}", name))
}

pub(super) fn sanitize_conv_weight(w: UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(&w);
    if shape.len() == 4 && shape[1] > shape[2] {
        mlxcel_core::transpose_axes(&w, &[0, 2, 3, 1])
    } else {
        w
    }
}

pub(super) fn nearest_upsample_nchw(
    x: &MlxArray,
    target_h: i32,
    target_w: i32,
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let b = shape[0];
    let c = shape[1];
    let h = shape[2];
    let w = shape[3];

    let scale_h = target_h / h;
    let scale_w = target_w / w;

    let x = mlxcel_core::reshape(x, &[b, c, h, 1, w, 1]);
    let x = mlxcel_core::broadcast_to(&x, &[b, c, h, scale_h, w, scale_w]);
    mlxcel_core::reshape(&x, &[b, c, h * scale_h, w * scale_w])
}
