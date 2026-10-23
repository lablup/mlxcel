// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Objects follows the pinned pure-MLX resize graph, including F32 rounding.
use super::*;

fn axis(x: &MlxArray, table: &Weights1d) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    interpolate_axis(
        x,
        table,
        1,
        &[shape[0], table.out as i32, table.taps as i32, shape[2]],
        &[1, table.out as i32, table.taps as i32, 1],
    )
}

/// Match pinned kernels.py::_separable_interpolate_mlx's axis ordering.
/// The two orders are mathematically equivalent but round differently in F32.
pub(crate) fn resize_bilinear_objects_nhwc(
    x: &MlxArray,
    new_size: (usize, usize),
) -> Result<UniquePtr<MlxArray>> {
    let input = checked_nhwc_shape(x)?;
    let (out_h, out_w) = checked_size(new_size)?;
    if (input[1], input[2]) == (out_h, out_w) {
        return Ok(mlxcel_core::copy(x));
    }
    let y = bilinear_weights_1d(input[1], out_h, false, true);
    let w = bilinear_weights_1d(input[2], out_w, false, true);
    // u128 keeps the upstream cost comparison exact even for extreme shapes.
    let [_, height, width, _] = input.map(|v| v as u128);
    let (oh, ow, ty, tx) = (out_h as u128, out_w as u128, y.taps as u128, w.taps as u128);
    let h_first = oh * width * (ty + 1) + oh * ow * (tx + 1);
    let w_first = height * width + height * ow * (tx + 1) + oh * ow * ty;
    let [n, h, width, c] = input.map(|v| v as i32);
    let (oh, ow) = (out_h as i32, out_w as i32);
    // MLX reshape dimensions are I32, including each flattened inner axis.
    let inner = |a: i32, b: i32| -> Result<i32> {
        a.checked_mul(b)
            .ok_or_else(|| anyhow::anyhow!("Objects interpolation inner axis exceeds I32"))
    };
    let x = mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32);
    if h_first <= w_first {
        let x = mlxcel_core::reshape(&x, &[n, h, inner(width, c)?]);
        let x = axis(&x, &y);
        let x = mlxcel_core::reshape(&x, &[n, oh, width, c]);
        let x = mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]);
        let x = mlxcel_core::reshape(&x, &[n, width, inner(oh, c)?]);
        let x = axis(&x, &w);
        let x = mlxcel_core::reshape(&x, &[n, ow, oh, c]);
        Ok(mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]))
    } else {
        let x = mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]);
        let x = mlxcel_core::reshape(&x, &[n, width, inner(h, c)?]);
        let x = axis(&x, &w);
        let x = mlxcel_core::reshape(&x, &[n, ow, h, c]);
        let x = mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]);
        let x = mlxcel_core::reshape(&x, &[n, h, inner(ow, c)?]);
        Ok(mlxcel_core::reshape(&axis(&x, &y), &[n, oh, ow, c]))
    }
}

/// Match nearest_indices: round the Python ratio to F32 before multiplication.
pub(crate) fn resize_nearest_objects_nhwc(
    x: &MlxArray,
    new_size: (usize, usize),
) -> Result<UniquePtr<MlxArray>> {
    let input = checked_nhwc_shape(x)?;
    let (out_h, out_w) = checked_size(new_size)?;
    let mut output = mlxcel_core::copy(x);
    for (axis, size) in [(1, out_h), (2, out_w)] {
        if input[axis] == size {
            continue;
        }
        let ratio = (input[axis] as f64 / size as f64) as f32;
        let indices: Vec<i32> = (0..size)
            .map(|i| ((i as f32 * ratio).floor() as i32).min(input[axis] as i32 - 1))
            .collect();
        let indices = mlxcel_core::from_slice_i32(&indices, &[size as i32]);
        output = mlxcel_core::take(&output, &indices, axis as i32);
    }
    Ok(output)
}

#[cfg(test)]
#[path = "interpolate_objects_tests.rs"]
mod tests;
