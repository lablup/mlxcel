// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Objects follows the pinned pure-MLX resize graph, including F32 rounding.
use super::*;
use mlxcel_core as mx;

struct ObjectsWeights {
    indices: UniquePtr<MlxArray>,
    weights: UniquePtr<MlxArray>,
    out: usize,
    taps: usize,
}
impl From<Weights1d> for ObjectsWeights {
    fn from(table: Weights1d) -> Self {
        Self {
            indices: mx::from_slice_i32(&table.indices, &[table.indices.len() as i32]),
            weights: mx::from_slice_f32(&table.weights, &[table.out as i32, table.taps as i32]),
            out: table.out,
            taps: table.taps,
        }
    }
}
fn axis(x: &MlxArray, table: &ObjectsWeights) -> UniquePtr<MlxArray> {
    let shape = mx::array_shape(x);
    let gathered = mx::take(x, &table.indices, 1);
    let gathered = mx::reshape(
        &gathered,
        &[shape[0], table.out as i32, table.taps as i32, shape[2]],
    );
    let weights = mx::reshape(&table.weights, &[1, table.out as i32, table.taps as i32, 1]);
    mx::sum_axis(&mx::multiply(&gathered, &weights), 2, false)
}

// Keep the Divide and cubic arithmetic on the same device as upstream's
// _bicubic_weights_1d, rather than replacing them with host F32 calculations.
fn bicubic_objects_weights(in_size: usize, out: usize, scale: f64) -> ObjectsWeights {
    let scalar = |x: f32| mx::from_slice_f32(&[x], &[]);
    let positions = mx::from_slice_f32(
        &(0..out).map(|i| i as f32).collect::<Vec<_>>(),
        &[out as i32],
    );
    let src = mx::subtract(
        &mx::divide(&mx::add(&positions, &scalar(0.5)), &scalar(scale as f32)),
        &scalar(0.5),
    );
    let base = mx::reshape(&mx::floor(&src), &[out as i32, 1]);
    let taps = mx::add(&base, &mx::from_slice_f32(&[-1., 0., 1., 2.], &[1, 4]));
    let at = mx::abs(&mx::subtract(&mx::reshape(&src, &[out as i32, 1]), &taps));
    let at2 = mx::multiply(&at, &at);
    let at3 = mx::multiply(&at2, &at);
    let w1 = mx::add(
        &mx::subtract(
            &mx::multiply(&scalar(1.25), &at3),
            &mx::multiply(&scalar(2.25), &at2),
        ),
        &scalar(1.),
    );
    let w2 = mx::subtract(
        &mx::add(
            &mx::subtract(
                &mx::multiply(&scalar(-0.75), &at3),
                &mx::multiply(&scalar(-3.75), &at2),
            ),
            &mx::multiply(&scalar(-6.), &at),
        ),
        &scalar(-3.),
    );
    let weights = mx::where_cond(
        &mx::less_equal(&at, &scalar(1.)),
        &w1,
        &mx::where_cond(
            &mx::less(&at, &scalar(2.)),
            &w2,
            &mx::zeros(&[out as i32, 4], mx::dtype::FLOAT32),
        ),
    );
    let indices = mx::astype(&taps, mx::dtype::INT32);
    let indices = mx::clip(
        &indices,
        &mx::from_slice_i32(&[0], &[]),
        &mx::from_slice_i32(&[(in_size - 1) as i32], &[]),
    );
    ObjectsWeights {
        indices: mx::reshape(&indices, &[-1]),
        weights,
        out,
        taps: 4,
    }
}

/// Match pinned kernels.py::_separable_interpolate_mlx's axis ordering.
/// The two orders are mathematically equivalent but round differently in F32.
pub(crate) fn resize_bilinear_objects_nhwc(
    x: &MlxArray,
    new_size: (usize, usize),
) -> Result<UniquePtr<MlxArray>> {
    resize_bilinear_objects_nhwc_with_antialias(x, new_size, true)
}

/// Pinned pure-MLX bilinear graph with the caller's antialias setting.
pub(crate) fn resize_bilinear_objects_nhwc_with_antialias(
    x: &MlxArray,
    new_size: (usize, usize),
    antialias: bool,
) -> Result<UniquePtr<MlxArray>> {
    let input = checked_nhwc_shape(x)?;
    let (out_h, out_w) = checked_size(new_size)?;
    if (input[1], input[2]) == (out_h, out_w) {
        return Ok(mlxcel_core::copy(x));
    }
    let y = bilinear_weights_1d(input[1], out_h, false, antialias).into();
    let w = bilinear_weights_1d(input[2], out_w, false, antialias).into();
    separable_objects(x, input, &y, &w)
}

fn separable_objects(
    x: &MlxArray,
    input: [usize; 4],
    y: &ObjectsWeights,
    w: &ObjectsWeights,
) -> Result<UniquePtr<MlxArray>> {
    let (out_h, out_w) = (y.out, w.out);
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
        let x = axis(&x, y);
        let x = mlxcel_core::reshape(&x, &[n, oh, width, c]);
        let x = mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]);
        let x = mlxcel_core::reshape(&x, &[n, width, inner(oh, c)?]);
        let x = axis(&x, w);
        let x = mlxcel_core::reshape(&x, &[n, ow, oh, c]);
        Ok(mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]))
    } else {
        let x = mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]);
        let x = mlxcel_core::reshape(&x, &[n, width, inner(h, c)?]);
        let x = axis(&x, w);
        let x = mlxcel_core::reshape(&x, &[n, ow, h, c]);
        let x = mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]);
        let x = mlxcel_core::reshape(&x, &[n, h, inner(ow, c)?]);
        Ok(mlxcel_core::reshape(&axis(&x, y), &[n, oh, ow, c]))
    }
}

/// Match Python's double output-size calculation and F32 tensor scalars.
/// Tap reductions use the same adaptive axis order and flattened layout.
pub(crate) fn resize_bicubic_objects_nhwc(
    x: &MlxArray,
    new_size: Option<(usize, usize)>,
    scale_factor: Option<(f64, f64)>,
    antialias: bool,
) -> Result<UniquePtr<MlxArray>> {
    let input = checked_nhwc_shape(x)?;
    let (size, scales) = match (new_size, scale_factor) {
        (Some(size), None) => {
            let size = checked_size(size)?;
            (
                size,
                (
                    size.0 as f64 / input[1] as f64,
                    size.1 as f64 / input[2] as f64,
                ),
            )
        }
        (None, Some(scales)) => {
            ensure!(
                scales.0.is_finite() && scales.1.is_finite() && scales.0 > 0.0 && scales.1 > 0.0,
                "bicubic scale factors must be finite and positive"
            );
            let size = checked_size((
                (input[1] as f64 * scales.0) as usize,
                (input[2] as f64 * scales.1) as usize,
            ))?;
            (size, scales)
        }
        _ => bail!("supply exactly one of bicubic new_size or scale_factor"),
    };
    if (input[1], input[2]) == size {
        return Ok(mlxcel_core::copy(x));
    }
    let y = if antialias {
        bicubic_aa_weights_1d(input[1], size.0).into()
    } else {
        bicubic_objects_weights(input[1], size.0, scales.0)
    };
    let w = if antialias {
        bicubic_aa_weights_1d(input[2], size.1).into()
    } else {
        bicubic_objects_weights(input[2], size.1, scales.1)
    };
    separable_objects(x, input, &y, &w)
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
