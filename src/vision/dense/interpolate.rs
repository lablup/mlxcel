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

//! PyTorch-compatible interpolation for channel-last MLX tensors.
//!
//! The coordinate and tap calculations are a direct port of mlx-vlm's
//! `models/interpolate.py`. Tap tables are small host values; gathering and
//! weighted reduction remain MLX graph operations on the input device.
//!
//! Used by: VideoDepthAnything, MoGe3, Sapiens2, Sam3dObjects.

use anyhow::{Result, bail, ensure};
use mlxcel_core::{MlxArray, UniquePtr};

#[derive(Debug)]
struct Weights1d {
    indices: Vec<i32>,
    weights: Vec<f32>,
    out: usize,
    taps: usize,
}

fn checked_nhwc_shape(x: &MlxArray) -> Result<[usize; 4]> {
    let shape = mlxcel_core::array_shape(x);
    ensure!(
        shape.len() == 4,
        "dense interpolation expects NHWC rank 4, got {shape:?}"
    );
    let mut out = [0; 4];
    for (index, dim) in shape.into_iter().enumerate() {
        ensure!(dim > 0, "dense interpolation dimensions must be positive");
        out[index] = dim as usize;
    }
    Ok(out)
}

fn checked_size(new_size: (usize, usize)) -> Result<(usize, usize)> {
    ensure!(
        new_size.0 > 0 && new_size.1 > 0,
        "interpolation output dimensions must be positive"
    );
    ensure!(
        new_size.0 <= i32::MAX as usize && new_size.1 <= i32::MAX as usize,
        "interpolation output dimensions exceed MLX limits"
    );
    Ok(new_size)
}

fn bilinear_weights_1d(
    in_size: usize,
    out_size: usize,
    align_corners: bool,
    antialias: bool,
) -> Weights1d {
    let scale = if align_corners {
        if out_size > 1 {
            (in_size - 1) as f32 / (out_size - 1) as f32
        } else {
            0.0
        }
    } else {
        in_size as f32 / out_size as f32
    };

    if antialias && scale > 1.0 {
        let taps = (2.0 * scale).ceil() as usize;
        let mut indices = Vec::with_capacity(out_size * taps);
        let mut weights = Vec::with_capacity(out_size * taps);
        for output in 0..out_size {
            let src = if align_corners {
                output as f32 * scale
            } else {
                (output as f32 + 0.5) * scale - 0.5
            };
            let lo = (src - scale + 1.0).floor().max(0.0);
            let hi = (src + scale).floor().min((in_size - 1) as f32);
            let start = weights.len();
            let mut total = 0.0f32;
            for tap in 0..taps {
                let pos = lo + tap as f32;
                let weight = if pos <= hi {
                    (1.0 - (pos - src).abs() / scale).max(0.0)
                } else {
                    0.0
                };
                indices.push(pos.min((in_size - 1) as f32) as i32);
                weights.push(weight);
                total += weight;
            }
            for weight in &mut weights[start..] {
                *weight /= total;
            }
        }
        return Weights1d {
            indices,
            weights,
            out: out_size,
            taps,
        };
    }

    let taps = 2;
    let mut indices = Vec::with_capacity(out_size * taps);
    let mut weights = Vec::with_capacity(out_size * taps);
    for output in 0..out_size {
        let src = if align_corners {
            output as f32 * scale
        } else {
            (output as f32 + 0.5) * scale - 0.5
        }
        .max(0.0);
        let base = src.floor();
        let frac = src - base;
        indices.push((base as i32).clamp(0, in_size as i32 - 1));
        indices.push((base as i32 + 1).clamp(0, in_size as i32 - 1));
        weights.extend_from_slice(&[1.0 - frac, frac]);
    }
    Weights1d {
        indices,
        weights,
        out: out_size,
        taps,
    }
}

fn cubic_weight(value: f32, a: f32) -> f32 {
    let at = value.abs();
    let at2 = at * at;
    let at3 = at2 * at;
    if at <= 1.0 {
        (a + 2.0) * at3 - (a + 3.0) * at2 + 1.0
    } else if at < 2.0 {
        a * at3 - 5.0 * a * at2 + 8.0 * a * at - 4.0 * a
    } else {
        0.0
    }
}

fn bicubic_weights_1d(in_size: usize, out_size: usize, scale: f32) -> Weights1d {
    let taps = 4;
    let mut indices = Vec::with_capacity(out_size * taps);
    let mut weights = Vec::with_capacity(out_size * taps);
    for output in 0..out_size {
        let src = (output as f32 + 0.5) / scale - 0.5;
        let base = src.floor();
        for offset in -1..=2 {
            let pos = base + offset as f32;
            indices.push((pos as i32).clamp(0, in_size as i32 - 1));
            weights.push(cubic_weight(src - pos, -0.75));
        }
    }
    Weights1d {
        indices,
        weights,
        out: out_size,
        taps,
    }
}

fn bicubic_aa_weights_1d(in_size: usize, out_size: usize) -> Weights1d {
    let scale = in_size as f32 / out_size as f32;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let taps = support.ceil() as usize * 2 + 1;
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let mut indices = Vec::with_capacity(out_size * taps);
    let mut weights = Vec::with_capacity(out_size * taps);
    for output in 0..out_size {
        let center = scale * (output as f32 + 0.5);
        let xmin = (center - support + 0.5).floor().max(0.0);
        let xmax = (center + support + 0.5).floor().min(in_size as f32);
        let xsize = (xmax - xmin).clamp(0.0, taps as f32);
        let start = weights.len();
        let mut total = 0.0f32;
        for tap in 0..taps {
            let pos = xmin + tap as f32;
            let weight = if (tap as f32) < xsize {
                cubic_weight((pos - center + 0.5) * invscale, -0.5)
            } else {
                0.0
            };
            indices.push(pos.min((in_size - 1) as f32) as i32);
            weights.push(weight);
            total += weight;
        }
        if total > 0.0 {
            for weight in &mut weights[start..] {
                *weight /= total.max(1e-30);
            }
        }
    }
    Weights1d {
        indices,
        weights,
        out: out_size,
        taps,
    }
}

fn interpolate_axis(
    x: &MlxArray,
    table: &Weights1d,
    axis: i32,
    shape_with_taps: &[i32],
    weight_shape: &[i32],
) -> UniquePtr<MlxArray> {
    let indices = mlxcel_core::from_slice_i32(&table.indices, &[table.indices.len() as i32]);
    let gathered = mlxcel_core::take(x, &indices, axis);
    let gathered = mlxcel_core::reshape(&gathered, shape_with_taps);
    let weights = mlxcel_core::from_slice_f32(&table.weights, weight_shape);
    mlxcel_core::sum_axis(&mlxcel_core::multiply(&gathered, &weights), axis + 1, false)
}

fn separable_interpolate(
    x: &MlxArray,
    input: [usize; 4],
    y: &Weights1d,
    x_weights: &Weights1d,
) -> UniquePtr<MlxArray> {
    let [batch, _, width, channels] = input.map(|dim| dim as i32);
    let as_f32 = mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32);
    let height = interpolate_axis(
        &as_f32,
        y,
        1,
        &[batch, y.out as i32, y.taps as i32, width, channels],
        &[1, y.out as i32, y.taps as i32, 1, 1],
    );
    interpolate_axis(
        &height,
        x_weights,
        2,
        &[
            batch,
            y.out as i32,
            x_weights.out as i32,
            x_weights.taps as i32,
            channels,
        ],
        &[1, 1, x_weights.out as i32, x_weights.taps as i32, 1],
    )
}

/// Resize `[N, H, W, C]` with PyTorch `bilinear`, `align_corners` and
/// antialias semantics.
///
/// Interpolation is computed in f32. When the requested spatial size already
/// matches the input, the upstream-compatible identity path returns a copy in
/// the input dtype.
pub fn resize_bilinear_nhwc(
    x: &MlxArray,
    new_size: (usize, usize),
    align_corners: bool,
    antialias: bool,
) -> Result<UniquePtr<MlxArray>> {
    let input = checked_nhwc_shape(x)?;
    let (out_h, out_w) = checked_size(new_size)?;
    if (input[1], input[2]) == (out_h, out_w) {
        return Ok(mlxcel_core::copy(x));
    }
    let y = bilinear_weights_1d(input[1], out_h, align_corners, antialias);
    let x_weights = bilinear_weights_1d(input[2], out_w, align_corners, antialias);
    Ok(separable_interpolate(x, input, &y, &x_weights))
}

/// Resize `[N, H, W, C]` with PyTorch `bicubic` semantics.
///
/// Supply exactly one of `new_size` or `scale_factor`. A scale factor floors
/// the output dimensions. Non-antialiased sampling retains the supplied
/// factor; antialiased sampling intentionally derives its scale from the
/// floored output size, matching upstream. As with bilinear resize, the
/// identity-size path preserves the input dtype and other paths produce f32.
pub fn resize_bicubic_nhwc(
    x: &MlxArray,
    new_size: Option<(usize, usize)>,
    scale_factor: Option<(f32, f32)>,
    antialias: bool,
) -> Result<UniquePtr<MlxArray>> {
    let input = checked_nhwc_shape(x)?;
    let ((out_h, out_w), (scale_h, scale_w)) = match (new_size, scale_factor) {
        (Some(size), None) => {
            let size = checked_size(size)?;
            (
                size,
                (
                    size.0 as f32 / input[1] as f32,
                    size.1 as f32 / input[2] as f32,
                ),
            )
        }
        (None, Some(scale)) => {
            ensure!(
                scale.0.is_finite() && scale.1.is_finite() && scale.0 > 0.0 && scale.1 > 0.0,
                "bicubic scale factors must be finite and positive"
            );
            let size = checked_size((
                (input[1] as f32 * scale.0) as usize,
                (input[2] as f32 * scale.1) as usize,
            ))?;
            (size, scale)
        }
        _ => bail!("supply exactly one of bicubic new_size or scale_factor"),
    };
    if (input[1], input[2]) == (out_h, out_w) {
        return Ok(mlxcel_core::copy(x));
    }
    let y = if antialias {
        bicubic_aa_weights_1d(input[1], out_h)
    } else {
        bicubic_weights_1d(input[1], out_h, scale_h)
    };
    let x_weights = if antialias {
        bicubic_aa_weights_1d(input[2], out_w)
    } else {
        bicubic_weights_1d(input[2], out_w, scale_w)
    };
    Ok(separable_interpolate(x, input, &y, &x_weights))
}

/// Resize `[N, H, W, C]` with PyTorch's `nearest` source-index rule.
pub fn resize_nearest_nhwc(x: &MlxArray, new_size: (usize, usize)) -> Result<UniquePtr<MlxArray>> {
    let input = checked_nhwc_shape(x)?;
    let (out_h, out_w) = checked_size(new_size)?;
    let mut output = mlxcel_core::copy(x);
    if input[1] != out_h {
        let indices: Vec<i32> = (0..out_h)
            .map(|index| {
                ((index as f32 * input[1] as f32 / out_h as f32).floor() as i32)
                    .min(input[1] as i32 - 1)
            })
            .collect();
        let indices = mlxcel_core::from_slice_i32(&indices, &[out_h as i32]);
        output = mlxcel_core::take(&output, &indices, 1);
    }
    if input[2] != out_w {
        let indices: Vec<i32> = (0..out_w)
            .map(|index| {
                ((index as f32 * input[2] as f32 / out_w as f32).floor() as i32)
                    .min(input[2] as i32 - 1)
            })
            .collect();
        let indices = mlxcel_core::from_slice_i32(&indices, &[out_w as i32]);
        output = mlxcel_core::take(&output, &indices, 2);
    }
    Ok(output)
}

/// Bilinear sample of NHWC images using half-pixel normalized coordinates
/// and zero padding, exactly as mlx-vlm `kernels.grid_sample`.
///
/// Used by: Sapiens2.
/// Coordinates are read on the host; pixel gathering stays on the device.
/// Call `grid_sample_host` when coordinates already originate on the host.
pub fn grid_sample(x: &MlxArray, grid: &MlxArray) -> Result<UniquePtr<MlxArray>> {
    let [batch, _, _, _] = checked_nhwc_shape(x)?;
    let shape = mlxcel_core::array_shape(grid);
    ensure!(
        shape.len() == 4 && shape[0] == batch as i32 && shape[3] == 2,
        "grid_sample expects grid [N,H,W,2] matching image batch"
    );
    ensure!(
        shape[1] > 0 && shape[2] > 0,
        "grid dimensions must be positive"
    );
    let coordinates = mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(
        grid,
        mlxcel_core::dtype::FLOAT32,
    ));
    grid_sample_host(x, &coordinates, (shape[1] as usize, shape[2] as usize))
}

/// Host-coordinate variant avoiding a host/device round trip for crop grids.
/// Coordinates have shape `[N, grid_size.0, grid_size.1, 2]`.
/// Used by: Sapiens2.
pub fn grid_sample_host(
    x: &MlxArray,
    grid: &[f32],
    grid_size: (usize, usize),
) -> Result<UniquePtr<MlxArray>> {
    let [batch, height, width, channels] = checked_nhwc_shape(x)?;
    let (out_h, out_w) = checked_size(grid_size)?;
    let shape = [batch as i32, out_h as i32, out_w as i32, 2];
    let pixels = batch
        .checked_mul(height)
        .and_then(|n| n.checked_mul(width))
        .filter(|&n| n <= i32::MAX as usize)
        .ok_or_else(|| anyhow::anyhow!("grid_sample image exceeds index limits"))?;
    let count = batch
        .checked_mul(shape[1] as usize)
        .and_then(|n| n.checked_mul(shape[2] as usize))
        .filter(|&n| n <= i32::MAX as usize / 4)
        .ok_or_else(|| anyhow::anyhow!("grid_sample output exceeds index limits"))?;
    ensure!(
        grid.len() == count * 2,
        "grid coordinate count does not match requested shape"
    );
    let mut indices = Vec::with_capacity(count * 4);
    let mut weights = Vec::with_capacity(count * 4);
    let per_batch = shape[1] as usize * shape[2] as usize;
    for (i, xy) in grid.chunks_exact(2).enumerate() {
        ensure!(
            xy.iter().all(|v| v.is_finite()),
            "grid coordinates must be finite"
        );
        let sx = ((xy[0] + 1.0) * width as f32 - 1.0) / 2.0;
        let sy = ((xy[1] + 1.0) * height as f32 - 1.0) / 2.0;
        // Entirely outside the support: avoid saturated integer conversion.
        if sx < -1.0 || sx >= width as f32 || sy < -1.0 || sy >= height as f32 {
            indices.extend_from_slice(&[0; 4]);
            weights.extend_from_slice(&[0.0; 4]);
            continue;
        }
        let x0 = sx.floor() as i32;
        let y0 = sy.floor() as i32;
        let dx = sx - x0 as f32;
        let dy = sy - y0 as f32;
        for (xx, yy, weight) in [
            (x0, y0, (1.0 - dx) * (1.0 - dy)),
            (x0 + 1, y0, dx * (1.0 - dy)),
            (x0, y0 + 1, (1.0 - dx) * dy),
            (x0 + 1, y0 + 1, dx * dy),
        ] {
            let valid = xx >= 0 && xx < width as i32 && yy >= 0 && yy < height as i32;
            indices.push(if valid {
                ((i / per_batch * height + yy as usize) * width + xx as usize) as i32
            } else {
                0
            });
            weights.push(if valid { weight } else { 0.0 });
        }
    }
    let flat = mlxcel_core::reshape(x, &[pixels as i32, channels as i32]);
    let indices = mlxcel_core::from_slice_i32(&indices, &[(count * 4) as i32]);
    let gathered = mlxcel_core::take(&flat, &indices, 0);
    let gathered = mlxcel_core::reshape(&gathered, &[count as i32, 4, channels as i32]);
    let weights = mlxcel_core::from_slice_f32(&weights, &[count as i32, 4, 1]);
    let result = mlxcel_core::sum_axis(&mlxcel_core::multiply(&gathered, &weights), 1, false);
    Ok(mlxcel_core::reshape(
        &result,
        &[batch as i32, shape[1], shape[2], channels as i32],
    ))
}

#[cfg(test)]
#[path = "interpolate_tests.rs"]
mod tests;

#[path = "interpolate_objects.rs"]
mod objects;
pub(crate) use objects::{
    resize_bicubic_objects_nhwc, resize_bilinear_objects_nhwc,
    resize_bilinear_objects_nhwc_with_antialias, resize_nearest_objects_nhwc,
};
