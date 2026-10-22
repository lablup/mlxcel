// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0

//! Top-down UDP crop geometry and DARK sub-pixel heatmap decoding.
//! Numerical conventions follow mlx-vlm Sapiens2 pose.py.

use super::super::interpolate::grid_sample_host;
use anyhow::{Result, ensure};
use mlxcel_core::{MlxArray, UniquePtr};

pub fn bbox_xyxy2cs(bbox: [f32; 4], padding: f32) -> Result<([f32; 2], [f32; 2])> {
    ensure!(
        bbox.iter().all(|v| v.is_finite()) && padding.is_finite() && padding > 0.0,
        "box and padding must be finite, with positive padding"
    );
    ensure!(
        bbox[2] > bbox[0] && bbox[3] > bbox[1],
        "box must have positive area"
    );
    Ok((
        [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5],
        [(bbox[2] - bbox[0]) * padding, (bbox[3] - bbox[1]) * padding],
    ))
}

pub fn fix_aspect_ratio(scale: [f32; 2], aspect: f32) -> Result<[f32; 2]> {
    ensure!(
        aspect.is_finite() && aspect > 0.0 && scale.iter().all(|v| v.is_finite() && *v > 0.0),
        "aspect and scale must be finite and positive"
    );
    Ok(if scale[0] > scale[1] * aspect {
        [scale[0], scale[0] / aspect]
    } else {
        [scale[1] * aspect, scale[1]]
    })
}

pub fn clip_boxes(boxes: &[[f32; 4]], width: u32, height: u32) -> Result<Vec<[f32; 4]>> {
    ensure!(width > 0 && height > 0, "source image must not be empty");
    boxes
        .iter()
        .map(|bbox| {
            ensure!(
                bbox.iter().all(|v| v.is_finite()),
                "box coordinates must be finite"
            );
            let b = [
                bbox[0].clamp(0.0, width as f32),
                bbox[1].clamp(0.0, height as f32),
                bbox[2].clamp(0.0, width as f32),
                bbox[3].clamp(0.0, height as f32),
            ];
            bbox_xyxy2cs(b, 1.0)?;
            Ok(b)
        })
        .collect()
}

pub type WarpedCrop = (UniquePtr<MlxArray>, [f32; 2], [f32; 2]);

/// `input_size` is (width,height); image and returned crop are NHWC batch one.
pub fn warp_to_input(
    image: &MlxArray,
    bbox: [f32; 4],
    input_size: (usize, usize),
) -> Result<WarpedCrop> {
    let shape = mlxcel_core::array_shape(image);
    ensure!(
        shape.len() == 4 && shape[0] == 1 && shape[1] > 0 && shape[2] > 0 && shape[3] > 0,
        "pose crop expects one NHWC image"
    );
    let (w, h) = input_size;
    ensure!(
        w > 1 && h > 1 && w <= i32::MAX as usize && h <= i32::MAX as usize,
        "pose crop dimensions must be at least two and fit MLX"
    );
    let count = w
        .checked_mul(h)
        .and_then(|v| v.checked_mul(2))
        .filter(|&v| v <= i32::MAX as usize)
        .ok_or_else(|| anyhow::anyhow!("pose crop too large"))?;
    let (center, scale) = bbox_xyxy2cs(bbox, 1.25)?;
    let scale = fix_aspect_ratio(scale, w as f32 / h as f32)?;
    let mut grid = Vec::with_capacity(count);
    for y in 0..h {
        for x in 0..w {
            let sx = center[0] - scale[0] * 0.5 + scale[0] / (w - 1) as f32 * x as f32;
            let sy = center[1] - scale[1] * 0.5 + scale[1] / (h - 1) as f32 * y as f32;
            grid.extend_from_slice(&[
                (2.0 * sx + 1.0) / shape[2] as f32 - 1.0,
                (2.0 * sy + 1.0) / shape[1] as f32 - 1.0,
            ]);
        }
    }
    Ok((grid_sample_host(image, &grid, (h, w))?, center, scale))
}

pub fn flip_indices_from_pairs(num: usize, pairs: &[[usize; 2]]) -> Result<Vec<i32>> {
    ensure!(
        num <= i32::MAX as usize,
        "keypoint count exceeds MLX limits"
    );
    let mut indices: Vec<i32> = (0..num as i32).collect();
    let mut used = vec![false; num];
    for &[left, right] in pairs {
        ensure!(
            left < num && right < num && left != right,
            "flip pair index out of range or self-paired"
        );
        ensure!(!used[left] && !used[right], "flip pairs must not overlap");
        used[left] = true;
        used[right] = true;
        indices.swap(left, right);
    }
    Ok(indices)
}

/// Gaussian blur stays on the device as upstream's separable matrix products.
/// Only maxima, indices and seven DARK neighbors per keypoint reach the host.
pub fn udp_decode_batch(
    heatmaps: &MlxArray,
    input_size: (usize, usize),
    kernel_size: usize,
) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
    let shape = mlxcel_core::array_shape(heatmaps);
    ensure!(
        shape.len() == 4 && shape[0] > 0 && shape[1] > 1 && shape[2] > 1 && shape[3] > 0,
        "UDP heatmaps must be [N,H,W,K] with H,W > 1"
    );
    ensure!(
        kernel_size > 0 && kernel_size % 2 == 1 && kernel_size <= 255,
        "UDP blur kernel must be odd and between 1 and 255"
    );
    ensure!(
        input_size.0 > 0 && input_size.1 > 0,
        "UDP input dimensions must be positive"
    );
    let [batch, h, w, k] = [
        shape[0] as usize,
        shape[1] as usize,
        shape[2] as usize,
        shape[3] as usize,
    ];
    ensure!(
        h.checked_mul(w).is_some_and(|n| n <= 16_777_216)
            && batch
                .checked_mul(h)
                .and_then(|n| n.checked_mul(w))
                .and_then(|n| n.checked_mul(k))
                .is_some_and(|n| n <= i32::MAX as usize),
        "UDP heatmap dimensions exceed index limits"
    );
    ensure!(
        h <= 4096 && w <= 4096,
        "UDP heatmaps exceed supported blur dimensions"
    );
    let heatmaps = mlxcel_core::astype(heatmaps, mlxcel_core::dtype::FLOAT32);
    let finite = mlxcel_core::astype(
        &mlxcel_core::all_all(&mlxcel_core::isfinite(&heatmaps)),
        mlxcel_core::dtype::FLOAT32,
    );
    ensure!(
        mlxcel_core::utils::array_to_vec_f32(&finite).first() == Some(&1.0),
        "pose heatmaps contain non-finite values"
    );
    let flat = mlxcel_core::reshape(&heatmaps, &[batch as i32, (h * w) as i32, k as i32]);
    let score_array = mlxcel_core::max_axis(&flat, 1, false);
    let scores = mlxcel_core::utils::array_to_vec_f32(&score_array);
    let indices = mlxcel_core::astype(
        &mlxcel_core::argmax(&flat, 1, false),
        mlxcel_core::dtype::FLOAT32,
    );
    let indices = mlxcel_core::utils::array_to_vec_f32(&indices);
    let blurred = blur_heatmaps(&heatmaps, [batch, h, w, k], kernel_size);
    let blurred_flat = mlxcel_core::reshape(&blurred, &[batch as i32, (h * w) as i32, k as i32]);
    let blur_max =
        mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::max_axis(&blurred_flat, 1, false));
    let offsets = [(0, 0), (1, 0), (0, 1), (1, 1), (-1, -1), (-1, 0), (0, -1)];
    let mut gather = Vec::with_capacity(batch * k * 7);
    for b in 0..batch {
        for c in 0..k {
            let index = indices[b * k + c] as usize;
            let (x, y) = ((index % w) as isize, (index / w) as isize);
            for (dx, dy) in offsets {
                let (xx, yy) = (
                    (x + dx).clamp(0, w as isize - 1) as usize,
                    (y + dy).clamp(0, h as isize - 1) as usize,
                );
                gather.push(((b * h * w + yy * w + xx) * k + c) as i32);
            }
        }
    }
    let gather = mlxcel_core::from_slice_i32(&gather, &[(batch * k * 7) as i32]);
    let blurred_flat = mlxcel_core::reshape(&blurred, &[(batch * h * w * k) as i32]);
    let neighbors =
        mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::take(&blurred_flat, &gather, 0));
    let mut keys = vec![0.0; batch * k * 2];
    for j in 0..batch * k {
        let factor = if blur_max[j] > 0.0 {
            scores[j] / blur_max[j]
        } else {
            0.0
        };
        let at = |n: usize| (neighbors[j * 7 + n] * factor).clamp(1e-3, 50.0).ln();
        let (i, xp, yp, pp, mm, xm, ym) = (at(0), at(1), at(2), at(3), at(4), at(5), at(6));
        let (dx, dy) = (0.5 * (xp - xm), 0.5 * (yp - ym));
        let (a, d) = (
            xp - 2.0 * i + xm + f32::EPSILON,
            yp - 2.0 * i + ym + f32::EPSILON,
        );
        let cross = 0.5 * (pp - xp - yp + i + i - xm - ym + mm);
        let det = a * d - cross * cross;
        let offset = [(d * dx - cross * dy) / det, (a * dy - cross * dx) / det];
        let index = indices[j] as usize;
        for (axis, &offset) in offset.iter().enumerate() {
            let loc = if scores[j] <= 0.0 {
                -1.0
            } else if axis == 0 {
                (index % w) as f32
            } else {
                (index / w) as f32
            };
            let scale = if axis == 0 {
                input_size.0 as f32 / (w - 1) as f32
            } else {
                input_size.1 as f32 / (h - 1) as f32
            };
            keys[j * 2 + axis] = (loc - offset) * scale;
        }
    }
    Ok((
        mlxcel_core::from_slice_f32(&keys, &[batch as i32, k as i32, 2]),
        score_array,
    ))
}

fn blur_heatmaps(
    heatmaps: &MlxArray,
    [batch, h, w, k]: [usize; 4],
    kernel_size: usize,
) -> UniquePtr<MlxArray> {
    let radius = kernel_size / 2;
    // Python math.exp computes in double before the upstream f32 kernel cast.
    let sigma = 0.3 * ((kernel_size - 1) as f64 * 0.5 - 1.0) + 0.8;
    let kernel: Vec<f64> = (0..kernel_size)
        .map(|i| (-((i as f64 - radius as f64).powi(2)) / (2.0 * sigma * sigma)).exp())
        .collect();
    let total: f64 = kernel.iter().sum();
    let matrix = |n: usize| {
        let mut values = vec![0.0f32; n * n];
        for row in 0..n {
            for (tap, weight) in kernel.iter().enumerate() {
                let col = row as isize + tap as isize - radius as isize;
                if col >= 0 && col < n as isize {
                    values[row * n + col as usize] = (*weight / total) as f32;
                }
            }
        }
        mlxcel_core::from_slice_f32(&values, &[n as i32, n as i32])
    };
    let x = mlxcel_core::reshape(heatmaps, &[batch as i32, h as i32, (w * k) as i32]);
    let x = mlxcel_core::matmul(&matrix(h), &x);
    let x = mlxcel_core::reshape(&x, &[batch as i32, h as i32, w as i32, k as i32]);
    let x = mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]);
    let x = mlxcel_core::reshape(&x, &[batch as i32, w as i32, (h * k) as i32]);
    let x = mlxcel_core::matmul(&matrix(w), &x);
    let x = mlxcel_core::reshape(&x, &[batch as i32, w as i32, h as i32, k as i32]);
    mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3])
}

pub fn keypoints_to_image(
    keypoints: &MlxArray,
    input_size: (usize, usize),
    centers: &[[f32; 2]],
    scales: &[[f32; 2]],
) -> Result<UniquePtr<MlxArray>> {
    let shape = mlxcel_core::array_shape(keypoints);
    ensure!(
        shape.len() == 3
            && shape[2] == 2
            && shape[0] as usize == centers.len()
            && centers.len() == scales.len(),
        "keypoint batch does not match crop geometry"
    );
    ensure!(
        input_size.0 > 0 && input_size.1 > 0,
        "keypoint input dimensions must be positive"
    );
    ensure!(
        centers.iter().flatten().all(|v| v.is_finite())
            && scales.iter().flatten().all(|v| v.is_finite() && *v > 0.0),
        "invalid keypoint crop geometry"
    );
    let mut values = mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(
        keypoints,
        mlxcel_core::dtype::FLOAT32,
    ));
    for b in 0..centers.len() {
        for k in 0..shape[1] as usize {
            for axis in 0..2 {
                let pos = (b * shape[1] as usize + k) * 2 + axis;
                let size = if axis == 0 {
                    input_size.0
                } else {
                    input_size.1
                };
                values[pos] = values[pos] / size as f32 * scales[b][axis] + centers[b][axis]
                    - 0.5 * scales[b][axis];
            }
        }
    }
    Ok(mlxcel_core::from_slice_f32(&values, &shape))
}

#[cfg(test)]
#[path = "pose_tests.rs"]
mod tests;
