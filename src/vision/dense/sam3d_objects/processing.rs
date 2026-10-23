// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Object crops, camera point normalization, and trained condition dropout.
use super::super::interpolate::{resize_bilinear_objects_nhwc, resize_nearest_objects_nhwc};
use anyhow::{Result, ensure};
use image::DynamicImage;
use mlxcel_core::{self as mx, MlxArray, UniquePtr};
use std::collections::BTreeMap;
#[path = "processing_budget.rs"]
mod budget;

pub type Inputs = BTreeMap<String, UniquePtr<MlxArray>>;
pub struct Metadata {
    pub shift: [f32; 3],
    pub scale: f32,
    pub conditioned: bool,
}

pub fn rgb(image: &DynamicImage) -> Result<UniquePtr<MlxArray>> {
    let values = image.to_rgb32f().into_raw();
    ensure!(
        values
            .iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
        "image values must be finite and in [0,1]"
    );
    Ok(mx::from_slice_f32(
        &values,
        &[image.height() as i32, image.width() as i32, 3],
    ))
}

pub fn mask_values(image: &DynamicImage, mask: Option<&DynamicImage>) -> Result<Vec<f32>> {
    let values = if let Some(mask) = mask {
        ensure!(
            mask.width() == image.width() && mask.height() == image.height(),
            "mask must have the same height and width as the image"
        );
        // The file CLI selects source alpha when present, even when fully opaque.
        let rgba = mask.to_rgba8();
        let alpha = mask.color().has_alpha();
        rgba.pixels()
            .map(|p| f32::from(p[if alpha { 3 } else { 0 }] > 0))
            .collect()
    } else {
        ensure!(
            image.color().has_alpha(),
            "provide an object mask or an RGBA image"
        );
        let alpha: Vec<u8> = image.to_rgba8().pixels().map(|p| p[3]).collect();
        let threshold = if alpha.iter().copied().max().unwrap_or(0) > 1 {
            127
        } else {
            0
        };
        alpha
            .into_iter()
            .map(|v| f32::from(v > threshold))
            .collect()
    };
    Ok(values)
}

/// Reject cheap input errors before loading the optional depth model.
pub fn validate_image_mask(
    image: &DynamicImage,
    mask: Option<&DynamicImage>,
    size: usize,
    points: Option<&MlxArray>,
    estimate_depth: bool,
) -> Result<()> {
    let (h, w) = (image.height() as usize, image.width() as usize);
    ensure!(
        h > 0 && w > 0 && h.checked_mul(w).is_some_and(|n| n <= 32 * 1024 * 1024),
        "image exceeds Objects pixel limit"
    );
    budget::source_preflight(h, w)?;
    let values = mask_values(image, mask)?;
    let bounds = mask_bounds(&values, h, w)?;
    let point_shape = if let Some(points) = points {
        Some(point_shape(points)?)
    } else {
        estimate_depth.then_some([h, w])
    };
    budget::preflight(h, w, bounds, size, point_shape)?;
    if let Some(rgb) = image.as_rgb32f() {
        ensure!(
            rgb.as_raw()
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
            "image values must be finite and in [0,1]"
        );
    }
    if let Some(rgba) = image.as_rgba32f() {
        ensure!(
            rgba.as_raw().chunks_exact(4).all(|p| p[..3]
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))),
            "image values must be finite and in [0,1]"
        );
    }
    Ok(())
}

fn point_shape(points: &MlxArray) -> Result<[usize; 2]> {
    let s = mx::array_shape(points);
    ensure!(
        s.len() == 3 && s[0] > 0 && s[1] > 0 && s[2] == 3,
        "point map must be HWC XYZ"
    );
    Ok([s[0] as usize, s[1] as usize])
}

fn mask_bounds(mask: &[f32], h: usize, w: usize) -> Result<[i32; 4]> {
    let (mut x0, mut x1, mut y0, mut y1) = (w, 0, h, 0);
    for (i, &value) in mask.iter().enumerate() {
        if value > 0.0 {
            x0 = x0.min(i % w);
            x1 = x1.max(i % w);
            y0 = y0.min(i / w);
            y1 = y1.max(i / w);
        }
    }
    ensure!(
        x1 > x0 && y1 > y0,
        "object mask must span at least two rows and columns"
    );
    let side = (((x1 - x0).max(y1 - y0).max(2)) as f64 * 1.2) as i32;
    let (cx, cy) = ((x0 + x1) as f64 / 2.0, (y0 + y1) as f64 / 2.0);
    let half = f64::from(side / 2);
    let bounds = [
        (cx - half) as i32,
        (cy - half) as i32,
        (cx + half) as i32,
        (cy + half) as i32,
    ];
    Ok(bounds)
}

fn square(x: &MlxArray, fill: f32) -> UniquePtr<MlxArray> {
    let s = mx::array_shape(x);
    let side = s[0].max(s[1]);
    let top = (side - s[0]) / 2;
    let left = (side - s[1]) / 2;
    mx::pad(
        x,
        &[top, side - s[0] - top, left, side - s[1] - left, 0, 0],
        fill,
    )
}
fn crop(x: &MlxArray, b: [i32; 4]) -> UniquePtr<MlxArray> {
    let s = mx::array_shape(x);
    let [left, top, right, bottom] = b;
    let padded = mx::pad(
        x,
        &[
            (-top).max(0),
            (bottom - s[0]).max(0),
            (-left).max(0),
            (right - s[1]).max(0),
            0,
            0,
        ],
        0.0,
    );
    let y = top.max(0);
    let z = left.max(0);
    mx::slice(
        &padded,
        &[y, z, 0],
        &[y + bottom - top, z + right - left, s[2]],
    )
}
fn median(mut values: Vec<f32>) -> f32 {
    values.retain(|v| !v.is_nan());
    if values.is_empty() {
        return f32::NAN;
    }
    values.sort_by(f32::total_cmp);
    values[(values.len() - 1) / 2]
}

pub fn prepare_inputs(
    image: &DynamicImage,
    mask: Option<&DynamicImage>,
    points: Option<&MlxArray>,
    size: usize,
) -> Result<(Inputs, Metadata)> {
    let (h, w) = (image.height() as usize, image.width() as usize);
    ensure!(
        h > 0 && w > 0 && h.checked_mul(w).is_some_and(|n| n <= 32 * 1024 * 1024),
        "image exceeds Objects pixel limit"
    );
    budget::source_preflight(h, w)?;
    let mask = mask_values(image, mask)?;
    let bounds = mask_bounds(&mask, h, w)?;
    budget::preflight(h, w, bounds, size, points.map(point_shape).transpose()?)?;
    let mut inputs = Inputs::new();
    let image_array = rgb(image)?;
    let mask_array = mx::from_slice_f32(&mask, &[h as i32, w as i32, 1]);
    for (name, full, value, nearest) in [
        ("image", "rgb_image", &image_array, false),
        ("mask", "rgb_image_mask", &mask_array, true),
    ] {
        for (key, value) in [(name, crop(value, bounds)), (full, mx::copy(value))] {
            let x = mx::expand_dims(&square(&value, 0.0), 0);
            let x = if nearest {
                resize_nearest_objects_nhwc(&x, (size, size))?
            } else {
                resize_bilinear_objects_nhwc(&x, (size, size))?
            };
            inputs.insert(key.into(), x);
        }
    }
    let mut meta = Metadata {
        shift: [0.0; 3],
        scale: 1.0,
        conditioned: points.is_some(),
    };
    if let Some(points) = points {
        let s = mx::array_shape(points);
        ensure!(
            s.len() == 3 && s[0] > 0 && s[1] > 0 && s[2] == 3,
            "point map must be HWC XYZ"
        );
        let points = resize_nearest_objects_nhwc(
            &mx::expand_dims(&mx::astype(points, mx::dtype::FLOAT32), 0),
            (h, w),
        )?;
        let points = mx::squeeze_axis(&points, 0);
        let values = mx::utils::array_to_vec_f32(&points);
        for c in 0..3 {
            meta.shift[c] = median(
                values
                    .chunks_exact(3)
                    .zip(&mask)
                    .filter_map(|(p, m)| (*m > 0.0).then_some(p[c]))
                    .collect(),
            );
        }
        ensure!(
            meta.shift.iter().all(|v| v.is_finite()),
            "object mask has no finite point-map center"
        );
        // MLX max propagates NaN: preserve that behavior before the lower median.
        meta.scale = median(
            values
                .chunks_exact(3)
                .map(|p| {
                    let d = [
                        (p[0] - meta.shift[0]).abs(),
                        (p[1] - meta.shift[1]).abs(),
                        (p[2] - meta.shift[2]).abs(),
                    ];
                    if d.iter().any(|v| v.is_nan()) {
                        f32::NAN
                    } else {
                        d[0].max(d[1]).max(d[2])
                    }
                })
                .collect(),
        );
        ensure!(
            meta.scale.is_finite() && meta.scale > 0.0,
            "point map has invalid or zero spatial scale"
        );
        let normalized = mx::divide(
            &mx::subtract(&points, &mx::from_slice_f32(&meta.shift, &[3])),
            &mx::from_slice_f32(&[meta.scale], &[]),
        );
        for (name, value, fill) in [
            ("pointmap", crop(&normalized, bounds), f32::NAN),
            ("rgb_pointmap", normalized, 0.0),
        ] {
            inputs.insert(
                name.into(),
                resize_nearest_objects_nhwc(
                    &mx::expand_dims(&square(&value, fill), 0),
                    (size, size),
                )?,
            );
        }
    }
    Ok((inputs, meta))
}

#[cfg(test)]
#[path = "processing_tests.rs"]
mod tests;
