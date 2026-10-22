// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Task-specific Sapiens2 image preprocessing.

use anyhow::Result;
use image::DynamicImage;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::{Sapiens2Config, Sapiens2Task};
use crate::vision::dense::interpolate::{resize_bicubic_nhwc, resize_bilinear_nhwc};

const IMAGENET_MEAN_255: [f32; 3] = [123.675, 116.28, 103.53];
const IMAGENET_STD_255: [f32; 3] = [58.395, 57.12, 57.375];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Padding {
    pub left: usize,
    pub right: usize,
    pub top: usize,
    pub bottom: usize,
}

pub struct ProcessedImage {
    pub pixels: UniquePtr<MlxArray>,
    pub padding: Option<Padding>,
}

/// Decode through `image`'s RGB8 path and keep the upstream `[0, 255]`
/// scale through resize/warp. The order matters for BF16 boundary rounding.
pub(super) fn image_to_nhwc_255(image: &DynamicImage) -> UniquePtr<MlxArray> {
    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();
    let values: Vec<f32> = rgb.as_raw().iter().map(|&value| f32::from(value)).collect();
    mlxcel_core::from_slice_f32(&values, &[1, height as i32, width as i32, 3])
}

pub(super) fn normalize_sapiens2(x: &MlxArray) -> Result<UniquePtr<MlxArray>> {
    let shape = mlxcel_core::array_shape(x);
    anyhow::ensure!(
        shape.len() == 4 && shape[3] == 3,
        "Sapiens2 normalization expects NHWC RGB, got {shape:?}"
    );
    let x = mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32);
    let mean = mlxcel_core::from_slice_f32(&IMAGENET_MEAN_255, &[1, 1, 1, 3]);
    let std = mlxcel_core::from_slice_f32(&IMAGENET_STD_255, &[1, 1, 1, 3]);
    Ok(mlxcel_core::divide(&mlxcel_core::subtract(&x, &mean), &std))
}

pub fn preprocess_image(image: &DynamicImage, config: &Sapiens2Config) -> Result<ProcessedImage> {
    let pixels = image_to_nhwc_255(image);
    let shape = mlxcel_core::array_shape(&pixels);
    let source = (shape[1] as usize, shape[2] as usize);
    let target = config.image_size;
    let padded = matches!(config.task, Sapiens2Task::Normal | Sapiens2Task::Pointmap);
    let (pixels, padding) = if padded {
        resize_and_pad(&pixels, source, target)?
    } else {
        (resize_stretched(&pixels, source, target)?, None)
    };
    Ok(ProcessedImage {
        pixels: normalize_sapiens2(&pixels)?,
        padding,
    })
}

fn resize_stretched(
    x: &MlxArray,
    source: (usize, usize),
    target: (usize, usize),
) -> Result<UniquePtr<MlxArray>> {
    if source == target {
        return Ok(mlxcel_core::copy(x));
    }
    // Upstream selects its resize kernel from horizontal scale: antialiased
    // bilinear when shrinking, bicubic otherwise.
    if target.1 < source.1 {
        resize_bilinear_nhwc(x, target, false, true)
    } else {
        resize_bicubic_nhwc(x, Some(target), None, false)
    }
}

fn resize_and_pad(
    x: &MlxArray,
    source: (usize, usize),
    target: (usize, usize),
) -> Result<(UniquePtr<MlxArray>, Option<Padding>)> {
    let scale = (target.1 as f64 / source.1 as f64).min(target.0 as f64 / source.0 as f64);
    let resized = (
        (source.0 as f64 * scale) as usize,
        (source.1 as f64 * scale) as usize,
    );
    let top = (target.0 - resized.0) / 2;
    let left = (target.1 - resized.1) / 2;
    let padding = Padding {
        left,
        right: target.1 - resized.1 - left,
        top,
        bottom: target.0 - resized.0 - top,
    };
    let resized = resize_bilinear_nhwc(x, resized, false, false)?;
    let padded = mlxcel_core::pad(
        &resized,
        &[
            0,
            0,
            padding.top as i32,
            padding.bottom as i32,
            padding.left as i32,
            padding.right as i32,
            0,
            0,
        ],
        0.0,
    );
    Ok((padded, Some(padding)))
}

pub fn remove_padding(x: &MlxArray, padding: Padding) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    mlxcel_core::slice(
        x,
        &[0, padding.top as i32, padding.left as i32, 0],
        &[
            shape[0],
            shape[1] - padding.bottom as i32,
            shape[2] - padding.right as i32,
            shape[3],
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remove_padding_restores_inner_shape() {
        let x = mlxcel_core::zeros(&[1, 8, 10, 3], mlxcel_core::dtype::FLOAT32);
        let out = remove_padding(
            &x,
            Padding {
                left: 2,
                right: 3,
                top: 1,
                bottom: 2,
            },
        );
        assert_eq!(mlxcel_core::array_shape(&out), vec![1, 5, 5, 3]);
    }
}
