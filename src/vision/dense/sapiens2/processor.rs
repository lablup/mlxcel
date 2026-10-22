// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Task-specific Sapiens2 image preprocessing.

use anyhow::Result;
use image::DynamicImage;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::{Sapiens2Config, Sapiens2Task};
use crate::vision::dense::interpolate::{resize_bicubic_nhwc, resize_bilinear_nhwc};
use crate::vision::dense::preprocess::{image_to_nhwc_f32, normalize_imagenet};

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

pub fn preprocess_image(image: &DynamicImage, config: &Sapiens2Config) -> Result<ProcessedImage> {
    let pixels = image_to_nhwc_f32(image);
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
        pixels: normalize_imagenet(&pixels)?,
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
