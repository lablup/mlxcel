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

//! Shared dense-model image preprocessing.

use anyhow::{Result, ensure};
use image::DynamicImage;
use mlxcel_core::{MlxArray, UniquePtr};

use super::interpolate::resize_bilinear_nhwc;

pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
pub const DINOV2_PATCH_SIZE: usize = 14;

/// Convert an image to `[1, H, W, 3]` RGB f32 in `[0, 1]`.
///
/// `image`'s RGB32F conversion scales u8 by 255 and u16 by 65535, preserving
/// the integer dtype's full range rather than treating every source as u8.
pub fn image_to_nhwc_f32(image: &DynamicImage) -> UniquePtr<MlxArray> {
    let rgb = image.to_rgb32f();
    let (width, height) = rgb.dimensions();
    mlxcel_core::from_slice_f32(rgb.as_raw(), &[1, height as i32, width as i32, 3])
}

/// Apply ImageNet RGB channel normalization to an NHWC tensor.
pub fn normalize_imagenet(x: &MlxArray) -> Result<UniquePtr<MlxArray>> {
    let shape = mlxcel_core::array_shape(x);
    ensure!(
        shape.len() == 4 && shape[3] == 3,
        "ImageNet normalization expects NHWC RGB, got {shape:?}"
    );
    let x = mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32);
    let mean = mlxcel_core::from_slice_f32(&IMAGENET_MEAN, &[1, 1, 1, 3]);
    let std = mlxcel_core::from_slice_f32(&IMAGENET_STD, &[1, 1, 1, 3]);
    Ok(mlxcel_core::divide(&mlxcel_core::subtract(&x, &mean), &std))
}

/// Round a positive spatial size to the nearest positive patch multiple.
pub fn size_to_patch_multiple(size: (usize, usize), patch_size: usize) -> Result<(usize, usize)> {
    ensure!(patch_size > 0, "patch size must be positive");
    ensure!(
        size.0 > 0 && size.1 > 0,
        "image dimensions must be positive"
    );
    let align = |value: usize| {
        let rounded = ((value + patch_size / 2) / patch_size) * patch_size;
        rounded.max(patch_size)
    };
    Ok((align(size.0), align(size.1)))
}

/// Bilinearly resize NHWC input so both spatial dimensions are patch-aligned.
pub fn resize_to_patch_multiple(x: &MlxArray, patch_size: usize) -> Result<UniquePtr<MlxArray>> {
    let shape = mlxcel_core::array_shape(x);
    ensure!(shape.len() == 4, "patch resize expects NHWC rank 4");
    let size = size_to_patch_multiple((shape[1] as usize, shape[2] as usize), patch_size)?;
    resize_bilinear_nhwc(x, size, false, true)
}

/// Convert a PyTorch OIHW convolution kernel to MLX OHWI once.
///
/// A trailing `expected_in_channels` axis proves the input is already OHWI and
/// returns a copy. A matching second axis identifies OIHW. Other layouts are
/// rejected instead of being silently transposed. As with other mlxcel vision
/// sanitizers, a shape where both axes equal `expected_in_channels` is treated
/// as already converted; callers should combine this guard with checkpoint-key
/// layout detection for ambiguous RGB 3x3 kernels.
pub fn conv_weight_to_channels_last(
    weight: &MlxArray,
    expected_in_channels: i32,
) -> Result<UniquePtr<MlxArray>> {
    let shape = mlxcel_core::array_shape(weight);
    ensure!(
        shape.len() == 4,
        "convolution weight must be rank 4, got {shape:?}"
    );
    if shape[3] == expected_in_channels {
        return Ok(mlxcel_core::copy(weight));
    }
    ensure!(
        shape[1] == expected_in_channels,
        "convolution weight has no input-channel axis of {expected_in_channels}: {shape:?}"
    );
    Ok(mlxcel_core::transpose_axes(weight, &[0, 2, 3, 1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u16_images_scale_by_their_dtype_range() {
        let image = DynamicImage::ImageRgb16(
            image::ImageBuffer::<image::Rgb<u16>, Vec<u16>>::from_raw(
                1,
                1,
                vec![u16::MAX, 32768, 0],
            )
            .unwrap(),
        );
        let array = image_to_nhwc_f32(&image);
        let values = mlxcel_core::utils::array_to_vec_f32(&array);
        assert_eq!(values[0], 1.0);
        assert!((values[1] - 32768.0 / 65535.0).abs() < 1e-6);
        assert_eq!(values[2], 0.0);
    }

    #[test]
    fn patch_alignment_uses_nearest_multiple_with_one_patch_minimum() {
        assert_eq!(size_to_patch_multiple((7, 21), 14).unwrap(), (14, 28));
        assert_eq!(size_to_patch_multiple((28, 35), 14).unwrap(), (28, 42));
    }

    #[test]
    fn convolution_permutation_is_idempotent() {
        let raw = mlxcel_core::from_slice_f32(&[0.0; 2 * 4 * 3 * 5], &[2, 4, 3, 5]);
        let converted = conv_weight_to_channels_last(&raw, 4).unwrap();
        assert_eq!(mlxcel_core::array_shape(&converted), vec![2, 3, 5, 4]);
        let twice = conv_weight_to_channels_last(&converted, 4).unwrap();
        assert_eq!(mlxcel_core::array_shape(&twice), vec![2, 3, 5, 4]);
    }
}
