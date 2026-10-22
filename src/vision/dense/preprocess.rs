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
use image::{DynamicImage, RgbImage};
use mlxcel_core::{MlxArray, UniquePtr};

use super::interpolate::resize_bilinear_nhwc;

pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
pub const DINOV2_PATCH_SIZE: usize = 14;

const PILLOW_PRECISION_BITS: i32 = 22;
const PILLOW_ROUNDING: i64 = 1 << (PILLOW_PRECISION_BITS - 1);

/// Resampling kernels implemented with Pillow's uint8 fixed-point arithmetic.
#[derive(Clone, Copy, Debug)]
pub enum PillowResizeFilter {
    /// Pillow `Resampling.BILINEAR`.
    Bilinear,
    /// Pillow `Resampling.BICUBIC`.
    Bicubic,
}

impl PillowResizeFilter {
    fn support(self) -> f64 {
        match self {
            Self::Bilinear => 1.0,
            Self::Bicubic => 2.0,
        }
    }

    fn evaluate(self, mut value: f64) -> f64 {
        value = value.abs();
        match self {
            Self::Bilinear if value < 1.0 => 1.0 - value,
            Self::Bilinear => 0.0,
            Self::Bicubic if value < 1.0 => ((1.5 * value - 2.5) * value * value) + 1.0,
            Self::Bicubic if value < 2.0 => (((value - 5.0) * value + 8.0) * value - 4.0) * -0.5,
            Self::Bicubic => 0.0,
        }
    }
}

#[derive(Debug)]
struct PillowCoefficients {
    bounds: Vec<(usize, usize)>,
    values: Vec<i32>,
    taps: usize,
}

fn pillow_coefficients(
    input: usize,
    output: usize,
    filter: PillowResizeFilter,
) -> Result<PillowCoefficients> {
    ensure!(
        input > 0 && output > 0,
        "Pillow resize dimensions must be positive"
    );
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = filter.support() * filter_scale;
    let taps = support.ceil() as usize * 2 + 1;
    let count = output
        .checked_mul(taps)
        .ok_or_else(|| anyhow::anyhow!("Pillow resize coefficient count overflowed"))?;
    let mut bounds = Vec::with_capacity(output);
    let mut values = vec![0; count];
    for position in 0..output {
        let center = (position as f64 + 0.5) * scale;
        let start = ((center - support + 0.5) as isize).max(0) as usize;
        let end = ((center + support + 0.5) as usize).min(input);
        let used = end.saturating_sub(start);
        let mut raw = Vec::with_capacity(used);
        let mut total = 0.0;
        for tap in 0..used {
            let value = filter.evaluate(((tap + start) as f64 - center + 0.5) / filter_scale);
            raw.push(value);
            total += value;
        }
        for (tap, value) in raw.into_iter().enumerate() {
            let normalized = if total == 0.0 { value } else { value / total };
            let scaled = normalized * f64::from(1 << PILLOW_PRECISION_BITS);
            values[position * taps + tap] = if scaled < 0.0 {
                (scaled - 0.5) as i32
            } else {
                (scaled + 0.5) as i32
            };
        }
        bounds.push((start, used));
    }
    Ok(PillowCoefficients {
        bounds,
        values,
        taps,
    })
}

fn pillow_clip(value: i64) -> u8 {
    (value >> PILLOW_PRECISION_BITS).clamp(0, 255) as u8
}

fn resize_pillow_horizontal(
    input: &RgbImage,
    output_width: u32,
    filter: PillowResizeFilter,
) -> Result<RgbImage> {
    let (input_width, height) = input.dimensions();
    let table = pillow_coefficients(input_width as usize, output_width as usize, filter)?;
    let mut output = RgbImage::new(output_width, height);
    let output_data: &mut [u8] = output.as_mut();
    for row in 0..height as usize {
        for column in 0..output_width as usize {
            let (start, used) = table.bounds[column];
            let weights = &table.values[column * table.taps..][..used];
            for channel in 0..3 {
                let mut sum = PILLOW_ROUNDING;
                for (tap, weight) in weights.iter().enumerate() {
                    let value =
                        input.as_raw()[((row * input_width as usize + start + tap) * 3) + channel];
                    sum += i64::from(value) * i64::from(*weight);
                }
                output_data[((row * output_width as usize + column) * 3) + channel] =
                    pillow_clip(sum);
            }
        }
    }
    Ok(output)
}

fn resize_pillow_vertical(
    input: &RgbImage,
    output_height: u32,
    filter: PillowResizeFilter,
) -> Result<RgbImage> {
    let (width, input_height) = input.dimensions();
    let table = pillow_coefficients(input_height as usize, output_height as usize, filter)?;
    let mut output = RgbImage::new(width, output_height);
    let output_data: &mut [u8] = output.as_mut();
    for row in 0..output_height as usize {
        let (start, used) = table.bounds[row];
        let weights = &table.values[row * table.taps..][..used];
        for column in 0..width as usize {
            for channel in 0..3 {
                let mut sum = PILLOW_ROUNDING;
                for (tap, weight) in weights.iter().enumerate() {
                    let value =
                        input.as_raw()[(((start + tap) * width as usize + column) * 3) + channel];
                    sum += i64::from(value) * i64::from(*weight);
                }
                output_data[((row * width as usize + column) * 3) + channel] = pillow_clip(sum);
            }
        }
    }
    Ok(output)
}

/// Resize RGB8 pixels with Pillow's fixed-point coefficient and pass ordering.
pub fn resize_pillow_rgb8(
    image: &RgbImage,
    width: u32,
    height: u32,
    filter: PillowResizeFilter,
) -> Result<RgbImage> {
    ensure!(
        width > 0 && height > 0,
        "Pillow resize dimensions must be positive"
    );
    let (input_width, input_height) = image.dimensions();
    let horizontal = width != input_width;
    let vertical = height != input_height;
    let height_reduction = i64::from(input_height) - i64::from(height);
    let width_reduction = i64::from(input_width) - i64::from(width);
    let horizontal_first = !(height_reduction > 0 && height_reduction > width_reduction * 2);
    match (horizontal, vertical, horizontal_first) {
        (false, false, _) => Ok(image.clone()),
        (true, false, _) => resize_pillow_horizontal(image, width, filter),
        (false, true, _) => resize_pillow_vertical(image, height, filter),
        (true, true, true) => resize_pillow_vertical(
            &resize_pillow_horizontal(image, width, filter)?,
            height,
            filter,
        ),
        (true, true, false) => resize_pillow_horizontal(
            &resize_pillow_vertical(image, height, filter)?,
            width,
            filter,
        ),
    }
}

/// Convert an image to `[1, H, W, 3]` RGB f32 in `[0, 1]`.
///
/// `image`'s RGB32F conversion scales u8 by 255 and u16 by 65535, preserving
/// the integer dtype's full range rather than treating every source as u8.
pub fn image_to_nhwc_f32(image: &DynamicImage) -> UniquePtr<MlxArray> {
    let rgb = image.to_rgb32f();
    let (width, height) = rgb.dimensions();
    mlxcel_core::from_slice_f32(rgb.as_raw(), &[1, height as i32, width as i32, 3])
}

/// Resize RGB bytes with Pillow-compatible bilinear arithmetic before
/// converting to normalized NHWC floats.
pub fn resize_rgb8_bilinear_nhwc_f32(
    image: &DynamicImage,
    width: u32,
    height: u32,
) -> Result<UniquePtr<MlxArray>> {
    ensure!(
        width > 0 && height > 0,
        "RGB resize dimensions must be positive"
    );
    let rgb = image.to_rgb8();
    let resized = resize_pillow_rgb8(&rgb, width, height, PillowResizeFilter::Bilinear)?;
    let values: Vec<f32> = resized
        .as_raw()
        .iter()
        .map(|value| f32::from(*value) / 255.0)
        .collect();
    Ok(mlxcel_core::from_slice_f32(
        &values,
        &[1, height as i32, width as i32, 3],
    ))
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

    #[test]
    fn bilinear_resize_matches_pillow_uint8_rounding() {
        let input = (0..3 * 4 * 3)
            .map(|value| (value * 7 + 11) as u8)
            .collect::<Vec<_>>();
        let input = RgbImage::from_raw(4, 3, input).unwrap();
        let output = resize_pillow_rgb8(&input, 7, 5, PillowResizeFilter::Bilinear).unwrap();
        assert_eq!(
            output.as_raw(),
            &[
                11, 18, 25, 19, 26, 33, 31, 38, 45, 43, 50, 57, 54, 61, 68, 66, 73, 80, 74, 81, 88,
                45, 52, 59, 53, 60, 67, 65, 72, 79, 77, 84, 91, 88, 95, 102, 100, 107, 114, 108,
                115, 122, 95, 102, 109, 103, 110, 117, 115, 122, 129, 127, 134, 141, 138, 145, 152,
                150, 157, 164, 158, 165, 172, 145, 152, 159, 153, 160, 167, 165, 172, 179, 177,
                184, 191, 188, 195, 192, 200, 207, 116, 208, 215, 69, 179, 186, 193, 187, 194, 201,
                199, 206, 213, 211, 218, 225, 222, 229, 218, 234, 241, 84, 242, 249, 0,
            ]
        );
    }
}
