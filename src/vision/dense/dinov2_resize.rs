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

//! Pillow-compatible uint8 RGB bicubic resizing for Hugging Face processors.
//!
//! Hugging Face's slow `BitImageProcessor` delegates resize to Pillow. Pillow
//! quantizes normalized bicubic coefficients to 22-bit fixed point and rounds
//! after each separable pass. Keeping those details avoids input drift from
//! floating-point image resizers before DINOv2 inference.

use anyhow::{Result, ensure};
use image::RgbImage;

const PRECISION_BITS: i32 = 22;
const ROUNDING: i64 = 1 << (PRECISION_BITS - 1);

#[derive(Debug)]
struct Coefficients {
    bounds: Vec<(usize, usize)>,
    values: Vec<i32>,
    taps: usize,
}

fn bicubic(mut value: f64) -> f64 {
    value = value.abs();
    if value < 1.0 {
        return ((1.5 * value - 2.5) * value * value) + 1.0;
    }
    if value < 2.0 {
        return (((value - 5.0) * value + 8.0) * value - 4.0) * -0.5;
    }
    0.0
}

fn coefficients(input: usize, output: usize) -> Result<Coefficients> {
    ensure!(
        input > 0 && output > 0,
        "Pillow resize dimensions must be positive"
    );
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
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
            let value = bicubic(((tap + start) as f64 - center + 0.5) / filter_scale);
            raw.push(value);
            total += value;
        }
        for (tap, value) in raw.into_iter().enumerate() {
            let normalized = if total == 0.0 { value } else { value / total };
            let scaled = normalized * f64::from(1 << PRECISION_BITS);
            values[position * taps + tap] = if scaled < 0.0 {
                (scaled - 0.5) as i32
            } else {
                (scaled + 0.5) as i32
            };
        }
        bounds.push((start, used));
    }
    Ok(Coefficients {
        bounds,
        values,
        taps,
    })
}

fn clip(value: i64) -> u8 {
    (value >> PRECISION_BITS).clamp(0, 255) as u8
}

fn resize_horizontal(input: &RgbImage, output_width: u32) -> Result<RgbImage> {
    let (input_width, height) = input.dimensions();
    let table = coefficients(input_width as usize, output_width as usize)?;
    let mut output = RgbImage::new(output_width, height);
    let output_data: &mut [u8] = output.as_mut();
    for row in 0..height as usize {
        for column in 0..output_width as usize {
            let (start, used) = table.bounds[column];
            let weights = &table.values[column * table.taps..][..used];
            for channel in 0..3 {
                let mut sum = ROUNDING;
                for (tap, weight) in weights.iter().enumerate() {
                    let value =
                        input.as_raw()[((row * input_width as usize + start + tap) * 3) + channel];
                    sum += i64::from(value) * i64::from(*weight);
                }
                output_data[((row * output_width as usize + column) * 3) + channel] = clip(sum);
            }
        }
    }
    Ok(output)
}

fn resize_vertical(input: &RgbImage, output_height: u32) -> Result<RgbImage> {
    let (width, input_height) = input.dimensions();
    let table = coefficients(input_height as usize, output_height as usize)?;
    let mut output = RgbImage::new(width, output_height);
    let output_data: &mut [u8] = output.as_mut();
    for row in 0..output_height as usize {
        let (start, used) = table.bounds[row];
        let weights = &table.values[row * table.taps..][..used];
        for column in 0..width as usize {
            for channel in 0..3 {
                let mut sum = ROUNDING;
                for (tap, weight) in weights.iter().enumerate() {
                    let value =
                        input.as_raw()[(((start + tap) * width as usize + column) * 3) + channel];
                    sum += i64::from(value) * i64::from(*weight);
                }
                output_data[((row * width as usize + column) * 3) + channel] = clip(sum);
            }
        }
    }
    Ok(output)
}

pub(super) fn resize_pil_bicubic_rgb(
    image: &RgbImage,
    width: u32,
    height: u32,
) -> Result<RgbImage> {
    ensure!(
        width > 0 && height > 0,
        "Pillow resize dimensions must be positive"
    );
    let (input_width, input_height) = image.dimensions();
    let horizontal = width != input_width;
    let vertical = height != input_height;
    let horizontal_first = !(input_height.saturating_sub(height) > 0
        && input_height.saturating_sub(height)
            > input_width.saturating_sub(width).saturating_mul(2));
    match (horizontal, vertical, horizontal_first) {
        (false, false, _) => Ok(image.clone()),
        (true, false, _) => resize_horizontal(image, width),
        (false, true, _) => resize_vertical(image, height),
        (true, true, true) => resize_vertical(&resize_horizontal(image, width)?, height),
        (true, true, false) => resize_horizontal(&resize_vertical(image, height)?, width),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pillow_bicubic_uint8_rounding() {
        let input = (0..3 * 4 * 3)
            .map(|value| (value * 7 + 11) as u8)
            .collect::<Vec<_>>();
        let input = RgbImage::from_raw(4, 3, input).unwrap();
        let output = resize_pil_bicubic_rgb(&input, 7, 5).unwrap();
        assert_eq!(
            output.as_raw(),
            &[
                3, 10, 17, 11, 18, 25, 24, 31, 38, 37, 44, 51, 49, 56, 63, 62, 69, 76, 70, 77, 84,
                35, 42, 49, 43, 50, 57, 56, 63, 70, 69, 76, 82, 81, 88, 95, 94, 101, 116, 102, 109,
                127, 93, 100, 107, 101, 108, 115, 114, 121, 128, 127, 134, 141, 139, 146, 153, 152,
                159, 166, 160, 167, 174, 151, 158, 165, 159, 166, 173, 172, 179, 186, 185, 192,
                210, 197, 204, 204, 210, 217, 109, 218, 225, 65, 183, 190, 197, 191, 198, 205, 204,
                211, 218, 217, 224, 249, 229, 236, 230, 242, 249, 65, 250, 255, 0,
            ]
        );
    }
}
