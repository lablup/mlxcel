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

//! Tests for [`super::GotOcrImageProcessor`].

use super::*;
use image::{Rgb, RgbImage};

fn solid(width: u32, height: u32, rgb: [u8; 3]) -> DynamicImage {
    DynamicImage::ImageRgb8(RgbImage::from_pixel(width, height, Rgb(rgb)))
}

/// Any input shape lands on the tower's fixed 1024x1024 canvas, channels-last.
/// The reference passes a `(h, w)` pair to `transforms.Resize`, so nothing is
/// padded and the aspect ratio is not preserved.
#[test]
fn preprocess_emits_a_fixed_1024_square_regardless_of_aspect() {
    let processor = GotOcrImageProcessor::default();
    for (w, h) in [(64u32, 64u32), (1600, 400), (300, 1200)] {
        let pixels = processor.preprocess(&[solid(w, h, [128, 128, 128])]);
        assert_eq!(
            mlxcel_core::array_shape(&pixels),
            vec![1, 1024, 1024, 3],
            "{w}x{h}"
        );
    }
}

/// A batch keeps one canvas per image.
#[test]
fn preprocess_batches_along_the_leading_axis() {
    let processor = GotOcrImageProcessor::default();
    let pixels = processor.preprocess(&[solid(32, 32, [0, 0, 0]), solid(90, 10, [255, 255, 255])]);
    assert_eq!(mlxcel_core::array_shape(&pixels), vec![2, 1024, 1024, 3]);
}

/// Normalization is `(pixel / 255 - mean) / std` with the CLIP constants, per
/// channel. A pure-black and a pure-white pixel pin both ends, and the two
/// differ per channel, so a swapped mean/std pair or an RGB/BGR mix-up shows.
#[test]
fn normalization_uses_clip_mean_and_std_per_channel() {
    let processor = GotOcrImageProcessor::default();
    for (rgb, scaled) in [([0u8, 0, 0], 0.0f32), ([255, 255, 255], 1.0)] {
        let pixels = processor.preprocess(&[solid(8, 8, rgb)]);
        for channel in 0..3usize {
            let cell = mlxcel_core::slice(
                &pixels,
                &[0, 0, 0, channel as i32],
                &[1, 1, 1, channel as i32 + 1],
            );
            mlxcel_core::eval(&cell);
            let expected = (scaled - GOT_IMAGE_MEAN[channel]) / GOT_IMAGE_STD[channel];
            let actual = mlxcel_core::item_f32(&cell);
            assert!(
                (actual - expected).abs() < 1e-4,
                "channel {channel} for {rgb:?}: {actual} != {expected}"
            );
        }
    }
}

/// The CLIP constants are the ones `GOTImageEvalProcessor` defaults to.
///
/// Compared in f64 against the reference's published decimals, so the check
/// does not restate whatever f32 spelling the constant happens to carry.
#[test]
fn clip_constants_match_the_reference_defaults() {
    let mean = [0.48145466_f64, 0.4578275, 0.40821073];
    let std = [0.26862954_f64, 0.26130258, 0.27577711];
    for channel in 0..3 {
        assert!((GOT_IMAGE_MEAN[channel] as f64 - mean[channel]).abs() < 1e-7);
        assert!((GOT_IMAGE_STD[channel] as f64 - std[channel]).abs() < 1e-7);
    }
    assert_eq!(GOT_IMAGE_SIZE, 1024);
}

/// A non-square source is stretched, not letterboxed: the resize covers the
/// whole canvas, so no background band survives at the edges.
#[test]
fn resize_stretches_instead_of_padding() {
    let mut image = RgbImage::from_pixel(200, 50, Rgb([255, 0, 0]));
    // A distinct bottom row would land inside the canvas under a stretch and at
    // the very edge under letterboxing with a grey/black band beyond it.
    for x in 0..200 {
        image.put_pixel(x, 49, Rgb([0, 0, 255]));
    }
    let processor = GotOcrImageProcessor::default();
    let pixels = processor.preprocess(&[DynamicImage::ImageRgb8(image)]);

    // Bottom-right corner: blue under a stretch (the last source row fills the
    // last canvas row), background under letterboxing.
    let corner = mlxcel_core::slice(&pixels, &[0, 1023, 1023, 2], &[1, 1024, 1024, 3]);
    mlxcel_core::eval(&corner);
    let blue_channel = mlxcel_core::item_f32(&corner);
    let expected_blue = (1.0 - GOT_IMAGE_MEAN[2]) / GOT_IMAGE_STD[2];
    assert!(
        (blue_channel - expected_blue).abs() < 0.2,
        "expected the stretched blue row at the canvas edge, got {blue_channel}"
    );
}
