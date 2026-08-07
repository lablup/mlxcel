// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Falcon-OCR image-processor tests.

use super::*;

fn proc() -> FalconOcrProcessor {
    FalconOcrProcessor::default()
}

#[test]
fn an_image_already_inside_the_band_is_not_clamped() {
    assert_eq!(proc().clamp_dimensions(640, 224), None);
    assert_eq!(proc().clamp_dimensions(64, 1024), None);
}

#[test]
fn an_oversized_image_is_clamped_on_its_long_side() {
    // Landscape: the height is pinned to max_dimension and the width follows
    // the aspect ratio, then the width cap pulls it back.
    let (w, h) = proc().clamp_dimensions(4000, 2000).expect("clamped");
    assert!(w <= 1024 && h <= 1024);
    assert_eq!(w, 1024);
    assert_eq!(h, 512);
}

#[test]
fn a_tiny_image_is_scaled_up_to_the_minimum() {
    let (w, h) = proc().clamp_dimensions(40, 20).expect("clamped");
    assert!(w >= 64 || h >= 64, "got {w}x{h}");
    assert_eq!(h, 64);
    assert_eq!(w, 128);
}

#[test]
fn smart_resize_snaps_to_the_patch_grid() {
    // 224x640 is already patch-aligned and inside the pixel band.
    assert_eq!(proc().smart_resize(224, 640), (224, 640));
    // 230 -> 14.375 patches rounds down to 14 -> 224.
    assert_eq!(proc().smart_resize(230, 640).0, 224);
}

/// Python's `round` is half-to-even. At `height % 16 == 8` the two rounding
/// rules disagree and the image-token budget would differ from the reference.
#[test]
fn smart_resize_uses_bankers_rounding_on_ties() {
    // 232 / 16 == 14.5 -> banker's rounds to 14 -> 224 (not 15 -> 240).
    assert_eq!(proc().smart_resize(232, 640).0, 224);
    // 248 / 16 == 15.5 -> banker's rounds to 16 -> 256.
    assert_eq!(proc().smart_resize(248, 640).0, 256);
    assert_eq!(round_half_even(14.5), 14);
    assert_eq!(round_half_even(15.5), 16);
    assert_eq!(round_half_even(14.6), 15);
}

#[test]
fn smart_resize_shrinks_past_the_pixel_ceiling() {
    let p = proc();
    let (h, w) = p.smart_resize(1024, 1024);
    assert!(
        (h as u64) * (w as u64) <= p.max_pixels as u64,
        "got {h}x{w}"
    );
    assert_eq!(h % p.spatial_patch_size, 0);
    assert_eq!(w % p.spatial_patch_size, 0);
}

#[test]
fn smart_resize_grows_past_the_pixel_floor() {
    let p = proc();
    let (h, w) = p.smart_resize(16, 16);
    assert!(
        (h as u64) * (w as u64) >= p.min_pixels as u64,
        "got {h}x{w}"
    );
}

#[test]
fn the_patch_matrix_is_row_major_over_the_grid_and_hwc_within_a_patch() {
    // 64x64 is the smallest square the clamp stage leaves alone: 4x4 patches.
    let mut img = image::RgbImage::new(64, 64);
    for (x, y, px) in img.enumerate_pixels_mut() {
        // Encode the pixel coordinate so the flattening order is observable.
        *px = image::Rgb([x as u8, y as u8, 0]);
    }
    let p = proc();
    let (values, grids) = p.preprocess_values_with_grid(&[image::DynamicImage::ImageRgb8(img)]);
    assert_eq!(grids, vec![(4, 4)]);
    assert_eq!(values.len(), 4 * 4 * 16 * 16 * 3);

    let denorm = |v: f32| ((v * IMAGE_STD + IMAGE_MEAN) * 255.0).round() as i32;
    let patch = 16 * 16 * 3;
    // Patch (0,0) pixel (0,0) -> x=0, y=0.
    assert_eq!(denorm(values[0]), 0);
    assert_eq!(denorm(values[1]), 0);
    // Patch (0,1) is the next one in memory and covers x in 16..32.
    assert_eq!(denorm(values[patch]), 16);
    assert_eq!(denorm(values[patch + 1]), 0);
    // Patch (1,0) is one grid row later, so it appears after 4 patches.
    assert_eq!(denorm(values[4 * patch]), 0);
    assert_eq!(denorm(values[4 * patch + 1]), 16);
    // Within a patch the next element is the next x, not the next channel row.
    assert_eq!(denorm(values[3]), 1);
}

#[test]
fn normalization_maps_the_byte_range_onto_minus_one_to_one() {
    let mut img = image::RgbImage::new(16, 16);
    for px in img.pixels_mut() {
        *px = image::Rgb([0, 255, 128]);
    }
    let (values, _) = proc().preprocess_values_with_grid(&[image::DynamicImage::ImageRgb8(img)]);
    assert!((values[0] - -1.0).abs() < 1e-6);
    assert!((values[1] - 1.0).abs() < 1e-6);
    assert!((values[2] - (128.0 / 255.0 - 0.5) / 0.5).abs() < 1e-6);
}

#[test]
fn the_grid_matches_what_the_patch_matrix_produces() {
    let img = image::DynamicImage::ImageRgb8(image::RgbImage::new(200, 100));
    let p = proc();
    let grid = p.grid_for(&img);
    let (values, grids) = p.preprocess_values_with_grid(&[img]);
    assert_eq!(grids, vec![grid]);
    let patch_dim = 16 * 16 * 3;
    assert_eq!(values.len(), (grid.0 * grid.1) as usize * patch_dim);
}
