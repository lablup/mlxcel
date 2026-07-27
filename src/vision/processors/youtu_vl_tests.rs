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

use super::*;
use image::{DynamicImage, RgbImage};

fn solid_image(h: u32, w: u32, rgb: [u8; 3]) -> DynamicImage {
    let mut img = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            img.put_pixel(x, y, image::Rgb(rgb));
        }
    }
    DynamicImage::ImageRgb8(img)
}

fn synthetic_processor() -> YoutuVLProcessor {
    // patch_size=16, spatial_merge_size=2 → factor = 32.
    // Use very tight pixel bounds so a small synthetic image survives
    // smart_resize untouched.
    YoutuVLProcessor::new(16, 2)
        .with_pixel_bounds(32 * 32, 256 * 256)
        .with_norm([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
}

#[test]
fn smart_resize_aligns_to_patch_merge_factor() {
    let p = synthetic_processor();

    let cases = vec![
        // Inputs that are already multiples of 32 should pass through as-is.
        (64, 64, 64, 64),
        // Inputs slightly off should round to the nearest multiple.
        (60, 100, 64, 96),
        // Tiny inputs must be lifted to satisfy the min_pixels lower bound.
        (16, 16, 32, 32),
    ];
    for (h, w, exp_h, exp_w) in cases {
        let (rh, rw) = p.smart_resize(h, w);
        assert!(
            rh % 32 == 0 && rw % 32 == 0,
            "smart_resize output ({}, {}) not aligned to 32 for input ({}, {})",
            rh,
            rw,
            h,
            w
        );
        assert_eq!(rh, exp_h, "h mismatch for input ({h}, {w})");
        assert_eq!(rw, exp_w, "w mismatch for input ({h}, {w})");
    }
}

#[test]
fn preprocess_emits_expected_patch_shape() {
    let p = synthetic_processor();
    let img = solid_image(64, 96, [128, 128, 128]);
    let (pixel_values, spatial_shapes) = p.preprocess_with_spatial(&[img]);

    // 64x96 image at patch_size=16 → 4x6 = 24 patches.
    assert_eq!(spatial_shapes, vec![(4, 6)]);

    let shape = mlxcel_core::array_shape(&pixel_values);
    let expected_patches = 4 * 6;
    let expected_features = 16 * 16 * 3;
    assert_eq!(shape, vec![expected_patches, expected_features]);
}

#[test]
fn preprocess_concatenates_multi_image_batches() {
    let p = synthetic_processor();
    let img_a = solid_image(64, 64, [10, 20, 30]);
    let img_b = solid_image(32, 64, [200, 100, 50]);
    let (pixel_values, spatial_shapes) = p.preprocess_with_spatial(&[img_a, img_b]);

    // 64x64 → (4, 4) = 16 patches; 32x64 → (2, 4) = 8 patches.
    assert_eq!(spatial_shapes, vec![(4, 4), (2, 4)]);

    let total_patches = 16 + 8;
    let shape = mlxcel_core::array_shape(&pixel_values);
    assert_eq!(shape, vec![total_patches, 16 * 16 * 3]);
}

#[test]
fn smart_resize_honors_num_patches_cap() {
    let p = YoutuVLProcessor::new(16, 2)
        .with_max_patches_per_image(4096)
        .with_pixel_bounds(32 * 32, usize::MAX);
    let shapes = p.compute_spatial_shapes(&[solid_image(4096, 4096, [1, 2, 3])]);
    let (h_patches, w_patches) = shapes[0];
    assert!(
        (h_patches as usize) * (w_patches as usize) <= 4096,
        "patch grid {:?} exceeds num_patches cap",
        shapes[0]
    );
}

#[test]
fn try_preprocess_rejects_cap_below_alignment_floor() {
    let p = YoutuVLProcessor::new(16, 2)
        .with_max_patches_per_image(3)
        .with_pixel_bounds(1, usize::MAX);
    let err = match p.try_preprocess_with_spatial(&[solid_image(64, 64, [1, 2, 3])]) {
        Ok(_) => panic!("expected preprocessing to reject a cap below the aligned patch floor"),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        YoutuVLPreprocessError::TooManyPatches {
            patches: 4,
            max_patches: 3,
            ..
        }
    ));
}

#[test]
fn try_preprocess_rejects_resize_factor_above_u32() {
    let p = YoutuVLProcessor::new(u32::MAX as usize + 1, 1);
    let err = match p.try_preprocess_with_spatial(&[]) {
        Ok(_) => panic!("expected preprocessing to reject an oversized resize factor"),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        YoutuVLPreprocessError::DimensionTooLarge { .. }
    ));
}

#[test]
fn normalization_matches_siglip_default() {
    let p = synthetic_processor();
    // A pure mid-gray image should normalize close to zero (val - 0.5)/0.5.
    let img = solid_image(64, 64, [128, 128, 128]);
    let (pixel_values, _) = p.preprocess_with_spatial(&[img]);
    mlxcel_core::eval(&pixel_values);

    let max_abs = mlxcel_core::max_all(&mlxcel_core::abs(&pixel_values));
    mlxcel_core::eval(&max_abs);
    // 128/255 ≈ 0.502; (0.502 - 0.5)/0.5 ≈ 0.0039
    assert!(
        mlxcel_core::item_f32(&max_abs) < 0.02,
        "mid-gray normalized values should sit close to 0; saw {}",
        mlxcel_core::item_f32(&max_abs)
    );
}

#[test]
fn flattened_patch_layout_matches_hf_merge_groups_and_channel_order() {
    let processor = YoutuVLProcessor::new(1, 2)
        .with_pixel_bounds(4 * 4, 4 * 4)
        .with_norm([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let mut image = RgbImage::new(4, 4);
    for y in 0..4 {
        for x in 0..4 {
            let base = (y * 4 + x) as u8;
            image.put_pixel(x, y, image::Rgb([base, base + 32, base + 64]));
        }
    }

    let (rows, shapes, width) = processor
        .try_preprocess_values_with_spatial(&[DynamicImage::ImageRgb8(image)])
        .unwrap();
    assert_eq!(shapes, vec![(4, 4)]);
    assert_eq!(width, 3);

    let observed_pixels = rows
        .chunks_exact(3)
        .map(|rgb| (rgb[0] * 255.0).round() as u8)
        .collect::<Vec<_>>();
    assert_eq!(
        observed_pixels,
        vec![0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
    );
    assert_eq!(
        rows[..3]
            .iter()
            .map(|value| (value * 255.0).round() as u8)
            .collect::<Vec<_>>(),
        vec![0, 32, 64]
    );
}
