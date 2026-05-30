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

//! Unit tests for the Nemotron H Nano Omni image processor.

use super::*;
use image::{DynamicImage, RgbImage};

fn make_solid_image(w: u32, h: u32, r: u8, g: u8, b: u8) -> DynamicImage {
    let mut img = RgbImage::new(w, h);
    for px in img.pixels_mut() {
        *px = image::Rgb([r, g, b]);
    }
    DynamicImage::ImageRgb8(img)
}

#[test]
fn compute_target_patches_respects_min_floor() {
    let proc = NemotronHNanoOmniImageProcessor::with_defaults();
    // Tiny input but a generous budget should still produce at least
    // `min_num_patches` (1024 by default) when the budget allows.
    let (w_patches, h_patches) = proc.compute_target_patches(64, 64, 4096);
    assert!(
        w_patches * h_patches >= proc.config.min_num_patches,
        "expected at least min_num_patches = {}, got {} = {} * {}",
        proc.config.min_num_patches,
        w_patches * h_patches,
        w_patches,
        h_patches,
    );
}

#[test]
fn compute_target_patches_divisible_by_downsample() {
    let proc = NemotronHNanoOmniImageProcessor::with_defaults();
    // downsample_factor defaults to 2 — both dimensions must be even.
    let (w_patches, h_patches) = proc.compute_target_patches(640, 480, 4096);
    assert_eq!(w_patches % 2, 0, "w_patches {w_patches} not divisible by 2");
    assert_eq!(h_patches % 2, 0, "h_patches {h_patches} not divisible by 2");
}

#[test]
fn compute_target_patches_caps_at_budget() {
    let proc = NemotronHNanoOmniImageProcessor::with_defaults();
    // Tight budget should hold (w * h) <= budget.
    let budget = 1024;
    let (w, h) = proc.compute_target_patches(2048, 2048, budget);
    assert!(
        w * h <= budget,
        "patch count {w}*{h}={} exceeds budget {budget}",
        w * h
    );
}

#[test]
fn preprocess_returns_one_entry_per_image() {
    let proc = NemotronHNanoOmniImageProcessor::with_defaults();
    let images = vec![
        make_solid_image(224, 224, 255, 0, 0),
        make_solid_image(112, 112, 0, 255, 0),
    ];
    let processed = proc.preprocess_batch(&images);
    assert_eq!(processed.len(), 2);
    for entry in &processed {
        let shape = mlxcel_core::array_shape(entry.pixel_values.as_ref().unwrap());
        assert_eq!(shape.len(), 4);
        assert_eq!(shape[0], 1, "batch dim must be 1, got {shape:?}");
        assert_eq!(shape[1], 3, "must be channel-first, got {shape:?}");
        assert!(shape[2] > 0 && shape[3] > 0);
        assert!(entry.num_tokens > 0);
        assert_eq!(
            entry.patch_grid.0 * proc.config.patch_size,
            shape[2] as usize,
            "patch_grid.h * patch_size must match H"
        );
        assert_eq!(
            entry.patch_grid.1 * proc.config.patch_size,
            shape[3] as usize,
            "patch_grid.w * patch_size must match W"
        );
    }
}

#[test]
fn preprocess_normalizes_pixels() {
    let proc = NemotronHNanoOmniImageProcessor::with_defaults();
    // Pure-red 224x224 patch — channel 0 should become
    // (1.0 - mean[0]) / std[0]; channels 1/2 should be -mean[c] / std[c].
    let image = make_solid_image(224, 224, 255, 0, 0);
    let processed = proc.preprocess_batch(&[image]);
    let entry = processed.into_iter().next().unwrap();

    let pixel_values = entry.pixel_values.as_ref().unwrap();
    // Read first element of each channel.
    mlxcel_core::eval(pixel_values);
    let shape = mlxcel_core::array_shape(pixel_values);
    let h = shape[2];
    let w = shape[3];

    let cfg = NemotronHNanoOmniProcessorConfig::default();
    let expected_r = (1.0 - cfg.norm_mean[0]) / cfg.norm_std[0];
    let expected_g = (0.0 - cfg.norm_mean[1]) / cfg.norm_std[1];
    let expected_b = (0.0 - cfg.norm_mean[2]) / cfg.norm_std[2];

    let r_pixel = mlxcel_core::slice(pixel_values, &[0, 0, 0, 0], &[1, 1, 1, 1]);
    let g_pixel = mlxcel_core::slice(pixel_values, &[0, 1, 0, 0], &[1, 2, 1, 1]);
    let b_pixel = mlxcel_core::slice(pixel_values, &[0, 2, 0, 0], &[1, 3, 1, 1]);
    let r_pixel = mlxcel_core::reshape(&r_pixel, &[1]);
    let g_pixel = mlxcel_core::reshape(&g_pixel, &[1]);
    let b_pixel = mlxcel_core::reshape(&b_pixel, &[1]);

    let r = read_first_f32(&r_pixel);
    let g = read_first_f32(&g_pixel);
    let b = read_first_f32(&b_pixel);

    let _ = (h, w);
    assert!(
        (r - expected_r).abs() < 1e-3,
        "R: got {r}, expected {expected_r}"
    );
    assert!(
        (g - expected_g).abs() < 1e-3,
        "G: got {g}, expected {expected_g}"
    );
    assert!(
        (b - expected_b).abs() < 1e-3,
        "B: got {b}, expected {expected_b}"
    );
}

fn read_first_f32(arr: &mlxcel_core::UniquePtr<mlxcel_core::MlxArray>) -> f32 {
    let raw = arr.as_ref().unwrap();
    mlxcel_core::eval(raw);
    let casted = mlxcel_core::astype(raw, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&casted);
    mlxcel_core::item_f32(casted.as_ref().unwrap())
}
