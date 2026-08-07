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

//! Unit tests for the LocateAnything native-resolution image processor.

use super::*;

fn solid(w: u32, h: u32, rgb: [u8; 3]) -> image::DynamicImage {
    image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(w, h, image::Rgb(rgb)))
}

#[test]
fn rounds_each_side_up_to_a_merge_patch_multiple() {
    // patch 14, merge 2x2 -> pad unit 28. A 60x60 image grows to 84x84
    // (ceil(60/28) = 3), giving a (6, 6) patch grid. Kimi-VL would have
    // cropped down to 56x56 / (4, 4); LocateAnything rounds up instead.
    let proc = LocateAnythingProcessor::new(14, [2, 2], 25_600);
    let (pixels, grids) = proc.preprocess_with_grid(&[solid(60, 60, [128, 128, 128])]);
    assert_eq!(grids, vec![(6, 6)]);
    assert_eq!(mlxcel_core::array_shape(&pixels), vec![36, 3, 14, 14]);
}

#[test]
fn exact_multiple_is_left_untouched() {
    // 56x56 is already a multiple of 28 on both axes -> (4, 4) grid.
    let proc = LocateAnythingProcessor::new(14, [2, 2], 25_600);
    let (_pixels, grids) = proc.preprocess_with_grid(&[solid(56, 56, [10, 20, 30])]);
    assert_eq!(grids, vec![(4, 4)]);
}

#[test]
fn non_square_images_keep_independent_axes() {
    // 100x40 -> ceil(100/28)*28 = 112 wide, ceil(40/28)*28 = 56 tall
    // -> (grid_h, grid_w) = (4, 8).
    let proc = LocateAnythingProcessor::new(14, [2, 2], 25_600);
    let (_pixels, grids) = proc.preprocess_with_grid(&[solid(100, 40, [0, 0, 0])]);
    assert_eq!(grids, vec![(4, 8)]);
}

#[test]
fn downscales_when_over_the_token_budget() {
    // A 560x560 image is 40x40 = 1600 patches; with a 16-patch budget it must
    // shrink close to the budget rather than stay at 1600.
    let proc = LocateAnythingProcessor::new(14, [2, 2], 16);
    let (_pixels, grids) = proc.preprocess_with_grid(&[solid(560, 560, [128, 128, 128])]);
    let (gh, gw) = grids[0];
    assert!(
        (gh * gw) as usize <= 4 * 16,
        "patch count {} should be reduced toward the 16-patch budget",
        gh * gw
    );
}

#[test]
fn normalization_is_the_plain_half_rescale() {
    // mean = std = 0.5 -> value maps to 2*(v/255) - 1.
    let proc = LocateAnythingProcessor::new(14, [2, 2], 25_600);
    let (pixels, _) = proc.preprocess_with_grid(&[solid(28, 28, [128, 128, 128])]);
    mlxcel_core::eval(&pixels);
    let first = mlxcel_core::slice(&pixels, &[0, 0, 0, 0], &[1, 1, 1, 1]);
    let expected = (128.0f32 / 255.0 - 0.5) / 0.5;
    assert!(
        (mlxcel_core::item_f32(&first) - expected).abs() < 1e-4,
        "LocateAnything normalization mismatch"
    );
}

#[test]
fn merged_token_count_divides_by_the_merge_kernel() {
    let proc = LocateAnythingProcessor::new(14, [2, 2], 25_600);
    assert_eq!(proc.merged_token_count((6, 6)), 9);
    assert_eq!(proc.merged_token_count((4, 8)), 8);
}

#[test]
fn multiple_images_concatenate_in_order() {
    let proc = LocateAnythingProcessor::new(14, [2, 2], 25_600);
    let (pixels, grids) =
        proc.preprocess_with_grid(&[solid(56, 56, [0, 0, 0]), solid(28, 28, [255, 255, 255])]);
    assert_eq!(grids, vec![(4, 4), (2, 2)]);
    // 16 + 4 = 20 patches total.
    assert_eq!(mlxcel_core::array_shape(&pixels), vec![20, 3, 14, 14]);

    mlxcel_core::eval(&pixels);
    let first_of_image_0 = mlxcel_core::slice(&pixels, &[0, 0, 0, 0], &[1, 1, 1, 1]);
    let first_of_image_1 = mlxcel_core::slice(&pixels, &[16, 0, 0, 0], &[17, 1, 1, 1]);
    assert!(
        mlxcel_core::item_f32(&first_of_image_0) < 0.0,
        "black image"
    );
    assert!(
        mlxcel_core::item_f32(&first_of_image_1) > 0.0,
        "white image"
    );
}

#[test]
fn grid_side_is_clamped_below_the_position_embedding_limit() {
    // An extreme aspect ratio whose long side would exceed 511 patches even
    // though the total patch count stays inside the budget.
    let proc = LocateAnythingProcessor::new(14, [2, 2], 25_600);
    let (_pixels, grids) = proc.preprocess_with_grid(&[solid(14 * 600, 28, [128, 128, 128])]);
    let (gh, gw) = grids[0];
    assert!(gw < 512, "grid width {gw} must stay under the 512 limit");
    assert!(gh < 512, "grid height {gh} must stay under the 512 limit");
}
