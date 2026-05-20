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

//! Unit tests for the InternVL dynamic-tiling image processor.

use super::*;

fn make_processor() -> InternVLProcessor {
    // internvl3-1b defaults.
    InternVLProcessor::new(448, 1, 12, true)
}

fn solid_image(w: u32, h: u32) -> image::DynamicImage {
    image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(w, h, image::Rgb([128, 64, 32])))
}

#[test]
fn square_image_produces_single_tile_no_thumbnail() {
    // A 224x224 square (the repo test image size) is closest to the 1x1
    // aspect ratio -> 1 tile. With blocks == 1 the thumbnail is NOT added.
    let proc = make_processor();
    let img = solid_image(224, 224);
    let (pixels, tiles) = proc.preprocess_with_tiles(std::slice::from_ref(&img));

    assert_eq!(tiles, vec![1], "square image should yield exactly 1 tile");
    let shape = mlxcel_core::array_shape(&pixels);
    assert_eq!(
        shape,
        vec![1, 3, 448, 448],
        "channels-first single-tile shape"
    );
}

#[test]
fn wide_image_splits_into_multiple_tiles_plus_thumbnail() {
    // A 2:1 wide image is closest to the (2, 1) ratio -> 2 tiles, and since
    // blocks > 1 a thumbnail tile is appended (total 3).
    let proc = make_processor();
    let img = solid_image(896, 448);
    let (pixels, tiles) = proc.preprocess_with_tiles(std::slice::from_ref(&img));

    assert_eq!(tiles, vec![3], "2:1 image -> 2 tiles + 1 thumbnail");
    let shape = mlxcel_core::array_shape(&pixels);
    assert_eq!(shape[0], 3, "3 tiles flattened on axis 0");
    assert_eq!(&shape[1..], &[3, 448, 448]);
}

#[test]
fn normalization_uses_imagenet_statistics() {
    // For a solid pixel the normalized value must equal (v/255 - mean)/std.
    let proc = make_processor();
    let img = solid_image(448, 448);
    let (pixels, _) = proc.preprocess_with_tiles(std::slice::from_ref(&img));
    mlxcel_core::eval(&pixels);

    // Channels-first layout: index 0 is channel-0, pixel (0,0).
    let first = mlxcel_core::slice(&pixels, &[0, 0, 0, 0], &[1, 1, 1, 1]);
    mlxcel_core::eval(&first);
    let got = mlxcel_core::item_f32(&first);
    let expected = (128.0f32 / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    assert!(
        (got - expected).abs() < 1e-4,
        "channel-0 normalization mismatch: got {got}, expected {expected}"
    );
}

#[test]
fn multiple_images_report_per_image_tile_counts() {
    let proc = make_processor();
    let square = solid_image(448, 448);
    let wide = solid_image(896, 448);
    let (_pixels, tiles) = proc.preprocess_with_tiles(&[square, wide]);
    // Square -> 1 tile, wide 2:1 -> 2 tiles + thumbnail = 3.
    assert_eq!(tiles, vec![1, 3]);
}
