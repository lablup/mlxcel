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

#[test]
fn an_aspect_ratio_tie_follows_the_upstream_area_threshold() {
    // A square image ties every `n x n` candidate at ratio difference 0, so the
    // tile count is decided entirely by the tie-break. Upstream trades up to a
    // finer grid when the source area clears `0.5 * image_size^2 * i * j`.
    //
    // At `image_size` 512 (the LLM-jp-VL tile size), 768x768 has area
    // 2.25 * 512^2: above the 2x2 threshold (2.0) and below the 3x3 one (4.5),
    // so upstream picks 2x2 and appends a thumbnail. The checkpoints' own
    // `processing_llmjpvl.LLMjpVLProcessor` returns 5 `pixel_values` rows for
    // this image, which is what this pins.
    let proc = InternVLProcessor::new(512, 1, 12, true);
    let (_, tiles) = proc.preprocess_with_tiles(std::slice::from_ref(&solid_image(768, 768)));
    assert_eq!(
        tiles,
        vec![5],
        "768x768 at tile 512 -> 2x2 plus a thumbnail"
    );

    // 512x512 has area exactly 1.0 * 512^2, below the 2x2 threshold, so it
    // stays a single tile and takes no thumbnail. The reference agrees.
    let (_, tiles) = proc.preprocess_with_tiles(std::slice::from_ref(&solid_image(512, 512)));
    assert_eq!(tiles, vec![1], "512x512 at tile 512 -> a single tile");

    // 224x224 is far below every threshold.
    let (_, tiles) = proc.preprocess_with_tiles(std::slice::from_ref(&solid_image(224, 224)));
    assert_eq!(tiles, vec![1], "224x224 at tile 512 -> a single tile");

    // A 4:3 image at 1024x768 hits (4, 3) exactly, so no tie is involved:
    // 12 tiles plus a thumbnail, which is also what the reference reports.
    let (_, tiles) = proc.preprocess_with_tiles(std::slice::from_ref(&solid_image(1024, 768)));
    assert_eq!(
        tiles,
        vec![13],
        "1024x768 at tile 512 -> 12 tiles plus a thumbnail"
    );
}
