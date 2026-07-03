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

fn granite_pinpoints() -> Vec<(i32, i32)> {
    let mut pins = Vec::new();
    // [384, 384] .. [384, 3840]
    for w in (384..=3840).step_by(384) {
        pins.push((384, w));
    }
    // [768, 384] .. [768, 1920]
    for w in (384..=1920).step_by(384) {
        pins.push((768, w));
    }
    // [1152, 384] .. [1152, 1152]
    for w in (384..=1152).step_by(384) {
        pins.push((1152, w));
    }
    // Remaining tall candidates.
    for h in [1536, 1920] {
        for w in [384, 768] {
            pins.push((h, w));
        }
    }
    for h in [2304, 2688, 3072, 3456, 3840] {
        pins.push((h, 384));
    }
    pins
}

#[test]
fn select_best_resolution_matches_reference() {
    let pins = granite_pinpoints();
    // Wide image resolves to a 2x3 grid at 768x1152.
    assert_eq!(select_best_resolution(500, 1000, &pins), (768, 1152));
    // Square-ish resolves to the square 1152x1152 grid.
    assert_eq!(select_best_resolution(1152, 1152, &pins), (1152, 1152));
    // Exact single-tile square.
    assert_eq!(select_best_resolution(384, 384, &pins), (384, 384));
}

#[test]
fn tile_info_grid_and_count() {
    let p = AnyResProcessor::new(granite_pinpoints(), 384);
    let info = p.tile_info(500, 1000);
    assert_eq!((info.n_tiles_h, info.n_tiles_w), (2, 3));
    assert_eq!(info.num_tiles, 1 + 2 * 3);
}

#[test]
fn num_image_tokens_matches_reference() {
    let p = AnyResProcessor::new(granite_pinpoints(), 384);
    // Golden counts computed from the reference packing math (side 27, base 729).
    let cases = [
        (500i32, 1000i32, 4009i32),
        (384, 384, 1485),
        (1000, 500, 4050),
        (1152, 1152, 7371),
        (2000, 300, 4941),
        (300, 2000, 4804),
        (700, 700, 3699),
    ];
    for (oh, ow, want) in cases {
        let info = p.tile_info(oh, ow);
        assert_eq!(num_image_tokens(&info, 27, 729), want, "size ({oh},{ow})");
    }
}

#[test]
fn token_count_equals_base_plus_unpad_area() {
    // The count helper and the unpad helper must agree for any size: this is the
    // invariant that keeps feature packing and prompt expansion in lockstep.
    let p = AnyResProcessor::new(granite_pinpoints(), 384);
    for oh in [200, 384, 500, 900, 1500, 2600] {
        for ow in [200, 384, 640, 1100, 2100, 3600] {
            let info = p.tile_info(oh, ow);
            let (h, w) =
                unpadded_token_hw(info.orig_h, info.orig_w, info.n_tiles_h, info.n_tiles_w, 27);
            assert_eq!(num_image_tokens(&info, 27, 729), 729 + h * (w + 1));
        }
    }
}

#[test]
fn preprocess_emits_base_tile_first_and_channels_last() {
    let p = AnyResProcessor::new(granite_pinpoints(), 384);
    let img = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        1000,
        500,
        image::Rgb([200, 100, 50]),
    ));
    let (pixels, infos) = p.preprocess_with_tiles(std::slice::from_ref(&img));
    assert_eq!(infos.len(), 1);
    assert_eq!(infos[0].num_tiles, 7); // 1 base + 2x3 grid
    let shape = mlxcel_core::array_shape(&pixels);
    assert_eq!(shape, vec![7, 384, 384, 3]);
    // A solid-color image normalizes deterministically; the base tile's first
    // pixel channel-0 is (200/255 - 0.5) / 0.5.
    let cell = mlxcel_core::slice(&pixels, &[0, 0, 0, 0], &[1, 1, 1, 1]);
    mlxcel_core::eval(&cell);
    let expected = (200.0f32 / 255.0 - 0.5) / 0.5;
    assert!((mlxcel_core::item_f32(&cell) - expected).abs() < 1e-4);
}
