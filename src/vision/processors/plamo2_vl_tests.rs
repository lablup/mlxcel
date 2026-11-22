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

//! PLaMo 2.1 VL tiler tests. Expected grids and tile counts were computed with
//! the checkpoint's own `_dynamic_preprocess_eagle2` rules (a line-for-line
//! Python replica of `processing_plamo2_vl.py`), not derived from this port.

use super::Plamo2VlProcessor;
use crate::vision::plamo2_vl_config::Plamo2VlProcessorConfig;

fn processor() -> Plamo2VlProcessor {
    Plamo2VlProcessor::new(&Plamo2VlProcessorConfig::default())
}

fn solid(width: u32, height: u32, value: u8) -> image::DynamicImage {
    image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        width,
        height,
        image::Rgb([value, value, value]),
    ))
}

#[test]
fn num_image_token_is_729_for_the_384px_patch14_tower() {
    let p = processor();
    assert_eq!(p.image_size, 384);
    assert_eq!(p.num_image_token(), 729);
}

#[test]
fn square_image_yields_one_tile_no_thumbnail() {
    let p = processor();
    for size in [224, 384, 512] {
        let (_, counts) = p.tile_budget(&[(size, size)]);
        assert_eq!(counts, vec![1], "{size}x{size}");
    }
    // A large square picks 3x3 at the full cap, which the budget shrinks
    // back to a single tile.
    assert_eq!(p.grid_for(1024, 1024, 24), (3, 3));
    assert_eq!(p.tile_budget(&[(1024, 1024)]), (2, vec![1]));
}

#[test]
fn wide_image_adds_thumbnail() {
    let p = processor();
    // 1000x300 (ratio 3.33) settles on (3, 1) crops + thumbnail = 4 tiles at
    // cap 12. The issue text predicted (2, 1) + thumbnail; the reference rules
    // give 4, which is what this pins.
    assert_eq!(p.tile_budget(&[(1000, 300)]), (12, vec![4]));
    assert_eq!(p.grid_for(1000, 300, 12), (3, 1));
    // 16:9 photo: (2, 1) + thumbnail = 3 tiles.
    assert_eq!(p.tile_budget(&[(1920, 1080)]), (6, vec![3]));
    // Tall mirror of the wide case.
    assert_eq!(p.tile_budget(&[(300, 1000)]), (12, vec![4]));

    let (pixels, counts) = p.preprocess_with_tiles(&[solid(1000, 300, 10)]).unwrap();
    assert_eq!(counts, vec![4]);
    assert_eq!(mlxcel_core::array_shape(&pixels), vec![4, 3, 384, 384]);
}

#[test]
fn context_budget_caps_total_tiles() {
    let p = processor();
    let per_tile = p.num_image_token();
    // 4:3 and 4:1 images pick multi-tile grids at cap 24 and must shrink until
    // `2 + tiles * 729 < 3072`.
    for (w, h, expected) in [(800, 600, 1), (2000, 500, 3), (4000, 100, 3)] {
        let (_, counts) = p.tile_budget(&[(w, h)]);
        assert_eq!(counts, vec![expected], "{w}x{h}");
        assert!(2 + counts[0] * per_tile < p.tile_context_length);
        assert!(counts[0] <= 4);
    }
    // Two images share one budget.
    assert_eq!(p.tile_budget(&[(1000, 300), (512, 512)]), (2, vec![3, 1]));
    assert_eq!(p.tile_budget(&[(384, 384), (384, 384)]), (24, vec![1, 1]));
}

#[test]
fn pixel_values_shape_and_normalization() {
    let p = processor();
    let (pixels, counts) = p.preprocess_with_tiles(&[solid(512, 512, 128)]).unwrap();
    assert_eq!(counts, vec![1]);
    assert_eq!(mlxcel_core::array_shape(&pixels), vec![1, 3, 384, 384]);
    let expected = (128.0 / 255.0 - 0.5) / 0.5;
    let lo = mlxcel_core::item_f32(&mlxcel_core::min_all(&pixels));
    let hi = mlxcel_core::item_f32(&mlxcel_core::max_all(&pixels));
    assert!(
        (lo - expected).abs() < 1e-6 && (hi - expected).abs() < 1e-6,
        "{lo} {hi}"
    );
}

#[test]
fn empty_image_is_rejected() {
    let p = processor();
    assert!(p.preprocess_with_tiles(&[]).is_err());
}
