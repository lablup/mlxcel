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

//! Unit tests for the Jina VLM image processor.

use super::{JinaVlmProcessor, get_patches_from_tiling, select_tiling, smart_resize};
use image::{DynamicImage, RgbImage};

fn processor() -> JinaVlmProcessor {
    JinaVlmProcessor::default()
}

/// A gradient image so downstream shape checks are not fooled by a constant.
fn test_image(width: u32, height: u32) -> DynamicImage {
    let mut buf = RgbImage::new(width, height);
    for (x, y, pixel) in buf.enumerate_pixels_mut() {
        *pixel = image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8]);
    }
    DynamicImage::ImageRgb8(buf)
}

#[test]
fn smart_resize_snaps_both_sides_to_the_patch_grid() {
    // 480x640 is inside the pixel budget, so it only snaps to multiples of 14.
    assert_eq!(smart_resize(480, 640, 14, 3136, 1_003_520), (476, 644));
    assert_eq!(476 % 14, 0);
    assert_eq!(644 % 14, 0);
}

#[test]
fn smart_resize_uses_bankers_rounding_like_python() {
    // 21 / 14 = 1.5 rounds to 2 under round-half-up and to 2 under
    // round-half-even (2 is even); 7 / 14 = 0.5 is where they part company,
    // and Python's `round` gives 0 there.
    assert_eq!(smart_resize(7, 7, 14, 1, 1_000_000).0, 14);
    // Below `min_pixels` the ceil branch takes over, so probe the rounding on a
    // pair that stays inside the budget: 63 / 14 = 4.5 -> 4 (even), so 56.
    assert_eq!(smart_resize(63, 63, 14, 1, 1_000_000), (56, 56));
}

#[test]
fn smart_resize_shrinks_past_the_maximum_pixel_budget() {
    let (h, w) = smart_resize(4000, 4000, 14, 3136, 1_003_520);
    assert!(h * w <= 1_003_520, "got {h}x{w}");
    assert_eq!(h % 14, 0);
    assert_eq!(w % 14, 0);
}

#[test]
fn select_tiling_prefers_the_least_upscaling_that_still_covers() {
    // 476x644 minus the 112 margin pixels, against a 266-pixel crop window.
    assert_eq!(select_tiling(364, 532, 266, 12), (2, 2));
    // A wide image lands on a single row of crops.
    assert_eq!(select_tiling(154, 1064, 266, 12).0, 1);
}

#[test]
fn the_token_grid_rounds_each_crop_window_up_separately() {
    // crop_patches = 27, window = 19, margins 4/4, pooling 2.
    // One tile: ceil(27 / 2) * 2 = 28.
    assert_eq!(get_patches_from_tiling(1, 2, 27, 19, 4, 4), 28);
    // Two tiles: ceil(23 / 2) * 2 twice = 48. The `tiles * window + margins`
    // shortcut would say 46, which is one pooled column short per axis.
    assert_eq!(get_patches_from_tiling(2, 2, 27, 19, 4, 4), 48);
    // Three tiles: 24 + 20 + 24.
    assert_eq!(get_patches_from_tiling(3, 2, 27, 19, 4, 4), 68);
}

#[test]
fn a_small_image_produces_a_thumbnail_plus_one_crop() {
    let out = processor().preprocess_image(&test_image(100, 100));

    let [crops, patches, patch_dim] = out.pixel_values_shape;
    assert_eq!(patches, 27 * 27);
    assert_eq!(patch_dim, 14 * 14 * 3);
    assert_eq!(
        out.pixel_values.len(),
        (crops * patches * patch_dim) as usize
    );
    assert_eq!(out.image_masks_shape, [crops, patches]);
    assert_eq!(out.image_masks.len(), (crops * patches) as usize);
}

#[test]
fn every_pooled_patch_maps_to_a_distinct_image_patch_token() {
    let processor = processor();
    let out = processor.preprocess_image(&test_image(640, 480));

    let patch_id = processor.tokens.image_patch_id;
    let patch_positions: Vec<usize> = out
        .image_token_ids
        .iter()
        .enumerate()
        .filter_map(|(i, &t)| (t == patch_id).then_some(i))
        .collect();

    let valid: Vec<i32> = out
        .image_input_idx
        .iter()
        .copied()
        .filter(|&v| v >= 0)
        .collect();

    // One target per `<im_patch>` token, and the mapping is a bijection.
    assert_eq!(valid.len(), patch_positions.len());
    let mut sorted = valid.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), valid.len(), "two patches share a target slot");
    for v in &valid {
        assert!(
            out.image_token_ids[*v as usize] == patch_id,
            "target {v} is not an <im_patch> token"
        );
    }

    // The pooled feature count the connector will emit.
    let [crops, _, _] = out.pixel_values_shape;
    assert_eq!(
        out.image_input_idx.len(),
        crops as usize * processor.tokens_per_image()
    );
}

#[test]
fn the_token_block_is_framed_and_column_delimited() {
    let processor = processor();
    let out = processor.preprocess_image(&test_image(300, 300));
    let t = processor.tokens;

    assert_eq!(out.image_token_ids.first(), Some(&t.image_start_id));
    assert_eq!(out.image_token_ids.last(), Some(&t.image_end_id));

    // Two blocks (thumbnail, then crops), each framed.
    let starts = out
        .image_token_ids
        .iter()
        .filter(|&&x| x == t.image_start_id)
        .count();
    let ends = out
        .image_token_ids
        .iter()
        .filter(|&&x| x == t.image_end_id)
        .count();
    assert_eq!((starts, ends), (2, 2));

    // The thumbnail block is exactly `<im_start> [14 patch + col] * 14 <im_end>`.
    let thumb_len = 2 + processor.token_length_h * (processor.token_length_w + 1);
    assert_eq!(out.image_token_ids[thumb_len - 1], t.image_end_id);
    let cols = out.image_token_ids[..thumb_len]
        .iter()
        .filter(|&&x| x == t.image_col_id)
        .count();
    assert_eq!(cols, processor.token_length_h);
}

#[test]
fn the_coverage_mask_ends_with_the_upstream_sentinel_row() {
    // Upstream appends a `-1` row after the crop masks (the thumbnail row is
    // never prepended), which is what makes the connector treat the final crop
    // as partially padded. Reproduced deliberately, so pin it.
    let out = processor().preprocess_image(&test_image(640, 480));
    let [crops, patches] = out.image_masks_shape;
    let last_row = &out.image_masks[((crops - 1) * patches) as usize..];
    assert!(last_row.iter().all(|&v| v == -1.0));
    let earlier = &out.image_masks[..((crops - 1) * patches) as usize];
    assert!(earlier.iter().all(|&v| v == 1.0));
}

#[test]
fn pixels_are_normalized_into_the_configured_min_max_range() {
    let out = processor().preprocess_image(&test_image(200, 150));
    let min = out.pixel_values.iter().copied().fold(f32::MAX, f32::min);
    let max = out.pixel_values.iter().copied().fold(f32::MIN, f32::max);
    assert!(min >= -1.0001, "min {min}");
    assert!(max <= 1.0001, "max {max}");
    // A gradient image must actually span most of the range; a broken
    // normalization that left values in [0, 1] would fail here.
    assert!(
        min < -0.5,
        "min {min} suggests the [-1, 1] mapping was skipped"
    );
}

#[test]
fn a_wide_image_gets_more_columns_of_crops_than_rows() {
    let out = processor().preprocess_image(&test_image(1200, 300));
    let [crops, _, _] = out.pixel_values_shape;
    // Thumbnail plus at least a 1xN tiling.
    assert!(crops >= 2, "expected multiple crops, got {crops}");

    let processor = processor();
    let patch_id = processor.tokens.image_patch_id;
    let per_row: Vec<usize> = out
        .image_token_ids
        .split(|&t| t == processor.tokens.image_col_id)
        .map(|row| row.iter().filter(|&&t| t == patch_id).count())
        .filter(|&n| n > 0)
        .collect();
    // The crop rows must be wider than the 14-wide thumbnail rows.
    assert!(
        per_row.iter().any(|&n| n > processor.token_length_w),
        "crop rows were not wider than the thumbnail: {per_row:?}"
    );
}
