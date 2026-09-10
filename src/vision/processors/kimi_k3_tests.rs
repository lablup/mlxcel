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
use image::{Rgb, Rgba};

#[test]
fn navit_resize_rule() {
    let cfg = KimiK3NavitConfig::default();

    // 4000x3000: s = min(1, sqrt(65536 / (285 * 214)) = 1.036, 7168 / 4000,
    // 7168 / 3000) = 1, so the size is kept and padded to the next multiples
    // of 28.
    let plan = cfg.plan(4000, 3000).unwrap();
    assert_eq!((plan.new_w, plan.new_h), (4000, 3000));
    assert_eq!(plan.padded(), (4004, 3024));
    assert_eq!((plan.grid_h, plan.grid_w), (216, 286));
    assert_eq!(plan.num_tokens, 216 * 286 / 4);
    assert_eq!(plan.num_tokens, 15_444);

    // 10000x100 is capped by the side limit: s = 7168 / 10000.
    let plan = cfg.plan(10_000, 100).unwrap();
    assert_eq!(plan.new_w, 7168);
    assert_eq!(plan.new_h, (100.0f64 * (7168.0 / 10_000.0)).trunc() as u32);
    assert_eq!(plan.new_h, 71);
    assert_eq!(plan.padded(), (7168, 84));
    assert_eq!((plan.grid_h, plan.grid_w), (6, 512));
    assert_eq!(plan.num_tokens, 3 * 256);

    // 100x100: s = 1, padded to 112x112, grid (8, 8), 16 tokens.
    let plan = cfg.plan(100, 100).unwrap();
    assert_eq!((plan.new_w, plan.new_h), (100, 100));
    assert_eq!(plan.padded(), (112, 112));
    assert_eq!((plan.grid_h, plan.grid_w), (8, 8));
    assert_eq!(plan.num_tokens, 16);

    // The patch budget: 8000x8000 has 571 * 571 patches, so
    // s = sqrt(65536 / 326041) = 0.4483 and the image lands under the budget.
    let plan = cfg.plan(8000, 8000).unwrap();
    assert_eq!(plan.new_w, 3586);
    assert_eq!(plan.padded(), (3612, 3612));
    assert_eq!(plan.num_tokens, 129 * 129);
    // The budget bounds the pre-padding patch count; padding can push the
    // grid slightly over it (258 * 258 = 66564 here), as in the reference.
    assert_eq!((plan.grid_h, plan.grid_w), (258, 258));

    assert!(cfg.plan(0, 10).is_err());

    // Geometry out of range is refused by name before any arithmetic.
    for (field, cfg) in [
        (
            "patch_size",
            KimiK3NavitConfig {
                patch_size: 100_000,
                ..cfg
            },
        ),
        (
            "merge_kernel_size",
            KimiK3NavitConfig {
                merge_kernel_size: 0,
                ..cfg
            },
        ),
        (
            "in_patch_limit",
            KimiK3NavitConfig {
                in_patch_limit: u32::MAX,
                ..cfg
            },
        ),
        (
            "patch_limit_on_one_side",
            KimiK3NavitConfig {
                patch_limit_on_one_side: 0,
                ..cfg
            },
        ),
    ] {
        let err = cfg.plan(100, 100).unwrap_err();
        assert!(err.contains(field), "{field}: {err}");
    }

    // In-range but large geometry still cannot wrap the grid arithmetic into
    // an allocation size: the patch tensor has to fit an i32 shape.
    let big = KimiK3NavitConfig {
        patch_size: MAX_PATCH_SIZE,
        merge_kernel_size: 2,
        in_patch_limit: MAX_IN_PATCH_LIMIT,
        patch_limit_on_one_side: MAX_PATCH_LIMIT_ON_ONE_SIDE,
    };
    let err = big.plan(1_000_000, 1_000_000).unwrap_err();
    assert!(err.contains("past what one tensor can hold"), "{err}");
}

#[test]
fn chessboard_fill() {
    // A half-transparent pixel over the top-left square (white, 255) blends
    // with 255: alpha 128 / 255 = 0.50196, out = 0.50196 * 40 + 0.49804 * 255
    // = 147.08 -> 147 (truncated like `astype(np.uint8)`).
    let mut rgba = RgbaImage::new(16, 16);
    for px in rgba.pixels_mut() {
        *px = Rgba([40, 40, 40, 128]);
    }
    let cfg = ChessboardConfig::default();
    let rgb = composite_onto_background(&rgba, TransparentBackground::Chessboard(cfg));
    let alpha = 128.0f32 / 255.0;
    let expect_white = (alpha * 40.0 + (1.0 - alpha) * 255.0) as u8;
    let expect_gray = (alpha * 40.0 + (1.0 - alpha) * 180.0) as u8;
    assert_eq!(expect_white, 147);
    assert_eq!(expect_gray, 109);
    assert_eq!(rgb.get_pixel(0, 0), &Rgb([147, 147, 147]));
    assert_eq!(rgb.get_pixel(7, 7), &Rgb([147, 147, 147]));
    // The square to the right and the square below are gray (180).
    assert_eq!(rgb.get_pixel(8, 0), &Rgb([109, 109, 109]));
    assert_eq!(rgb.get_pixel(0, 8), &Rgb([109, 109, 109]));
    // Diagonal neighbour is white again.
    assert_eq!(rgb.get_pixel(8, 8), &Rgb([147, 147, 147]));

    // Fully opaque pixels are untouched, fully transparent ones become the
    // background.
    let mut rgba = RgbaImage::new(2, 1);
    rgba.put_pixel(0, 0, Rgba([10, 20, 30, 255]));
    rgba.put_pixel(1, 0, Rgba([10, 20, 30, 0]));
    let rgb = composite_onto_background(&rgba, TransparentBackground::Chessboard(cfg));
    assert_eq!(rgb.get_pixel(0, 0), &Rgb([10, 20, 30]));
    assert_eq!(rgb.get_pixel(1, 0), &Rgb([255, 255, 255]));

    // `white_on_top_left = false` flips the parity.
    let flipped = ChessboardConfig {
        white_on_top_left: false,
        ..cfg
    };
    assert_eq!(flipped.value_at(0, 0), 180);
    assert_eq!(flipped.value_at(8, 0), 255);
}

#[test]
fn published_config_parses_from_media_proc_cfg() {
    let cfg = serde_json::json!({
        "in_patch_limit": 65536,
        "patch_size": 14,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "merge_kernel_size": 2,
        "fixed_output_tokens": null,
        "patch_limit_on_one_side": 512,
        "in_patch_limit_each_frame": 16384,
        "sample_fps": 8.0,
        "temporal_merge_kernel_size": 4,
        "transparent_bg_config": {
            "pattern": "chessboard",
            "chessboard_square_size": 8,
            "chessboard_square_on_top_left": true,
            "chessboard_white_value": 255,
            "chessboard_gray_value": 180
        },
        "transparent_bg_fill_stage": "after_resize",
        "config_type": "media_proc.processors.moonvit.MoonViTMediaProcessorConfig"
    });
    let parsed = KimiK3ImageProcessor::from_media_proc_cfg(&cfg).unwrap();
    assert_eq!(parsed, KimiK3ImageProcessor::default());

    let bad = serde_json::json!({ "transparent_bg_fill_stage": "sometime" });
    assert!(KimiK3ImageProcessor::from_media_proc_cfg(&bad).is_err());
    let no_bg = serde_json::json!({ "transparent_bg_config": null });
    let parsed = KimiK3ImageProcessor::from_media_proc_cfg(&no_bg).unwrap();
    assert_eq!(parsed.background, None);
    assert_eq!(parsed.fill_stage, FillStage::BeforeResize);
}

#[test]
fn patchify_is_row_major_channel_first_and_pads_black() {
    // 30x16 RGB -> plan keeps the size (s = 1) and pads to 56x28: grid (2, 4),
    // 8 patches, 2 tokens. Left half red, right half blue.
    let mut rgb = RgbImage::new(30, 16);
    for (x, _, px) in rgb.enumerate_pixels_mut() {
        *px = if x < 15 {
            Rgb([255, 0, 0])
        } else {
            Rgb([0, 0, 255])
        };
    }
    let proc = KimiK3ImageProcessor::default();
    let prepared = proc
        .preprocess_with_grid(&[DynamicImage::ImageRgb8(rgb)])
        .unwrap();
    assert_eq!(prepared.images.len(), 1);
    let item = prepared.images[0];
    assert_eq!(item.original_size, (30, 16));
    assert_eq!(item.grid, MoonViT3DGrid::image(2, 4));
    assert_eq!(item.plan.num_tokens, 2);
    assert_eq!(
        mlxcel_core::array_shape(&prepared.pixel_values),
        vec![8, 3, 14, 14]
    );
    mlxcel_core::eval(&prepared.pixel_values);
    let values = mlxcel_core::utils::array_to_vec_f32(&prepared.pixel_values);
    let at =
        |patch: usize, c: usize, y: usize, x: usize| values[((patch * 3 + c) * 14 + y) * 14 + x];
    // Patch 0 (row 0, col 0) is red: channel 0 = (1 - 0.5) / 0.5 = 1, others -1.
    assert_eq!(at(0, 0, 0, 0), 1.0);
    assert_eq!(at(0, 1, 0, 0), -1.0);
    assert_eq!(at(0, 2, 0, 0), -1.0);
    // Patch 2 (row 0, col 2) covers x in 28..42: pixels 28 and 29 are blue,
    // the rest is black padding (-1 on every channel).
    assert_eq!(at(2, 2, 0, 0), 1.0);
    assert_eq!(at(2, 0, 0, 0), -1.0);
    assert_eq!(at(2, 2, 0, 2), -1.0);
    // Patch 4 is (row 1, col 0): y in 14..28, rows 14 and 15 red, then padding.
    assert_eq!(at(4, 0, 0, 0), 1.0);
    assert_eq!(at(4, 0, 2, 0), -1.0);

    // The channels-last layout the tower reads holds the same values with the
    // channel as the fastest axis, and the array form carries the same grid.
    let mut last = Vec::new();
    let same_item = proc
        .prepare_into(
            &DynamicImage::ImageRgb8(RgbImage::from_fn(30, 16, |x, _| {
                if x < 15 {
                    Rgb([255, 0, 0])
                } else {
                    Rgb([0, 0, 255])
                }
            })),
            PatchLayout::ChannelsLast,
            &mut last,
        )
        .unwrap();
    assert_eq!(same_item.grid, item.grid);
    assert_eq!(last.len(), values.len());
    for patch in 0..8 {
        for c in 0..3 {
            for y in 0..14 {
                for x in 0..14 {
                    let nhwc = last[((patch * 14 + y) * 14 + x) * 3 + c];
                    assert_eq!(nhwc, at(patch, c, y, x), "patch {patch} c {c} y {y} x {x}");
                }
            }
        }
    }
}

#[test]
fn media_token_budget_bounds_a_request() {
    let budget = max_media_tokens_per_request();
    assert_eq!(budget, DEFAULT_MAX_MEDIA_TOKENS_PER_REQUEST);

    // One worst-case image fits; the navit rule alone tops out just under
    // 17,000 merged tokens.
    assert!(check_media_token_budget([16_698]).is_ok());
    assert!(check_media_token_budget([budget]).is_ok());
    assert!(check_media_token_budget(std::iter::empty()).is_ok());

    // Sixteen 4000x3000 photos are 247,104 media tokens, well past it, and
    // the error names the total and the override.
    let photo = KimiK3NavitConfig::default().plan(4000, 3000).unwrap();
    assert_eq!(photo.num_tokens, 15_444);
    let err = check_media_token_budget(std::iter::repeat_n(photo.num_tokens, 16)).unwrap_err();
    assert!(err.contains("247104"), "{err}");
    assert!(err.contains(MAX_MEDIA_TOKENS_ENV), "{err}");

    // The sum is taken in u64, so a batch that would wrap u32 still refuses.
    let err = check_media_token_budget(std::iter::repeat_n(u32::MAX, 4)).unwrap_err();
    assert!(err.contains("over the per-request budget"), "{err}");
}

#[test]
fn after_resize_fill_composites_the_resized_alpha() {
    // A 60x60 RGBA image with alpha 0 everywhere becomes the chessboard after
    // the resize (s = 1 here, so no scaling) and pads to 84x84.
    let rgba = RgbaImage::from_pixel(60, 60, Rgba([0, 0, 0, 0]));
    let proc = KimiK3ImageProcessor::default();
    let mut out = Vec::new();
    let item = proc
        .prepare(&DynamicImage::ImageRgba8(rgba), &mut out)
        .unwrap();
    assert_eq!(item.grid, MoonViT3DGrid::image(6, 6));
    // Patch 0 pixel (0, 0) is a white square: (255/255 - 0.5) / 0.5 = 1.
    assert_eq!(out[0], 1.0);
    // Pixel (8, 0) of the image is in patch 0 (x < 14): gray 180.
    let expect_gray = (180.0f32 / 255.0 - 0.5) / 0.5;
    assert!((out[8] - expect_gray).abs() < 1e-6);

    // With no background the alpha is dropped: black pixels stay black.
    let plain = KimiK3ImageProcessor {
        background: None,
        ..KimiK3ImageProcessor::default()
    };
    let mut out = Vec::new();
    plain
        .prepare(
            &DynamicImage::ImageRgba8(RgbaImage::from_pixel(28, 28, Rgba([0, 0, 0, 0]))),
            &mut out,
        )
        .unwrap();
    assert_eq!(out[0], -1.0);
}

#[test]
fn image_prompt_text_names_the_original_size_and_repeats_the_pad() {
    let text = image_prompt_text(4000, 3000, 3);
    assert_eq!(
        text,
        "<|media_begin|>image 4000x3000<|media_content|><|media_pad|><|media_pad|><|media_pad|><|media_end|>"
    );
}
