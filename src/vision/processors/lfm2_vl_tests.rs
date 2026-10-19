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

use super::*;

fn solid(w: u32, h: u32) -> DynamicImage {
    DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        w,
        h,
        image::Rgb([255, 255, 255]),
    ))
}

fn processor() -> Lfm2VlProcessor {
    Lfm2VlProcessor::new(
        16,
        2,
        64,
        256,
        Lfm2VlTilingPolicy::default(),
        FilterType::Triangle,
    )
}

#[test]
fn smart_resize_divisible_by_total_and_clamped() {
    let p = processor();
    // Small image scaled up so token count >= min (64).
    let (h1, w1) = p.smart_resize(20, 20);
    assert_eq!(h1 % 32, 0);
    assert_eq!(w1 % 32, 0);
    let tokens = (h1 / 32) * (w1 / 32);
    assert!(tokens >= 64, "tokens {tokens} below min");
    // Large image scaled down so token count <= max (256).
    let (h2, w2) = p.smart_resize(4000, 4000);
    assert_eq!(h2 % 32, 0);
    assert_eq!(w2 % 32, 0);
    assert!((h2 / 32) * (w2 / 32) <= 256, "over max tokens");
}

#[test]
fn target_tile_ratios_for_2_to_10() {
    let ratios = target_tile_ratios(2, 10);
    for expected in [(1, 2), (2, 1), (3, 3), (2, 5), (5, 2)] {
        assert!(ratios.contains(&expected), "missing ratio {expected:?}");
    }
    assert!(ratios.iter().all(|(w, h)| (2..=10).contains(&(w * h))));
    assert!(!ratios.contains(&(1, 1)));
}

#[test]
fn closest_ratio_prefers_area_on_ties() {
    let ratios = target_tile_ratios(2, 10);
    assert_eq!(find_closest_aspect_ratio(1024, 1024, &ratios, 512), (2, 2));
    assert_eq!(find_closest_aspect_ratio(2048, 2048, &ratios, 512), (3, 3));
}

#[test]
fn large_image_splits_into_row_major_tiles_plus_optional_thumbnail() {
    let mut policy = Lfm2VlTilingPolicy {
        use_thumbnail: true,
        ..Lfm2VlTilingPolicy::default()
    };
    let p = Lfm2VlProcessor::new(16, 2, 64, 256, policy, FilterType::Triangle);
    let (_pixels, layouts) = p.preprocess_with_grid(std::slice::from_ref(&solid(2048, 1024)));
    assert_eq!(layouts.len(), 1);
    assert_eq!((layouts[0].cols, layouts[0].rows), (4, 2));
    assert_eq!(layouts[0].views.len(), 9);
    assert_eq!(&layouts[0].views[..8], &[(32, 32); 8]);
    assert_eq!(layouts[0].views[8], {
        let (h, w) = p.smart_resize(1024, 2048);
        ((h / 16) as i32, (w / 16) as i32)
    });
    assert!(layouts[0].has_thumbnail());

    policy.use_thumbnail = false;
    let p = Lfm2VlProcessor::new(16, 2, 64, 256, policy, FilterType::Triangle);
    let (_pixels, layouts) = p.preprocess_with_grid(std::slice::from_ref(&solid(2048, 1024)));
    assert_eq!((layouts[0].cols, layouts[0].rows), (4, 2));
    assert_eq!(layouts[0].views.len(), 8);
    assert!(!layouts[0].has_thumbnail());
}

#[test]
fn split_views_follow_row_major_tile_order() {
    let p = processor();
    let mut image = image::RgbImage::new(1024, 1024);
    for y in 0..1024 {
        for x in 0..1024 {
            let pixel = match (x >= 512, y >= 512) {
                (false, false) => image::Rgb([255, 0, 0]),
                (true, false) => image::Rgb([0, 255, 0]),
                (false, true) => image::Rgb([0, 0, 255]),
                (true, true) => image::Rgb([255, 255, 0]),
            };
            image.put_pixel(x, y, pixel);
        }
    }
    let (views, rows, cols) = p.views_for_image(&DynamicImage::ImageRgb8(image));
    assert_eq!((rows, cols), (2, 2));
    assert_eq!(views.len(), 4);
    let first_pixels: Vec<[u8; 3]> = views.iter().map(|view| view.get_pixel(0, 0).0).collect();
    assert_eq!(
        first_pixels,
        vec![[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]
    );
}

#[test]
fn small_image_keeps_single_view() {
    let p = processor();
    let image = solid(640, 480);
    let mut legacy = Vec::new();
    let (h1, w1) = p.smart_resize(480, 640);
    let legacy_img = image.resize_exact(w1, h1, p.resample).to_rgb8();
    let legacy_grid = p.pack_rgb_image(&legacy_img, &mut legacy);

    let mut current = Vec::new();
    let current_grid = p.pack_image(&image, &mut current);
    assert_eq!(current_grid, legacy_grid);
    assert_eq!(current, legacy);

    let (_pixels, layouts) = p.preprocess_with_grid(std::slice::from_ref(&image));
    assert_eq!(layouts[0].views, vec![legacy_grid]);
    assert_eq!((layouts[0].rows, layouts[0].cols), (1, 1));
}

#[test]
fn packs_patch_dim_and_grid() {
    let p = processor();
    let (pixels, layouts) = p.preprocess_with_grid(std::slice::from_ref(&solid(64, 64)));
    assert_eq!(layouts.len(), 1);
    let (gh, gw) = layouts[0].views[0];
    let shape = mlxcel_core::array_shape(&pixels);
    assert_eq!(shape[0], 1);
    assert_eq!(shape[1], gh * gw);
    assert_eq!(shape[2], 16 * 16 * 3);
}

#[test]
fn white_pixel_normalizes_to_one() {
    let p = processor();
    let (pixels, _) = p.preprocess_with_grid(std::slice::from_ref(&solid(64, 64)));
    let first = mlxcel_core::slice(&pixels, &[0, 0, 0], &[1, 1, 1]);
    mlxcel_core::eval(&first);
    assert!((mlxcel_core::item_f32(&first) - 1.0).abs() < 1e-6);
}

fn json(text: &str) -> serde_json::Value {
    serde_json::from_str(text).expect("valid json")
}

#[test]
fn resample_two_selects_bilinear_and_three_selects_bicubic() {
    // LFM2.5-VL nests the setting under `image_processor`.
    let nested = json(r#"{"image_processor": {"resample": 2}}"#);
    assert_eq!(
        resample_filter(Some(&nested), None),
        Ok(FilterType::Triangle)
    );
    // LFM2-VL keeps it at the top level of preprocessor_config.json.
    let top = json(r#"{"resample": 3}"#);
    assert_eq!(
        resample_filter(None, Some(&top)),
        Ok(FilterType::CatmullRom)
    );
    let top = json(r#"{"resample": 2}"#);
    assert_eq!(resample_filter(None, Some(&top)), Ok(FilterType::Triangle));
    assert_eq!(pil_resample_filter(0), Ok(FilterType::Nearest));
    assert_eq!(pil_resample_filter(1), Ok(FilterType::Lanczos3));
}

#[test]
fn absent_resample_defaults_to_bicubic() {
    assert_eq!(resample_filter(None, None), Ok(FilterType::CatmullRom));
    let tiling_only = json(r#"{"do_image_splitting": true, "tile_size": 512}"#);
    let no_resample = json(r#"{"image_mean": [0.5, 0.5, 0.5]}"#);
    assert_eq!(
        resample_filter(Some(&tiling_only), Some(&no_resample)),
        Ok(FilterType::CatmullRom)
    );
    let null = json(r#"{"resample": null}"#);
    assert_eq!(
        resample_filter(Some(&null), None),
        Ok(FilterType::CatmullRom)
    );
}

#[test]
fn processor_config_resample_wins_over_preprocessor_config() {
    let processor_config = json(r#"{"image_processor": {"resample": 2}}"#);
    let preprocessor_config = json(r#"{"resample": 3}"#);
    assert_eq!(
        resample_filter(Some(&processor_config), Some(&preprocessor_config)),
        Ok(FilterType::Triangle)
    );
    // A processor_config without the key falls through to the other file.
    let tiling_only = json(r#"{"tile_size": 512}"#);
    assert_eq!(
        resample_filter(Some(&tiling_only), Some(&preprocessor_config)),
        Ok(FilterType::CatmullRom)
    );
}

#[test]
fn unsupported_resample_is_a_load_error() {
    for bad in [
        r#"{"resample": 7}"#,
        r#"{"resample": 4}"#,
        r#"{"resample": "bicubic"}"#,
    ] {
        let config = json(bad);
        assert!(
            resample_filter(None, Some(&config)).is_err(),
            "{bad} must be rejected"
        );
    }
    assert!(pil_resample_filter(7).is_err());
}

/// A non-uniform image, so the resize filter shows in the pixels.
fn gradient(w: u32, h: u32) -> DynamicImage {
    DynamicImage::ImageRgb8(image::RgbImage::from_fn(w, h, |x, y| {
        image::Rgb([
            ((x * 7 + y * 3) % 256) as u8,
            ((x * 13) % 251) as u8,
            ((y * 11 + x) % 241) as u8,
        ])
    }))
}

#[test]
fn configured_filter_reaches_every_resize_site() {
    let policy = Lfm2VlTilingPolicy {
        use_thumbnail: true,
        ..Lfm2VlTilingPolicy::default()
    };
    let small = gradient(613, 389);
    let large = gradient(2048, 1024);
    let mut seen = Vec::new();
    for filter in [FilterType::Triangle, FilterType::CatmullRom] {
        let p = Lfm2VlProcessor::new(16, 2, 64, 256, policy, filter);

        // Single-view smart resize.
        let (views, _, _) = p.views_for_image(&small);
        let (h1, w1) = p.smart_resize(389, 613);
        assert_eq!(views[0], small.resize_exact(w1, h1, filter).to_rgb8());

        // Tiled grid and thumbnail.
        let (views, rows, cols) = p.views_for_image(&large);
        assert_eq!((rows, cols), (2, 4));
        let grid = large.resize_exact(512 * 4, 512 * 2, filter).to_rgb8();
        let tile = image::imageops::crop_imm(&grid, 512, 0, 512, 512).to_image();
        assert_eq!(views[1], tile);
        let (th, tw) = p.smart_resize(1024, 2048);
        assert_eq!(
            views[8],
            large.resize_exact(tw, th, filter).to_rgb8(),
            "thumbnail"
        );
        seen.push((views[0].clone(), views[8].clone()));
    }
    assert_ne!(
        seen[0], seen[1],
        "the two filters must produce different pixels"
    );
}
