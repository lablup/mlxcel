use image::{DynamicImage, Rgb, RgbImage};

use super::*;

fn processor() -> MuseGlimmerImageProcessor {
    MuseGlimmerImageProcessor::from_vision_config(&MuseGlimmerVisionConfig::default())
}

fn patch_id_image() -> DynamicImage {
    let mut image = RgbImage::new(28, 28);
    for patch_y in 0..2u32 {
        for patch_x in 0..2u32 {
            let value = (patch_y * 2 + patch_x) as u8 * 64;
            for y in patch_y * 14..(patch_y + 1) * 14 {
                for x in patch_x * 14..(patch_x + 1) * 14 {
                    image.put_pixel(x, y, Rgb([value, value, value]));
                }
            }
        }
    }
    DynamicImage::ImageRgb8(image)
}

#[test]
fn smart_resize_uses_28_pixel_grid_and_4096_visual_token_cap() {
    let p = processor();
    assert_eq!(p.factor(), 28);
    assert_eq!(p.min_pixels(), 28 * 28);
    assert_eq!(p.max_pixels(), 4096 * 28 * 28);
    assert_eq!(p.smart_resize(448, 448), (448, 448));
    assert_eq!(p.smart_resize(4000, 4000), (1792, 1792));

    let (h, w) = p.smart_resize(9000, 1000);
    assert_eq!(h % 28, 0);
    assert_eq!(w % 28, 0);
    assert!((h as usize / 28) * (w as usize / 28) <= 4096);
}

#[test]
fn grid_and_placeholder_counts_match_merge_size() {
    let p = processor();
    let grids = p.compute_grid_thw(&[DynamicImage::new_rgb8(448, 448)]);
    assert_eq!(grids, vec![(1, 32, 32)]);
    assert_eq!(p.visual_token_count(grids[0]).unwrap(), 256);
    assert_eq!(merged_visual_token_count((1, 128, 128), 2).unwrap(), 4096);
    assert!(merged_visual_token_count((1, 31, 32), 2).is_err());
}

#[test]
fn normalization_maps_black_and_white_to_minus_one_and_one() {
    let p = processor();
    let (black, black_grid) = p.preprocess_values_with_grid(&[DynamicImage::ImageRgb8(
        RgbImage::from_pixel(28, 28, Rgb([0, 0, 0])),
    )]);
    let (white, white_grid) = p.preprocess_values_with_grid(&[DynamicImage::ImageRgb8(
        RgbImage::from_pixel(28, 28, Rgb([255, 255, 255])),
    )]);
    assert_eq!(black_grid, vec![(1, 2, 2)]);
    assert_eq!(white_grid, vec![(1, 2, 2)]);
    assert!((black[0] + 1.0).abs() < 1e-6);
    assert!((white[0] - 1.0).abs() < 1e-6);
}

#[test]
fn patch_layout_is_row_major_with_temporal_duplication_inside_rows() {
    let p = processor();
    let (values, grids) = p.preprocess_values_with_grid(&[patch_id_image()]);
    assert_eq!(grids, vec![(1, 2, 2)]);
    let patch_dim = p.temporal_patch_size * 3 * p.patch_size * p.patch_size;
    assert_eq!(values.len(), 4 * patch_dim);

    let first_scalars = (0..4)
        .map(|patch| values[patch * patch_dim])
        .collect::<Vec<_>>();
    let expected = [0u8, 64, 128, 192]
        .map(|v| (v as f32 / 255.0 - 0.5) / 0.5)
        .to_vec();
    for (actual, expected) in first_scalars.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6);
    }

    let second_temporal_frame = 3 * p.patch_size * p.patch_size;
    assert_eq!(values[0], values[second_temporal_frame]);
    assert_eq!(values[patch_dim], values[patch_dim + second_temporal_frame]);
}

#[test]
fn processor_config_fixture_overrides_pinned_fields() {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("mlxcel_muse_processor_{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("processor_config.json"),
        r#"{
          "image_processor": {
            "do_convert_rgb": true,
            "do_normalize": true,
            "do_rescale": true,
            "do_resize": true,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "max_image_tokens": 4096,
            "merge_size": 2,
            "patch_size": 14,
            "resample": 1,
            "rescale_factor": 0.00392156862745098,
            "temporal_patch_size": 2
          }
        }"#,
    )
    .unwrap();
    let p = MuseGlimmerImageProcessor::from_model_dir(&dir, &MuseGlimmerVisionConfig::default())
        .unwrap();
    assert_eq!(p.patch_size, 14);
    assert_eq!(p.temporal_patch_size, 2);
    assert_eq!(p.merge_size, 2);
    assert_eq!(p.max_image_tokens, 4096);
    assert_eq!(p.mean, [0.5; 3]);
    assert_eq!(p.std, [0.5; 3]);
    std::fs::remove_dir_all(dir).unwrap();
}
