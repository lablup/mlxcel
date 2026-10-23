// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
use super::*;
use image::{GrayImage, Luma, RgbImage, Rgba, RgbaImage};
#[test]
fn masks_reject_missing_wrong_size_and_degenerate_foreground() {
    let image = DynamicImage::ImageRgb8(RgbImage::new(4, 4));
    assert!(mask_values(&image, None).is_err());
    assert!(
        mask_values(
            &image,
            Some(&DynamicImage::ImageLuma8(GrayImage::new(2, 2)))
        )
        .is_err()
    );
    let mut mask = GrayImage::new(4, 4);
    mask.put_pixel(1, 1, Luma([255]));
    mask.put_pixel(2, 1, Luma([255]));
    let error = prepare_inputs(&image, Some(&DynamicImage::ImageLuma8(mask)), None, 14)
        .err()
        .unwrap();
    assert!(error.to_string().contains("two rows and columns"));
}
#[test]
fn rgba_alpha_and_explicit_cli_masks_keep_distinct_thresholds() {
    let mut rgba = RgbaImage::from_pixel(2, 2, Rgba([0, 0, 0, 127]));
    rgba.put_pixel(0, 0, Rgba([0, 0, 0, 128]));
    let image = DynamicImage::ImageRgba8(rgba);
    assert_eq!(mask_values(&image, None).unwrap(), [1.0, 0.0, 0.0, 0.0]);
    let explicit = DynamicImage::ImageLuma8(GrayImage::from_pixel(2, 2, Luma([1])));
    assert_eq!(mask_values(&image, Some(&explicit)).unwrap(), [1.0; 4]);
}
#[test]
fn lower_median_discards_nan_but_retains_invalid_infinity() {
    assert_eq!(median(vec![4.0, 1.0, f32::NAN, 2.0, 3.0]), 2.0);
    assert!(median(vec![f32::NAN]).is_nan());
    assert_eq!(median(vec![f32::INFINITY]), f32::INFINITY);
}
#[test]
fn pointmap_dropout_retains_zero_shift_and_unit_scale() {
    let image = DynamicImage::ImageRgba8(RgbaImage::from_pixel(4, 4, Rgba([10, 20, 30, 255])));
    let (inputs, meta) = prepare_inputs(&image, None, None, 14).unwrap();
    assert!(!meta.conditioned);
    assert_eq!(meta.shift, [0.0; 3]);
    assert_eq!(meta.scale, 1.0);
    assert!(!inputs.contains_key("pointmap"));
    assert_eq!(mx::array_shape(&inputs["image"]), [1, 14, 14, 3]);
}

#[test]
fn explicit_opaque_rgba_mask_uses_alpha_not_dark_rgb() {
    let image = DynamicImage::ImageRgb8(RgbImage::new(2, 2));
    let mask = DynamicImage::ImageRgba8(RgbaImage::from_pixel(2, 2, Rgba([0, 0, 0, 255])));
    assert_eq!(mask_values(&image, Some(&mask)).unwrap(), [1.0; 4]);
}
