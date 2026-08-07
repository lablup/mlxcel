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

//! Florence-2 preprocessing configuration and pixel layout.

use std::path::Path;

use image::{DynamicImage, Rgb, RgbImage};

use super::*;

const MODEL_DIR: &str = "models/Florence-2-base-ft-bf16";

/// The checkpoint's own `preprocessor_config.json`, verbatim.
const REAL_CONFIG: &str = r#"{
  "crop_size": {"height": 768, "width": 768},
  "do_center_crop": false,
  "do_convert_rgb": null,
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [0.485, 0.456, 0.406],
  "image_processor_type": "CLIPImageProcessor",
  "image_seq_length": 577,
  "image_std": [0.229, 0.224, 0.225],
  "resample": 3,
  "rescale_factor": 0.00392156862745098,
  "size": {"height": 768, "width": 768}
}"#;

fn from_json(json: &str) -> Result<Florence2ImageProcessor, String> {
    let parsed: RawPreprocessor =
        serde_json::from_str(json).map_err(|e| format!("parse error: {e}"))?;
    Florence2ImageProcessor::from_raw(parsed)
}

fn solid(width: u32, height: u32, rgb: [u8; 3]) -> DynamicImage {
    let mut image = RgbImage::new(width, height);
    for pixel in image.pixels_mut() {
        *pixel = Rgb(rgb);
    }
    DynamicImage::ImageRgb8(image)
}

#[test]
fn parses_the_checkpoint_configuration() {
    let processor = from_json(REAL_CONFIG).expect("parse");
    assert_eq!((processor.width, processor.height), (768, 768));
    assert!(processor.do_resize && processor.do_rescale && processor.do_normalize);
    // ImageNet statistics, not CLIP's 0.481/0.457/0.408 and not SigLIP's 0.5s.
    assert_eq!(processor.image_mean, [0.485, 0.456, 0.406]);
    assert_eq!(processor.image_std, [0.229, 0.224, 0.225]);
    assert!((processor.rescale_factor - 1.0 / 255.0).abs() < 1e-9);
}

/// The on-disk file must agree with the transcription above, so the pinned
/// constants cannot drift from the checkpoint unnoticed.
#[test]
fn the_real_file_matches_the_pinned_configuration() {
    if !Path::new(MODEL_DIR).exists() {
        eprintln!("skipping: {MODEL_DIR} not present");
        return;
    }
    let from_disk = Florence2ImageProcessor::from_pretrained(Path::new(MODEL_DIR)).expect("load");
    assert_eq!(from_disk, from_json(REAL_CONFIG).expect("parse"));
}

#[test]
fn rejects_configurations_it_would_silently_mishandle() {
    // Center cropping would change which region the location bins address.
    let cropping = REAL_CONFIG.replace("\"do_center_crop\": false", "\"do_center_crop\": true");
    assert!(from_json(&cropping).is_err());

    // A different resampling filter changes the pixels the tower sees.
    let nearest = REAL_CONFIG.replace("\"resample\": 3", "\"resample\": 0");
    assert!(from_json(&nearest).is_err());

    // A zero standard deviation would divide every channel by zero.
    let zero_std = REAL_CONFIG.replace("[0.229, 0.224, 0.225]", "[0.0, 0.224, 0.225]");
    assert!(from_json(&zero_std).is_err());

    // Wrong-length statistics, and no size at all.
    let short_mean = REAL_CONFIG.replace("[0.485, 0.456, 0.406]", "[0.485, 0.456]");
    assert!(from_json(&short_mean).is_err());
    assert!(from_json(r#"{"do_normalize": true}"#).is_err());
    let zero_size = REAL_CONFIG.replace(
        "\"size\": {\"height\": 768, \"width\": 768}",
        "\"size\": {\"height\": 0, \"width\": 0}",
    );
    assert!(from_json(&zero_size).is_err());

    // `size` is the only field here that becomes an allocation, so an
    // oversized one has to be refused at parse time rather than turned into a
    // multi-gigabyte host buffer, and a value large enough to wrap the
    // `batch * 3 * height * width` product has to be refused for the same
    // reason it would otherwise be dangerous: the write loop still runs to the
    // configured extent.
    let huge_size = REAL_CONFIG.replace(
        "\"size\": {\"height\": 768, \"width\": 768}",
        "\"size\": {\"height\": 65536, \"width\": 65536}",
    );
    assert!(from_json(&huge_size).is_err());
    let wrapping_size = REAL_CONFIG.replace(
        "\"size\": {\"height\": 768, \"width\": 768}",
        "\"size\": {\"height\": 4294967296, \"width\": 4294967296}",
    );
    assert!(from_json(&wrapping_size).is_err());

    // The shipped 768 stays inside the cap.
    assert!(from_json(REAL_CONFIG).is_ok());
}

#[test]
fn missing_file_is_an_error_rather_than_a_default() {
    let err = Florence2ImageProcessor::from_pretrained(Path::new("models/no-such-florence2"))
        .expect_err("must fail");
    assert!(err.contains("preprocessor_config.json"), "{err}");
}

#[test]
fn emits_nchw_at_the_configured_resolution() {
    let processor = from_json(REAL_CONFIG).expect("parse");
    let processed = processor.preprocess_with_sizes(&[solid(224, 224, [0, 0, 0])]);
    assert_eq!(
        mlxcel_core::array_shape(&processed.pixel_values),
        vec![1, 3, 768, 768]
    );
    assert_eq!(
        mlxcel_core::array_dtype(&processed.pixel_values),
        mlxcel_core::dtype::FLOAT32
    );
}

/// The original extent, not the resized one, is what the location bins are
/// relative to, and it is `(width, height)` in that order.
#[test]
fn reports_the_original_size_before_resizing() {
    let processor = from_json(REAL_CONFIG).expect("parse");
    let processed =
        processor.preprocess_with_sizes(&[solid(640, 480, [0, 0, 0]), solid(100, 200, [0, 0, 0])]);
    assert_eq!(processed.original_sizes, vec![(640, 480), (100, 200)]);
    assert_eq!(
        mlxcel_core::array_shape(&processed.pixel_values),
        vec![2, 3, 768, 768]
    );
}

/// A solid image resamples to itself, so the normalization can be checked in
/// closed form: `(value / 255 - mean) / std` per channel.
#[test]
fn normalizes_each_channel_with_its_own_statistics() {
    let processor = from_json(REAL_CONFIG).expect("parse");
    let processed = processor.preprocess_with_sizes(&[solid(64, 64, [255, 0, 128])]);
    let values = to_vec_f32(&processed.pixel_values);

    let plane = 768 * 768;
    let expected = [
        (1.0 - 0.485) / 0.229,
        (0.0 - 0.456) / 0.224,
        (128.0 / 255.0 - 0.406) / 0.225,
    ];
    for (channel, want) in expected.iter().enumerate() {
        let got = values[channel * plane];
        assert!(
            (got - want).abs() < 1e-4,
            "channel {channel}: got {got}, want {want}"
        );
    }
}

/// A red/green split image pins the CHW ordering: channel planes must be
/// contiguous and rows must run before columns. A transposed layout would
/// still produce plausible statistics.
#[test]
fn lays_out_channels_then_rows_then_columns() {
    let mut image = RgbImage::new(2, 2);
    image.put_pixel(0, 0, Rgb([255, 0, 0]));
    image.put_pixel(1, 0, Rgb([0, 255, 0]));
    image.put_pixel(0, 1, Rgb([0, 255, 0]));
    image.put_pixel(1, 1, Rgb([0, 255, 0]));

    let mut processor = from_json(REAL_CONFIG).expect("parse");
    processor.height = 2;
    processor.width = 2;
    processor.do_normalize = false;
    let processed = processor.preprocess_with_sizes(&[DynamicImage::ImageRgb8(image)]);
    let values = to_vec_f32(&processed.pixel_values);
    assert_eq!(values.len(), 12);

    // Red plane first: only the top-left pixel is red.
    assert!((values[0] - 1.0).abs() < 1e-5, "{values:?}");
    assert!(
        values[1] < 1e-5 && values[2] < 1e-5 && values[3] < 1e-5,
        "{values:?}"
    );
    // Green plane next: the other three pixels.
    assert!(values[4] < 1e-5, "{values:?}");
    assert!((values[5] - 1.0).abs() < 1e-5, "{values:?}");
    // Blue plane is empty throughout.
    assert!(values[8..].iter().all(|v| *v < 1e-5), "{values:?}");
}

/// A non-square image is squashed to the square target rather than cropped or
/// letterboxed, because `size` gives an explicit height and width.
#[test]
fn resizes_without_preserving_aspect_ratio() {
    let mut processor = from_json(REAL_CONFIG).expect("parse");
    processor.height = 8;
    processor.width = 8;
    processor.do_normalize = false;
    // A wide image whose left half is white: after a squash the left half of
    // the 8x8 output is white, which letterboxing would not produce.
    let mut image = RgbImage::new(40, 10);
    for y in 0..10 {
        for x in 0..20 {
            image.put_pixel(x, y, Rgb([255, 255, 255]));
        }
    }
    let processed = processor.preprocess_with_sizes(&[DynamicImage::ImageRgb8(image)]);
    let values = to_vec_f32(&processed.pixel_values);
    // Row 4 of the red plane: columns 0..3 white, columns 4..7 black.
    let row = 4 * 8;
    assert!(values[row] > 0.9, "{}", values[row]);
    assert!(values[row + 7] < 0.1, "{}", values[row + 7]);
}

#[test]
fn an_empty_batch_produces_an_empty_tensor() {
    let processor = from_json(REAL_CONFIG).expect("parse");
    let processed = processor.preprocess_with_sizes(&[]);
    assert_eq!(
        mlxcel_core::array_shape(&processed.pixel_values),
        vec![0, 3, 768, 768]
    );
    assert!(processed.original_sizes.is_empty());
}

fn to_vec_f32(array: &mlxcel_core::MlxArray) -> Vec<f32> {
    mlxcel_core::eval(array);
    mlxcel_core::array_to_raw_bytes(array)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
