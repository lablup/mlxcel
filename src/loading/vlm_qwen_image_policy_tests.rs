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

//! Gates for where the Qwen-VL loaders read their resize policy from.
//!
//! Picking the wrong sidecar or key changes the image token count without
//! failing, so the precedence and the resulting grid are pinned here.

use image::imageops::FilterType;

use super::QwenImageResizePolicy;
use crate::vision::encoders::qwen3_vl::Qwen3VLVisionConfig;
use crate::vision::processors::qwen2_vl::{DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS};

fn model_dir(files: &[(&str, &str)]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("temp dir");
    for (file, body) in files {
        std::fs::write(dir.path().join(file), body).expect("write sidecar");
    }
    dir
}

fn read(files: &[(&str, &str)]) -> Result<QwenImageResizePolicy, String> {
    let dir = model_dir(files);
    QwenImageResizePolicy::read(dir.path())
}

/// `preprocessor_config.json` exactly as `Qwen/Qwen3-VL-2B-Instruct` ships it
/// (no `processor_config.json`): explicit keys are null, `size` holds the
/// bounds.
const QWEN3_VL_PREPROCESSOR: &str = r#"{
    "max_pixels": null, "min_pixels": null, "merge_size": 2, "patch_size": 16,
    "temporal_patch_size": 2, "resample": 3,
    "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
    "size": {"longest_edge": 16777216, "shortest_edge": 65536}
}"#;

/// Acceptance gate for #3290: a 224x224 image through the real eager loader
/// helper resizes to 256x256 (grid [1, 16, 16]) and expands into 64 image
/// tokens, as HF's Qwen3-VL processor and the XLA host preprocessor do. The
/// family default bounds kept it at 224x224 (49 tokens).
#[test]
fn qwen3_vl_eager_processor_matches_the_checkpoint_token_count() {
    let dir = model_dir(&[("preprocessor_config.json", QWEN3_VL_PREPROCESSOR)]);
    let vision_config: Qwen3VLVisionConfig = serde_json::from_value(serde_json::json!({
        "hidden_size": 1024,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2
    }))
    .expect("vision config");
    let processor = super::super::qwen_vl_processor_with_norm(
        dir.path(),
        &vision_config,
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    )
    .expect("processor");
    assert_eq!(
        (processor.min_pixels, processor.max_pixels),
        (65_536, 16_777_216)
    );
    assert_eq!(processor.resize_filter(), FilterType::CatmullRom);

    let image = image::DynamicImage::new_rgb8(224, 224);
    let grid = processor.compute_grid_thw(std::slice::from_ref(&image));
    assert_eq!(grid, vec![(1, 16, 16)]);
    let merge = vision_config.spatial_merge_size as i32;
    let (_, h, w) = grid[0];
    assert_eq!((h / merge) * (w / merge), 64);
}

#[test]
fn qwen3_vl_preprocessor_size_block_is_read() {
    let policy = read(&[("preprocessor_config.json", QWEN3_VL_PREPROCESSOR)]).unwrap();
    assert_eq!(policy.pixel_bounds(), (65_536, 16_777_216));
    assert_eq!(
        policy.resize_filter().unwrap(),
        Some(FilterType::CatmullRom)
    );
}

/// Both sidecars as `mlx-community/North-Micro-Vision-Instruct-4bit` ships
/// them. HF's `AutoProcessor` builds the image processor from the nested
/// `processor_config.json` block, so 16384 / 3868706 is the reference and the
/// standalone 65536 / 16777216 is stale (taking it gives the 224x224 fixture
/// 64 tokens where the reference has 49). The nested block has no `resample`,
/// so the standalone bicubic code fills it in.
#[test]
fn the_nested_processor_config_block_wins_over_preprocessor_config() {
    let policy = read(&[
        (
            "processor_config.json",
            r#"{"image_processor": {"min_pixels": 16384, "max_pixels": 3868706,
                "size": {"shortest_edge": 16384, "longest_edge": 3868706}}}"#,
        ),
        (
            "preprocessor_config.json",
            r#"{"size": {"shortest_edge": 65536, "longest_edge": 16777216}, "resample": 3}"#,
        ),
    ])
    .unwrap();
    assert_eq!(policy.pixel_bounds(), (16_384, 3_868_706));
    assert_eq!(policy.resample, Some(3));
}

/// Qwen3.8's nested block states explicit keys and no `size`; the explicit
/// keys are enough, and a missing `resample` means HF's bicubic default.
#[test]
fn nested_explicit_keys_without_size_or_resample() {
    let policy = read(&[(
        "processor_config.json",
        r#"{"image_processor": {"min_pixels": 65536, "max_pixels": 16777216}}"#,
    )])
    .unwrap();
    assert_eq!(policy.pixel_bounds(), (65_536, 16_777_216));
    assert_eq!(
        policy.resize_filter().unwrap(),
        Some(FilterType::CatmullRom)
    );
}

/// Explicit `min_pixels` / `max_pixels` beat the `size` block in the same
/// file, which is the order the HF processor applies them in.
#[test]
fn explicit_pixel_keys_win_over_the_size_block() {
    let policy = read(&[(
        "preprocessor_config.json",
        r#"{"min_pixels": 1024, "max_pixels": 4096,
            "size": {"shortest_edge": 65536, "longest_edge": 16777216}}"#,
    )])
    .unwrap();
    assert_eq!(policy.pixel_bounds(), (1024, 4096));
}

/// A Qwen2.5-VL style sidecar with the family default bounds and no
/// `resample` keeps the default bounds and gets HF's bicubic resampler.
#[test]
fn qwen2_style_sidecar_keeps_the_family_default_bounds() {
    let policy = read(&[(
        "preprocessor_config.json",
        r#"{"min_pixels": 3136, "max_pixels": 12845056, "patch_size": 14}"#,
    )])
    .unwrap();
    assert_eq!(
        policy.pixel_bounds(),
        (DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS)
    );
}

/// No sidecar at all keeps the processor's historical defaults, filter
/// included, so synthetic checkpoints do not change behaviour.
#[test]
fn no_sidecar_keeps_the_historical_defaults() {
    let policy = read(&[]).unwrap();
    assert!(!policy.has_sidecar);
    assert_eq!(
        policy.pixel_bounds(),
        (DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS)
    );
    assert_eq!(policy.resize_filter().unwrap(), None);
}

#[test]
fn pil_resample_codes_map_to_image_filters() {
    for (code, filter) in [
        (0, FilterType::Nearest),
        (1, FilterType::Lanczos3),
        (2, FilterType::Triangle),
        (3, FilterType::CatmullRom),
    ] {
        let body = format!(r#"{{"resample": {code}}}"#);
        let policy = read(&[("preprocessor_config.json", &body)]).unwrap();
        assert_eq!(
            policy.resize_filter().unwrap(),
            Some(filter),
            "resample={code}"
        );
    }
    let err = read(&[("preprocessor_config.json", r#"{"resample": 5}"#)]).unwrap_err();
    assert!(err.contains("resample=5"), "{err}");
}

/// Malformed or inverted values fail the load instead of silently changing
/// the grid.
#[test]
fn malformed_or_inverted_bounds_fail_closed() {
    let inverted = read(&[(
        "preprocessor_config.json",
        r#"{"size": {"shortest_edge": 16777216, "longest_edge": 65536}}"#,
    )])
    .unwrap_err();
    assert!(inverted.contains("inverted"), "{inverted}");

    let string_bound = read(&[(
        "preprocessor_config.json",
        r#"{"size": {"shortest_edge": "65536", "longest_edge": 16777216}, "min_pixels": 65536}"#,
    )])
    .unwrap_err();
    assert!(string_bound.contains("positive integer"), "{string_bound}");

    let zero = read(&[("preprocessor_config.json", r#"{"min_pixels": 0}"#)]).unwrap_err();
    assert!(zero.contains("positive integer"), "{zero}");

    let size_not_object = read(&[("preprocessor_config.json", r#"{"size": 65536}"#)]).unwrap_err();
    assert!(
        size_not_object.contains("must be an object"),
        "{size_not_object}"
    );
}
