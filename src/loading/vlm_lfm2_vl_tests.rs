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

use serde_json::json;

use super::metadata::*;
use crate::vision::processors::lfm2_vl::Lfm2VlTilingPolicy;

#[test]
fn tiling_validation_uses_processor_max_num_patches_not_vision_table_len() {
    let policy = Lfm2VlTilingPolicy::default();
    assert!(validate_tiling_policy(policy, 16, 2, 64, 256, 1024).is_ok());
    assert!(validate_tiling_policy(policy, 16, 2, 64, 256, 256).is_err());
}

#[test]
fn processor_sidecar_unwraps_nested_image_processor_config() {
    let full = json!({
        "encoder_patch_size": 32,
        "max_num_patches": 256,
        "use_image_special_tokens": true
    });
    let sidecar = json!({
        "image_processor": {
            "encoder_patch_size": 16,
            "max_num_patches": 1024,
            "use_image_special_tokens": false
        }
    });
    let sidecar = lfm2_processor_sidecar(&sidecar);
    assert_eq!(
        sidecar_or_config_usize(Some(sidecar), &full, "encoder_patch_size", 8),
        16
    );
    assert_eq!(
        sidecar_or_config_usize(Some(sidecar), &full, "max_num_patches", 512),
        1024
    );
    assert!(!sidecar_or_config_bool(
        Some(sidecar),
        &full,
        "use_image_special_tokens",
        true
    ));
}

#[test]
fn tiling_policy_reads_reference_processor_defaults() {
    let sidecar = json!({
        "do_image_splitting": true,
        "tile_size": 512,
        "min_tiles": 2,
        "max_tiles": 10,
        "max_pixels_tolerance": 2.0,
        "use_thumbnail": false
    });
    assert_eq!(
        parse_tiling_policy(Some(&sidecar)),
        Lfm2VlTilingPolicy::default()
    );
    assert_eq!(parse_tiling_policy(None), Lfm2VlTilingPolicy::default());
}

#[test]
fn processor_metadata_rejects_malformed_json() {
    let dir = tempfile::tempdir().expect("temporary model dir");
    let path = dir.path().join("processor_config.json");
    std::fs::write(&path, "{not-json").expect("write processor config");

    let err = read_optional_json_file(&path, "processor_config.json").unwrap_err();
    assert!(err.to_string().contains("processor_config.json"));
}

#[test]
fn tiling_validation_bounds_canvas_allocation() {
    let policy = Lfm2VlTilingPolicy {
        tile_size: 2048,
        max_tiles: 10,
        ..Lfm2VlTilingPolicy::default()
    };
    let err = validate_tiling_policy(policy, 16, 2, 64, 256, MAX_LFM2_VL_VIEW_PATCHES).unwrap_err();
    assert!(err.to_string().contains("safety limit"));
}

#[test]
fn tiling_validation_bounds_single_view_patch_budget() {
    let err = validate_tiling_policy(
        Lfm2VlTilingPolicy::default(),
        16,
        2,
        64,
        MAX_LFM2_VL_VIEW_PATCHES,
        1024,
    )
    .unwrap_err();
    assert!(err.to_string().contains("max_image_tokens"));
}

#[test]
fn token_field_rejects_out_of_range_ids() {
    let config = json!({"image_token_index": i64::MAX});
    let err = token_field(&config, "image_token_index").unwrap_err();
    assert!(err.to_string().contains("outside i32 range"));
}

#[test]
fn positive_i32_field_rejects_invalid_downsample_factor() {
    let config = json!({"downsample_factor": -2});
    let err = positive_i32_field(&config, "downsample_factor").unwrap_err();
    assert!(err.to_string().contains("positive i32"));
}

#[test]
fn marker_resolution_uses_tokenizer_ids_and_requires_tiling_table() {
    let tokenizer = json!({
        "model": {
            "vocab": {
                "<|img_row_1_col_1|>": 900,
                "<|img_row_1_col_2|>": 901,
                "<|img_row_2_col_1|>": 910,
                "<|img_row_2_col_2|>": 911,
                "<|img_thumbnail|>": 999
            }
        }
    });
    let policy = Lfm2VlTilingPolicy {
        min_tiles: 2,
        max_tiles: 2,
        use_thumbnail: true,
        ..Lfm2VlTilingPolicy::default()
    };
    let (row_col_ids, thumbnail_id) =
        resolve_lfm2_vl_marker_ids(None, Some(&tokenizer), policy).unwrap();
    assert_eq!(row_col_ids[0][0], 900);
    assert_eq!(row_col_ids[0][1], 901);
    assert_eq!(row_col_ids[1][0], 910);
    assert_eq!(row_col_ids[1][1], 911);
    assert_eq!(thumbnail_id, 999);

    let missing = json!({
        "model": {
            "vocab": {
                "<|img_row_1_col_1|>": 900
            }
        }
    });
    let err = resolve_lfm2_vl_marker_ids(None, Some(&missing), policy).unwrap_err();
    assert!(err.to_string().contains("<|img_row_1_col_2|>"));
}
