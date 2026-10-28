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

use serde_json::{Map, json};

use super::*;

fn extra(value: serde_json::Value) -> Map<String, serde_json::Value> {
    value.as_object().cloned().unwrap_or_default()
}

#[test]
fn scheduler_defaults_follow_the_checkpoint_kind() {
    let turbo = SchedulerConfig::from_value(&json!({
        "use_uniform_sigmas": true, "shift": 3.0, "stochastic_sampling": true,
        "use_dynamic_shifting": false, "invert_sigmas": false
    }))
    .unwrap();
    assert_eq!((turbo.shift, turbo.stochastic_sampling), (3.0, true));
    assert_eq!(
        turbo.defaults(),
        ImageSamplingDefaults {
            steps: 4,
            guidance: 1.0
        }
    );
    assert_eq!(turbo.variant(), "llada-image-turbo");

    let base =
        SchedulerConfig::from_value(&json!({"shift": 1.0, "stochastic_sampling": false})).unwrap();
    assert!(!base.use_uniform_sigmas);
    assert_eq!(
        base.defaults(),
        ImageSamplingDefaults {
            steps: 50,
            guidance: 5.0
        }
    );
    assert_eq!(base.variant(), "llada-image");

    // Omitted keys: shift 3 / stochastic for uniform, shift 1 / Euler otherwise.
    let bare = SchedulerConfig::from_value(&json!({"use_uniform_sigmas": true})).unwrap();
    assert_eq!((bare.shift, bare.stochastic_sampling), (3.0, true));
    let bare = SchedulerConfig::from_value(&json!({})).unwrap();
    assert_eq!((bare.shift, bare.stochastic_sampling), (1.0, false));
}

#[test]
fn scheduler_rejects_dynamic_shifting_and_inverted_sigmas() {
    for key in ["use_dynamic_shifting", "invert_sigmas"] {
        let err = SchedulerConfig::from_value(&json!({ key: true })).unwrap_err();
        assert!(format!("{err}").contains("static, non-inverted"), "{err}");
    }
}

#[test]
fn transformer_config_checks_patch_size_and_heads() {
    let ok = TransformerConfig::from_json(r#"{"all_patch_size": [1], "all_f_patch_size": [1]}"#)
        .unwrap();
    assert_eq!(ok.head_dim(), 128);
    let err = TransformerConfig::from_json(r#"{"all_patch_size": [2]}"#).unwrap_err();
    assert!(format!("{err}").contains("patch size 1"));
    let err = TransformerConfig::from_json(r#"{"axes_dims": [32, 48, 32]}"#).unwrap_err();
    assert!(format!("{err}").contains("axes_dims"));
}

#[test]
fn options_parse_modes_per_task() {
    let generate = LladaOptions::from_extra(&Map::new(), false).unwrap();
    assert_eq!(generate.mode, GenerationMode::Text);
    let edit = LladaOptions::from_extra(&Map::new(), true).unwrap();
    assert_eq!(edit.mode, GenerationMode::Editing);
    let vq = LladaOptions::from_extra(
        &extra(json!({"generation_mode": "vq", "stochastic_sampling": false, "negative_prompt": "blur"})),
        false,
    )
    .unwrap();
    assert_eq!(vq.mode, GenerationMode::Vq);
    assert_eq!(vq.stochastic_sampling, Some(false));
    assert_eq!(vq.negative_prompt.as_deref(), Some("blur"));

    assert!(
        LladaOptions::from_extra(&extra(json!({"generation_mode": "editing"})), false).is_err()
    );
    assert!(LladaOptions::from_extra(&extra(json!({"generation_mode": "vq"})), true).is_err());
    assert!(LladaOptions::from_extra(&extra(json!({"generation_mode": 3})), false).is_err());
    assert!(
        LladaOptions::from_extra(&extra(json!({"stochastic_sampling": "yes"})), false).is_err()
    );
}

#[test]
fn dimensions_follow_the_mode() {
    assert!(validate_dimensions(1024, 768, GenerationMode::Text).is_ok());
    assert!(validate_dimensions(528, 528, GenerationMode::Vq).is_ok());
    assert!(validate_dimensions(1000, 1024, GenerationMode::Text).is_err());
    assert!(validate_dimensions(0, 1024, GenerationMode::Text).is_err());
    // 528 is a multiple of 16 but not of 32.
    assert!(validate_dimensions(528, 512, GenerationMode::Editing).is_err());
    assert!(validate_dimensions(512, 768, GenerationMode::Editing).is_ok());
}

#[test]
fn edit_size_matches_upstream_edit_array() {
    // No size: scale toward 1024x1024 of area, never up, floor to 32.
    assert_eq!(default_edit_size(512, 512), (512, 512));
    assert_eq!(default_edit_size(1000, 750), (992, 736));
    assert_eq!(default_edit_size(4000, 3000), (1152, 864));
    assert_eq!(default_edit_size(20, 20), (32, 32));
    // One side: keep the aspect ratio, round the other to 32 (ties to even).
    assert_eq!(edit_size(1000, 500, Some(512), None), (512, 256));
    assert_eq!(edit_size(1000, 500, None, Some(512)), (1024, 512));
    assert_eq!(edit_size(1000, 750, Some(640), Some(480)), (640, 480));
}

#[test]
fn layout_requires_the_pipeline_class_and_components() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("model_index.json"),
        r#"{"_class_name": "ZImagePipeline"}"#,
    )
    .unwrap();
    let err = validate_model_layout(dir.path()).unwrap_err();
    assert!(format!("{err}").contains("Not a LLaDA-Image checkpoint"));
    std::fs::write(
        dir.path().join("model_index.json"),
        r#"{"_class_name": "LLaDAImagePipeline"}"#,
    )
    .unwrap();
    let err = validate_model_layout(dir.path()).unwrap_err();
    assert!(format!("{err}").contains("text_encoder"), "{err}");
}
