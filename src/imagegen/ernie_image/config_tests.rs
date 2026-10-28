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

use serde_json::json;
use tempfile::TempDir;

use super::*;

fn options(extra: serde_json::Value) -> Result<ErnieOptions> {
    ErnieOptions::from_extra(extra.as_object().unwrap())
}

#[test]
fn image_strength_must_be_in_the_half_open_unit_interval() {
    assert_eq!(options(json!({})).unwrap().image_strength, 0.6);
    assert_eq!(
        options(json!({"image_strength": 0.7}))
            .unwrap()
            .image_strength,
        0.7
    );
    assert_eq!(
        options(json!({"image_strength": 1.0}))
            .unwrap()
            .image_strength,
        1.0
    );
    for bad in [json!(0.0), json!(-0.1), json!(1.01), json!("0.5")] {
        let err = options(json!({ "image_strength": bad })).unwrap_err();
        assert!(err.to_string().contains("image_strength"), "{bad}: {err}");
    }
}

#[test]
fn strength_is_an_alias_and_image_strength_wins() {
    assert_eq!(
        options(json!({"strength": 0.3})).unwrap().image_strength,
        0.3
    );
    assert_eq!(
        options(json!({"strength": 0.3, "image_strength": 0.9}))
            .unwrap()
            .image_strength,
        0.9
    );
    assert!(options(json!({"strength": 2.0})).is_err());
}

#[test]
fn enhancer_and_negative_prompt_options_are_typed() {
    let o = options(json!({"use_prompt_enhancer": false, "negative_prompt": "blurry"})).unwrap();
    assert_eq!(o.use_prompt_enhancer, Some(false));
    assert_eq!(o.negative_prompt.as_deref(), Some("blurry"));
    assert!(options(json!({"use_prompt_enhancer": "yes"})).is_err());
    assert!(options(json!({"negative_prompt": 3})).is_err());
    assert!(options(json!({"prompt_enhancer_max_tokens": 0})).is_err());
    assert_eq!(
        options(json!({"prompt_enhancer_max_tokens": 64}))
            .unwrap()
            .prompt_enhancer_max_tokens,
        Some(64)
    );
}

#[test]
fn dimensions_must_be_positive_multiples_of_16() {
    validate_dimensions(1024, 512).unwrap();
    validate_dimensions(16, 16).unwrap();
    assert!(validate_dimensions(1000, 1024).is_err());
    assert!(validate_dimensions(1024, 8).is_err());
    assert!(validate_dimensions(0, 16).is_err());
}

#[test]
fn edit_start_index_rounds_half_to_even_like_python() {
    // round(8 * 0.7) = 6 denoising steps -> start 2 (upstream's oracle run).
    assert_eq!(edit_start_index(8, 0.7), 2);
    assert_eq!(edit_start_index(8, 0.6), 3);
    // round(2.5) = 2 in Python, not 3.
    assert_eq!(edit_start_index(4, 0.625), 2);
    // At least one step, at most all of them.
    assert_eq!(edit_start_index(8, 0.01), 7);
    assert_eq!(edit_start_index(8, 1.0), 0);
}

#[test]
fn variant_defaults_follow_upstream() {
    let turbo = ErnieVariant::Turbo;
    assert_eq!(turbo.generate_defaults().steps, 8);
    assert_eq!(turbo.generate_defaults().guidance, 1.0);
    assert_eq!(turbo.edit_defaults().guidance, 3.0);
    let base = ErnieVariant::Base;
    assert_eq!(base.generate_defaults().steps, 50);
    assert_eq!(base.generate_defaults().guidance, 4.0);
    assert_eq!(base.edit_defaults().guidance, 4.0);
}

#[test]
fn variant_comes_from_metadata_or_the_directory_name() {
    let dir = TempDir::new().unwrap();
    let turbo = dir.path().join("ERNIE-Image-Turbo");
    let base = dir.path().join("ernie-image");
    std::fs::create_dir_all(&turbo).unwrap();
    std::fs::create_dir_all(&base).unwrap();
    assert_eq!(ErnieVariant::from_dir(&turbo).unwrap(), ErnieVariant::Turbo);
    assert_eq!(ErnieVariant::from_dir(&base).unwrap(), ErnieVariant::Base);
    std::fs::write(
        base.join("mlx_ernie_image.json"),
        r#"{"source": "baidu/ERNIE-Image-Turbo"}"#,
    )
    .unwrap();
    assert_eq!(ErnieVariant::from_dir(&base).unwrap(), ErnieVariant::Turbo);
    let snapshot = dir
        .path()
        .join("models--baidu--ERNIE-Image-Turbo/snapshots/abc123");
    std::fs::create_dir_all(&snapshot).unwrap();
    assert_eq!(
        ErnieVariant::from_dir(&snapshot).unwrap(),
        ErnieVariant::Turbo
    );
}

#[test]
fn default_edit_size_targets_one_megapixel() {
    assert_eq!(default_edit_size(1024, 1024), (1024, 1024));
    assert_eq!(default_edit_size(2048, 2048), (1024, 1024));
    assert_eq!(default_edit_size(512, 512), (512, 512));
    // Small sources grow to a 256 short side.
    assert_eq!(default_edit_size(100, 200), (256, 512));
    // Rounded down to 16.
    assert_eq!(
        default_edit_size(1000, 750),
        (1000 / 16 * 16, 750 / 16 * 16)
    );
}

#[test]
fn transformer_config_parses_the_checkpoint_json() {
    let c = ErnieTransformerConfig::from_json(
        r#"{"_class_name": "ErnieImageTransformer2DModel", "hidden_size": 4096,
            "num_attention_heads": 32, "rope_axes_dim": [32, 48, 48], "use_lora": false}"#,
    )
    .unwrap();
    assert_eq!(c.head_dim(), 128);
    assert_eq!(c.num_layers, 36);
    let err = ErnieTransformerConfig::from_json(r#"{"rope_axes_dim": [32, 48, 40]}"#).unwrap_err();
    assert!(err.to_string().contains("rope_axes_dim"), "{err}");
}
