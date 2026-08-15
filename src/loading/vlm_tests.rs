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

use super::{
    QwenVisionTokenIds, inherit_qwen_vision_quantization, qwen_vl_token_ids, qwen35_vl_token_ids,
    require_object_mut, rewrite_qwen3_vl_weight_key,
};
use crate::vision::encoders::qwen3_vl::Qwen3VLVisionConfig;
use serde_json::json;

#[test]
fn inherit_qwen_vision_quantization_uses_top_level_defaults() {
    let mut vision_config: Qwen3VLVisionConfig = serde_json::from_value(json!({
        "hidden_size": 1536
    }))
    .unwrap();
    let full_config = json!({
        "quantization": {
            "group_size": 128,
            "bits": 8
        }
    });

    inherit_qwen_vision_quantization(&mut vision_config, &full_config);

    assert_eq!(vision_config.quant_group_size, 128);
    assert_eq!(vision_config.quant_bits, 8);
}

#[test]
fn inherit_qwen_vision_quantization_preserves_existing_values() {
    let mut vision_config: Qwen3VLVisionConfig = serde_json::from_value(json!({
        "hidden_size": 1536,
        "quant_group_size": 32,
        "quant_bits": 6
    }))
    .unwrap();
    let full_config = json!({
        "quantization": {
            "group_size": 128,
            "bits": 8
        }
    });

    inherit_qwen_vision_quantization(&mut vision_config, &full_config);

    assert_eq!(vision_config.quant_group_size, 32);
    assert_eq!(vision_config.quant_bits, 6);
}

#[test]
fn rewrite_qwen3_vl_weight_key_rewrites_language_and_visual_prefixes() {
    assert_eq!(
        rewrite_qwen3_vl_weight_key(
            "model.language_model.layers.0.self_attn.q_proj.weight".into(),
            false
        ),
        "model.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(
        rewrite_qwen3_vl_weight_key("model.visual.blocks.0.attn.qkv.weight".into(), false),
        "vision_tower.blocks.0.attn.qkv.weight"
    );
    assert_eq!(
        rewrite_qwen3_vl_weight_key("language_model.lm_head.weight".into(), false),
        "lm_head.weight"
    );
}

#[test]
fn rewrite_qwen3_vl_weight_key_sanitizes_moe_expert_weights() {
    assert_eq!(
        rewrite_qwen3_vl_weight_key(
            "model.language_model.layers.0.mlp.experts.up_proj".into(),
            true
        ),
        "model.layers.0.mlp.switch_mlp.up_proj.weight"
    );
    assert_eq!(
        rewrite_qwen3_vl_weight_key(
            "model.language_model.layers.0.mlp.experts.up_proj.weight".into(),
            true
        ),
        "model.layers.0.mlp.switch_mlp.up_proj.weight"
    );
}

#[test]
fn require_object_mut_rejects_non_object_values() {
    let mut value = json!(7);
    let err = require_object_mut(&mut value, "test config")
        .unwrap_err()
        .to_string();
    assert!(err.contains("Expected test config to be a JSON object"));
}

#[test]
fn qwen_vl_token_ids_applies_defaults_and_overrides() {
    let defaults = QwenVisionTokenIds {
        image_token_id: 10,
        video_token_id: 11,
        vision_start_token_id: 12,
    };

    let ids = qwen_vl_token_ids(
        &json!({
            "image_token_id": 20,
            "vision_start_token_id": 22
        }),
        defaults,
    );

    assert_eq!(
        ids,
        QwenVisionTokenIds {
            image_token_id: 20,
            video_token_id: 11,
            vision_start_token_id: 22,
        }
    );
}

/// The older Qwen VL families still resolve their token ids through
/// [`qwen_vl_token_ids`] with per-family defaults, and #1163 must not change
/// that. The `vision_start_token_id` requirement is scoped to the Qwen3.5
/// family, where the historical default was stale.
///
/// The default triples below are exactly the ones the loaders pass in
/// `src/loading/vlm_qwen.rs`, and they match the shipped checkpoints
/// (`qwen2-vl-2b-4bit`, `qwen2.5-vl-3b-4bit`, `qwen3-vl-4b-4bit` all declare
/// 151652).
#[test]
fn older_qwen_vl_families_still_resolve_token_ids_from_defaults() {
    let qwen_vl = QwenVisionTokenIds {
        image_token_id: 151655,
        video_token_id: 151656,
        vision_start_token_id: 151652,
    };
    let glm4v = QwenVisionTokenIds {
        image_token_id: 151363,
        video_token_id: 151364,
        vision_start_token_id: 151339,
    };

    for (family, defaults) in [
        ("qwen2_vl", qwen_vl),
        ("qwen2_5_vl", qwen_vl),
        ("qwen3_vl", qwen_vl),
        ("qwen3_vl_moe", qwen_vl),
        ("glm4v", glm4v),
    ] {
        // A config that omits every id must still load, from the defaults.
        assert_eq!(
            qwen_vl_token_ids(&json!({}), defaults),
            defaults,
            "{family}: an empty config must fall back to the family defaults"
        );

        // A config that supplies them must win over the defaults.
        let overridden = qwen_vl_token_ids(
            &json!({
                "image_token_id": 7,
                "video_token_id": 8,
                "vision_start_token_id": 9
            }),
            defaults,
        );
        assert_eq!(
            overridden,
            QwenVisionTokenIds {
                image_token_id: 7,
                video_token_id: 8,
                vision_start_token_id: 9,
            },
            "{family}: explicit ids must be read from the config"
        );
    }
}

/// The Qwen3.5 family requires `vision_start_token_id` from the config. The
/// removed 248045 default was stale for every shipped checkpoint in the
/// family, and a wrong start id mis-segments MRoPE vision spans silently
/// rather than failing.
#[test]
fn qwen35_vl_token_ids_requires_vision_start_from_config() {
    let err = qwen35_vl_token_ids(&json!({
        "image_token_id": 248056,
        "video_token_id": 248057
    }))
    .expect_err("a Qwen3.5 config without vision_start_token_id must not load");
    let message = err.to_string();
    assert!(
        message.contains("vision_start_token_id"),
        "error must name the missing key, got: {message}"
    );
    assert!(
        message.contains("248045"),
        "error must name the stale default it replaces, got: {message}"
    );
}

/// Pinned to the published `config.json` of `mlx-community/Qwen3.8-27B-4bit`
/// and `mlx-community/Qwen3.5-27B-4bit`, which agree on all three ids.
#[test]
fn qwen35_vl_token_ids_read_the_published_checkpoint_ids() {
    let ids = qwen35_vl_token_ids(&json!({
        "image_token_id": 248056,
        "video_token_id": 248057,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054
    }))
    .expect("the published Qwen3.8-27B config supplies vision_start_token_id");

    assert_eq!(
        ids,
        QwenVisionTokenIds {
            image_token_id: 248056,
            video_token_id: 248057,
            // Not the old 248045 default.
            vision_start_token_id: 248053,
        }
    );
}

/// `image_token_id` and `video_token_id` keep their defaults for this family:
/// those constants do match every shipped checkpoint.
#[test]
fn qwen35_vl_token_ids_default_only_the_image_and_video_ids() {
    let ids = qwen35_vl_token_ids(&json!({ "vision_start_token_id": 248053 }))
        .expect("vision_start_token_id alone is enough to load");
    assert_eq!(
        ids,
        QwenVisionTokenIds {
            image_token_id: 248056,
            video_token_id: 248057,
            vision_start_token_id: 248053,
        }
    );
}
