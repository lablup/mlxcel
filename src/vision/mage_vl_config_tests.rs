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

//! Mage-VL config gates: the decoder RoPE base must come from the nested
//! `rope_parameters`, and the vision switches this port does not implement
//! must be refused by name.

use serde_json::json;

use super::*;
use crate::models::qwen3::ModelArgs;

/// The released `text_config`, trimmed to what `ModelArgs` needs plus the
/// nested rope block and no flat `rope_theta`.
fn released_text_config() -> Value {
    json!({
        "model_type": "qwen3",
        "hidden_size": 2560,
        "num_hidden_layers": 36,
        "intermediate_size": 9728,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "vocab_size": 151936,
        "max_position_embeddings": 262144,
        "rope_scaling": null,
        "rope_parameters": {"rope_theta": 5000000, "rope_type": "default"},
        "tie_word_embeddings": false
    })
}

#[test]
fn decoder_rope_theta_is_lifted_from_rope_parameters() {
    let mut config = released_text_config();
    lift_rope_parameters(&mut config);
    let args: ModelArgs = serde_json::from_value(config).unwrap();
    assert_eq!(args.rope_theta, 5_000_000.0);
    assert!(args.rope_scaling.is_none());
    assert!(!args.tie_word_embeddings);
}

#[test]
fn without_the_lift_the_decoder_would_silently_run_at_base_10000() {
    // The failure mode the lift exists for: serde fills the default.
    let args: ModelArgs = serde_json::from_value(released_text_config()).unwrap();
    assert_eq!(args.rope_theta, 10000.0);
}

#[test]
fn flat_rope_theta_wins_over_rope_parameters() {
    let mut config = released_text_config();
    config["rope_theta"] = json!(1_000_000);
    lift_rope_parameters(&mut config);
    let args: ModelArgs = serde_json::from_value(config).unwrap();
    assert_eq!(args.rope_theta, 1_000_000.0);
}

#[test]
fn non_default_rope_type_becomes_rope_scaling() {
    let mut config = released_text_config();
    config["rope_parameters"] = json!({
        "rope_theta": 5000000,
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768
    });
    lift_rope_parameters(&mut config);
    assert_eq!(config["rope_scaling"]["type"], "yarn");
    assert_eq!(config["rope_scaling"]["factor"], 4.0);
    assert!(config["rope_scaling"].get("rope_theta").is_none());
    assert_eq!(config["rope_theta"], 5000000);
}

#[test]
fn released_vision_config_parses_and_validates() {
    let config: MageVlVisionConfig = serde_json::from_value(json!({
        "frame_windows_size": 4,
        "hidden_act": "gelu",
        "hidden_size": 1024,
        "image_size": 448,
        "intermediate_size": 4096,
        "layer_norm_eps": 1e-06,
        "layer_norm_type": "layer_norm",
        "model_type": "mage_vl_vision",
        "num_attention_heads": 16,
        "num_channels": 3,
        "num_hidden_layers": 24,
        "out_hidden_size": 2560,
        "patch_size": 16,
        "rope_theta": 10000.0,
        "spatial_merge_size": 2,
        "temporal_patch_size": 1,
        "use_head": false,
        "max_position_embeddings": 8192
    }))
    .unwrap();
    assert_eq!(config.head_dim(), 64);
    assert!(!config.use_patch_position_encoding);
    config.validate().unwrap();
}

#[test]
fn unsupported_vision_switches_are_rejected_by_name() {
    let cases = [
        (
            json!({"use_patch_position_encoding": true}),
            "use_patch_position_encoding",
        ),
        (json!({"use_head": true}), "use_head"),
        (json!({"layer_norm_type": "rms_norm"}), "layer_norm_type"),
        (json!({"hidden_act": "gelu_pytorch_tanh"}), "hidden_act"),
        (
            json!({"hidden_size": 1152, "num_attention_heads": 16}),
            "4:6:6",
        ),
        (json!({"temporal_patch_size": 2}), "temporal_patch_size"),
    ];
    for (raw, needle) in cases {
        let config: MageVlVisionConfig = serde_json::from_value(raw.clone()).unwrap();
        let err = config.validate().unwrap_err();
        assert!(err.contains(needle), "{raw}: {err}");
    }
}

#[test]
fn token_ids_fall_back_to_released_values_and_reject_garbage() {
    let ids = MageVlTokenIds::from_config(&json!({})).unwrap();
    assert_eq!(ids.image_token_id, 151655);
    assert_eq!(ids.video_token_id, 151656);
    assert_eq!(ids.vision_start_token_id, 151652);
    assert_eq!(ids.vision_end_token_id, 151653);

    let err = MageVlTokenIds::from_config(&json!({"image_token_id": -1})).unwrap_err();
    assert!(err.contains("image_token_id"), "{err}");
    let err = MageVlTokenIds::from_config(&json!({"video_token_id": 1u64 << 40})).unwrap_err();
    assert!(err.contains("video_token_id"), "{err}");
}

#[test]
fn eos_ids_merge_generation_config_config_and_defaults() {
    let ids = resolve_eos_token_ids(
        Some(&json!({"eos_token_id": [151645, 7]})),
        &json!({"eos_token_id": 151645}),
    );
    assert_eq!(ids, vec![151645, 7, 151643]);
}

#[test]
fn pixel_bounds_and_normalization_come_from_the_preprocessor_config() {
    let pre = json!({
        "min_pixels": 3136,
        "max_pixels": 4000000,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711]
    });
    assert_eq!(resolve_pixel_bounds(Some(&pre)), (3136, 4_000_000));
    assert_eq!(resolve_pixel_bounds(None), (3136, 4_000_000));
    assert_eq!(
        resolve_pixel_bounds(Some(
            &json!({"size": {"shortest_edge": 100, "longest_edge": 200}})
        )),
        (100, 200)
    );
    let (mean, std) = resolve_normalization(Some(&pre));
    assert_eq!(mean, MAGE_VL_IMAGE_MEAN);
    assert_eq!(std, MAGE_VL_IMAGE_STD);
}
