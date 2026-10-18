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

use super::lift_rope_parameters;
use serde::Deserialize;
use serde_json::json;

#[test]
fn nested_theta_and_partial_factor_fill_absent_flat_keys() {
    let mut config = json!({
        "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.4, "rope_type": "default"}
    });
    lift_rope_parameters(&mut config);
    assert_eq!(config["rope_theta"], 10000.0);
    assert_eq!(config["partial_rotary_factor"], 0.4);
    assert!(config.get("rope_scaling").is_none());
}

#[test]
fn flat_keys_win_and_null_flat_keys_do_not() {
    let mut config = json!({
        "rope_theta": 500000.0,
        "partial_rotary_factor": null,
        "rope_parameters": {"rope_theta": 123456.0, "partial_rotary_factor": 0.5}
    });
    lift_rope_parameters(&mut config);
    assert_eq!(config["rope_theta"], 500000.0);
    assert_eq!(config["partial_rotary_factor"], 0.5);
}

#[test]
fn llama3_block_becomes_rope_scaling_under_type_only() {
    // The shape transformers 5.20 writes for a Llama 3.2 config.
    let mut config = json!({
        "rope_parameters": {
            "factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192, "rope_type": "llama3",
            "rope_theta": 500000.0
        }
    });
    lift_rope_parameters(&mut config);
    assert_eq!(config["rope_theta"], 500000.0);
    let scaling = &config["rope_scaling"];
    assert_eq!(scaling["type"], "llama3");
    assert_eq!(scaling["factor"], 32.0);
    assert_eq!(scaling["original_max_position_embeddings"], 8192);
    assert!(scaling.get("rope_theta").is_none());
    // One spelling only, so an alias-both field does not see a duplicate.
    assert!(scaling.get("rope_type").is_none());

    #[derive(Deserialize)]
    struct AliasBoth {
        #[serde(alias = "type", alias = "rope_type")]
        scaling_type: Option<String>,
    }
    let parsed: AliasBoth = serde_json::from_value(scaling.clone()).unwrap();
    assert_eq!(parsed.scaling_type.as_deref(), Some("llama3"));
}

#[test]
fn existing_rope_scaling_is_kept() {
    let mut config = json!({
        "rope_scaling": {"type": "linear", "factor": 2.0},
        "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "yarn", "factor": 4.0}
    });
    lift_rope_parameters(&mut config);
    assert_eq!(
        config["rope_scaling"],
        json!({"type": "linear", "factor": 2.0})
    );
    assert_eq!(config["rope_theta"], 1000000.0);
}

#[test]
fn per_layer_type_block_lifts_full_and_sliding_bases() {
    // The shape transformers 5.20 writes for a Gemma 3 text config.
    let mut config = json!({
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "linear", "factor": 8.0, "rope_theta": 1000000.0}
        }
    });
    lift_rope_parameters(&mut config);
    assert_eq!(config["rope_theta"], 1000000.0);
    assert_eq!(config["rope_local_base_freq"], 10000.0);
    assert_eq!(
        config["rope_scaling"],
        json!({"type": "linear", "factor": 8.0})
    );
}

#[test]
fn default_mrope_block_still_becomes_rope_scaling() {
    // transformers 5.20 Qwen2-VL / Qwen3-VL text configs: the scheme is
    // "default" but `mrope_section` used to live in `rope_scaling`.
    let mut qwen2_vl = json!({
        "rope_parameters": {
            "type": "mrope", "mrope_section": [16, 24, 24],
            "rope_theta": 1000000.0, "rope_type": "default"
        }
    });
    lift_rope_parameters(&mut qwen2_vl);
    assert_eq!(qwen2_vl["rope_theta"], 1000000.0);
    assert_eq!(
        qwen2_vl["rope_scaling"],
        json!({"type": "mrope", "mrope_section": [16, 24, 24]})
    );

    let mut qwen3_vl = json!({
        "rope_parameters": {
            "rope_type": "default", "mrope_section": [24, 20, 20],
            "mrope_interleaved": true, "rope_theta": 5000000.0
        }
    });
    lift_rope_parameters(&mut qwen3_vl);
    assert_eq!(
        qwen3_vl["rope_scaling"],
        json!({"type": "default", "mrope_section": [24, 20, 20], "mrope_interleaved": true})
    );
}

#[test]
fn non_object_and_unusable_shapes_are_left_alone() {
    let mut array = json!([1, 2]);
    lift_rope_parameters(&mut array);
    assert_eq!(array, json!([1, 2]));

    let mut no_block = json!({"rope_parameters": null, "rope_theta": 10.0});
    let before = no_block.clone();
    lift_rope_parameters(&mut no_block);
    assert_eq!(no_block, before);

    // A non-numeric base is not copied, and a per-layer-type block with no
    // `full_attention` entry leaves the flat keys alone.
    let mut string_theta =
        json!({"rope_parameters": {"rope_theta": "1e6", "rope_type": "default"}});
    lift_rope_parameters(&mut string_theta);
    assert!(string_theta.get("rope_theta").is_none());
    assert!(string_theta.get("rope_scaling").is_none());

    let mut sliding_only = json!({
        "rope_parameters": {"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}}
    });
    lift_rope_parameters(&mut sliding_only);
    assert!(sliding_only.get("rope_theta").is_none());
    assert!(sliding_only.get("rope_scaling").is_none());
    assert_eq!(sliding_only["rope_local_base_freq"], 10000.0);
}

#[test]
fn null_type_key_does_not_hide_rope_type() {
    let mut config = json!({
        "rope_parameters": {"type": null, "rope_type": "yarn", "factor": 4.0, "rope_theta": 1000000.0}
    });
    lift_rope_parameters(&mut config);
    assert_eq!(
        config["rope_scaling"],
        json!({"type": "yarn", "factor": 4.0})
    );
}

#[test]
fn deepseek_v4_main_and_compress_blocks_lift() {
    // The shape transformers 5.20 writes for DeepSeek-V4: the flat bases stay,
    // `rope_scaling` moves under `compress`.
    let mut config = json!({
        "rope_theta": 10000.0,
        "compress_rope_theta": 160000.0,
        "rope_parameters": {
            "main": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.125},
            "compress": {
                "type": "yarn", "factor": 16.0, "original_max_position_embeddings": 65536,
                "beta_fast": 32, "beta_slow": 1, "rope_theta": 160000.0, "rope_type": "yarn",
                "partial_rotary_factor": 0.125, "attention_factor": 1.0
            }
        }
    });
    lift_rope_parameters(&mut config);
    assert_eq!(config["rope_theta"], 10000.0);
    assert_eq!(config["compress_rope_theta"], 160000.0);
    let scaling = &config["rope_scaling"];
    assert_eq!(scaling["type"], "yarn");
    assert_eq!(scaling["factor"], 16.0);
    assert_eq!(scaling["original_max_position_embeddings"], 65536);
    assert!(scaling.get("rope_type").is_none());

    // With the flat bases dropped, both come from the nested blocks.
    let mut nested_only = json!({
        "rope_parameters": {
            "main": {"rope_type": "default", "rope_theta": 10000.0},
            "compress": {"rope_type": "yarn", "factor": 16.0, "rope_theta": 160000.0}
        }
    });
    lift_rope_parameters(&mut nested_only);
    assert_eq!(nested_only["rope_theta"], 10000.0);
    assert_eq!(nested_only["compress_rope_theta"], 160000.0);
}
