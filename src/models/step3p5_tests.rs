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

use super::{Cache, Step3p5Config, Step3p5Model};

fn parse_config(json: serde_json::Value) -> Step3p5Config {
    serde_json::from_value(json).expect("valid Step3p5Config")
}

fn assert_cache_kinds(caches: &[Cache], expected_rotating: &[bool]) {
    assert_eq!(caches.len(), expected_rotating.len());
    for (cache, expect_rotating) in caches.iter().zip(expected_rotating) {
        match (cache, expect_rotating) {
            (Cache::Rotating(_), true) | (Cache::Standard(_), false) => {}
            _ => panic!("unexpected cache kind"),
        }
    }
}

#[test]
fn build_layer_caches_uses_rotating_for_explicit_sliding_layers() {
    let config = parse_config(serde_json::json!({
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "vocab_size": 1024,
        "num_attention_heads": 4,
        "num_attention_groups": 2,
        "head_dim": 64,
        "intermediate_size": 512,
        "sliding_window": 16,
        "layer_types": [
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention"
        ]
    }));

    let caches = Step3p5Model::build_layer_caches(&config);
    assert_cache_kinds(&caches, &[false, true, true, false]);
}

#[test]
fn build_layer_caches_uses_fallback_even_odd_pattern_without_layer_types() {
    let config = parse_config(serde_json::json!({
        "hidden_size": 256,
        "num_hidden_layers": 5,
        "vocab_size": 1024,
        "num_attention_heads": 4,
        "num_attention_groups": 2,
        "head_dim": 64,
        "intermediate_size": 512,
        "sliding_window": 32
    }));

    let caches = Step3p5Model::build_layer_caches(&config);
    assert_cache_kinds(&caches, &[true, false, true, false, true]);
}

// step3p7 config-gated extensions (issue #543).

fn base_text_config_json() -> serde_json::Value {
    serde_json::json!({
        "hidden_size": 4096,
        "num_hidden_layers": 45,
        "vocab_size": 128896,
        "num_attention_heads": 64,
        "num_attention_groups": 8,
        "head_dim": 128,
        "intermediate_size": 11264,
        "moe_num_experts": 288,
        "moe_top_k": 8,
        "moe_intermediate_size": 1280,
        "share_expert_dim": 1280
    })
}

#[test]
fn moe_layers_enum_accepts_comma_separated_string() {
    let mut json = base_text_config_json();
    json["moe_layers_enum"] = serde_json::json!("3, 4 ,7");
    let config = parse_config(json);
    let mut indices: Vec<usize> = config.moe_layer_indices().into_iter().collect();
    indices.sort_unstable();
    assert_eq!(indices, vec![3, 4, 7]);
}

#[test]
fn moe_layers_enum_accepts_json_int_array() {
    let mut json = base_text_config_json();
    json["moe_layers_enum"] = serde_json::json!([3, 4, 7]);
    let config = parse_config(json);
    let mut indices: Vec<usize> = config.moe_layer_indices().into_iter().collect();
    indices.sort_unstable();
    assert_eq!(indices, vec![3, 4, 7]);
}

#[test]
fn moe_layers_enum_absent_uses_flat_all_except_zero_fallback() {
    // Flat step3p5 config: fallback is every layer except 0.
    let mut json = base_text_config_json();
    json["num_hidden_layers"] = serde_json::json!(4);
    let config = parse_config(json);
    let mut indices: Vec<usize> = config.moe_layer_indices().into_iter().collect();
    indices.sort_unstable();
    assert_eq!(indices, vec![1, 2, 3]);
}

#[test]
fn nested_text_config_uses_step3p7_moe_layer_fallback() {
    // Nested step3p7 text_config without moe_layers_enum falls back to 3..=44.
    let config = Step3p5Config::from_nested_text_config(&base_text_config_json())
        .expect("nested text_config parses");
    let indices = config.moe_layer_indices();
    assert_eq!(indices.len(), 42);
    assert!(!indices.contains(&2));
    assert!(indices.contains(&3));
    assert!(indices.contains(&44));
    assert!(!indices.contains(&45));
}

#[test]
fn flat_config_keeps_step3p5_defaults() {
    // Flat parse must not flip the step3p5 defaults (regression gate).
    let config = parse_config(base_text_config_json());
    assert!(config.use_head_wise_attn_gate, "flat default stays true");
    assert_eq!(
        config.moe_router_scaling_factor, 3.0,
        "flat default stays 3.0"
    );
}

#[test]
fn nested_text_config_flips_defaults_when_keys_absent() {
    let config = Step3p5Config::from_nested_text_config(&base_text_config_json())
        .expect("nested text_config parses");
    assert!(
        !config.use_head_wise_attn_gate,
        "step3p7 default flips to false"
    );
    assert_eq!(
        config.moe_router_scaling_factor, 1.0,
        "step3p7 default flips to 1.0"
    );
}

#[test]
fn nested_text_config_explicit_values_win_over_flips() {
    let mut json = base_text_config_json();
    json["use_head_wise_attn_gate"] = serde_json::json!(true);
    json["moe_router_scaling_factor"] = serde_json::json!(2.5);
    let config = Step3p5Config::from_nested_text_config(&json).expect("nested text_config parses");
    assert!(config.use_head_wise_attn_gate, "explicit true wins");
    assert_eq!(config.moe_router_scaling_factor, 2.5, "explicit value wins");
}

#[test]
fn resolve_eos_prefers_top_level_int_then_list_then_text_config() {
    // Top-level int.
    let cfg = serde_json::json!({ "eos_token_id": 128805 });
    assert_eq!(
        Step3p5Config::resolve_step3p7_eos_token_ids(&cfg),
        vec![128805]
    );

    // Top-level list.
    let cfg = serde_json::json!({ "eos_token_id": [128805, 128806] });
    assert_eq!(
        Step3p5Config::resolve_step3p7_eos_token_ids(&cfg),
        vec![128805, 128806]
    );

    // Falls back to text_config.eos_token_id when top-level absent.
    let cfg = serde_json::json!({ "text_config": { "eos_token_id": 7 } });
    assert_eq!(Step3p5Config::resolve_step3p7_eos_token_ids(&cfg), vec![7]);

    // Last resort [2].
    let cfg = serde_json::json!({ "text_config": {} });
    assert_eq!(Step3p5Config::resolve_step3p7_eos_token_ids(&cfg), vec![2]);
}

#[test]
fn resolve_eos_treats_null_top_level_as_absent_and_uses_text_config() {
    // Composite VLM configs commonly serialize the top-level `eos_token_id` as
    // `null` with the real ids under `text_config`. A present-but-null value
    // must be treated as absent so the text_config source wins over the `[2]`
    // last resort.
    let cfg = serde_json::json!({
        "eos_token_id": serde_json::Value::Null,
        "text_config": { "eos_token_id": [128805, 128806] }
    });
    assert_eq!(
        Step3p5Config::resolve_step3p7_eos_token_ids(&cfg),
        vec![128805, 128806]
    );

    // Null at both levels falls through to the `[2]` last resort.
    let cfg = serde_json::json!({
        "eos_token_id": serde_json::Value::Null,
        "text_config": { "eos_token_id": serde_json::Value::Null }
    });
    assert_eq!(Step3p5Config::resolve_step3p7_eos_token_ids(&cfg), vec![2]);

    // An empty list is not usable ids; fall through to text_config.
    let cfg = serde_json::json!({
        "eos_token_id": [],
        "text_config": { "eos_token_id": 7 }
    });
    assert_eq!(Step3p5Config::resolve_step3p7_eos_token_ids(&cfg), vec![7]);
}
