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

use super::*;
use crate::distributed::pipeline::partition::StageAssignment;

// ---------------------------------------------------------------------------
// LayerFilter construction
// ---------------------------------------------------------------------------

#[test]
fn layer_filter_from_stage_first() {
    let stage = StageAssignment {
        stage_index: 0,
        device_id: "dev0".into(),
        layer_range: 0..16,
        has_embedding: true,
        has_lm_head: false,
        estimated_memory_bytes: 1_000_000,
    };
    let filter = LayerFilter::from_stage(&stage);
    assert_eq!(filter.layer_range, 0..16);
    assert!(filter.has_embedding);
    assert!(!filter.has_lm_head);
    assert_eq!(filter.num_layers(), 16);
}

#[test]
fn layer_filter_from_stage_last() {
    let stage = StageAssignment {
        stage_index: 1,
        device_id: "dev1".into(),
        layer_range: 16..32,
        has_embedding: false,
        has_lm_head: true,
        estimated_memory_bytes: 1_000_000,
    };
    let filter = LayerFilter::from_stage(&stage);
    assert_eq!(filter.layer_range, 16..32);
    assert!(!filter.has_embedding);
    assert!(filter.has_lm_head);
}

#[test]
fn layer_filter_is_full_model() {
    let filter = LayerFilter {
        layer_range: 0..32,
        has_embedding: true,
        has_lm_head: true,
    };
    assert!(filter.is_full_model(32));
    assert!(!filter.is_full_model(64));

    let partial = LayerFilter {
        layer_range: 0..16,
        has_embedding: true,
        has_lm_head: false,
    };
    assert!(!partial.is_full_model(32));
}

// ---------------------------------------------------------------------------
// Weight key classification
// ---------------------------------------------------------------------------

#[test]
fn classify_standard_layer_key() {
    assert_eq!(
        classify_weight_key("model.layers.5.self_attn.q_proj.weight"),
        WeightClass::Layer(5)
    );
    assert_eq!(
        classify_weight_key("model.layers.31.mlp.gate_proj.weight"),
        WeightClass::Layer(31)
    );
}

#[test]
fn classify_language_model_prefix() {
    assert_eq!(
        classify_weight_key("language_model.model.layers.0.self_attn.q_proj.weight"),
        WeightClass::Layer(0)
    );
}

#[test]
fn classify_transformer_prefix() {
    assert_eq!(
        classify_weight_key("transformer.h.12.attn.weight"),
        WeightClass::Layer(12)
    );
    assert_eq!(
        classify_weight_key("transformer.layers.3.norm.weight"),
        WeightClass::Layer(3)
    );
}

#[test]
fn classify_embedding_keys() {
    assert_eq!(
        classify_weight_key("model.embed_tokens.weight"),
        WeightClass::Embedding
    );
    assert_eq!(
        classify_weight_key("language_model.model.embed_tokens.weight"),
        WeightClass::Embedding
    );
    assert_eq!(
        classify_weight_key("transformer.wte.weight"),
        WeightClass::Embedding
    );
}

#[test]
fn classify_lm_head_keys() {
    assert_eq!(classify_weight_key("lm_head.weight"), WeightClass::LmHead);
    assert_eq!(classify_weight_key("lm_head.scales"), WeightClass::LmHead);
    assert_eq!(
        classify_weight_key("language_model.lm_head.weight"),
        WeightClass::LmHead
    );
}

#[test]
fn classify_norm_keys() {
    assert_eq!(classify_weight_key("model.norm.weight"), WeightClass::Norm);
    assert_eq!(
        classify_weight_key("model.final_layernorm.weight"),
        WeightClass::Norm
    );
}

#[test]
fn classify_other_keys() {
    // Vision encoder, connector, etc.
    assert_eq!(
        classify_weight_key("vision_tower.encoder.layers.0.weight"),
        WeightClass::Other
    );
    assert_eq!(
        classify_weight_key("multi_modal_projector.linear.weight"),
        WeightClass::Other
    );
}

// ---------------------------------------------------------------------------
// should_load_key
// ---------------------------------------------------------------------------

#[test]
fn should_load_first_stage() {
    let filter = LayerFilter {
        layer_range: 0..16,
        has_embedding: true,
        has_lm_head: false,
    };
    // Layers in range
    assert!(should_load_key(
        "model.layers.0.self_attn.q_proj.weight",
        &filter
    ));
    assert!(should_load_key(
        "model.layers.15.mlp.gate_proj.weight",
        &filter
    ));
    // Layers out of range
    assert!(!should_load_key(
        "model.layers.16.self_attn.q_proj.weight",
        &filter
    ));
    assert!(!should_load_key(
        "model.layers.31.mlp.gate_proj.weight",
        &filter
    ));
    // Embedding yes
    assert!(should_load_key("model.embed_tokens.weight", &filter));
    // lm_head no
    assert!(!should_load_key("lm_head.weight", &filter));
    // Norm no (only loaded with lm_head)
    assert!(!should_load_key("model.norm.weight", &filter));
    // Other yes (vision encoder on first stage)
    assert!(should_load_key("vision_tower.weight", &filter));
}

#[test]
fn should_load_last_stage() {
    let filter = LayerFilter {
        layer_range: 16..32,
        has_embedding: false,
        has_lm_head: true,
    };
    // Layers in range
    assert!(should_load_key("model.layers.16.weight", &filter));
    assert!(should_load_key("model.layers.31.weight", &filter));
    // Layers out of range
    assert!(!should_load_key("model.layers.0.weight", &filter));
    assert!(!should_load_key("model.layers.15.weight", &filter));
    // Embedding no
    assert!(!should_load_key("model.embed_tokens.weight", &filter));
    // lm_head yes
    assert!(should_load_key("lm_head.weight", &filter));
    // Norm yes (with lm_head)
    assert!(should_load_key("model.norm.weight", &filter));
    // Other no (no embedding)
    assert!(!should_load_key("vision_tower.weight", &filter));
}

#[test]
fn should_load_middle_stage() {
    let filter = LayerFilter {
        layer_range: 8..16,
        has_embedding: false,
        has_lm_head: false,
    };
    assert!(should_load_key("model.layers.8.weight", &filter));
    assert!(should_load_key("model.layers.15.weight", &filter));
    assert!(!should_load_key("model.layers.7.weight", &filter));
    assert!(!should_load_key("model.layers.16.weight", &filter));
    assert!(!should_load_key("model.embed_tokens.weight", &filter));
    assert!(!should_load_key("lm_head.weight", &filter));
    assert!(!should_load_key("model.norm.weight", &filter));
}

// ---------------------------------------------------------------------------
// filter_weight_keys
// ---------------------------------------------------------------------------

#[test]
fn filter_weight_keys_basic() {
    let keys = vec![
        "model.embed_tokens.weight",
        "model.layers.0.weight",
        "model.layers.1.weight",
        "model.layers.2.weight",
        "model.layers.3.weight",
        "model.norm.weight",
        "lm_head.weight",
    ];
    let filter = LayerFilter {
        layer_range: 0..2,
        has_embedding: true,
        has_lm_head: false,
    };
    let result = filter_weight_keys(keys.into_iter(), &filter);
    assert_eq!(
        result,
        vec![
            "model.embed_tokens.weight",
            "model.layers.0.weight",
            "model.layers.1.weight",
        ]
    );
}

// ---------------------------------------------------------------------------
// SafeTensorsIndex
// ---------------------------------------------------------------------------

#[test]
fn safetensors_index_parse() {
    let json = r#"{
        "metadata": {"total_size": 1000000},
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.self_attn.v_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
            "model.layers.1.self_attn.v_proj.weight": "model-00002-of-00002.safetensors",
            "model.norm.weight": "model-00002-of-00002.safetensors",
            "lm_head.weight": "model-00002-of-00002.safetensors"
        }
    }"#;

    let index = SafeTensorsIndex::from_json(json).unwrap();
    assert_eq!(index.weight_to_shard.len(), 7);
}

#[test]
fn safetensors_index_required_shards_first_stage() {
    let json = r#"{
        "weight_map": {
            "model.embed_tokens.weight": "shard-1.safetensors",
            "model.layers.0.weight": "shard-1.safetensors",
            "model.layers.1.weight": "shard-1.safetensors",
            "model.layers.2.weight": "shard-2.safetensors",
            "model.layers.3.weight": "shard-2.safetensors",
            "model.norm.weight": "shard-2.safetensors",
            "lm_head.weight": "shard-2.safetensors"
        }
    }"#;
    let index = SafeTensorsIndex::from_json(json).unwrap();

    let filter = LayerFilter {
        layer_range: 0..2,
        has_embedding: true,
        has_lm_head: false,
    };
    let shards = index.required_shards(&filter);
    assert_eq!(shards.len(), 1);
    assert!(shards.contains("shard-1.safetensors"));
}

#[test]
fn safetensors_index_required_shards_last_stage() {
    let json = r#"{
        "weight_map": {
            "model.embed_tokens.weight": "shard-1.safetensors",
            "model.layers.0.weight": "shard-1.safetensors",
            "model.layers.1.weight": "shard-1.safetensors",
            "model.layers.2.weight": "shard-2.safetensors",
            "model.layers.3.weight": "shard-2.safetensors",
            "model.norm.weight": "shard-2.safetensors",
            "lm_head.weight": "shard-2.safetensors"
        }
    }"#;
    let index = SafeTensorsIndex::from_json(json).unwrap();

    let filter = LayerFilter {
        layer_range: 2..4,
        has_embedding: false,
        has_lm_head: true,
    };
    let shards = index.required_shards(&filter);
    assert_eq!(shards.len(), 1);
    assert!(shards.contains("shard-2.safetensors"));
}

#[test]
fn safetensors_index_required_shards_spans_both() {
    let json = r#"{
        "weight_map": {
            "model.embed_tokens.weight": "shard-1.safetensors",
            "model.layers.0.weight": "shard-1.safetensors",
            "model.layers.1.weight": "shard-2.safetensors",
            "model.layers.2.weight": "shard-2.safetensors",
            "model.layers.3.weight": "shard-3.safetensors",
            "model.norm.weight": "shard-3.safetensors",
            "lm_head.weight": "shard-3.safetensors"
        }
    }"#;
    let index = SafeTensorsIndex::from_json(json).unwrap();

    // Middle stage covering layers 1-2
    let filter = LayerFilter {
        layer_range: 1..3,
        has_embedding: false,
        has_lm_head: false,
    };
    let shards = index.required_shards(&filter);
    assert_eq!(shards.len(), 1);
    assert!(shards.contains("shard-2.safetensors"));
}

#[test]
fn safetensors_index_invalid_json() {
    let result = SafeTensorsIndex::from_json("not json");
    assert!(result.is_err());
}

#[test]
fn safetensors_index_missing_weight_map() {
    let result = SafeTensorsIndex::from_json(r#"{"metadata": {}}"#);
    assert!(result.is_err());
}

#[test]
fn safetensors_index_rejects_path_traversal_slash() {
    let json = r#"{
        "weight_map": {
            "model.layers.0.weight": "../../../etc/passwd"
        }
    }"#;
    let result = SafeTensorsIndex::from_json(json);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("path traversal"), "error was: {err}");
}

#[test]
fn safetensors_index_rejects_path_traversal_dotdot() {
    let json = r#"{
        "weight_map": {
            "model.layers.0.weight": "..\\malicious.safetensors"
        }
    }"#;
    let result = SafeTensorsIndex::from_json(json);
    assert!(result.is_err());
}

#[test]
fn safetensors_index_rejects_path_traversal_subdir() {
    let json = r#"{
        "weight_map": {
            "model.layers.0.weight": "subdir/model.safetensors"
        }
    }"#;
    let result = SafeTensorsIndex::from_json(json);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("path traversal"), "error was: {err}");
}

// ---------------------------------------------------------------------------
// Memory estimation
// ---------------------------------------------------------------------------

fn test_profile() -> ModelProfile {
    ModelProfile::uniform(32, 100_000_000, 50_000_000, 50_000_000)
}

#[test]
fn estimate_partial_memory_first_stage() {
    let profile = test_profile();
    let filter = LayerFilter {
        layer_range: 0..16,
        has_embedding: true,
        has_lm_head: false,
    };
    let mem = estimate_partial_memory(&filter, &profile);
    // 16 layers * 100MB + 50MB embedding = 1650MB
    assert_eq!(mem, 1_650_000_000);
}

#[test]
fn estimate_partial_memory_last_stage() {
    let profile = test_profile();
    let filter = LayerFilter {
        layer_range: 16..32,
        has_embedding: false,
        has_lm_head: true,
    };
    let mem = estimate_partial_memory(&filter, &profile);
    // 16 layers * 100MB + 50MB lm_head = 1650MB
    assert_eq!(mem, 1_650_000_000);
}

#[test]
fn estimate_partial_memory_middle_stage() {
    let profile = test_profile();
    let filter = LayerFilter {
        layer_range: 8..16,
        has_embedding: false,
        has_lm_head: false,
    };
    let mem = estimate_partial_memory(&filter, &profile);
    // 8 layers * 100MB = 800MB
    assert_eq!(mem, 800_000_000);
}

#[test]
fn estimate_partial_memory_full_model() {
    let profile = test_profile();
    let filter = LayerFilter {
        layer_range: 0..32,
        has_embedding: true,
        has_lm_head: true,
    };
    let mem = estimate_partial_memory(&filter, &profile);
    assert_eq!(mem, profile.total_param_bytes());
}

// ---------------------------------------------------------------------------
// Memory validation
// ---------------------------------------------------------------------------

#[test]
fn validate_partial_memory_ok() {
    let profile = test_profile();
    let filter = LayerFilter {
        layer_range: 0..16,
        has_embedding: true,
        has_lm_head: false,
    };
    let result = validate_partial_memory(&filter, &profile, 2_000_000_000);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 1_650_000_000);
}

#[test]
fn validate_partial_memory_exact_fit() {
    let profile = test_profile();
    let filter = LayerFilter {
        layer_range: 0..16,
        has_embedding: true,
        has_lm_head: false,
    };
    let result = validate_partial_memory(&filter, &profile, 1_650_000_000);
    assert!(result.is_ok());
}

#[test]
fn validate_partial_memory_insufficient() {
    let profile = test_profile();
    let filter = LayerFilter {
        layer_range: 0..16,
        has_embedding: true,
        has_lm_head: false,
    };
    let result = validate_partial_memory(&filter, &profile, 1_000_000_000);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("1650000000 bytes"));
    assert!(err.contains("1000000000 bytes available"));
}

// ---------------------------------------------------------------------------
// parse_leading_usize
// ---------------------------------------------------------------------------

#[test]
fn parse_leading_usize_valid() {
    assert_eq!(parse_leading_usize("0.weight"), Some(0));
    assert_eq!(parse_leading_usize("31.mlp.gate"), Some(31));
    assert_eq!(parse_leading_usize("100"), Some(100));
}

#[test]
fn parse_leading_usize_invalid() {
    assert_eq!(parse_leading_usize(""), None);
    assert_eq!(parse_leading_usize(".weight"), None);
    assert_eq!(parse_leading_usize("abc"), None);
}

// ---------------------------------------------------------------------------
// filter_weight_map (requires mlxcel_core)
// ---------------------------------------------------------------------------

#[test]
fn filter_weight_map_removes_unneeded() {
    use mlxcel_core::weights::WeightMap;

    let mut weights = WeightMap::new();
    let dummy = || mlxcel_core::ones(&[2, 2], mlxcel_core::dtype::FLOAT32);

    weights.insert("model.embed_tokens.weight".into(), dummy());
    weights.insert("model.layers.0.weight".into(), dummy());
    weights.insert("model.layers.1.weight".into(), dummy());
    weights.insert("model.layers.2.weight".into(), dummy());
    weights.insert("model.layers.3.weight".into(), dummy());
    weights.insert("model.norm.weight".into(), dummy());
    weights.insert("lm_head.weight".into(), dummy());

    let filter = LayerFilter {
        layer_range: 0..2,
        has_embedding: true,
        has_lm_head: false,
    };

    let removed = filter_weight_map(&mut weights, &filter);
    // Should remove: layers 2, 3, norm, lm_head = 4 keys
    assert_eq!(removed, 4);
    assert_eq!(weights.len(), 3);
    assert!(weights.contains_key("model.embed_tokens.weight"));
    assert!(weights.contains_key("model.layers.0.weight"));
    assert!(weights.contains_key("model.layers.1.weight"));
}

#[test]
fn filter_weight_map_last_stage_keeps_norm() {
    use mlxcel_core::weights::WeightMap;

    let mut weights = WeightMap::new();
    let dummy = || mlxcel_core::ones(&[2, 2], mlxcel_core::dtype::FLOAT32);

    weights.insert("model.embed_tokens.weight".into(), dummy());
    weights.insert("model.layers.0.weight".into(), dummy());
    weights.insert("model.layers.1.weight".into(), dummy());
    weights.insert("model.norm.weight".into(), dummy());
    weights.insert("lm_head.weight".into(), dummy());

    let filter = LayerFilter {
        layer_range: 1..2,
        has_embedding: false,
        has_lm_head: true,
    };

    let removed = filter_weight_map(&mut weights, &filter);
    // Should remove: embed_tokens, layers.0 = 2 keys
    assert_eq!(removed, 2);
    assert_eq!(weights.len(), 3);
    assert!(weights.contains_key("model.layers.1.weight"));
    assert!(weights.contains_key("model.norm.weight"));
    assert!(weights.contains_key("lm_head.weight"));
}
