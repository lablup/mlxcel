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

use super::validate_iree_processor_metadata;
use super::{
    LlavaTextBackend, detect_bunny_text_backend, infer_llama_config_from_weights,
    inherit_text_quantization_if_missing, is_llava_host_preprocessor_weight, llava_text_backend,
    parse_bunny_vision_config, require_llava_host_family, rewrite_bunny_weight_key,
};
use crate::multimodal::host_preprocessor::HostPreprocessorError;
use crate::vision::config::VLMConfig;
use mlxcel_core::dtype;
use mlxcel_core::weights::WeightMap;
use serde_json::json;

fn test_weight_map() -> WeightMap {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.norm.weight".to_string(),
        mlxcel_core::ones(&[256], dtype::FLOAT32),
    );
    weights.insert(
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        mlxcel_core::ones(&[512, 256], dtype::FLOAT32),
    );
    weights.insert(
        "model.layers.2.self_attn.q_proj.weight".to_string(),
        mlxcel_core::ones(&[256, 256], dtype::FLOAT32),
    );
    weights
}

#[test]
fn infer_llama_config_from_weights_populates_missing_fields() {
    let mut config = json!({});

    infer_llama_config_from_weights(&mut config, &test_weight_map());

    assert_eq!(config["hidden_size"], 256);
    assert_eq!(config["num_hidden_layers"], 3);
    assert_eq!(config["intermediate_size"], 512);
    assert_eq!(config["num_attention_heads"], 2);
}

#[test]
fn infer_llama_config_from_weights_preserves_existing_values() {
    let mut config = json!({
        "hidden_size": 1024,
        "num_hidden_layers": 8,
        "intermediate_size": 4096,
        "num_attention_heads": 16
    });

    infer_llama_config_from_weights(&mut config, &test_weight_map());

    assert_eq!(config["hidden_size"], 1024);
    assert_eq!(config["num_hidden_layers"], 8);
    assert_eq!(config["intermediate_size"], 4096);
    assert_eq!(config["num_attention_heads"], 16);
}

#[test]
fn rewrite_bunny_weight_key_rewrites_expected_prefixes() {
    assert_eq!(
        rewrite_bunny_weight_key("vision_tower.vision_tower.vision_model.encoder.layers.0"),
        Some("vision_tower.vision_model.encoder.layers.0".to_string())
    );
    assert_eq!(
        rewrite_bunny_weight_key("mm_projector.0.weight"),
        Some("multi_modal_projector.0.weight".to_string())
    );
    assert_eq!(
        rewrite_bunny_weight_key("model.lm_head.weight"),
        Some("lm_head.weight".to_string())
    );
    assert_eq!(
        rewrite_bunny_weight_key("model.layers.0.self_attn.q_proj"),
        None
    );
}

#[test]
fn detect_bunny_text_backend_prefers_nested_text_config_then_top_level_hint() {
    assert_eq!(
        detect_bunny_text_backend(&json!({
            "text_config": { "model_type": "qwen2" },
            "model_type": "llama"
        })),
        LlavaTextBackend::Qwen2
    );
    assert_eq!(
        detect_bunny_text_backend(&json!({
            "model_type": "bunny-qwen2"
        })),
        LlavaTextBackend::Qwen2
    );
    assert_eq!(
        detect_bunny_text_backend(&json!({
            "model_type": "bunny-llama"
        })),
        LlavaTextBackend::Llama
    );
}

#[test]
fn llava_text_backend_rejects_unknown_backends() {
    let err = llava_text_backend("gemma", "LLaVA").unwrap_err();
    assert!(err.to_string().contains("Unsupported LLaVA text backend"));
}

#[test]
fn parse_bunny_vision_config_uses_defaults_for_sparse_configs() {
    let config = parse_bunny_vision_config(
        &json!({
            "mm_hidden_size": 1408
        }),
        &json!({
            "intermediate_size": 5120
        }),
    );

    assert_eq!(config.model_type, "siglip_vision_model");
    assert_eq!(config.hidden_size, 1408);
    assert_eq!(config.intermediate_size, 5120);
    assert_eq!(config.patch_size, 14);
    assert_eq!(config.image_size, 384);
}

#[test]
fn inherit_text_quantization_if_missing_rejects_non_object_configs() {
    let mut text_config = json!("bad");
    let full_config = json!({
        "quantization": {"group_size": 128, "bits": 8}
    });

    let err = inherit_text_quantization_if_missing(&mut text_config, &full_config)
        .unwrap_err()
        .to_string();
    assert!(err.contains("LLaVA text_config"));
}

#[test]
fn parse_bunny_vision_config_preserves_full_configs() {
    let config = parse_bunny_vision_config(
        &json!({}),
        &json!({
            "model_type": "siglip_vision_model",
            "num_hidden_layers": 2,
            "hidden_size": 768,
            "intermediate_size": 1536,
            "num_attention_heads": 12,
            "patch_size": 16,
            "image_size": 224,
            "num_channels": 3,
            "layer_norm_eps": 1e-5
        }),
    );

    assert_eq!(config.hidden_size, 768);
    assert_eq!(config.intermediate_size, 1536);
    assert_eq!(config.patch_size, 16);
    assert_eq!(config.image_size, 224);
}

#[test]
fn host_weight_filter_keeps_no_decoder_layer_or_lm_head() {
    for key in [
        "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
        "multi_modal_projector.linear_1.weight",
        "model.embed_tokens.weight",
        "model.embed_tokens.scales",
        "language_model.model.embed_tokens.biases",
    ] {
        assert!(is_llava_host_preprocessor_weight(key), "{key}");
    }
    for key in [
        "model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.31.mlp.down_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    ] {
        assert!(!is_llava_host_preprocessor_weight(key), "{key}");
    }
}

#[test]
fn host_family_validation_rejects_incompatible_processor_family() {
    let config: VLMConfig = serde_json::from_value(json!({
        "model_type": "qwen2_vl",
        "text_config": {"model_type": "qwen2", "hidden_size": 8},
        "vision_config": {
            "model_type": "qwen2_vl",
            "num_hidden_layers": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_attention_heads": 1,
            "patch_size": 2,
            "image_size": 4,
            "num_channels": 3
        }
    }))
    .unwrap();

    let error = require_llava_host_family(&config).unwrap_err();
    assert!(matches!(
        error,
        HostPreprocessorError::FamilyMismatch { .. }
    ));
}

#[test]
fn host_family_validation_accepts_floor_patch_grid_used_by_llava_interleave() {
    let config: VLMConfig = serde_json::from_value(json!({
        "model_type": "llava",
        "text_config": {"model_type": "qwen2", "hidden_size": 8},
        "vision_config": {
            "model_type": "siglip_vision_model",
            "num_hidden_layers": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_attention_heads": 1,
            "patch_size": 14,
            "image_size": 384,
            "num_channels": 3
        }
    }))
    .unwrap();

    require_llava_host_family(&config).unwrap();
}

#[test]
fn iree_vision_contract_processor_metadata_must_match_host_pixel_producer() {
    let model = tempfile::tempdir().unwrap();
    let path = model.path().join("preprocessor_config.json");
    std::fs::write(
        &path,
        serde_json::to_vec(&json!({
            "do_resize": true,
            "do_rescale": true,
            "do_normalize": true,
            "rescale_factor": 1.0 / 255.0,
            "size": {"height": 384, "width": 384},
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5]
        }))
        .unwrap(),
    )
    .unwrap();
    let processor = crate::vision::processors::siglip::SigLipProcessor::new(384);
    validate_iree_processor_metadata(model.path(), &processor).unwrap();

    let mut drifted: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    drifted["image_mean"] = json!([0.4, 0.5, 0.5]);
    std::fs::write(&path, serde_json::to_vec(&drifted).unwrap()).unwrap();
    let error = validate_iree_processor_metadata(model.path(), &processor).unwrap_err();
    assert!(error.to_string().contains("image_mean[0]"));
}
