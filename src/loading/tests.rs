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
    Qwen35VlmKind, conv1d_weight_is_channel_last, conv2d_weight_is_channel_last, model_path_str,
    parse_eos_token_ids, qwen35_vlm_kind, read_eos_token_ids, require_qwen35_vlm_kind,
    resolve_model_dir,
};
use crate::model_metadata::{
    DirectoryLoadRoute, ModelCapabilities, ModelKind, ModelLoadPolicy, WeightLoadRoute,
    directory_load_route, is_ministral3_config, is_mistral4_config, is_vlm_model_type,
    model_capabilities, model_load_policy, static_model_descriptor, weight_load_route,
};
use crate::models::ModelType;
use serde_json::json;
#[cfg(unix)]
use std::ffi::OsString;
use std::fs;
#[cfg(unix)]
use std::os::unix::ffi::OsStringExt;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_path(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("mlxcel_loader_test_{name}_{nanos}"))
}

#[test]
fn parse_eos_token_ids_supports_single_number() {
    let config = json!({ "eos_token_id": 42 });
    assert_eq!(parse_eos_token_ids(&config), vec![42]);
}

#[test]
fn parse_eos_token_ids_supports_number_arrays() {
    let config = json!({ "eos_token_id": [1, 2, 3] });
    assert_eq!(parse_eos_token_ids(&config), vec![1, 2, 3]);
}

#[test]
fn parse_eos_token_ids_ignores_invalid_entries() {
    let config = json!({ "eos_token_id": [7, "bad", null, 9] });
    assert_eq!(parse_eos_token_ids(&config), vec![7, 9]);
}

#[test]
fn read_eos_token_ids_returns_empty_for_missing_file() {
    let missing_dir = temp_path("missing_generation_config");
    assert!(read_eos_token_ids(&missing_dir).is_empty());
}

#[test]
fn read_eos_token_ids_reads_generation_config() {
    let model_dir = temp_path("generation_config");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("generation_config.json"),
        r#"{ "eos_token_id": [10, 11] }"#,
    )
    .unwrap();

    assert_eq!(read_eos_token_ids(&model_dir), vec![10, 11]);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn resolve_model_dir_uses_parent_for_model_files() {
    let model_dir = temp_path("model_dir");
    fs::create_dir_all(&model_dir).unwrap();
    let model_file = model_dir.join("model.safetensors");
    fs::write(&model_file, b"").unwrap();

    assert_eq!(resolve_model_dir(&model_file), model_dir);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn resolve_model_dir_keeps_directory_paths() {
    let model_dir = temp_path("directory_passthrough");
    fs::create_dir_all(&model_dir).unwrap();

    assert_eq!(resolve_model_dir(&model_dir), model_dir);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn model_path_str_accepts_utf8_paths() {
    let model_dir = temp_path("utf8_path");
    fs::create_dir_all(&model_dir).unwrap();

    assert_eq!(
        model_path_str(&model_dir).unwrap(),
        model_dir.to_str().unwrap()
    );

    fs::remove_dir_all(model_dir).unwrap();
}

#[cfg(unix)]
#[test]
fn model_path_str_rejects_non_utf8_paths() {
    let path = PathBuf::from(OsString::from_vec(vec![0xff, b'm', b'o', b'd', b'e', b'l']));
    let err = model_path_str(&path).unwrap_err().to_string();
    assert!(err.contains("invalid UTF-8"));
}

#[test]
fn is_ministral3_config_detects_nested_model_type() {
    let config = json!({
        "text_config": {
            "model_type": "ministral3"
        }
    });

    assert!(is_ministral3_config(&config));
}

#[test]
fn is_ministral3_config_returns_false_without_matching_text_model() {
    let config = json!({
        "text_config": {
            "model_type": "llama"
        }
    });

    assert!(!is_ministral3_config(&config));
}

#[test]
fn qwen35_vlm_kind_matches_supported_model_types() {
    assert_eq!(
        qwen35_vlm_kind(ModelType::Qwen35VLM),
        Some(Qwen35VlmKind::Dense)
    );
    assert_eq!(
        qwen35_vlm_kind(ModelType::Qwen35MoeVLM),
        Some(Qwen35VlmKind::Moe)
    );
    assert_eq!(qwen35_vlm_kind(ModelType::Qwen35), None);
}

#[test]
fn require_qwen35_vlm_kind_rejects_non_vlm_variants() {
    let err = require_qwen35_vlm_kind(ModelType::Qwen35)
        .unwrap_err()
        .to_string();
    assert!(err.contains("Expected a Qwen3.5 VLM variant"));
}

#[test]
fn is_vlm_model_type_distinguishes_multimodal_variants() {
    assert!(is_vlm_model_type(ModelType::Qwen3VL));
    assert!(is_vlm_model_type(ModelType::LlavaVLM));
    assert!(is_vlm_model_type(ModelType::Gemma4VLM));
    assert!(!is_vlm_model_type(ModelType::Qwen35));
    assert!(!is_vlm_model_type(ModelType::Llama));
}

#[test]
fn model_capabilities_distinguish_kind_and_adapter_support() {
    assert_eq!(
        model_capabilities(ModelType::Llama),
        ModelCapabilities {
            kind: ModelKind::Text,
            adapter_unsupported_message: None,
        }
    );

    let qwen_vl = model_capabilities(ModelType::Qwen3VL);
    assert_eq!(qwen_vl.kind, ModelKind::Vlm);
    assert!(qwen_vl.adapter_unsupported_message.is_some());

    let gemma4 = model_capabilities(ModelType::Gemma4);
    assert_eq!(gemma4.kind, ModelKind::Text);
    assert_eq!(gemma4.adapter_unsupported_message, None);

    let gemma4_vlm = model_capabilities(ModelType::Gemma4VLM);
    assert_eq!(gemma4_vlm.kind, ModelKind::Vlm);
    assert!(gemma4_vlm.adapter_unsupported_message.is_some());
}

#[test]
fn directory_load_route_handles_mistral3_text_subtype() {
    let config = json!({
        "text_config": {
            "model_type": "ministral3"
        }
    });

    assert_eq!(
        directory_load_route(ModelType::Mistral3, Some(&config)).unwrap(),
        DirectoryLoadRoute::Mistral3TextWrapper
    );
}

#[test]
fn directory_load_route_handles_mistral3_llama_fallback() {
    let config = json!({
        "text_config": {
            "model_type": "llama"
        }
    });

    assert_eq!(
        directory_load_route(ModelType::Mistral3, Some(&config)).unwrap(),
        DirectoryLoadRoute::Mistral3LlamaFallback
    );
}

#[test]
fn directory_load_route_distinguishes_vlm_nonstandard_and_config_backed() {
    assert_eq!(
        directory_load_route(ModelType::Qwen3VL, None).unwrap(),
        DirectoryLoadRoute::Vlm
    );
    assert_eq!(
        directory_load_route(ModelType::Qwen35, None).unwrap(),
        DirectoryLoadRoute::Nonstandard
    );
    assert_eq!(
        directory_load_route(ModelType::Llama, None).unwrap(),
        DirectoryLoadRoute::ConfigBacked
    );
}

#[test]
fn weight_load_route_distinguishes_loader_strategies() {
    assert_eq!(
        weight_load_route(ModelType::Mistral3).unwrap(),
        WeightLoadRoute::LlamaFamily
    );
    assert_eq!(
        weight_load_route(ModelType::Gemma3n).unwrap(),
        WeightLoadRoute::Special
    );
    assert_eq!(
        weight_load_route(ModelType::Qwen3).unwrap(),
        WeightLoadRoute::ConfigBacked
    );
}

#[test]
fn model_load_policy_combines_routes_and_capabilities() {
    let text_policy = model_load_policy(ModelType::Qwen35, None).unwrap();
    assert_eq!(
        text_policy,
        ModelLoadPolicy {
            descriptor: static_model_descriptor(ModelType::Qwen35),
            capabilities: ModelCapabilities {
                kind: ModelKind::Text,
                adapter_unsupported_message: None,
            },
            directory_route: DirectoryLoadRoute::Nonstandard,
            weight_route: Some(WeightLoadRoute::Special),
        }
    );

    let vlm_policy = model_load_policy(ModelType::Qwen3VL, None).unwrap();
    assert_eq!(vlm_policy.capabilities.kind, ModelKind::Vlm);
    assert_eq!(vlm_policy.directory_route, DirectoryLoadRoute::Vlm);
    assert_eq!(vlm_policy.weight_route, None);
}

#[test]
fn model_load_policy_handles_mistral3_text_wrapper_config() {
    let config = json!({
        "text_config": {
            "model_type": "ministral3"
        }
    });

    let policy = model_load_policy(ModelType::Mistral3, Some(&config)).unwrap();
    assert_eq!(policy.capabilities.kind, ModelKind::Text);
    assert_eq!(
        policy.directory_route,
        DirectoryLoadRoute::Mistral3TextWrapper
    );
    assert_eq!(policy.weight_route, Some(WeightLoadRoute::LlamaFamily));
}

#[test]
fn is_mistral4_config_detects_nested_model_type() {
    let config = json!({
        "text_config": {
            "model_type": "mistral4"
        }
    });

    assert!(is_mistral4_config(&config));
}

#[test]
fn is_mistral4_config_returns_false_without_matching_text_model() {
    let config = json!({
        "text_config": {
            "model_type": "llama"
        }
    });

    assert!(!is_mistral4_config(&config));
}

#[test]
fn directory_load_route_handles_mistral3_mistral4_subtype() {
    let config = json!({
        "text_config": {
            "model_type": "mistral4"
        }
    });

    assert_eq!(
        directory_load_route(ModelType::Mistral3, Some(&config)).unwrap(),
        DirectoryLoadRoute::Mistral3Mistral4Wrapper
    );
}

#[test]
fn model_load_policy_handles_mistral3_mistral4_wrapper_config() {
    let config = json!({
        "text_config": {
            "model_type": "mistral4"
        }
    });

    let policy = model_load_policy(ModelType::Mistral3, Some(&config)).unwrap();
    assert_eq!(policy.capabilities.kind, ModelKind::Text);
    assert_eq!(
        policy.directory_route,
        DirectoryLoadRoute::Mistral3Mistral4Wrapper
    );
    assert_eq!(policy.weight_route, Some(WeightLoadRoute::LlamaFamily));
}

#[test]
fn conv2d_weight_is_channel_last_detects_mlx_layout() {
    // MLX channel-last [out, kH, kW, in]: out dominates and kernels are square.
    assert!(conv2d_weight_is_channel_last(&[128, 3, 3, 1]));
    assert!(conv2d_weight_is_channel_last(&[32, 3, 3, 128]));
    // PyTorch [out, in, kH, kW]: in breaks dim1 == dim2 (or out >= dim1).
    assert!(!conv2d_weight_is_channel_last(&[128, 1, 3, 3]));
    assert!(!conv2d_weight_is_channel_last(&[32, 128, 3, 3]));
    // Non-4D shapes are never treated as conv2d channel-last.
    assert!(!conv2d_weight_is_channel_last(&[128, 3, 3]));
    assert!(!conv2d_weight_is_channel_last(&[4, 4]));
}

#[test]
fn conv1d_weight_is_channel_last_detects_depthwise_mlx_layout() {
    // MLX depthwise [out, kW, in=1]: trailing in dim is 1.
    assert!(conv1d_weight_is_channel_last(&[1024, 5, 1]));
    // PyTorch depthwise [out, in=1, kW]: middle in dim is 1, kW > 1.
    assert!(!conv1d_weight_is_channel_last(&[1024, 1, 5]));
    // A regular conv1d [out, in, kW] with in, kW > 1 is treated as PyTorch.
    assert!(!conv1d_weight_is_channel_last(&[4, 8, 3]));
    // Non-3D shapes are never treated as depthwise conv1d channel-last.
    assert!(!conv1d_weight_is_channel_last(&[1024, 5, 1, 1]));
    assert!(!conv1d_weight_is_channel_last(&[4, 4]));
}
