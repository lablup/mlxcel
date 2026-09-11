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

use crate::model_metadata::{
    DirectoryLoadRoute, DirectoryRouteFamily, ModelKind, WeightLoadRoute,
    adapter_loading_unsupported_message, has_config_backed_registration,
    is_config_backed_model_type, is_nonstandard_model_type, is_special_weight_model_type,
    is_vlm_model_type, model_load_policy, static_model_descriptor,
};
use crate::models::ModelType;
use serde_json::json;

#[test]
fn model_metadata_distinguishes_text_and_vlm_routes() {
    let text_policy = model_load_policy(ModelType::Qwen35, None).unwrap();
    assert_eq!(text_policy.capabilities.kind, ModelKind::Text);
    assert_eq!(text_policy.directory_route, DirectoryLoadRoute::Nonstandard);
    assert_eq!(text_policy.weight_route, Some(WeightLoadRoute::Special));

    let vlm_policy = model_load_policy(ModelType::Qwen3VL, None).unwrap();
    assert_eq!(vlm_policy.capabilities.kind, ModelKind::Vlm);
    assert_eq!(vlm_policy.directory_route, DirectoryLoadRoute::Vlm);
    assert_eq!(vlm_policy.weight_route, None);

    let phi4_siglip_policy = model_load_policy(ModelType::Phi4SigLipVLM, None).unwrap();
    assert_eq!(phi4_siglip_policy.capabilities.kind, ModelKind::Vlm);
    assert_eq!(phi4_siglip_policy.directory_route, DirectoryLoadRoute::Vlm);
    assert_eq!(phi4_siglip_policy.weight_route, None);

    let phi4mm_policy = model_load_policy(ModelType::Phi4MMVLM, None).unwrap();
    assert_eq!(phi4mm_policy.capabilities.kind, ModelKind::Vlm);
    assert_eq!(phi4mm_policy.directory_route, DirectoryLoadRoute::Vlm);
    assert_eq!(phi4mm_policy.weight_route, None);

    let minicpmo_policy = model_load_policy(ModelType::MiniCPMOVLM, None).unwrap();
    assert_eq!(minicpmo_policy.capabilities.kind, ModelKind::Vlm);
    assert_eq!(minicpmo_policy.directory_route, DirectoryLoadRoute::Vlm);
    assert_eq!(minicpmo_policy.weight_route, None);

    let moondream3_policy = model_load_policy(ModelType::Moondream3VLM, None).unwrap();
    assert_eq!(moondream3_policy.capabilities.kind, ModelKind::Vlm);
    assert_eq!(moondream3_policy.directory_route, DirectoryLoadRoute::Vlm);
    assert_eq!(moondream3_policy.weight_route, None);
}

#[test]
fn model_metadata_preserves_mistral3_nested_text_wrapper_rule() {
    let config = json!({
        "text_config": {
            "model_type": "ministral3"
        }
    });

    let policy = model_load_policy(ModelType::Mistral3, Some(&config)).unwrap();
    assert_eq!(
        policy.directory_route,
        DirectoryLoadRoute::Mistral3TextWrapper
    );
    assert_eq!(policy.weight_route, Some(WeightLoadRoute::LlamaFamily));
}

#[test]
fn model_metadata_preserves_mistral3_mistral4_text_wrapper_rule() {
    let config = json!({
        "text_config": {
            "model_type": "mistral4"
        }
    });

    let policy = model_load_policy(ModelType::Mistral3, Some(&config)).unwrap();
    assert_eq!(
        policy.directory_route,
        DirectoryLoadRoute::Mistral3Mistral4Wrapper
    );
    assert_eq!(policy.weight_route, Some(WeightLoadRoute::LlamaFamily));
}

#[test]
fn model_metadata_mistral4_standalone_is_config_backed() {
    assert!(is_config_backed_model_type(ModelType::Mistral4));
    assert!(has_config_backed_registration(ModelType::Mistral4));
}

#[test]
fn is_vlm_model_type_matches_control_plane_capabilities() {
    assert!(is_vlm_model_type(ModelType::Gemma3VLM));
    assert!(is_vlm_model_type(ModelType::Gemma4VLM));
    assert!(is_vlm_model_type(ModelType::Phi4MMVLM));
    assert!(is_vlm_model_type(ModelType::Phi4SigLipVLM));
    assert!(is_vlm_model_type(ModelType::MiniCPMOVLM));
    assert!(is_vlm_model_type(ModelType::Moondream3VLM));
    assert!(is_vlm_model_type(ModelType::Phi3VLM));
    assert!(!is_vlm_model_type(ModelType::Gemma3));
    assert!(!is_vlm_model_type(ModelType::Gemma4));
    assert!(!is_vlm_model_type(ModelType::Mamba2));
}

#[test]
fn static_model_descriptor_centralizes_directory_and_weight_families() {
    let llama = static_model_descriptor(ModelType::Llama);
    assert_eq!(llama.kind, ModelKind::Text);
    assert_eq!(llama.directory_family, DirectoryRouteFamily::ConfigBacked);
    assert_eq!(
        llama.adapter_weight_route,
        Some(WeightLoadRoute::ConfigBacked)
    );

    let mistral3 = static_model_descriptor(ModelType::Mistral3);
    assert_eq!(
        mistral3.directory_family,
        DirectoryRouteFamily::Mistral3Dynamic
    );
    assert_eq!(
        mistral3.adapter_weight_route,
        Some(WeightLoadRoute::LlamaFamily)
    );

    let qwen_vl = static_model_descriptor(ModelType::Qwen3VL);
    assert_eq!(qwen_vl.kind, ModelKind::Vlm);
    assert_eq!(qwen_vl.directory_family, DirectoryRouteFamily::Vlm);
    assert_eq!(qwen_vl.adapter_weight_route, None);
}

#[test]
fn descriptor_backed_support_helpers_stay_in_sync() {
    assert!(is_config_backed_model_type(ModelType::Gemma3));
    assert!(is_config_backed_model_type(ModelType::Gemma4));
    assert!(has_config_backed_registration(ModelType::Gemma3));
    assert!(has_config_backed_registration(ModelType::Gemma4));
    assert!(is_nonstandard_model_type(ModelType::Gemma3n));
    assert!(is_special_weight_model_type(ModelType::Qwen35));
    assert!(!has_config_backed_registration(ModelType::Gemma3n));
    assert!(!has_config_backed_registration(ModelType::Qwen3VL));

    assert_eq!(
        adapter_loading_unsupported_message(ModelType::Gemma4VLM),
        Some("Gemma4 VLM cannot be loaded with LoRA adapters yet")
    );
    assert_eq!(
        adapter_loading_unsupported_message(ModelType::Phi3VLM),
        Some("Phi3V VLM does not support adapter loading; use load_model() instead")
    );
    assert_eq!(
        adapter_loading_unsupported_message(ModelType::Phi4MMVLM),
        Some("Phi4MM VLM does not support adapter loading; use load_model() instead")
    );
    assert_eq!(
        adapter_loading_unsupported_message(ModelType::Phi4SigLipVLM),
        Some("Phi4-SigLIP VLM does not support adapter loading; use load_model() instead")
    );
    assert_eq!(
        adapter_loading_unsupported_message(ModelType::MiniCPMOVLM),
        Some("MiniCPM-o VLM does not support adapter loading; use load_model() instead")
    );
    assert_eq!(
        adapter_loading_unsupported_message(ModelType::Moondream3VLM),
        Some("Moondream3 VLM does not support adapter loading; use load_model() instead")
    );
    assert_eq!(adapter_loading_unsupported_message(ModelType::Llama), None);
}
