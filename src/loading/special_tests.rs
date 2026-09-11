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

use super::{
    SpecialWeightLoaderKind, adapter_loading_unsupported_message, qwen35_text_config,
    special_weight_loader_kind,
};
use crate::models::ModelType;
use serde_json::json;

#[test]
fn qwen35_text_config_merges_top_level_quantization() {
    let config = json!({
        "text_config": { "hidden_size": 1024 },
        "quantization": { "group_size": 128 }
    });

    let merged = qwen35_text_config(&config).unwrap();
    assert_eq!(merged["hidden_size"], 1024);
    assert_eq!(merged["quantization"]["group_size"], 128);
}

#[test]
fn qwen35_text_config_preserves_nested_quantization() {
    let config = json!({
        "text_config": {
            "hidden_size": 1024,
            "quantization": { "group_size": 64 }
        },
        "quantization": { "group_size": 128 }
    });

    let merged = qwen35_text_config(&config).unwrap();
    assert_eq!(merged["quantization"]["group_size"], 64);
}

#[test]
fn adapter_loading_unsupported_message_groups_vlm_families() {
    assert_eq!(
        adapter_loading_unsupported_message(ModelType::Qwen35VLM),
        Some("Qwen3.5 VLM does not support adapter loading")
    );
    assert_eq!(
        adapter_loading_unsupported_message(ModelType::Qwen3VL),
        Some("Qwen VL models cannot be loaded with LoRA adapters yet")
    );
    assert_eq!(
        adapter_loading_unsupported_message(ModelType::Phi3VLM),
        Some("Phi3V VLM does not support adapter loading; use load_model() instead")
    );
}

#[test]
fn adapter_loading_unsupported_message_returns_none_for_text_models() {
    assert_eq!(adapter_loading_unsupported_message(ModelType::Llama), None);
    assert_eq!(adapter_loading_unsupported_message(ModelType::Qwen35), None);
    assert_eq!(
        adapter_loading_unsupported_message(ModelType::Gemma3n),
        None
    );
}

#[test]
fn special_weight_loader_kind_covers_special_families() {
    assert_eq!(
        special_weight_loader_kind(ModelType::Qwen35),
        Some(SpecialWeightLoaderKind::Qwen35)
    );
    assert_eq!(
        special_weight_loader_kind(ModelType::Mamba2),
        Some(SpecialWeightLoaderKind::OwnedConfig)
    );
    assert_eq!(
        special_weight_loader_kind(ModelType::LongcatFlashNgram),
        Some(SpecialWeightLoaderKind::Longcat)
    );
    assert_eq!(
        special_weight_loader_kind(ModelType::Rwkv7),
        Some(SpecialWeightLoaderKind::Rwkv7)
    );
}

#[test]
fn special_weight_loader_kind_returns_none_for_config_backed_models() {
    assert_eq!(special_weight_loader_kind(ModelType::Llama), None);
    assert_eq!(special_weight_loader_kind(ModelType::Qwen3), None);
    assert_eq!(special_weight_loader_kind(ModelType::Step3p5), None);
}
