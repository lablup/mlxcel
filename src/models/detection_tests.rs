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

use super::ModelType;
use super::detection::{detect_hunyuan_model_type, detect_text_or_vlm, has_vision_config};
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn has_vision_config_detects_vlm_configs() {
    assert!(has_vision_config(&json!({ "vision_config": {} })));
    assert!(!has_vision_config(&json!({ "text_config": {} })));
}

#[test]
fn detect_text_or_vlm_prefers_vlm_when_vision_config_exists() {
    let vlm = detect_text_or_vlm(
        &json!({ "vision_config": {} }),
        ModelType::Gemma3,
        ModelType::Gemma3VLM,
    );
    let text = detect_text_or_vlm(&json!({}), ModelType::Gemma3, ModelType::Gemma3VLM);

    assert_eq!(vlm, ModelType::Gemma3VLM);
    assert_eq!(text, ModelType::Gemma3);
}

#[test]
fn detect_hunyuan_model_type_uses_num_experts() {
    assert_eq!(
        detect_hunyuan_model_type(&json!({ "num_experts": 4 })),
        ModelType::HunyuanMoe
    );
    assert_eq!(
        detect_hunyuan_model_type(&json!({ "num_experts": 1 })),
        ModelType::HunyuanV1Dense
    );
    assert_eq!(
        detect_hunyuan_model_type(&json!({})),
        ModelType::HunyuanV1Dense
    );
}

fn temp_path(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("mlxcel_detection_test_{name}_{nanos}"))
}

#[test]
fn whisper_model_type_is_detected() {
    let model_dir = temp_path("whisper_asr");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "whisper",
            "num_mel_bins": 80,
            "d_model": 384,
            "encoder_attention_heads": 6,
            "encoder_layers": 4,
            "decoder_attention_heads": 6,
            "decoder_layers": 4,
            "vocab_size": 51865
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::Whisper);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn mllama_model_type_is_detected() {
    // Llama 3.2 Vision: a `mllama` checkpoint must resolve to the VLM route
    // instead of erroring with "Unsupported model type".
    let model_dir = temp_path("llama_3_2_vision");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "mllama",
            "image_token_index": 128256,
            "text_config": {
                "model_type": "mllama",
                "hidden_size": 4096,
                "num_hidden_layers": 40,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "cross_attention_layers": [3, 8, 13, 18, 23, 28, 33, 38]
            },
            "vision_config": {
                "image_size": 560,
                "patch_size": 14,
                "hidden_size": 1280,
                "num_hidden_layers": 32,
                "num_global_layers": 8
            }
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::MllamaVLM);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn mellum_model_type_is_detected() {
    let model_dir = temp_path("mellum_code");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "mellum",
            "architectures": ["MellumForCausalLM"],
            "hidden_size": 2304,
            "head_dim": 128,
            "num_hidden_layers": 28,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "num_experts": 64,
            "vocab_size": 98304
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::Mellum);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn gemma4_detection_stays_on_text_route_without_vision_weights() {
    let model_dir = temp_path("gemma4_text_route");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "gemma4",
            "vision_config": {},
            "text_config": { "model_type": "gemma4_text" }
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::Gemma4);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn gemma4_detection_uses_vlm_route_when_vision_weights_exist() {
    let model_dir = temp_path("gemma4_vlm_route");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "gemma4",
            "vision_config": {},
            "text_config": { "model_type": "gemma4_text" }
        }"#,
    )
    .unwrap();
    fs::write(
        model_dir.join("model.safetensors.index.json"),
        r#"{
            "weight_map": {
                "vision_tower.encoder.layers.0.input_layernorm.weight": "model-00001-of-00001.safetensors"
            }
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::Gemma4VLM);

    fs::remove_dir_all(model_dir).unwrap();
}
