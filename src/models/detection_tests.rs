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
fn deepseek_vl2_model_type_is_detected() {
    let model_dir = temp_path("deepseek_vl2");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "deepseek_vl_v2",
            "tile_tag": "2D",
            "global_view_pos": "head",
            "candidate_resolutions": [[384, 384]],
            "language_config": { "model_type": "deepseek_v2", "hidden_size": 2048 },
            "vision_config": { "model_type": "vision", "width": 1152, "layers": 27, "patch_size": 14 },
            "projector_config": { "model_type": "mlp_projector" }
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::DeepSeekVL2);

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

#[test]
fn idefics3_smolvlm_instruct_model_type_is_detected() {
    // SmolVLM-Instruct ships as an Idefics3 checkpoint: top-level
    // `model_type: "idefics3"` (`Idefics3ForConditionalGeneration`) with a Llama
    // `text_config` and a SigLIP-style `vision_config` that is itself tagged
    // `idefics3`. It must resolve to the SmolVLM runtime instead of erroring with
    // "Unsupported model type: idefics3". Config shape mirrors the released
    // HuggingFaceTB/SmolVLM-Instruct config.json.
    let model_dir = temp_path("smolvlm_instruct_idefics3");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "architectures": ["Idefics3ForConditionalGeneration"],
            "model_type": "idefics3",
            "image_token_id": 49153,
            "image_seq_len": 81,
            "scale_factor": 3,
            "tie_word_embeddings": false,
            "text_config": {
                "model_type": "llama",
                "hidden_size": 2048,
                "intermediate_size": 8192,
                "num_hidden_layers": 24,
                "num_attention_heads": 32,
                "num_key_value_heads": 32,
                "head_dim": 64,
                "rms_norm_eps": 1e-05,
                "rope_theta": 273768.0,
                "vocab_size": 49155,
                "tie_word_embeddings": false
            },
            "vision_config": {
                "model_type": "idefics3",
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "patch_size": 14,
                "image_size": 384
            },
            "vocab_size": 49155
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::SmolVLM);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn lfm2_vl_model_type_is_detected_both_spellings() {
    // LFM2-VL ships model_type "lfm2-vl" (hyphen); the underscore alias must also
    // resolve. Both map to the LFM2-VL runtime, not "Unsupported model type".
    for mt in ["lfm2-vl", "lfm2_vl"] {
        let model_dir = temp_path(&format!("lfm2_vl_{}", mt.replace('-', "_")));
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(
            model_dir.join("config.json"),
            format!(
                r#"{{
                    "model_type": "{mt}",
                    "image_token_index": 396,
                    "downsample_factor": 2,
                    "text_config": {{ "model_type": "lfm2", "hidden_size": 1024, "num_hidden_layers": 16 }},
                    "vision_config": {{ "model_type": "siglip2_vision_model", "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "patch_size": 16, "num_patches": 256 }}
                }}"#
            ),
        )
        .unwrap();
        let detected = super::detection::get_model_type(&model_dir).unwrap();
        assert_eq!(detected, ModelType::Lfm2VL, "spelling {mt}");
    }
}

#[test]
fn granite_vision_model_type_is_detected() {
    // MLX conversions ship `model_type: "granite_vision"`.
    let model_dir = temp_path("granite_vision");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "granite_vision",
            "image_token_index": 49155,
            "vision_feature_layer": [-24, -20, -12, -1],
            "text_config": {"model_type": "granite", "hidden_size": 2048},
            "vision_config": {"model_type": "siglip_vision_model", "num_hidden_layers": 27,
                "hidden_size": 1152, "intermediate_size": 4304, "num_attention_heads": 16,
                "patch_size": 14}
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::GraniteVisionVLM);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn llava_next_with_granite_text_routes_to_granite_vision() {
    // The original IBM checkpoint ships `llava_next` + a `granite` text config.
    let model_dir = temp_path("llava_next_granite");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "llava_next",
            "image_token_index": 49155,
            "text_config": {"model_type": "granite", "hidden_size": 2048},
            "vision_config": {"model_type": "siglip_vision_model", "num_hidden_layers": 27,
                "hidden_size": 1152, "intermediate_size": 4304, "num_attention_heads": 16,
                "patch_size": 14}
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::GraniteVisionVLM);

    fs::remove_dir_all(model_dir).unwrap();
}

#[test]
fn llava_next_without_granite_stays_llava() {
    // A vanilla LLaVA-Next (llama/mistral/qwen2 text) must still route to LLaVA.
    let model_dir = temp_path("llava_next_vanilla");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("config.json"),
        r#"{
            "model_type": "llava_next",
            "text_config": {"model_type": "llama", "hidden_size": 4096},
            "vision_config": {"model_type": "clip_vision_model"}
        }"#,
    )
    .unwrap();

    let detected = super::detection::get_model_type(&model_dir).unwrap();
    assert_eq!(detected, ModelType::LlavaVLM);

    fs::remove_dir_all(model_dir).unwrap();
}
