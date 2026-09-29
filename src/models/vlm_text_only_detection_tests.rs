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

//! Text-only routing of vision-stripped VLM checkpoints (#1367).

use super::ModelType;
use super::detection::{get_model_type, vlm_has_vision};
use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("mlxcel_text_only_{name}_{nanos}_{seq}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_config(dir: &Path, config: &Value) {
    fs::write(dir.join("config.json"), config.to_string()).unwrap();
}

fn write_index(dir: &Path, keys: &[&str]) {
    let weight_map: serde_json::Map<String, Value> = keys
        .iter()
        .map(|key| {
            (
                (*key).to_string(),
                json!("model-00001-of-00001.safetensors"),
            )
        })
        .collect();
    fs::write(
        dir.join("model.safetensors.index.json"),
        json!({ "weight_map": weight_map }).to_string(),
    )
    .unwrap();
}

fn detect(config: Value, index_keys: Option<&[&str]>) -> ModelType {
    let dir = temp_dir("detect");
    write_config(&dir, &config);
    if let Some(keys) = index_keys {
        write_index(&dir, keys);
    }
    let detected = get_model_type(&dir).unwrap();
    fs::remove_dir_all(dir).unwrap();
    detected
}

const LM_KEYS: &[&str] = &["language_model.model.embed_tokens.weight"];

#[test]
fn empty_vision_config_routes_to_text() {
    let config = json!({ "model_type": "qwen3_5", "vision_config": {}, "text_config": {} });
    assert_eq!(detect(config, None), ModelType::Qwen35);
}

#[test]
fn null_vision_config_routes_to_text() {
    let config = json!({ "model_type": "qwen3_5_moe", "vision_config": null, "text_config": {} });
    assert_eq!(detect(config, None), ModelType::Qwen35Moe);
}

#[test]
fn language_model_only_routes_to_text() {
    let config = json!({
        "model_type": "qwen3_5",
        "language_model_only": true,
        "vision_config": { "depth": 2 },
        "text_config": {}
    });
    assert_eq!(detect(config.clone(), None), ModelType::Qwen35);
    // The flag wins even when the checkpoint ships vision weights.
    let with_tower = &["model.visual.blocks.0.attn.qkv.weight"];
    assert_eq!(detect(config, Some(with_tower)), ModelType::Qwen35);
}

#[test]
fn language_model_only_false_keeps_the_vlm_route() {
    let config = json!({
        "model_type": "qwen3_5",
        "language_model_only": false,
        "vision_config": { "depth": 2 },
        "text_config": {}
    });
    assert_eq!(detect(config, None), ModelType::Qwen35VLM);
}

#[test]
fn no_vision_weights_in_index_routes_to_text() {
    for (model_type, text) in [
        ("qwen3_5", ModelType::Qwen35),
        ("qwen3_5_moe", ModelType::Qwen35Moe),
        ("gemma3", ModelType::Gemma3),
    ] {
        let config = json!({
            "model_type": model_type,
            "vision_config": { "depth": 2 },
            "text_config": {}
        });
        assert_eq!(detect(config, Some(LM_KEYS)), text, "{model_type}");
    }
}

#[test]
fn every_vision_weight_prefix_keeps_the_vlm_route() {
    for prefix in [
        "vision_tower.",
        "model.visual.",
        "model.vision_tower.",
        "visual.",
        "vision_model.",
    ] {
        let key = format!("{prefix}blocks.0.weight");
        let config = json!({
            "model_type": "qwen3_5",
            "vision_config": { "depth": 2 },
            "text_config": {}
        });
        assert_eq!(
            detect(config, Some(&[LM_KEYS[0], key.as_str()])),
            ModelType::Qwen35VLM,
            "{prefix}"
        );
    }
}

#[test]
fn single_file_checkpoint_header_is_scanned_when_there_is_no_index() {
    let config = json!({ "model_type": "qwen3_5", "vision_config": { "depth": 2 } });
    for (tensor, expected) in [
        ("language_model.model.norm.weight", false),
        ("vision_tower.blocks.0.weight", true),
    ] {
        let dir = temp_dir("header");
        let header = json!({
            tensor: { "dtype": "F32", "shape": [1], "data_offsets": [0, 4] }
        })
        .to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0_u8; 4]);
        fs::write(dir.join("model.safetensors"), bytes).unwrap();
        assert_eq!(vlm_has_vision(&config, &dir), expected, "{tensor}");
        fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn unreadable_weights_keep_the_config_decision() {
    let dir = temp_dir("no_weights");
    let config = json!({ "model_type": "qwen3_5", "vision_config": { "depth": 2 } });
    assert!(vlm_has_vision(&config, &dir));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn qwen_vl_families_keep_their_model_type() {
    // These four load with vision_encoder None instead of a text ModelType,
    // because their decoders carry MRoPE and cannot reuse Qwen2 / Qwen3.
    for (model_type, expected) in [
        ("qwen2_vl", ModelType::Qwen2VL),
        ("qwen2_5_vl", ModelType::Qwen25VL),
        ("qwen3_vl", ModelType::Qwen3VL),
        ("qwen3_vl_moe", ModelType::Qwen3VLMoe),
    ] {
        let config = json!({ "model_type": model_type, "text_config": {} });
        assert_eq!(detect(config, Some(LM_KEYS)), expected, "{model_type}");
    }
}
