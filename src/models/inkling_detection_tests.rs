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

use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::json;

use super::{ModelType, get_model_type};

fn checkpoint(name: &str, with_vision_config: bool) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("mlxcel_inkling_{name}_{unique}"));
    fs::create_dir_all(&path).unwrap();
    let mut config = json!({
        "model_type": "inkling_mm_model",
        "text_config": {"model_type": "inkling"}
    });
    if with_vision_config {
        config["vision_config"] = json!({"model_type": "inkling_vision"});
    }
    fs::write(
        path.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    path
}

#[test]
fn inkling_requires_both_vision_config_and_indexed_visual_weights() {
    let path = checkpoint("indexed", true);
    fs::write(
        path.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({
            "weight_map": {
                "model.visual.layers.linear_0.weight": "model-00001-of-00002.safetensors",
                "model.llm.embed_tokens.weight": "model-00002-of-00002.safetensors"
            }
        }))
        .unwrap(),
    )
    .unwrap();
    assert_eq!(get_model_type(&path).unwrap(), ModelType::InklingVLM);
    fs::remove_dir_all(path).unwrap();

    let path = checkpoint("no_config", false);
    fs::write(
        path.join("model.safetensors.index.json"),
        r#"{"weight_map":{"model.visual.final_norm.weight":"model.safetensors"}}"#,
    )
    .unwrap();
    assert_eq!(get_model_type(&path).unwrap(), ModelType::Inkling);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn unindexed_safetensors_header_detects_vision_without_reading_tensor_data() {
    let path = checkpoint("header", true);
    let header = serde_json::to_vec(&json!({
        "model.visual.final_norm.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [0, 4]
        }
    }))
    .unwrap();
    let mut file = Vec::with_capacity(8 + header.len());
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(&header);
    fs::write(path.join("model.safetensors"), file).unwrap();
    assert_eq!(get_model_type(&path).unwrap(), ModelType::InklingVLM);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn vision_config_without_visual_weights_stays_text_only() {
    let path = checkpoint("no_weights", true);
    assert_eq!(get_model_type(&path).unwrap(), ModelType::Inkling);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn audio_weights_do_not_hide_or_fabricate_the_vlm_shell() {
    let path = checkpoint("audio_with_vision", true);
    let mut config: serde_json::Value =
        serde_json::from_slice(&fs::read(path.join("config.json")).unwrap()).unwrap();
    config["audio_config"] = json!({"model_type": "inkling_audio"});
    fs::write(
        path.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    fs::write(
        path.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({
            "weight_map": {
                "model.visual.final_norm.weight": "model-00001-of-00002.safetensors",
                "model.audio.encoder.weight": "model-00002-of-00002.safetensors"
            }
        }))
        .unwrap(),
    )
    .unwrap();
    assert_eq!(get_model_type(&path).unwrap(), ModelType::InklingVLM);
    fs::remove_dir_all(path).unwrap();

    let path = checkpoint("audio_without_vision", false);
    let mut config: serde_json::Value =
        serde_json::from_slice(&fs::read(path.join("config.json")).unwrap()).unwrap();
    config["audio_config"] = json!({"model_type": "inkling_audio"});
    fs::write(
        path.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    fs::write(
        path.join("model.safetensors.index.json"),
        r#"{"weight_map":{"model.audio.encoder.weight":"model.safetensors"}}"#,
    )
    .unwrap();
    assert_eq!(get_model_type(&path).unwrap(), ModelType::Inkling);
    fs::remove_dir_all(path).unwrap();
}
