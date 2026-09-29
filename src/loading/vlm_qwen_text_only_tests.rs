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

//! Loader-level tests for vision-stripped Qwen-VL checkpoints (#1367).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::LoadedModel;
use crate::multimodal::qwen_vl::QwenVlRuntime;

fn temp_dir(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("mlxcel_qwen_text_only_{tag}_{nanos}_{seq}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_safetensors_f32(path: &Path, tensors: &[(String, Vec<i64>)]) {
    let mut data: Vec<u8> = Vec::new();
    let mut header = serde_json::Map::new();
    for (name, shape) in tensors {
        let start = data.len();
        let count: i64 = shape.iter().product();
        for i in 0..count {
            data.extend_from_slice(&(0.01_f32 * ((i % 7) as f32 + 1.0)).to_le_bytes());
        }
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let header = serde_json::to_string(&header).unwrap();
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    file.write_all(&data).unwrap();
}

/// One-layer Qwen3-VL text decoder (hidden 8, 2 heads of 4, tied embeddings)
/// under the mlx-community `language_model.` prefix, and no `vision_tower.*`.
fn write_stripped_qwen3_vl(dir: &Path, vision_config: Option<serde_json::Value>) {
    let mut config = serde_json::json!({
        "model_type": "qwen3_vl",
        "text_config": {
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "vocab_size": 16,
            "tie_word_embeddings": true
        }
    });
    if let Some(vision) = vision_config {
        config["vision_config"] = vision;
    }
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let p = "language_model.model";
    let tensors: Vec<(String, Vec<i64>)> = vec![
        (format!("{p}.embed_tokens.weight"), vec![16, 8]),
        (format!("{p}.norm.weight"), vec![8]),
        (format!("{p}.layers.0.input_layernorm.weight"), vec![8]),
        (
            format!("{p}.layers.0.post_attention_layernorm.weight"),
            vec![8],
        ),
        (format!("{p}.layers.0.self_attn.q_proj.weight"), vec![8, 8]),
        (format!("{p}.layers.0.self_attn.k_proj.weight"), vec![8, 8]),
        (format!("{p}.layers.0.self_attn.v_proj.weight"), vec![8, 8]),
        (format!("{p}.layers.0.self_attn.o_proj.weight"), vec![8, 8]),
        (format!("{p}.layers.0.self_attn.q_norm.weight"), vec![4]),
        (format!("{p}.layers.0.self_attn.k_norm.weight"), vec![4]),
        (format!("{p}.layers.0.mlp.gate_proj.weight"), vec![16, 8]),
        (format!("{p}.layers.0.mlp.up_proj.weight"), vec![16, 8]),
        (format!("{p}.layers.0.mlp.down_proj.weight"), vec![8, 16]),
    ];
    write_safetensors_f32(&dir.join("model.safetensors"), &tensors);
}

fn assert_text_only(loaded: &LoadedModel, dir: &Path) {
    let LoadedModel::Qwen3VL(model) = loaded else {
        panic!("qwen3_vl must load as the Qwen3VL variant");
    };
    assert!(model.vision_encoder.is_none());
    assert!(!loaded.has_vision_tower());
    assert!(loaded.is_vlm(), "the VLM runtime stays for text requests");
    assert_eq!(
        model.text_only_source(),
        Some(dir.display().to_string().as_str())
    );

    let image = image::DynamicImage::new_rgb8(64, 64);
    let mut tokens = vec![1, 2, 3];
    let err = crate::vlm_runtime::compute_qwen_vl_media_embeddings(
        model,
        &mut tokens,
        &[image],
        &[],
        None,
        None,
    )
    .err()
    .expect("an image request against a text-only model must be refused");
    let message = err.to_string();
    assert!(
        message.contains("was loaded without a vision tower (text-only checkpoint)")
            && message.contains(&dir.display().to_string()),
        "got: {message}"
    );
    assert_eq!(tokens, vec![1, 2, 3], "the prompt must be left untouched");
}

#[test]
fn qwen3_vl_without_vision_config_loads_text_only() {
    let dir = temp_dir("absent");
    write_stripped_qwen3_vl(&dir, None);
    let loaded = super::qwen::load_qwen3_vl(&dir).expect("stripped checkpoint must load");
    assert_text_only(&loaded, &dir);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn qwen3_vl_with_empty_vision_config_loads_text_only() {
    let dir = temp_dir("empty");
    write_stripped_qwen3_vl(&dir, Some(serde_json::json!({})));
    let loaded = super::qwen::load_qwen3_vl(&dir).expect("stripped checkpoint must load");
    assert_text_only(&loaded, &dir);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn qwen3_vl_with_vision_config_but_no_vision_weights_loads_text_only() {
    let dir = temp_dir("no_weights");
    write_stripped_qwen3_vl(
        &dir,
        Some(serde_json::json!({ "hidden_size": 8, "depth": 1 })),
    );
    let loaded = super::qwen::load_qwen3_vl(&dir).expect("stripped checkpoint must load");
    assert_text_only(&loaded, &dir);
    std::fs::remove_dir_all(dir).unwrap();
}
