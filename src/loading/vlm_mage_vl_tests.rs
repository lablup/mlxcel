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

//! Mage-VL loader gates: both weight layouts remap onto the same keys, the
//! decoder config is lifted and inherits quantization, and a vision-stripped
//! checkpoint loads text-only and refuses images (#1367 convention).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use mlxcel_core::generate::LanguageModel;
use serde_json::json;

use super::*;

#[test]
fn hub_and_converted_keys_remap_to_one_layout() {
    for (raw, expected) in [
        (
            "model.language_model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        ),
        ("language_model.model.norm.weight", "model.norm.weight"),
        ("lm_head.weight", "lm_head.weight"),
        ("language_model.lm_head.scales", "lm_head.scales"),
        (
            "model.visual.merger.mlp.0.weight",
            "vision_tower.merger.mlp.0.weight",
        ),
        (
            "vision_tower.layernorm_pre.bias",
            "vision_tower.layernorm_pre.bias",
        ),
    ] {
        assert_eq!(remap_mage_vl_weight_key(raw), expected, "{raw}");
    }
}

#[test]
fn text_config_lifts_rope_and_inherits_quantization() {
    let full = json!({
        "model_type": "mage_vl",
        "quantization": {"group_size": 64, "bits": 8, "mode": "affine"},
        "text_config": {
            "model_type": "qwen3",
            "hidden_size": 2560, "num_hidden_layers": 36, "intermediate_size": 9728,
            "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
            "rms_norm_eps": 1e-6, "vocab_size": 151936,
            "rope_parameters": {"rope_theta": 5000000, "rope_type": "default"},
            "tie_word_embeddings": false
        }
    });
    let text = mage_vl_text_config(&full).unwrap();
    let args: models::qwen3::ModelArgs = serde_json::from_value(text).unwrap();
    assert_eq!(args.rope_theta, 5_000_000.0);
    assert_eq!((args.group_size(), args.bits()), (64, 8));
    assert!(!args.tie_word_embeddings);

    let mut other = full.clone();
    other["text_config"]["model_type"] = json!("llama");
    let err = mage_vl_text_config(&other).unwrap_err().to_string();
    assert!(err.contains("llama"), "{err}");
    assert!(mage_vl_text_config(&json!({})).is_err());
}

fn temp_dir(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("mlxcel_mage_vl_{tag}_{nanos}_{seq}"));
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
            json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let header = serde_json::to_string(&header).unwrap();
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    file.write_all(&data).unwrap();
}

/// One-layer Qwen3 decoder (hidden 8, 2 heads of 4, untied head) under the
/// hub `model.language_model.` prefix, and no vision weights.
fn write_stripped_mage_vl(dir: &Path, top: serde_json::Value) {
    let mut config = json!({
        "model_type": "mage_vl",
        "eos_token_id": 151645,
        "text_config": {
            "model_type": "qwen3",
            "hidden_size": 8, "num_hidden_layers": 1, "intermediate_size": 16,
            "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 4,
            "rms_norm_eps": 1e-6, "vocab_size": 16,
            "rope_parameters": {"rope_theta": 5000000, "rope_type": "default"},
            "tie_word_embeddings": false
        }
    });
    for (k, v) in top.as_object().unwrap() {
        config[k] = v.clone();
    }
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let p = "model.language_model";
    let mut tensors: Vec<(String, Vec<i64>)> = vec![
        (format!("{p}.embed_tokens.weight"), vec![16, 8]),
        (format!("{p}.norm.weight"), vec![8]),
        ("lm_head.weight".to_string(), vec![16, 8]),
    ];
    let l = format!("{p}.layers.0");
    for (name, shape) in [
        ("input_layernorm.weight", vec![8]),
        ("post_attention_layernorm.weight", vec![8]),
        ("self_attn.q_proj.weight", vec![8, 8]),
        ("self_attn.k_proj.weight", vec![8, 8]),
        ("self_attn.v_proj.weight", vec![8, 8]),
        ("self_attn.o_proj.weight", vec![8, 8]),
        ("self_attn.q_norm.weight", vec![4]),
        ("self_attn.k_norm.weight", vec![4]),
        ("mlp.gate_proj.weight", vec![16, 8]),
        ("mlp.up_proj.weight", vec![16, 8]),
        ("mlp.down_proj.weight", vec![8, 16]),
    ] {
        tensors.push((format!("{l}.{name}"), shape));
    }
    write_safetensors_f32(&dir.join("model.safetensors"), &tensors);
}

fn assert_text_only(dir: &Path) {
    let loaded = load_mage_vl(dir).expect("stripped checkpoint must load");
    assert!(!loaded.has_vision_tower());
    assert!(loaded.is_vlm(), "the VLM runtime stays for text requests");
    let LoadedModel::MageVLM(model) = &loaded else {
        panic!("mage_vl must load as the MageVLM variant");
    };
    assert!(model.vision.is_none());
    assert_eq!(model.num_layers(), 1);
    assert!(model.eos_token_ids().contains(&151645));
    assert!(model.eos_token_ids().contains(&151643));
    assert_eq!(
        model.output_suppressed_token_ids(),
        vec![151652, 151653, 151655, 151656]
    );

    let image = image::DynamicImage::new_rgb8(64, 64);
    let mut tokens = vec![1, 2, 3];
    let err = crate::vlm_runtime::prepare_vlm_embeddings(
        &loaded,
        &mut tokens,
        "describe",
        &[image],
        |_: &str, _: bool| vec![1, 2, 3],
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
fn mage_vl_with_empty_vision_config_loads_text_only() {
    let dir = temp_dir("empty");
    write_stripped_mage_vl(&dir, json!({"vision_config": {}}));
    assert_text_only(&dir);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn mage_vl_with_language_model_only_loads_text_only() {
    let dir = temp_dir("lm_only");
    write_stripped_mage_vl(
        &dir,
        json!({"vision_config": {"hidden_size": 1024}, "language_model_only": true}),
    );
    assert_text_only(&dir);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn mage_vl_with_vision_config_but_no_vision_weights_loads_text_only() {
    let dir = temp_dir("no_weights");
    write_stripped_mage_vl(&dir, json!({"vision_config": {"hidden_size": 1024}}));
    assert_text_only(&dir);
    std::fs::remove_dir_all(dir).unwrap();
}
