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

//! Granite 4 Vision (`granite4_vision`) parity tests.
//!
//! Config-level detection runs in CI without a checkpoint. The forward-pass
//! tests are gated on a real Granite 4 Vision model under the shared model
//! directory and `eprintln!` + return when absent.

mod common;

use std::path::PathBuf;

use common::repo_model_dir;
use mlxcel::models::{ModelType, get_model_type};

const MODEL_NAME: &str = "granite-4.0-3b-vision-4bit";

fn model_dir() -> Option<PathBuf> {
    let dir = repo_model_dir(MODEL_NAME);
    if dir.exists() {
        Some(dir)
    } else {
        eprintln!("Skipping Granite 4 Vision test: model directory not found at {dir:?}");
        None
    }
}

#[test]
fn detects_granite4_vision_model_type() {
    let tmp = std::env::temp_dir().join(format!("mlxcel-granite4-detect-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("create temp dir");
    let config = r#"{
        "model_type": "granite4_vision",
        "image_token_index": 100352,
        "downsample_rate": "4/8",
        "deepstack_layer_map": [[-19, 9], [-13, 6], [-7, 3], [-1, 0]],
        "spatial_target_layers": [12, 15, 18, 21],
        "spatial_vision_layer": -1,
        "text_config": {"model_type": "granitemoehybrid", "hidden_size": 2560},
        "vision_config": {"model_type": "siglip_vision_model", "num_hidden_layers": 27,
            "hidden_size": 1152, "intermediate_size": 4304, "num_attention_heads": 16,
            "patch_size": 16, "image_size": 384}
    }"#;
    std::fs::write(tmp.join("config.json"), config).expect("write config.json");

    let model_type = get_model_type(&tmp).expect("detect model type");
    assert_eq!(model_type, ModelType::Granite4VisionVLM);

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn detects_and_loads_real_model_as_vlm() {
    let Some(dir) = model_dir() else { return };

    let model_type = get_model_type(&dir).expect("detect model type");
    assert_eq!(model_type, ModelType::Granite4VisionVLM);

    let (model, _tokenizer) = mlxcel::load_model(&dir).expect("load Granite 4 Vision");
    assert!(model.is_vlm(), "Granite 4 Vision must register as a VLM");
}

#[test]
fn text_only_forward_produces_finite_logits() {
    let Some(dir) = model_dir() else { return };

    let (model, tokenizer) = mlxcel::load_model(&dir).expect("load Granite 4 Vision");

    let tokens: Vec<i32> = tokenizer
        .encode("Hello, world.", true)
        .expect("tokenize")
        .iter()
        .map(|&t| t as i32)
        .collect();
    let input_ids = mlxcel_core::from_slice_i32(&tokens, &[1, tokens.len() as i32]);

    let mut caches = mlxcel_core::generate::LanguageModel::make_caches(&model);
    let logits =
        mlxcel_core::generate::LanguageModel::forward(&model, &input_ids, &mut caches, None);
    mlxcel_core::eval(&logits);

    let shape = mlxcel_core::array_shape(&logits);
    let vocab = *shape.last().unwrap();
    let last_pos = shape[1] - 1;
    let row = mlxcel_core::slice(&logits, &[0, last_pos, 0], &[1, last_pos + 1, vocab]);
    let max = mlxcel_core::max_all(&row);
    mlxcel_core::eval(&max);
    assert!(
        mlxcel_core::item_f32(&max).is_finite(),
        "text-only logits must be finite"
    );
}
