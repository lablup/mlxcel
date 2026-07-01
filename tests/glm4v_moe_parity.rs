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

//! GLM-4V MoE (`glm4v_moe`) parity / smoke tests.
//!
//! Config-level tests run in CI without any checkpoint; the forward-pass tests
//! are gated on a real GLM-4V MoE model under the shared model directory and
//! `eprintln!` + return when absent (inert in CI). GLM-4V MoE reuses the
//! GLM-4V ViT vision encoder and its config, so the vision-config assertions
//! mirror the GLM-4V ones.
//!
//! Run the gated portion with a model present via:
//! ```text
//! cargo test --release --test glm4v_moe_parity
//! ```

mod common;

use std::path::PathBuf;

use common::repo_model_dir;
use mlxcel::models::{ModelType, get_model_type};
use mlxcel::vision::encoders::glm4v::Glm4vVisionConfig;

const MODEL_NAME: &str = "GLM-4.5V";

fn model_dir() -> Option<PathBuf> {
    let dir = repo_model_dir(MODEL_NAME);
    if dir.exists() {
        Some(dir)
    } else {
        eprintln!("Skipping GLM-4V MoE test: model directory not found at {dir:?}");
        None
    }
}

#[test]
fn detects_glm4v_moe_model_type_from_config() {
    let tmp = std::env::temp_dir().join(format!("mlxcel-glm4vmoe-detect-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("create temp dir");
    let config = r#"{
        "model_type": "glm4v_moe",
        "image_token_id": 151363,
        "video_token_id": 151364,
        "vision_start_token_id": 151339,
        "text_config": {"model_type": "glm4v_moe_text", "hidden_size": 4096,
            "n_routed_experts": 128, "num_experts_per_tok": 8},
        "vision_config": {"model_type": "glm4v_moe", "depth": 24, "hidden_size": 1536,
            "intermediate_size": 13696, "num_heads": 12, "patch_size": 14}
    }"#;
    std::fs::write(tmp.join("config.json"), config).expect("write config.json");

    let model_type = get_model_type(&tmp).expect("detect model type");
    assert_eq!(model_type, ModelType::Glm4vMoe);

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn parses_shared_vision_config() {
    // GLM-4V MoE reuses the GLM-4V vision encoder/config.
    let json = r#"{
        "model_type": "glm4v_moe",
        "depth": 24,
        "hidden_size": 1536,
        "intermediate_size": 13696,
        "num_heads": 12,
        "patch_size": 14,
        "out_hidden_size": 4096,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2
    }"#;
    let config: Glm4vVisionConfig = serde_json::from_str(json).expect("parse vision config");
    assert_eq!(config.depth, 24);
    assert_eq!(config.out_hidden_size, 4096);
    assert_eq!(config.spatial_merge_size, 2);
    assert_eq!(config.hidden_size % config.num_heads, 0);
}

#[test]
fn detects_and_loads_real_model_as_vlm() {
    let Some(dir) = model_dir() else { return };

    let model_type = get_model_type(&dir).expect("detect model type");
    assert_eq!(model_type, ModelType::Glm4vMoe);

    let (model, _tokenizer) = mlxcel::load_model(&dir).expect("load GLM-4V MoE");
    assert!(model.is_vlm(), "GLM-4V MoE must register as a VLM");
}

#[test]
fn text_only_forward_produces_finite_logits() {
    let Some(dir) = model_dir() else { return };

    let (model, tokenizer) = mlxcel::load_model(&dir).expect("load GLM-4V MoE");

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
