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

//! GLM-OCR (`glm_ocr`) parity / smoke tests.
//!
//! The config-level tests run in CI without any checkpoint. The heavier
//! forward-pass tests are gated on the presence of a real GLM-OCR model under
//! the shared model directory; they `eprintln!` + return when absent, so they
//! are inert in CI and on machines without the checkpoint.
//!
//! Run the gated portion with a model present via:
//! ```text
//! cargo test --release --test glm_ocr_parity
//! ```

mod common;

use std::path::PathBuf;

use common::repo_model_dir;
use mlxcel::models::{ModelType, get_model_type};
use mlxcel::vision::encoders::glm4v::Glm4vVisionConfig;

const MODEL_NAME: &str = "glm-ocr-4bit";

fn model_dir() -> Option<PathBuf> {
    let dir = repo_model_dir(MODEL_NAME);
    if dir.exists() {
        Some(dir)
    } else {
        eprintln!("Skipping GLM-OCR test: model directory not found at {dir:?}");
        None
    }
}

#[test]
fn detects_glm_ocr_model_type_from_config() {
    let tmp = std::env::temp_dir().join(format!("mlxcel-glm-ocr-detect-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("create temp dir");
    let config = r#"{
        "model_type": "glm_ocr",
        "image_token_id": 59280,
        "video_token_id": 59281,
        "image_start_token_id": 59256,
        "image_end_token_id": 59257,
        "text_config": {"model_type": "glm_ocr_text", "hidden_size": 1536,
            "rope_parameters": {"mrope_section": [16, 24, 24], "partial_rotary_factor": 1.0,
                "rope_theta": 10000}},
        "vision_config": {"model_type": "glm_ocr_vision", "depth": 24, "hidden_size": 1024,
            "intermediate_size": 4096, "num_heads": 16, "patch_size": 14}
    }"#;
    std::fs::write(tmp.join("config.json"), config).expect("write config.json");

    let model_type = get_model_type(&tmp).expect("detect model type");
    assert_eq!(model_type, ModelType::GlmOcr);

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn parses_glm_ocr_vision_config() {
    // GLM-OCR reuses `Glm4vVisionConfig`; the only structural differences (q/k
    // norm, no position embed / post-conv norm) live in the encoder, not config.
    let json = r#"{
        "model_type": "glm_ocr_vision",
        "depth": 24,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_heads": 16,
        "patch_size": 14,
        "out_hidden_size": 1536,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "image_size": 336,
        "in_channels": 3,
        "rms_norm_eps": 1e-05
    }"#;
    let config: Glm4vVisionConfig = serde_json::from_str(json).expect("parse vision config");
    assert_eq!(config.depth, 24);
    assert_eq!(config.hidden_size, 1024);
    assert_eq!(config.num_heads, 16);
    assert_eq!(config.out_hidden_size, 1536);
    assert_eq!(config.rms_norm_eps, 1e-05);
    // head_dim = 1024 / 16 = 64, must be even for the vision RoPE.
    assert_eq!(config.hidden_size / config.num_heads, 64);
}

#[test]
fn detects_and_loads_real_model_as_vlm() {
    let Some(dir) = model_dir() else { return };

    let model_type = get_model_type(&dir).expect("detect model type");
    assert_eq!(model_type, ModelType::GlmOcr);

    let (model, _tokenizer) = mlxcel::load_model(&dir).expect("load GLM-OCR");
    assert!(model.is_vlm(), "GLM-OCR must register as a VLM");
}

#[test]
fn text_only_forward_produces_finite_logits() {
    let Some(dir) = model_dir() else { return };

    let (model, tokenizer) = mlxcel::load_model(&dir).expect("load GLM-OCR");

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
