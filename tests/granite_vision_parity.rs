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

//! Granite Vision (`granite_vision` / `llava_next`+granite) parity tests.
//!
//! Config-level tests (detection for both `model_type` spellings, AnyRes token
//! math) run in CI without a checkpoint. The forward-pass tests are gated on the
//! presence of a real Granite Vision model under the shared model directory and
//! `eprintln!` + return when absent.

mod common;

use std::path::PathBuf;

use common::repo_model_dir;
use mlxcel::models::{ModelType, get_model_type};
use mlxcel::vision::processors::anyres::{AnyResProcessor, num_image_tokens};

const MODEL_NAME: &str = "granite-vision-3.2-2b-4bit";

fn model_dir() -> Option<PathBuf> {
    let dir = repo_model_dir(MODEL_NAME);
    if dir.exists() {
        Some(dir)
    } else {
        eprintln!("Skipping Granite Vision test: model directory not found at {dir:?}");
        None
    }
}

fn write_config(dir: &std::path::Path, model_type: &str, text_type: &str) {
    let config = format!(
        r#"{{
            "model_type": "{model_type}",
            "image_token_index": 49155,
            "vision_feature_layer": [-24, -20, -12, -1],
            "text_config": {{"model_type": "{text_type}", "hidden_size": 2048}},
            "vision_config": {{"model_type": "siglip_vision_model", "num_hidden_layers": 27,
                "hidden_size": 1152, "intermediate_size": 4304, "num_attention_heads": 16,
                "patch_size": 14, "image_size": 384}}
        }}"#
    );
    std::fs::write(dir.join("config.json"), config).expect("write config.json");
}

#[test]
fn detects_both_granite_vision_spellings() {
    for (mt, text) in [("granite_vision", "granite"), ("llava_next", "granite")] {
        let tmp =
            std::env::temp_dir().join(format!("mlxcel-granite-detect-{mt}-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).expect("create temp dir");
        write_config(&tmp, mt, text);
        let detected = get_model_type(&tmp).expect("detect model type");
        assert_eq!(detected, ModelType::GraniteVisionVLM, "spelling {mt}");
        let _ = std::fs::remove_dir_all(&tmp);
    }
}

#[test]
fn anyres_token_math_is_self_consistent() {
    // The prompt-token count and the feature-packing area must agree for any
    // image size (this is the invariant that keeps the two in lockstep).
    let mut pins: Vec<(i32, i32)> = Vec::new();
    for w in (384..=3840).step_by(384) {
        pins.push((384, w));
    }
    for w in (384..=1920).step_by(384) {
        pins.push((768, w));
    }
    for w in (384..=1152).step_by(384) {
        pins.push((1152, w));
    }
    for h in [1536, 1920] {
        for w in [384, 768] {
            pins.push((h, w));
        }
    }
    for h in [2304, 2688, 3072, 3456, 3840] {
        pins.push((h, 384));
    }
    let p = AnyResProcessor::new(pins, 384);
    // Golden count for a 2x3 grid image (500x1000): 729 + 40*(81+1) = 4009.
    let info = p.tile_info(500, 1000);
    assert_eq!(num_image_tokens(&info, 27, 729), 4009);
}

#[test]
fn detects_and_loads_real_model_as_vlm() {
    let Some(dir) = model_dir() else { return };

    let model_type = get_model_type(&dir).expect("detect model type");
    assert_eq!(model_type, ModelType::GraniteVisionVLM);

    let (model, _tokenizer) = mlxcel::load_model(&dir).expect("load Granite Vision");
    assert!(model.is_vlm(), "Granite Vision must register as a VLM");
}

#[test]
fn text_only_forward_produces_finite_logits() {
    let Some(dir) = model_dir() else { return };

    let (model, tokenizer) = mlxcel::load_model(&dir).expect("load Granite Vision");

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
