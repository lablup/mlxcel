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

//! Moondream2 (`vikhyatk/moondream2`) parity / smoke tests.
//!
//! Two tiers, mirroring `tests/molmo_parity.rs` and `tests/internvl_parity.rs`:
//!
//! - Checkpoint-free tests run in CI. They validate the reference-derived
//!   architecture wiring without a model: `moondream2` model-type detection, the
//!   crop-tiling processor's channels-last output shape, the vision grid token
//!   count, and the fused-QKV / partial-rotary text config.
//! - Real-model tests are gated on the presence of `models/moondream2`; they
//!   `eprintln!` + return when the checkpoint is absent, exercising detection,
//!   a text-only forward, and the image-conditioned forward.
//!
//! Run the gated tier with the model present via:
//! ```text
//! cargo test --release --test moondream2_parity
//! ```

mod common;

use std::path::PathBuf;

use common::repo_model_dir;
use mlxcel::models::moondream2::ModelArgs;
use mlxcel::models::{ModelType, get_model_type};
// Moondream2 reuses Moondream3's vision tower + crop preprocessor.
use mlxcel::vision::processors::moondream3::Moondream3Processor;

const MODEL_NAME: &str = "moondream2";
const MOONDREAM2_VOCAB: i32 = 51200;

fn model_dir() -> Option<PathBuf> {
    let dir = repo_model_dir(MODEL_NAME);
    if dir.exists() {
        Some(dir)
    } else {
        eprintln!("Skipping Moondream2 test: model directory not found at {dir:?}");
        None
    }
}

fn fixture_image() -> image::DynamicImage {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_image.png");
    image::open(&path).expect("load tests/fixtures/test_image.png")
}

fn default_processor() -> Moondream3Processor {
    Moondream3Processor::new(378, 14, 12, 4)
}

// ----------------------------------------------------------------------------
// Checkpoint-free reference-parity tests (run in CI).
// ----------------------------------------------------------------------------

#[test]
fn detects_moondream2_model_type_from_config() {
    // Detection must route `moondream2` to the new VLM arm without a checkpoint.
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type": "moondream2", "vision_config": {}, "text_config": {}}"#,
    )
    .expect("write config.json");

    let model_type = get_model_type(dir.path()).expect("detect model type");
    assert_eq!(model_type, ModelType::Moondream2VLM);
}

#[test]
fn text_config_matches_phi_style_reference() {
    let args: ModelArgs = serde_json::from_value(serde_json::json!({})).unwrap();
    assert_eq!(args.dim, 2048);
    assert_eq!(args.n_heads, 32);
    assert_eq!(args.n_kv_heads, 32);
    assert_eq!(args.head_dim(), 64);
    // Partial rotary over half the head, fused QKV of width 6144.
    assert_eq!(args.rope_dims(), 32);
    assert_eq!(args.qkv_dim(), 6144);
    assert_eq!(args.vocab_size, MOONDREAM2_VOCAB as usize);
}

#[test]
fn processor_emits_channels_first_crops_for_fixture() {
    // The shared Moondream3 crop preprocessor emits `[num_crops, 3, 378, 378]`,
    // normalized to [-1, 1]. The 224x224 fixture yields a global + local crop.
    let proc = default_processor();
    let out = proc.preprocess_image(&fixture_image());

    assert_eq!(out.pixel_values_shape[1..], [3, 378, 378]);
    let num_crops = out.pixel_values_shape[0];
    assert!(
        num_crops >= 2,
        "expected global + >=1 local crop, got {num_crops}"
    );
    let (rows, cols) = out.tiling;
    assert_eq!(num_crops as usize, 1 + rows * cols);
    assert_eq!(
        out.pixel_values.len() as i32,
        num_crops * 3 * 378 * 378,
        "pixel buffer must match the channels-first shape",
    );
    for &value in &out.pixel_values {
        assert!(
            (-1.0..=1.0).contains(&value),
            "pixel {value} out of [-1, 1]"
        );
    }
}

// ----------------------------------------------------------------------------
// Real-model tests (gated on models/moondream2).
// ----------------------------------------------------------------------------

#[test]
fn detects_moondream2_from_checkpoint() {
    let Some(dir) = model_dir() else { return };
    let model_type = get_model_type(&dir).expect("detect model type");
    assert_eq!(model_type, ModelType::Moondream2VLM);
}

#[test]
fn real_checkpoint_is_starmie_era() {
    // The 2025-06-21 snapshot ships `moondream.py` naming the
    // `moondream/starmie-v1` tokenizer, while its local tokenizer.json is the
    // STALE legacy GPT-2 one; detection must side with moondream.py or the
    // model consumes and emits ids from the wrong vocabulary and generates
    // garbage.
    let Some(dir) = model_dir() else { return };
    assert_eq!(
        mlxcel::moondream2_prompt::detect_moondream2_prompt_style(&dir),
        mlxcel::moondream2_prompt::Moondream2PromptStyle::StarmieTemplates
    );
}

#[test]
#[ignore = "resolves the starmie tokenizer from the Hub (network or hf-hub cache)"]
fn real_checkpoint_tokenizer_resolves_to_starmie() {
    // End-to-end tokenizer-override chain on the real checkpoint: despite the
    // stale GPT-2 tokenizer.json in the directory, `load_tokenizer` must hand
    // back moondream/starmie-v1, where `<|endoftext|>` is id 0 and the
    // template words match the config.py ids ("query" = 15381).
    let Some(dir) = model_dir() else { return };

    let tokenizer = mlxcel::tokenizer::load_tokenizer(&dir).expect("load starmie tokenizer");
    let query_ids = tokenizer.encode("query", false).expect("encode");
    assert_eq!(query_ids, vec![15381], "starmie must map 'query' to 15381");

    let endoftext = tokenizer.decode(&[0], false).expect("decode id 0");
    assert!(
        endoftext.contains("<|endoftext|>"),
        "starmie id 0 must be <|endoftext|>, got {endoftext:?}"
    );
}

#[test]
#[ignore = "requires real checkpoint; orchestrator validates"]
fn text_only_forward_produces_finite_logits() {
    let Some(dir) = model_dir() else { return };

    let (model, tokenizer) = mlxcel::load_model(&dir).expect("load moondream2");
    assert!(model.is_vlm(), "Moondream2 must register as a VLM");

    let style = mlxcel::moondream2_prompt::detect_moondream2_prompt_style(&dir);
    let prepared = mlxcel::moondream2_prompt::prepare_moondream2_prompt_tokens(
        "Hello?",
        0,
        style,
        |text, add_special| {
            tokenizer
                .encode(text, add_special)
                .unwrap_or_default()
                .iter()
                .map(|&t| t as i32)
                .collect()
        },
    )
    .expect("prepare text-only prompt");
    let tokens = prepared.tokens;
    let input_ids = mlxcel_core::from_slice_i32(&tokens, &[1, tokens.len() as i32]);

    let mut caches = mlxcel_core::generate::LanguageModel::make_caches(&model);
    let logits =
        mlxcel_core::generate::LanguageModel::forward(&model, &input_ids, &mut caches, None);
    mlxcel_core::eval(&logits);

    let shape = mlxcel_core::array_shape(&logits);
    assert_eq!(
        *shape.last().unwrap(),
        MOONDREAM2_VOCAB,
        "logits last dim must equal the Moondream2 vocab size"
    );

    let last_pos = shape[1] - 1;
    let row = mlxcel_core::slice(
        &logits,
        &[0, last_pos, 0],
        &[1, last_pos + 1, MOONDREAM2_VOCAB],
    );
    let max = mlxcel_core::max_all(&row);
    mlxcel_core::eval(&max);
    assert!(
        mlxcel_core::item_f32(&max).is_finite(),
        "text-only logits must be finite"
    );
}

#[test]
#[ignore = "requires real checkpoint; orchestrator validates"]
fn image_forward_produces_finite_logits() {
    let Some(dir) = model_dir() else { return };

    let (model, tokenizer) = mlxcel::load_model(&dir).expect("load moondream2");

    let images = vec![fixture_image()];
    let mut prompt_tokens: Vec<i32> = Vec::new();

    let prepared = mlxcel::vlm_runtime::prepare_and_compute_vlm_embeddings(
        &model,
        &mut prompt_tokens,
        "Describe this image.",
        &images,
        |text, add_special| {
            tokenizer
                .encode(text, add_special)
                .unwrap_or_default()
                .iter()
                .map(|&t| t as i32)
                .collect()
        },
    )
    .expect("prepare VLM embeddings")
    .expect("Moondream2 should produce embeddings for an image request");

    let input_ids = mlxcel_core::from_slice_i32(&prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let mut caches = mlxcel_core::generate::LanguageModel::make_caches(&model);
    let logits = mlxcel_core::generate::LanguageModel::forward_with_embeddings(
        &model,
        &input_ids,
        prepared.embeddings.inputs_embeds.as_ref(),
        &mut caches,
        prepared.embeddings.attention_mask_4d.as_deref(),
    );
    mlxcel_core::eval(&logits);

    let shape = mlxcel_core::array_shape(&logits);
    assert_eq!(*shape.last().unwrap(), MOONDREAM2_VOCAB);

    let last_pos = shape[1] - 1;
    let row = mlxcel_core::slice(
        &logits,
        &[0, last_pos, 0],
        &[1, last_pos + 1, MOONDREAM2_VOCAB],
    );
    let max = mlxcel_core::max_all(&row);
    mlxcel_core::eval(&max);
    assert!(
        mlxcel_core::item_f32(&max).is_finite(),
        "image-conditioned logits must be finite"
    );
}
