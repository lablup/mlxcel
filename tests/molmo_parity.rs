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

//! Molmo v1 (`molmo-7b`) real-model parity / smoke tests.
//!
//! Gated on the presence of `models/molmo-7b`; each test `eprintln!`s and
//! returns when the checkpoint is absent, so they are inert in CI and on
//! machines without the model. With the model present they exercise:
//!
//! - model-type detection of `molmo`
//! - the multi-crop image processor's pixel / `image_input_idx` / mask shapes
//! - a text-only forward pass producing finite logits at the right vocab dim
//! - the image path (token expansion + additive merge + a forward pass)
//!
//! Run with the model present via:
//! ```text
//! cargo test --release --test molmo_parity
//! ```

mod common;

use std::path::PathBuf;

use common::repo_model_dir;
use mlxcel::models::{ModelType, get_model_type};
use mlxcel::vision::processors::molmo::{MolmoImageTokens, MolmoProcessor};

const MODEL_NAME: &str = "molmo-7b";
const MOLMO_VOCAB: i32 = 152064;
const PATCH_DIM: i32 = 14 * 14 * 3; // 588
const N_PATCHES: i32 = 24 * 24; // 576 patches per 336/14 crop

fn model_dir() -> Option<PathBuf> {
    let dir = repo_model_dir(MODEL_NAME);
    if dir.exists() {
        Some(dir)
    } else {
        eprintln!("Skipping Molmo test: model directory not found at {dir:?}");
        None
    }
}

fn fixture_image() -> image::DynamicImage {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/test_image.png");
    image::open(&path).expect("load tests/fixtures/test_image.png")
}

fn default_processor() -> MolmoProcessor {
    MolmoProcessor::new(
        12,
        Some((4, 4)),
        Some(14),
        Some((336, 336)),
        Some((12, 12)),
        None,
        None,
        MolmoImageTokens::default(),
    )
}

#[test]
fn detects_molmo_model_type() {
    let Some(dir) = model_dir() else { return };
    let model_type = get_model_type(&dir).expect("detect model type");
    assert_eq!(model_type, ModelType::MolmoVLM);
}

#[test]
fn processor_produces_consistent_crop_and_token_shapes() {
    // The 224x224 fixture picks a 1x1 high-res tiling -> 1 crop + 1 global crop.
    let proc = default_processor();
    let img = fixture_image();
    let out = proc.preprocess_image(&img);

    // pixel_values: [n_crops, n_patches, patch_dim]; at least the global + one crop.
    assert_eq!(out.pixel_values_shape[1], N_PATCHES);
    assert_eq!(out.pixel_values_shape[2], PATCH_DIM);
    let n_crops = out.pixel_values_shape[0];
    assert!(
        n_crops >= 2,
        "expected global + >=1 hi-res crop, got {n_crops}"
    );
    assert_eq!(
        out.pixel_values.len() as i32,
        n_crops * N_PATCHES * PATCH_DIM
    );

    // image_masks: [n_crops, n_patches].
    assert_eq!(out.image_masks_shape, [n_crops, N_PATCHES]);
    assert_eq!(out.image_masks.len() as i32, n_crops * N_PATCHES);

    // image_token_ids contain at least one <im_patch> and bracket tokens.
    let t = MolmoImageTokens::default();
    let n_patch_tokens = out
        .image_token_ids
        .iter()
        .filter(|&&id| id == t.image_patch_id)
        .count();
    assert!(n_patch_tokens > 0, "no <im_patch> tokens emitted");
    assert!(out.image_token_ids.contains(&t.image_start_id));
    assert!(out.image_token_ids.contains(&t.image_end_id));

    // image_input_idx length matches num_image * tokens_per_image (144 per image).
    assert_eq!(out.image_input_idx_len, out.image_input_idx.len() as i32);
    assert_eq!(out.image_input_idx_len % 144, 0);

    // Every non-negative image_input_idx points at an <im_patch> token slot.
    for &pos in &out.image_input_idx {
        if pos >= 0 {
            let id = out.image_token_ids[pos as usize];
            assert_eq!(id, t.image_patch_id, "image_input_idx must hit <im_patch>");
        }
    }
}

#[test]
fn text_only_forward_produces_finite_logits() {
    let Some(dir) = model_dir() else { return };

    let (model, tokenizer) = mlxcel::load_model(&dir).expect("load molmo-7b");
    assert!(model.is_vlm(), "Molmo must register as a VLM");

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
    assert_eq!(
        *shape.last().unwrap(),
        MOLMO_VOCAB,
        "logits last dim must equal the Molmo vocab size"
    );

    let last_pos = shape[1] - 1;
    let row = mlxcel_core::slice(&logits, &[0, last_pos, 0], &[1, last_pos + 1, MOLMO_VOCAB]);
    let max = mlxcel_core::max_all(&row);
    mlxcel_core::eval(&max);
    assert!(
        mlxcel_core::item_f32(&max).is_finite(),
        "text-only logits must be finite"
    );
}

#[test]
fn image_forward_produces_finite_logits() {
    let Some(dir) = model_dir() else { return };

    let (model, tokenizer) = mlxcel::load_model(&dir).expect("load molmo-7b");

    let images = vec![fixture_image()];
    let mut prompt_tokens: Vec<i32> = tokenizer
        .encode("Describe this image.", true)
        .expect("tokenize")
        .iter()
        .map(|&t| t as i32)
        .collect();

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
    .expect("Molmo should produce embeddings for an image request");

    let input_ids = mlxcel_core::from_slice_i32(&prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let mut caches = mlxcel_core::generate::LanguageModel::make_caches(&model);
    let logits = mlxcel_core::generate::LanguageModel::forward_with_embeddings(
        &model,
        &input_ids,
        prepared.embeddings.inputs_embeds.as_ref(),
        &mut caches,
        None,
    );
    mlxcel_core::eval(&logits);

    let shape = mlxcel_core::array_shape(&logits);
    assert_eq!(*shape.last().unwrap(), MOLMO_VOCAB);

    let last_pos = shape[1] - 1;
    let row = mlxcel_core::slice(&logits, &[0, last_pos, 0], &[1, last_pos + 1, MOLMO_VOCAB]);
    let max = mlxcel_core::max_all(&row);
    mlxcel_core::eval(&max);
    assert!(
        mlxcel_core::item_f32(&max).is_finite(),
        "image-conditioned logits must be finite"
    );
}
