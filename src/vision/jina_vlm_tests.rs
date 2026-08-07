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

//! Unit tests for the Jina VLM runtime: the image-feature scatter and the
//! cache-layout declaration the server path depends on.

use super::JinaVlmModel;
use crate::models::jina_vlm::tests::tiny_text_weights;
use crate::models::jina_vlm::{JinaVlmTextConfig, JinaVlmTextModel};
use crate::vision::encoders::jina_vlm::tests::build_tiny_vision_model;
use crate::vision::processors::jina_vlm::JinaVlmProcessor;
use mlxcel_core::MlxArray;
use mlxcel_core::cache::SequenceStateBackend;
use mlxcel_core::generate::LanguageModel;

const IMAGE_PROMPT_TOKEN_ID: i32 = 9;

fn tiny_text_config() -> JinaVlmTextConfig {
    JinaVlmTextConfig {
        hidden_size: 8,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        vocab_size: 12,
        additional_vocab_size: 4,
        intermediate_size: 6,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        use_qk_norm: true,
        tie_word_embeddings: false,
        quantization: None,
    }
}

fn build_model() -> JinaVlmModel {
    let text_config = tiny_text_config();
    let weights = tiny_text_weights(&text_config, "language_model");
    let text_model =
        JinaVlmTextModel::from_weights(&weights, &text_config, "language_model", vec![0])
            .expect("tiny text model builds");
    let (_vision_config, vision_model) = build_tiny_vision_model();

    JinaVlmModel {
        text_model,
        vision_model,
        processor: JinaVlmProcessor::default(),
        image_prompt_token_id: IMAGE_PROMPT_TOKEN_ID,
        always_start_with_space: true,
    }
}

fn to_vec_f32(a: &MlxArray) -> Vec<f32> {
    let f = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&f);
    mlxcel_core::array_to_raw_bytes(&f)
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

/// One crop's worth of synthetic pixels for the tiny 4x4 / patch-2 tower.
fn tiny_pixels(
    crops: i32,
) -> (
    mlxcel_core::UniquePtr<MlxArray>,
    mlxcel_core::UniquePtr<MlxArray>,
) {
    let patches = 4i32; // (4 / 2)^2
    let patch_dim = 12i32; // 2 * 2 * 3
    let data: Vec<f32> = (0..(crops * patches * patch_dim))
        .map(|i| ((i as f32 * 0.13).sin()) * 0.5)
        .collect();
    let pixels = mlxcel_core::from_slice_f32(&data, &[crops, patches, patch_dim]);
    let masks =
        mlxcel_core::from_slice_f32(&vec![1.0; (crops * patches) as usize], &[crops, patches]);
    (pixels, masks)
}

#[test]
fn image_features_land_only_on_the_named_positions() {
    let model = build_model();
    let hidden = 8usize;
    let seq_len = 6i32;

    let tokens: Vec<i32> = vec![1, 2, 3, 4, 5, 6];
    let ids = mlxcel_core::from_slice_i32(&tokens, &[1, seq_len]);
    let (pixels, masks) = tiny_pixels(1);

    // The tiny tower pools 2x2 -> one token per crop, so exactly one target.
    let merged = model.get_input_embeddings(&ids, &pixels, &[3], &masks);

    let base = to_vec_f32(&model.text_model.embedding.forward(&ids));
    let got = to_vec_f32(&merged.inputs_embeds);
    assert_eq!(got.len(), base.len());

    for position in 0..seq_len as usize {
        let range = position * hidden..(position + 1) * hidden;
        let changed = base[range.clone()]
            .iter()
            .zip(got[range].iter())
            .any(|(a, b)| (a - b).abs() > 1e-5);
        assert_eq!(
            changed,
            position == 3,
            "position {position} changed={changed}, expected only position 3 to change"
        );
    }
}

#[test]
fn the_scatter_adds_the_connector_output_to_the_placeholder_embedding() {
    // Upstream adds `embed(<im_patch>)` to every feature and then assigns; that
    // equals adding the feature onto the placeholder row, which is what this
    // path does. Pin the arithmetic so a future switch to a masked overwrite
    // has to be deliberate.
    let model = build_model();
    let hidden = 8usize;
    let ids = mlxcel_core::from_slice_i32(&[1, 2, 3], &[1, 3]);
    let (pixels, masks) = tiny_pixels(1);

    let merged = to_vec_f32(
        &model
            .get_input_embeddings(&ids, &pixels, &[1], &masks)
            .inputs_embeds,
    );
    let base = to_vec_f32(&model.text_model.embedding.forward(&ids));

    let images = mlxcel_core::reshape(&pixels, &[1, 1, 4, 12]);
    let crop_masks = mlxcel_core::reshape(&masks, &[1, 1, 4]);
    let features = to_vec_f32(&model.vision_model.forward(&images, &crop_masks));

    for c in 0..hidden {
        let expected = base[hidden + c] + features[c];
        assert!(
            (merged[hidden + c] - expected).abs() < 1e-4,
            "channel {c}: merged {} != base + feature {}",
            merged[hidden + c],
            expected
        );
    }
}

#[test]
fn negative_and_out_of_range_targets_are_skipped() {
    let model = build_model();
    let ids = mlxcel_core::from_slice_i32(&[1, 2, 3], &[1, 3]);
    let (pixels, masks) = tiny_pixels(2);
    // -10000 is the processor's padding sentinel; 99 is past the sequence.
    let merged = to_vec_f32(
        &model
            .get_input_embeddings(&ids, &pixels, &[-10000, 99], &masks)
            .inputs_embeds,
    );
    let base = to_vec_f32(&model.text_model.embedding.forward(&ids));
    for (a, b) in base.iter().zip(merged.iter()) {
        assert!((a - b).abs() < 1e-6, "an out-of-range target was written");
    }
}

#[test]
fn the_runtime_declares_the_dense_kv_layout_the_decoder_uses() {
    // Regression guard for the Falcon-OCR failure mode: a wrapper that lets the
    // trait default infer a model-owned layout makes the server allocate an
    // empty external cache, and `forward` then runs zero decoder layers with no
    // panic and no log line.
    let model = build_model();
    let layout = LanguageModel::sequence_state_layout(&model);
    assert_eq!(layout.backend, SequenceStateBackend::DenseKvCache);
    assert_eq!(layout.num_layers, LanguageModel::num_layers(&model));
    assert!(LanguageModel::supports_batching(&model));
}

#[test]
fn structural_image_tokens_are_suppressed_from_the_output() {
    let model = build_model();
    let suppressed = LanguageModel::output_suppressed_token_ids(&model);
    let tokens = JinaVlmProcessor::default().tokens;
    for id in [
        tokens.image_start_id,
        tokens.image_end_id,
        tokens.image_patch_id,
        tokens.image_col_id,
        IMAGE_PROMPT_TOKEN_ID,
    ] {
        assert!(suppressed.contains(&id), "token {id} was not suppressed");
    }
}
