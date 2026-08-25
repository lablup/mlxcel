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

//! Unit tests for the embedding and sequence-classification heads over the
//! deterministic fixture checkpoint from [`super::super::bert::bert_tests`].

use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::{MlxArray, UniquePtr};

use super::{BertEmbeddingModel, BertSequenceClassifier, DENSE_QUANTIZATION};
use crate::embeddings::model::{EmbeddingBatch, EmbeddingModel, ImageInput};
use crate::embeddings::pooling::PoolingMode;
use crate::models::bert::BertVariant;
use crate::models::bert::bert_tests::{HIDDEN, cosine, mlx_test_guard, tiny_args, tiny_weights};

fn embedding_model(variant: BertVariant, pooling: PoolingMode) -> BertEmbeddingModel {
    let args = tiny_args(variant);
    let weights = tiny_weights(&args, false);
    BertEmbeddingModel::from_weights(&weights, args, DENSE_QUANTIZATION, pooling)
        .expect("fixture embedding model builds")
}

fn classifier(variant: BertVariant) -> BertSequenceClassifier {
    let args = tiny_args(variant);
    let weights = tiny_weights(&args, true);
    BertSequenceClassifier::from_weights(&weights, args, DENSE_QUANTIZATION)
        .expect("fixture classifier builds")
}

fn i32_array(values: &[i32], shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(values, shape)
}

/// Run one batch and read the `[B, D]` result back row by row.
fn embed_rows(model: &BertEmbeddingModel, ids: &[i32], mask: &[i32], batch: i32) -> Vec<Vec<f32>> {
    let width = ids.len() as i32 / batch;
    let input_ids = i32_array(ids, &[batch, width]);
    let attention_mask = i32_array(mask, &[batch, width]);
    let output = model
        .embed(&EmbeddingBatch {
            input_ids: &input_ids,
            attention_mask: &attention_mask,
            token_type_ids: None,
            images: None,
        })
        .expect("forward succeeds");
    mlxcel_core::try_eval(&output.embeddings).unwrap();
    assert_eq!(
        mlxcel_core::array_shape(&output.embeddings),
        vec![batch, HIDDEN as i32]
    );
    array_to_vec_f32(&output.embeddings)
        .chunks(HIDDEN)
        .map(<[f32]>::to_vec)
        .collect()
}

#[test]
fn identical_inputs_embed_identically() {
    let _guard = mlx_test_guard();
    for variant in [BertVariant::Bert, BertVariant::XlmRoberta] {
        let model = embedding_model(variant, PoolingMode::Mean);
        let rows = embed_rows(&model, &[5, 6, 7, 5, 6, 7], &[1, 1, 1, 1, 1, 1], 2);
        assert!(
            cosine(&rows[0], &rows[1]) > 1.0 - 1e-6,
            "{variant:?}: repeated input must give the same vector"
        );
    }
}

#[test]
fn padding_row_does_not_change_the_real_row_embedding() {
    let _guard = mlx_test_guard();
    for variant in [BertVariant::Bert, BertVariant::XlmRoberta] {
        for pooling in [PoolingMode::Mean, PoolingMode::Cls] {
            let model = embedding_model(variant, pooling);
            let pad = model.args().pad_token_id;
            let alone = embed_rows(&model, &[5, 6, 7], &[1, 1, 1], 1);
            let batched = embed_rows(
                &model,
                &[5, 6, 7, pad, pad, pad, 11, 12, 13, 14, 15, 16],
                &[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                2,
            );
            assert!(
                cosine(&alone[0], &batched[0]) > 1.0 - 1e-5,
                "{variant:?}/{pooling}: padding must not move the real row"
            );
        }
    }
}

#[test]
fn different_token_sequences_give_different_vectors() {
    let _guard = mlx_test_guard();
    // A forward pass that dropped the input (a constant or all-zero output)
    // would still pass the two tests above; this one would not. The bound is
    // on the largest component difference rather than on cosine: over random
    // weights two pooled vectors stay nearly parallel because the LayerNorm
    // bias dominates, so semantic separation is gated on the real checkpoints
    // in `bert_real_checkpoint_tests`, not here.
    for variant in [BertVariant::Bert, BertVariant::XlmRoberta] {
        let model = embedding_model(variant, PoolingMode::Mean);
        let rows = embed_rows(&model, &[3, 4, 5, 20, 21, 22], &[1, 1, 1, 1, 1, 1], 2);
        let spread = rows[0]
            .iter()
            .zip(&rows[1])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            spread > 1e-6,
            "{variant:?}: different inputs produced the same vector (max delta {spread})"
        );
    }
}

#[test]
fn embedding_model_reports_its_width_pooling_and_segment_needs() {
    let _guard = mlx_test_guard();
    let bert = embedding_model(BertVariant::Bert, PoolingMode::Mean);
    assert_eq!(bert.embedding_dim(), HIDDEN);
    assert_eq!(bert.default_pooling(), PoolingMode::Mean);
    assert!(bert.needs_token_type_ids());
    assert!(bert.normalize());
    assert!(!bert.multi_vector());
    assert!(!bert.supports_images());
    assert_eq!(bert.pad_to_max_length(), None);
    // BERT indexes the table from 0: all rows usable.
    assert_eq!(
        bert.max_sequence_length(),
        Some(bert.args().max_position_embeddings)
    );

    let xlmr = embedding_model(BertVariant::XlmRoberta, PoolingMode::Cls);
    assert!(!xlmr.needs_token_type_ids());
    assert_eq!(
        xlmr.max_sequence_length(),
        Some(xlmr.args().max_position_embeddings - 2)
    );
}

#[test]
fn embedding_model_rejects_image_inputs() {
    let _guard = mlx_test_guard();
    let model = embedding_model(BertVariant::Bert, PoolingMode::Mean);
    let input_ids = i32_array(&[5, 6], &[1, 2]);
    let attention_mask = i32_array(&[1, 1], &[1, 2]);
    let images = [ImageInput {
        image: image::DynamicImage::new_rgb8(1, 1),
    }];
    let err = match model.embed(&EmbeddingBatch {
        input_ids: &input_ids,
        attention_mask: &attention_mask,
        token_type_ids: None,
        images: Some(&images),
    }) {
        Ok(_) => panic!("a text-only family must refuse images"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("image"), "{err}");
}

#[test]
fn classifier_head_produces_one_logit_row_per_input() {
    let _guard = mlx_test_guard();
    for variant in [BertVariant::Bert, BertVariant::XlmRoberta] {
        let model = classifier(variant);
        assert_eq!(model.args().num_labels, 1);
        assert_eq!(
            model.needs_token_type_ids(),
            variant == BertVariant::Bert,
            "{variant:?}"
        );
        let pad = model.args().pad_token_id;
        let input_ids = i32_array(&[5, 6, 7, 8, 9, pad], &[2, 3]);
        let attention_mask = i32_array(&[1, 1, 1, 1, 1, 0], &[2, 3]);
        let logits = model.logits(&input_ids, &attention_mask, None).unwrap();
        mlxcel_core::try_eval(&logits).unwrap();
        assert_eq!(
            mlxcel_core::array_shape(&logits),
            vec![2, 1],
            "{variant:?}: [B, num_labels]"
        );
        assert!(
            array_to_vec_f32(&logits).iter().all(|v| v.is_finite()),
            "{variant:?}: logits must be finite"
        );
    }
}

#[test]
fn classifier_head_scores_multi_label_configs() {
    let _guard = mlx_test_guard();
    let variant = BertVariant::XlmRoberta;
    let mut args = tiny_args(variant);
    args.num_labels = 4;
    let weights = tiny_weights(&args, true);
    let model = BertSequenceClassifier::from_weights(&weights, args, DENSE_QUANTIZATION).unwrap();
    let input_ids = i32_array(&[5, 6, 7], &[1, 3]);
    let attention_mask = i32_array(&[1, 1, 1], &[1, 3]);
    let logits = model.logits(&input_ids, &attention_mask, None).unwrap();
    mlxcel_core::try_eval(&logits).unwrap();
    assert_eq!(mlxcel_core::array_shape(&logits), vec![1, 4]);
}

#[test]
fn num_labels_comes_from_the_projection_tensor_not_the_config() {
    let _guard = mlx_test_guard();
    // The `/v1/rerank` single-label check keys off the width the head really
    // produces, so a config that disagrees with the tensor must not decide it
    // (#1356). Build the weights for one label and lie about it in the config.
    for variant in [BertVariant::Bert, BertVariant::XlmRoberta] {
        let weights = tiny_weights(&tiny_args(variant), true);
        let mut args = tiny_args(variant);
        args.num_labels = 7;
        let model =
            BertSequenceClassifier::from_weights(&weights, args, DENSE_QUANTIZATION).unwrap();
        assert_eq!(
            model.args().num_labels,
            7,
            "{variant:?}: the config's claim"
        );
        assert_eq!(
            model.num_labels(),
            1,
            "{variant:?}: the tensor decides the real head width"
        );

        let input_ids = i32_array(&[5, 6, 7], &[1, 3]);
        let attention_mask = i32_array(&[1, 1, 1], &[1, 3]);
        let logits = model.logits(&input_ids, &attention_mask, None).unwrap();
        mlxcel_core::try_eval(&logits).unwrap();
        assert_eq!(
            mlxcel_core::array_shape(&logits),
            vec![1, 1],
            "{variant:?}: the logits match the tensor, not the config"
        );
    }
}
