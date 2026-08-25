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

use super::engine::{EmbedOptions, EmbeddingEngine, EmbeddingEngineError};
use super::model::ImageInput;
use super::stub::{STUB_DIM, STUB_MAX_LENGTH, STUB_VOCAB_SIZE, stub_loaded_model};

fn engine(batch_size: usize) -> EmbeddingEngine {
    EmbeddingEngine::new(stub_loaded_model(false), batch_size)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[test]
fn engine_reports_stub_limits() {
    let e = engine(4);
    assert_eq!(e.dim(), STUB_DIM);
    assert_eq!(e.max_length(), STUB_MAX_LENGTH);
    assert_eq!(e.vocab_size(), STUB_VOCAB_SIZE);
    assert!(!e.multi_vector());
    assert!(!e.supports_images());
    assert_eq!(e.batch_size(), 4);
    assert_eq!(
        EmbeddingEngine::new(stub_loaded_model(false), 0).batch_size(),
        1
    );
}

#[test]
fn embed_tokens_returns_unit_vectors_in_request_order_across_micro_batches() {
    // batch_size 1 forces one micro-batch per row; the rows have different
    // lengths so the length sort reorders them and the write-back must undo it.
    let e = engine(1);
    let rows = vec![vec![2, 3, 4], vec![5], vec![2, 3]];
    let reply = e.embed_tokens(&rows, &EmbedOptions::default()).unwrap();
    assert_eq!(reply.prompt_tokens, 6);
    assert_eq!(reply.vectors.len(), 3);
    for (i, v) in reply.vectors.iter().enumerate() {
        assert_eq!(v.shape, vec![STUB_DIM], "row {i}");
        assert!((norm(&v.values) - 1.0).abs() < 1e-5, "row {i} is unit norm");
    }
    // Row 1 is one-hot on id 5; rows 0 and 2 share ids 2 and 3.
    assert!((reply.vectors[1].values[5] - 1.0).abs() < 1e-6);
    assert!(cosine(&reply.vectors[0].values, &reply.vectors[2].values) > 0.5);
    assert!(cosine(&reply.vectors[0].values, &reply.vectors[1].values).abs() < 1e-6);

    // A larger batch gives the same vectors: micro-batching is invisible.
    let wide = engine(16)
        .embed_tokens(&rows, &EmbedOptions::default())
        .unwrap();
    for (a, b) in reply.vectors.iter().zip(&wide.vectors) {
        for (x, y) in a.values.iter().zip(&b.values) {
            assert!((x - y).abs() < 1e-6);
        }
    }
}

#[test]
fn embed_texts_uses_the_tokenizer_and_counts_tokens() {
    let e = engine(8);
    let reply = e
        .embed_texts(&["hello".to_string()], &EmbedOptions::default())
        .unwrap();
    assert_eq!(reply.vectors.len(), 1);
    // [CLS] hello [SEP]: three real tokens, only `hello` (id 3) is one-hot.
    assert_eq!(reply.prompt_tokens, 3);
    assert!(
        (reply.vectors[0].values[3] - 1.0).abs() < 1e-6,
        "hello is id 3"
    );

    let empty = e.embed_texts(&[], &EmbedOptions::default()).unwrap();
    assert!(empty.vectors.is_empty());
    assert_eq!(empty.prompt_tokens, 0);
}

#[test]
fn embed_texts_rejects_empty_strings() {
    let e = engine(8);
    let err = e
        .embed_texts(
            &["hello".to_string(), String::new()],
            &EmbedOptions::default(),
        )
        .unwrap_err();
    assert!(matches!(err, EmbeddingEngineError::InvalidInput(ref m) if m.contains("input[1]")));
}

#[test]
fn embed_tokens_rejects_out_of_vocab_ids() {
    let e = engine(8);
    let err = e
        .embed_tokens(&[vec![1, STUB_VOCAB_SIZE as u32]], &EmbedOptions::default())
        .unwrap_err();
    match err {
        EmbeddingEngineError::InvalidInput(m) => assert!(m.contains("vocab_size"), "{m}"),
        other => panic!("expected InvalidInput, got {other:?}"),
    }
    let err = e
        .embed_tokens(&[vec![]], &EmbedOptions::default())
        .unwrap_err();
    assert!(matches!(err, EmbeddingEngineError::InvalidInput(_)));
}

#[test]
fn dimensions_truncates_and_renormalizes() {
    let e = engine(8);
    let opts = EmbedOptions {
        instruction: None,
        dimensions: Some(4),
    };
    let reply = e.embed_tokens(&[vec![2, 3]], &opts).unwrap();
    let v = &reply.vectors[0];
    assert_eq!(v.shape, vec![4]);
    assert!(
        (norm(&v.values) - 1.0).abs() < 1e-5,
        "re-normalized after truncation"
    );
    // ids 2 and 3 both survive the cut to 4 dims and share the mass equally.
    assert!((v.values[2] - v.values[3]).abs() < 1e-6);

    for bad in [0, STUB_DIM + 1] {
        let err = e
            .embed_tokens(
                &[vec![2]],
                &EmbedOptions {
                    instruction: None,
                    dimensions: Some(bad),
                },
            )
            .unwrap_err();
        assert!(
            matches!(err, EmbeddingEngineError::InvalidInput(_)),
            "{bad}"
        );
    }
}

#[test]
fn multi_vector_reply_has_one_row_per_real_token() {
    let e = EmbeddingEngine::new(stub_loaded_model(true), 8);
    assert!(e.multi_vector());
    let reply = e
        .embed_tokens(&[vec![2, 3, 4], vec![5]], &EmbedOptions::default())
        .unwrap();
    assert_eq!(reply.vectors[0].shape, vec![3, STUB_DIM]);
    assert_eq!(reply.vectors[1].shape, vec![1, STUB_DIM]);
    let rows: Vec<&[f32]> = reply.vectors[0].rows().collect();
    assert_eq!(rows.len(), 3);
    assert!((rows[0][2] - 1.0).abs() < 1e-6);
    assert!((rows[2][4] - 1.0).abs() < 1e-6);
    assert!(reply.vectors[0].is_multi_vector());

    // Truncation applies per row.
    let reply = e
        .embed_tokens(
            &[vec![2, 3]],
            &EmbedOptions {
                instruction: None,
                dimensions: Some(3),
            },
        )
        .unwrap();
    assert_eq!(reply.vectors[0].shape, vec![2, 3]);
}

#[test]
fn image_input_is_rejected_by_text_only_models() {
    let e = engine(8);
    let image = ImageInput {
        image: image::DynamicImage::new_rgb8(2, 2),
    };
    let err = e.embed_image(image, &EmbedOptions::default()).unwrap_err();
    assert!(matches!(err, EmbeddingEngineError::InvalidInput(ref m) if m.contains("image")));
}
