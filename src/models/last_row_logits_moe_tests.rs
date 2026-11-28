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

//! MoE-family coverage of the `last_row_logits` hook (#3070): each case builds
//! a tiny random-weight model and runs the shared harness
//! (`last_row_logits_harness`), which asserts the hook is taken and returns
//! `[1, 1, vocab]` logits equal to the full forward's row.

use crate::models::last_row_logits_harness::last_row_case;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::{LanguageModel, logits_at_position};
use mlxcel_core::{MlxArray, UniquePtr};

last_row_case!(
    deepseek_last_row_logits,
    crate::models::mixtral_style_batched_decode_tests::tiny_deepseek_model,
    embeddings
);

fn ids(tokens: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32])
}

fn assert_row(actual: &MlxArray, full: &MlxArray, pos: usize, what: &str) {
    let expected = logits_at_position(full, pos);
    let vocab = mlxcel_core::array_shape(full)[2];
    assert_eq!(
        mlxcel_core::array_shape(actual),
        vec![1, 1, vocab],
        "{what}: last-row logits must be [1, 1, vocab]"
    );
    let close = mlxcel_core::allclose(actual, &expected, 1e-4, 1e-4);
    mlxcel_core::eval(&close);
    assert!(
        mlxcel_core::item_bool(&close),
        "{what}: last-row logits differ from row {pos} of the full forward"
    );
}

/// The shared harness's checks with every forward under a prepared sequence
/// id, for a model-owned family whose id-less multi-token forward restarts
/// its internal state (AFMoE, BailingMoE-Linear): there an id-less second
/// prefill piece is a fresh prompt by design, so the chunked check only holds
/// under an id, which is how the engine always calls it.
pub(crate) fn assert_last_row_logits_under_sequence_ids<M: LanguageModel>(
    make: impl Fn() -> M,
    tokens: &[i32],
) {
    assert!(tokens.len() >= 6, "need at least six prompt tokens");
    let input = ids(tokens);
    let last = tokens.len() - 1;
    let prepared = |raw: u64| {
        let model = make();
        let seq = SequenceId::from_raw(raw);
        model.prepare_sequence_state(seq);
        (model, seq)
    };

    let (reference, seq) = prepared(307_100);
    let mut caches = reference.make_caches();
    let full = reference.forward_with_sequence_id(&input, Some(seq), &mut caches, None);
    mlxcel_core::eval(&full);

    let (model, seq) = prepared(307_101);
    let mut caches = model.make_caches();
    let row = model
        .last_row_logits(&input, None, Some(seq), &mut caches, None, last)
        .expect("the model must implement last_row_logits for its text forward");
    assert_row(&row, &full, last, "last_row_logits");

    // The id-less entry point, single pass.
    let model = make();
    let mut caches = model.make_caches();
    let row = model.forward_last_logits(&input, &mut caches, None, last);
    assert_row(&row, &full, last, "forward_last_logits");

    let (model, seq) = prepared(307_102);
    let mut caches = model.make_caches();
    let row = model
        .try_forward_last_logits_with_sequence_id(&input, Some(seq), &mut caches, None, 2)
        .expect("in-process prefill forward succeeds");
    assert_row(&row, &full, 2, "try_forward_last_logits_with_sequence_id");

    let (model, seq) = prepared(307_103);
    let split = tokens.len() / 2;
    let mut caches = model.make_caches();
    let head = model.forward_last_logits_with_sequence_id(
        &ids(&tokens[..split]),
        Some(seq),
        &mut caches,
        None,
        split - 1,
    );
    mlxcel_core::eval(&head);
    let tail = model.forward_last_logits_with_sequence_id(
        &ids(&tokens[split..]),
        Some(seq),
        &mut caches,
        None,
        tokens.len() - split - 1,
    );
    assert_row(&tail, &full, last, "second piece of a chunked prefill");
}
