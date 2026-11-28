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

//! Shared assertions for the `LanguageModel::last_row_logits` hook (#3070).
//!
//! A prefill samples one position. A model that implements the hook projects
//! only that hidden row through its final norm, LM head and any softcap or
//! scale, so the engine never materializes `[B, L, vocab]`. These helpers pin
//! the two things a wiring mistake breaks: the hook is actually taken (a model
//! that silently declines still passes an equality check through the full
//! forward), and the row it returns is the full forward's row.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::{LanguageModel, logits_at_position};
use mlxcel_core::{MlxArray, UniquePtr};

const TOL: f64 = 1e-4;

fn ids(tokens: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32])
}

fn assert_row(actual: &MlxArray, full: &MlxArray, pos: usize, what: &str) {
    let expected = logits_at_position(full, pos);
    mlxcel_core::eval(&expected);
    mlxcel_core::eval(actual);
    let vocab = mlxcel_core::array_shape(full)[2];
    assert_eq!(
        mlxcel_core::array_shape(actual),
        vec![1, 1, vocab],
        "{what}: last-row logits must be [1, 1, vocab]"
    );
    let close = mlxcel_core::allclose(actual, &expected, TOL, TOL);
    mlxcel_core::eval(&close);
    assert!(
        mlxcel_core::item_bool(&close),
        "{what}: last-row logits differ from row {pos} of the full forward"
    );
}

/// Text path: the hook answers, and every text entry point returns the full
/// forward's row, at the last position, at a mid position under a sequence
/// id, and across a two-piece chunked prefill.
///
/// `make` builds a fresh model per run so no cache or model-owned state leaks
/// between the reference and the path under test. `tokens` needs at least
/// six ids, all inside the model's vocabulary.
pub(crate) fn assert_last_row_logits<M: LanguageModel>(make: impl Fn() -> M, tokens: &[i32]) {
    assert!(tokens.len() >= 6, "need at least six prompt tokens");
    let input = ids(tokens);
    let last = tokens.len() - 1;

    let reference = make();
    let mut caches = reference.make_caches();
    let full = reference.forward_with_sequence_id(&input, None, &mut caches, None);
    mlxcel_core::eval(&full);
    assert_eq!(
        mlxcel_core::array_shape(&full)[1] as usize,
        tokens.len(),
        "reference forward must return every position"
    );

    // The hook itself, so a model that declines cannot pass through the
    // default's full forward.
    let model = make();
    let mut caches = model.make_caches();
    let row = model
        .last_row_logits(&input, None, None, &mut caches, None, last)
        .expect("the model must implement last_row_logits for its text forward");
    assert_row(&row, &full, last, "last_row_logits");

    let model = make();
    let mut caches = model.make_caches();
    let row = model.forward_last_logits(&input, &mut caches, None, last);
    assert_row(&row, &full, last, "forward_last_logits");

    // The engine's entry point, under a prepared sequence id and at a
    // position that is not the last one (a padded prefill's sampled row).
    let seq = SequenceId::from_raw(307_001);
    let model = make();
    let mut caches = model.make_caches();
    model.prepare_sequence_state(seq);
    let row = model
        .try_forward_last_logits_with_sequence_id(&input, Some(seq), &mut caches, None, 2)
        .expect("in-process prefill forward succeeds");
    assert_row(&row, &full, 2, "try_forward_last_logits_with_sequence_id");
    model.release_sequence_state_by_id(seq);

    // A chunked prefill continues the same state: the second piece's sampled
    // row is the single pass's last row.
    let model = make();
    if model.supports_chunked_prefill() {
        let split = tokens.len() / 2;
        let mut caches = model.make_caches();
        let head = model.forward_last_logits(&ids(&tokens[..split]), &mut caches, None, split - 1);
        mlxcel_core::eval(&head);
        let tail = model.forward_last_logits(
            &ids(&tokens[split..]),
            &mut caches,
            None,
            tokens.len() - split - 1,
        );
        assert_row(&tail, &full, last, "second piece of a chunked prefill");
    }
}

/// Embeddings path, for a model whose `embed_tokens` and
/// `forward_with_embeddings` are a plain text forward from embeddings: the
/// hook answers for embeddings, and the engine's embeddings entry point
/// returns the row of `forward_with_embeddings_and_sequence_id`.
pub(crate) fn assert_last_row_logits_with_embeddings<M: LanguageModel>(
    make: impl Fn() -> M,
    tokens: &[i32],
) {
    assert!(tokens.len() >= 6, "need at least six prompt tokens");
    let input = ids(tokens);
    let last = tokens.len() - 1;

    let reference = make();
    let embeds = reference
        .embed_tokens(&input)
        .expect("the model must expose embed_tokens for the embeddings path");
    mlxcel_core::eval(&embeds);
    let mut caches = reference.make_caches();
    let full = reference.forward_with_embeddings_and_sequence_id(
        &input,
        Some(&embeds),
        None,
        &mut caches,
        None,
    );
    mlxcel_core::eval(&full);

    assert_embedded_rows(&make, &input, &embeds, &full, last);
}

/// The embeddings half of [`assert_last_row_logits_with_embeddings`] for a
/// caller that built the embeddings and the full reference logits itself (a
/// VLM wrapper whose embeddings come from its own merge step). `make` must
/// rebuild the model in the same prepared state the reference ran in.
pub(crate) fn assert_embedded_rows<M: LanguageModel>(
    make: &impl Fn() -> M,
    input: &MlxArray,
    embeds: &MlxArray,
    full: &MlxArray,
    last: usize,
) {
    let model = make();
    let mut caches = model.make_caches();
    let row = model
        .last_row_logits(input, Some(embeds), None, &mut caches, None, last)
        .expect("the model must implement last_row_logits for its embeddings forward");
    assert_row(&row, full, last, "last_row_logits with embeddings");

    let seq = SequenceId::from_raw(307_002);
    let model = make();
    let mut caches = model.make_caches();
    model.prepare_sequence_state(seq);
    let row = model.forward_last_logits_with_embeddings_and_sequence_id(
        input,
        Some(embeds),
        Some(seq),
        &mut caches,
        None,
        2,
    );
    assert_row(
        &row,
        full,
        2,
        "forward_last_logits_with_embeddings_and_sequence_id",
    );
    model.release_sequence_state_by_id(seq);
}

/// Prompt every shared case runs: eleven ids below the smallest tiny vocab.
pub(crate) const TOKENS: &[i32] = &[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 7];

/// One `#[test]` running [`assert_last_row_logits`] over a fixture function
/// (`fn() -> impl LanguageModel`). The `embeddings` form also runs
/// [`assert_last_row_logits_with_embeddings`].
macro_rules! last_row_case {
    ($name:ident, $fixture:expr) => {
        #[test]
        fn $name() {
            let _guard = crate::models::embedding_test_support::mlx_test_guard();
            crate::models::last_row_logits_harness::assert_last_row_logits(
                $fixture,
                crate::models::last_row_logits_harness::TOKENS,
            );
        }
    };
    ($name:ident, $fixture:expr, embeddings) => {
        #[test]
        fn $name() {
            let _guard = crate::models::embedding_test_support::mlx_test_guard();
            crate::models::last_row_logits_harness::assert_last_row_logits(
                $fixture,
                crate::models::last_row_logits_harness::TOKENS,
            );
            crate::models::last_row_logits_harness::assert_last_row_logits_with_embeddings(
                $fixture,
                crate::models::last_row_logits_harness::TOKENS,
            );
        }
    };
}
pub(crate) use last_row_case;
