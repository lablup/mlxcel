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

//! Shared gates for per-sequence DeepStack state (#2567).
//!
//! The server prepares a request's DeepStack features at enqueue time and
//! prefills later, so two image requests admitted in one drain tick both write
//! the text model before either prefills, and a text-only prefill can land in
//! between. These helpers reproduce both orders on a synthetic text model and
//! demand that every sequence's prefill injects its own features.
//!
//! Used by: Qwen3VLModel, Qwen3VLMoeModel and CohereCompassTextModel tests.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::embedding_test_support::{Rng, max_abs_diff, to_vec};

/// Family-specific entry points the gates drive. All three DeepStack text
/// models expose these as inherent methods with the same shapes but no shared
/// trait, so the tests hand them in as closures.
pub(crate) struct DeepStackHooks<'a> {
    /// `set_deepstack_state(visual_pos_masks, embeds)`: the vision wrapper's
    /// write during request preparation.
    pub set_state: &'a dyn Fn(UniquePtr<MlxArray>, Vec<UniquePtr<MlxArray>>),
    /// `bind_mrope_state_to_sequence(seq_id)`: the scheduler's admission bind.
    pub bind: &'a dyn Fn(SequenceId),
    /// `get_embed_tokens(input_ids)`: the embeddings the prefill injects into.
    pub embed: &'a dyn Fn(&MlxArray) -> UniquePtr<MlxArray>,
}

const PROMPT: [i32; 6] = [1, 2, 3, 4, 5, 6];
/// Positions 2..=4 of [`PROMPT`] are the image run.
const IMAGE_POSITIONS: [usize; 3] = [2, 3, 4];

fn prompt_ids() -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(&PROMPT, &[1, PROMPT.len() as i32])
}

fn visual_pos_mask() -> UniquePtr<MlxArray> {
    let mut values = vec![0.0f32; PROMPT.len()];
    for position in IMAGE_POSITIONS {
        values[position] = 1.0;
    }
    let mask = mlxcel_core::from_slice_f32(&values, &[1, PROMPT.len() as i32]);
    mlxcel_core::astype(&mask, mlxcel_core::dtype::BOOL)
}

/// One DeepStack branch per layer, `[n_img, hidden]`, drawn from `seed` so two
/// entries differ in every element.
fn branches(seed: u64, hidden: i32, layers: usize) -> Vec<UniquePtr<MlxArray>> {
    let mut rng = Rng::new(seed);
    (0..layers)
        .map(|_| rng.tensor(&[IMAGE_POSITIONS.len() as i32, hidden], 0.5))
        .collect()
}

fn prefill_with_embeddings<M: LanguageModel>(
    model: &M,
    hooks: &DeepStackHooks<'_>,
    seq_id: SequenceId,
) -> Vec<f32> {
    let input = prompt_ids();
    let embeds = (hooks.embed)(&input);
    let mut caches = model.make_caches();
    let logits = model.forward_with_embeddings_and_sequence_id(
        &input,
        Some(&embeds),
        Some(seq_id),
        &mut caches,
        None,
    );
    to_vec(&logits)
}

/// Prepare and bind one entry, prefill it alone, and release it.
fn prefill_alone<M: LanguageModel>(
    model: &M,
    hooks: &DeepStackHooks<'_>,
    seed: u64,
    hidden: i32,
    layers: usize,
    seq_id: SequenceId,
) -> Vec<f32> {
    (hooks.set_state)(visual_pos_mask(), branches(seed, hidden, layers));
    (hooks.bind)(seq_id);
    let logits = prefill_with_embeddings(model, hooks, seq_id);
    model.release_sequence_state_by_id(seq_id);
    logits
}

fn assert_same(actual: &[f32], expected: &[f32], what: &str) {
    let diff = max_abs_diff(actual, expected);
    assert!(
        diff <= 1e-5,
        "{what}: prefill logits differ from the sequence run alone (max abs diff {diff:e})"
    );
}

/// Entries X and Y are prepared and bound to sequences 1 and 2 before either
/// prefills (two image requests admitted in one drain tick). Each sequence's
/// prefill must inject its own entry.
pub(crate) fn assert_bound_deepstack_entries_stay_with_their_sequences<M: LanguageModel>(
    model: &M,
    hidden: i32,
    layers: usize,
    hooks: DeepStackHooks<'_>,
) {
    let seed_x = 0x2567_0001;
    let seed_y = 0x2567_0002;
    let alone_x = prefill_alone(
        model,
        &hooks,
        seed_x,
        hidden,
        layers,
        SequenceId::from_raw(71),
    );
    let alone_y = prefill_alone(
        model,
        &hooks,
        seed_y,
        hidden,
        layers,
        SequenceId::from_raw(72),
    );
    assert!(
        max_abs_diff(&alone_x, &alone_y) > 1e-4,
        "the two DeepStack entries must change the logits differently for the gate to mean anything"
    );

    let seq_x = SequenceId::from_raw(81);
    let seq_y = SequenceId::from_raw(82);
    (hooks.set_state)(visual_pos_mask(), branches(seed_x, hidden, layers));
    (hooks.bind)(seq_x);
    (hooks.set_state)(visual_pos_mask(), branches(seed_y, hidden, layers));
    (hooks.bind)(seq_y);

    let got_x = prefill_with_embeddings(model, &hooks, seq_x);
    let got_y = prefill_with_embeddings(model, &hooks, seq_y);
    assert_same(&got_x, &alone_x, "sequence 1 after sequence 2 was bound");
    assert_same(&got_y, &alone_y, "sequence 2 after sequence 1 prefilled");

    model.release_sequence_state_by_id(seq_x);
    model.release_sequence_state_by_id(seq_y);
}

/// A text-only offset-0 forward on sequence 3, landing between binding
/// sequence 1 and prefilling it, must leave sequence 1's entry intact.
pub(crate) fn assert_text_only_forward_keeps_bound_deepstack_entry<M: LanguageModel>(
    model: &M,
    hidden: i32,
    layers: usize,
    hooks: DeepStackHooks<'_>,
) {
    let seed_x = 0x2567_0003;
    let alone_x = prefill_alone(
        model,
        &hooks,
        seed_x,
        hidden,
        layers,
        SequenceId::from_raw(73),
    );

    let seq_x = SequenceId::from_raw(83);
    let seq_text = SequenceId::from_raw(84);
    (hooks.set_state)(visual_pos_mask(), branches(seed_x, hidden, layers));
    (hooks.bind)(seq_x);
    (hooks.bind)(seq_text);

    let text_input = mlxcel_core::from_slice_i32(&[7, 8, 9], &[1, 3]);
    let mut text_caches = model.make_caches();
    let text_logits =
        model.forward_with_sequence_id(&text_input, Some(seq_text), &mut text_caches, None);
    mlxcel_core::eval(&text_logits);

    let got_x = prefill_with_embeddings(model, &hooks, seq_x);
    assert_same(
        &got_x,
        &alone_x,
        "sequence 1 after a text-only offset-0 forward on sequence 3",
    );

    model.release_sequence_state_by_id(seq_x);
    model.release_sequence_state_by_id(seq_text);
}
