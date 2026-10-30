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

//! Cache-alignment guards for [`FullModelDrafter`] (#2757): after every
//! event of the round-loop protocol the draft caches must hold exactly the
//! positions the module docs promise, measured on the caches themselves and
//! not on the bookkeeping that predicts them. The stream-level guards (temp-0
//! byte-identity, chi-square, Markov) run in
//! `engine::speculative_distribution_tests` with this drafter over the stub
//! models there.

use super::*;
use crate::{UniquePtr, dtype, ffi};

const VOCAB: usize = 6;

/// Fills the caches as a real model would, one position per input token.
fn advance_caches(input_ids: &MlxArray, caches: &mut [KVCache]) -> Vec<i32> {
    let shape = ffi::array_shape(input_ids);
    for cache in caches.iter_mut() {
        let k = ffi::ones(&[shape[0], 2, shape[1], 4], dtype::FLOAT32);
        let v = ffi::ones(&[shape[0], 2, shape[1], 4], dtype::FLOAT32);
        cache.update(k, v);
    }
    shape
}

/// Greedy next token is `t + step (mod 6)`; two cache layers so a per-layer
/// trim mismatch is visible.
struct Walk {
    step: i32,
    eos: Vec<i32>,
    trimmable: bool,
}

impl Walk {
    fn new(step: i32) -> Self {
        Self {
            step,
            eos: Vec::new(),
            trimmable: true,
        }
    }

    fn next(&self, token: i32) -> i32 {
        (token + self.step).rem_euclid(VOCAB as i32)
    }
}

impl LanguageModel for Walk {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = advance_caches(input_ids, caches);
        ffi::eval(input_ids);
        let tokens: Vec<i32> = ffi::array_to_raw_bytes(input_ids)
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut out = vec![0.0f32; tokens.len() * VOCAB];
        for (i, &tok) in tokens.iter().enumerate() {
            out[i * VOCAB + self.next(tok) as usize] = 8.0;
        }
        ffi::from_slice_f32(&out, &[shape[0], shape[1], VOCAB as i32])
    }

    fn make_caches(&self) -> Vec<KVCache> {
        vec![KVCache::new(), KVCache::new()]
    }

    fn num_layers(&self) -> usize {
        2
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos.clone()
    }

    fn supports_batching(&self) -> bool {
        self.trimmable
    }

    fn supports_padded_prefill(&self) -> bool {
        true
    }
}

/// Positions every cache layer holds; the layers must agree.
fn cache_len(state: &DraftModelState) -> usize {
    let lens: Vec<i32> = state.caches.iter().map(KVCache::seq_len).collect();
    assert!(
        lens.windows(2).all(|w| w[0] == w[1]),
        "layers disagree: {lens:?}"
    );
    lens[0] as usize
}

fn assert_aligned(state: &DraftModelState, expected: usize) {
    assert_eq!(state.cached_positions(), expected, "bookkeeping");
    assert_eq!(cache_len(state), expected, "caches");
}

/// Prefill, propose, accept a partial block, propose again, accept a full
/// block, and confirm the caches end each step where the invariant says.
#[test]
fn caches_track_the_context_through_partial_and_full_acceptance() {
    let model = Walk::new(1);
    let sampling = SamplingConfig::greedy();
    let hidden = ffi::from_slice_i32(&[0], &[1]);
    let mut state = DraftModelState::new(&model).unwrap();
    let mut drafter = FullModelDrafter::new(&model, &mut state);
    drafter
        .prefill_from_target_hidden(&[1, 2, 3], &hidden, 4, &sampling)
        .unwrap();
    // The prompt is cached; the first token is the uncached current token.
    assert_aligned(drafter.state, 3);

    let draft = drafter.draft_block(4, None, 3, &sampling).unwrap();
    assert_eq!(draft, vec![5, 0, 1]);
    // Fed: 4 (current), 5, 0. The last proposal is never fed.
    assert_aligned(drafter.state, 6);
    assert_eq!(drafter.stats().draft_forwards, 1 + 3);

    // The target confirms 5 and replaces 0 with 2.
    drafter
        .accept_verified_tokens(&hidden, &draft, 1, &[5, 2], &sampling)
        .unwrap();
    // Context [1,2,3,4,5,2]: caches hold everything but the current token.
    assert_aligned(drafter.state, 5);
    assert_eq!(drafter.stats().accepted_draft_tokens, 1);

    let draft = drafter.draft_block(2, None, 3, &sampling).unwrap();
    assert_eq!(draft, vec![3, 4, 5]);
    assert_aligned(drafter.state, 8);

    // Fully accepted plus the bonus token 0: d_{k-1} = 5 was never fed.
    drafter
        .accept_verified_tokens(&hidden, &draft, 3, &[3, 4, 5, 0], &sampling)
        .unwrap();
    // Context [1,2,3,4,5,2,3,4,5,0]: caches hold through 4, owe 5 and 0.
    assert_aligned(drafter.state, 8);

    // The next ask feeds the owed proposal with the current token in one
    // forward and proposes from the current token.
    let draft = drafter.draft_block(0, None, 2, &sampling).unwrap();
    assert_eq!(draft, vec![1, 2]);
    assert_aligned(drafter.state, 11);
    assert_eq!(drafter.stats().draft_forwards, 4 + 3 + 2);

    let counts = drafter.counts();
    assert_eq!(counts.rounds, 3);
    assert_eq!(counts.drafted_rounds, 3);
    assert_eq!(counts.proposed_draft_tokens, 8);
    assert_eq!(counts.accepted_draft_tokens, 4);
}

/// A retracted ask (the loop ran no forward for it) drops every fed proposal
/// and keeps the current token, so the next ask after one plain step feeds
/// only that step's token.
#[test]
fn retract_drops_the_fed_proposals_and_keeps_the_context() {
    let model = Walk::new(1);
    let sampling = SamplingConfig::greedy();
    let hidden = ffi::from_slice_i32(&[0], &[1]);
    let mut state = DraftModelState::new(&model).unwrap();
    let mut drafter = FullModelDrafter::new(&model, &mut state);
    drafter
        .prefill_from_target_hidden(&[1, 2], &hidden, 3, &sampling)
        .unwrap();
    let draft = drafter.draft_block(3, None, 4, &sampling).unwrap();
    assert_eq!(draft, vec![4, 5, 0, 1]);
    assert_aligned(drafter.state, 2 + 1 + 3);
    drafter.retract_draft(&draft);
    // Context [1,2,3] is fully cached; nothing of the proposal remains.
    assert_aligned(drafter.state, 3);
    assert_eq!(drafter.stats().rounds, 0);
    assert_eq!(drafter.stats().proposed_draft_tokens, 0);

    // A plain step emitted 4 (the drafter's own guess, as it happens).
    drafter.observe_emitted(&[4]);
    let forwards = drafter.stats().draft_forwards;
    let draft = drafter.draft_block(4, None, 2, &sampling).unwrap();
    assert_eq!(draft, vec![5, 0]);
    assert_eq!(drafter.stats().draft_forwards, forwards + 2);
    assert_aligned(drafter.state, 3 + 1 + 1);
}

/// A row that finishes inside the block (EOS, a stop, `max_tokens`) reports
/// fewer tokens than `accepted + 1`, and one whose finishing token equals the
/// next proposal reports more confirmed context than `accepted`; the
/// realignment is against the context, so both leave the caches a prefix of
/// it and the next ask still feeds a non-empty tail.
#[test]
fn early_finishes_realign_against_the_context_not_the_accept_count() {
    let model = Walk::new(1);
    let sampling = SamplingConfig::greedy();
    let hidden = ffi::from_slice_i32(&[0], &[1]);
    let mut state = DraftModelState::new(&model).unwrap();
    let mut drafter = FullModelDrafter::new(&model, &mut state);
    drafter
        .prefill_from_target_hidden(&[0], &hidden, 1, &sampling)
        .unwrap();
    let draft = drafter.draft_block(1, None, 3, &sampling).unwrap();
    assert_eq!(draft, vec![2, 3, 4]);
    assert_aligned(drafter.state, 1 + 1 + 2);

    // The target confirmed 2, then its token 3 finished the row (so it was
    // not counted as accepted). Context [0,1,2,3] is entirely cached.
    drafter
        .accept_verified_tokens(&hidden, &draft, 1, &[2, 3], &sampling)
        .unwrap();
    assert_aligned(drafter.state, 4);
    // Asked anyway: the current token is re-fed from one position back.
    let draft = drafter.draft_block(3, None, 1, &sampling).unwrap();
    assert_eq!(draft, vec![4]);
    assert_aligned(drafter.state, 4);

    // A round that emits only the accepted proposals (the finish fell on
    // the position after them and was not stored).
    drafter
        .accept_verified_tokens(&hidden, &draft, 0, &[], &sampling)
        .unwrap();
    assert_aligned(drafter.state, 4);
    assert_eq!(drafter.state.context, vec![0, 1, 2, 3]);
}

/// A proposal that is a draft-model EOS ends the block early.
#[test]
fn an_eos_proposal_ends_the_block() {
    let mut model = Walk::new(1);
    model.eos = vec![0];
    let sampling = SamplingConfig::greedy();
    let hidden = ffi::from_slice_i32(&[0], &[1]);
    let mut state = DraftModelState::new(&model).unwrap();
    let mut drafter = FullModelDrafter::new(&model, &mut state);
    drafter
        .prefill_from_target_hidden(&[3], &hidden, 4, &sampling)
        .unwrap();
    let draft = drafter.draft_block(4, None, 4, &sampling).unwrap();
    assert_eq!(draft, vec![5, 0]);
    assert_aligned(drafter.state, 1 + 1 + 1);
}

/// The one shared rule (#2495): a model whose state a trim cannot roll back
/// is refused before any forward, with the same reason a target gets.
#[test]
fn a_draft_model_whose_state_cannot_roll_back_is_refused() {
    let mut model = Walk::new(1);
    model.trimmable = false;
    let err = DraftModelState::new(&model).unwrap_err();
    let FullModelDrafterError::Unsupported(reason) = err;
    assert_eq!(
        Some(reason),
        prompt_lookup_unsupported_reason(&model),
        "the drafter and the target are held to one rule"
    );
}

/// `reset` and a second `prefill` start from empty caches.
#[test]
fn reset_and_reprefill_start_from_empty_caches() {
    let model = Walk::new(2);
    let sampling = SamplingConfig::greedy();
    let hidden = ffi::from_slice_i32(&[0], &[1]);
    let mut state = DraftModelState::new(&model).unwrap();
    let mut drafter = FullModelDrafter::new(&model, &mut state);
    drafter
        .prefill_from_target_hidden(&[1, 2, 3, 4], &hidden, 0, &sampling)
        .unwrap();
    drafter.draft_block(0, None, 2, &sampling).unwrap();
    drafter.reset(&model).unwrap();
    assert_aligned(drafter.state, 0);
    drafter
        .prefill_from_target_hidden(&[1], &hidden, 3, &sampling)
        .unwrap();
    assert_aligned(drafter.state, 1);
    assert_eq!(drafter.draft_block(3, None, 1, &sampling).unwrap(), vec![5]);
}
