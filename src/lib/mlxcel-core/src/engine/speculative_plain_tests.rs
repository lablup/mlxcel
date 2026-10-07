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

//! The drafter loop's pipelined plain rounds (#2176) against the same loop
//! kept synchronous and against `generate`: the same stream, the same
//! sequence state at every kind of finish, and a proposal found while a step
//! is in flight verified on the next round.

use std::cell::RefCell;

use super::*;
use crate::drafter::{Drafter, DrafterError, DrafterKind};
use crate::generate::{LanguageModel, SamplingConfig};
use crate::layers::KVCache;
use crate::utils::array_to_vec_f32;
use crate::weights::WeightMap;
use crate::{MlxArray, UniquePtr, ffi};

const VOCAB: usize = 8;
/// Never produced: the model cycles through `0..7`.
const EOS: i32 = 7;

fn host_tokens(input_ids: &MlxArray) -> Vec<i32> {
    ffi::eval(input_ids);
    array_to_vec_f32(&ffi::astype(input_ids, crate::dtype::FLOAT32))
        .into_iter()
        .map(|t| t as i32)
        .collect()
}

/// A dense model whose greedy continuation of `t` is `(t + 1) % 7`, so a
/// stream runs until its budget or a stop id; it counts forwards.
#[derive(Default)]
struct CycleModel {
    forwards: RefCell<usize>,
    /// Return a single position's logits for a multi-token input, as a model
    /// that breaks the `[B, T, V]` contract would.
    short_logits: bool,
}

impl LanguageModel for CycleModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        *self.forwards.borrow_mut() += 1;
        let shape = ffi::array_shape(input_ids);
        for cache in caches.iter_mut() {
            let k = ffi::zeros(&[1, 1, shape[1], 4], crate::dtype::FLOAT32);
            let v = ffi::zeros(&[1, 1, shape[1], 4], crate::dtype::FLOAT32);
            cache.update(k, v);
        }
        let tokens = host_tokens(input_ids);
        let mut logits = vec![0.0f32; tokens.len() * VOCAB];
        for (i, tok) in tokens.into_iter().enumerate() {
            logits[i * VOCAB + (tok + 1).rem_euclid(7) as usize] = 10.0;
        }
        if self.short_logits && shape[1] > 1 {
            return ffi::from_slice_f32(&logits[..VOCAB], &[shape[0], 1, VOCAB as i32]);
        }
        ffi::from_slice_f32(&logits, &[shape[0], shape[1], VOCAB as i32])
    }

    fn make_caches(&self) -> Vec<KVCache> {
        vec![KVCache::new()]
    }

    fn num_layers(&self) -> usize {
        1
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![EOS]
    }

    fn supports_batching(&self) -> bool {
        true
    }
}

/// Proposes the model's own next two tokens when the context reaches one of
/// `propose_at`, nothing otherwise, and says whether plain rounds pipeline.
struct ScriptedDrafter {
    context: Vec<i32>,
    propose_at: Vec<usize>,
    pipelines: bool,
    retracted: usize,
}

impl ScriptedDrafter {
    fn new(propose_at: &[usize], pipelines: bool) -> Self {
        Self {
            context: Vec::new(),
            propose_at: propose_at.to_vec(),
            pipelines,
            retracted: 0,
        }
    }
}

impl Drafter for ScriptedDrafter {
    fn bind(&mut self, _target: &dyn LanguageModel) -> Result<(), DrafterError> {
        Ok(())
    }

    fn drafts_from_tokens_only(&self) -> bool {
        true
    }

    fn prefill_from_target_hidden(
        &mut self,
        prompt_tokens: &[i32],
        _hidden: &MlxArray,
        first_bonus: i32,
        _sampler: &SamplingConfig,
    ) -> Result<(), DrafterError> {
        self.context = prompt_tokens.to_vec();
        self.context.push(first_bonus);
        Ok(())
    }

    fn draft_block(
        &mut self,
        last_bonus: i32,
        _hidden: Option<&MlxArray>,
        block_size: usize,
        _sampler: &SamplingConfig,
    ) -> Result<Vec<i32>, DrafterError> {
        assert_eq!(self.context.last(), Some(&last_bonus), "context is current");
        if !self.propose_at.contains(&self.context.len()) {
            return Ok(Vec::new());
        }
        let mut draft = vec![(last_bonus + 1) % 7, (last_bonus + 2) % 7];
        draft.truncate(block_size);
        Ok(draft)
    }

    fn accept_verified_tokens(
        &mut self,
        _verify_hidden: &MlxArray,
        _draft_tokens: &[i32],
        _accepted: usize,
        new_tokens: &[i32],
        _sampler: &SamplingConfig,
    ) -> Result<(), DrafterError> {
        self.context.extend_from_slice(new_tokens);
        Ok(())
    }

    fn pipelines_plain_rounds(&self) -> bool {
        self.pipelines
    }

    fn retract_draft(&mut self, _draft: &[i32]) {
        self.retracted += 1;
    }

    fn sanitize(&mut self, _weights: &mut WeightMap) -> Result<(), DrafterError> {
        Ok(())
    }

    fn kind(&self) -> DrafterKind {
        DrafterKind::PromptLookup
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Trace {
    tokens: Vec<i32>,
    delivered: Vec<i32>,
    offset: i32,
    kv_len: i32,
}

/// One drafter run through the client's own open / speculate / close, with
/// the sequence's pool offset and KV length right before the close, its
/// rounds, the forwards it ran and the drafter's retractions.
fn drafted(
    force_sync: bool,
    drafter: &mut ScriptedDrafter,
    prompt: &[i32],
    max_tokens: usize,
    sampling: &SamplingConfig,
    deliveries: Option<usize>,
) -> (Trace, SpeculativeRounds, usize) {
    let model = CycleModel::default();
    let mut client = DirectEngine::new(&model, 0).with_force_sync(force_sync);
    let id = client.open_sequence().unwrap();
    let mut delivered = Vec::new();
    let run = client
        .speculate(
            id,
            &DirectRequest::text(prompt, max_tokens, sampling),
            sampling,
            drafter,
            4,
            |t| {
                delivered.push(t);
                deliveries.is_none_or(|n| delivered.len() < n)
            },
        )
        .unwrap();
    let set = client.engine().pool().get(id).unwrap();
    let trace = Trace {
        tokens: run.run.tokens,
        delivered,
        offset: set.current_offset,
        kv_len: set.caches[0].seq_len(),
    };
    client.close_sequence(id);
    let forwards = *model.forwards.borrow();
    (trace, run.rounds, forwards)
}

fn generated(prompt: &[i32], max_tokens: usize, sampling: &SamplingConfig) -> Vec<i32> {
    let model = CycleModel::default();
    DirectEngine::new(&model, 0)
        .with_force_sync(true)
        .run(prompt, max_tokens, sampling)
        .unwrap()
}

/// Pipelined plain rounds reach the stream and the state the synchronous
/// rounds do, with no forward past a length finish.
#[test]
fn pipelined_plain_rounds_match_the_synchronous_rounds() {
    let greedy = SamplingConfig::greedy();
    for max_tokens in [1, 2, 5, 12] {
        let mut piped_drafter = ScriptedDrafter::new(&[], true);
        let (piped, piped_rounds, piped_fwd) =
            drafted(false, &mut piped_drafter, &[0], max_tokens, &greedy, None);
        let mut sync_drafter = ScriptedDrafter::new(&[], true);
        let (sync, sync_rounds, sync_fwd) =
            drafted(true, &mut sync_drafter, &[0], max_tokens, &greedy, None);
        assert_eq!(piped, sync, "max_tokens {max_tokens}");
        assert_eq!(piped.tokens, generated(&[0], max_tokens, &greedy));
        assert_eq!(piped_fwd, sync_fwd, "max_tokens {max_tokens}");
        assert_eq!(piped_rounds, sync_rounds, "one round per forward");
    }
}

/// A proposal found while a step is in flight is dropped for that round
/// (and retracted), and verified from the in-flight token next round.
#[test]
fn a_proposal_during_a_pipelined_run_is_verified_next_round() {
    let greedy = SamplingConfig::greedy();
    let mut piped_drafter = ScriptedDrafter::new(&[4, 5, 9], true);
    let (piped, piped_rounds, _) = drafted(false, &mut piped_drafter, &[0], 14, &greedy, None);
    let mut sync_drafter = ScriptedDrafter::new(&[4, 5, 9], true);
    let (sync, _, _) = drafted(true, &mut sync_drafter, &[0], 14, &greedy, None);
    assert_eq!(piped, sync);
    assert_eq!(piped.tokens, generated(&[0], 14, &greedy));
    assert!(
        piped_drafter.retracted > 0,
        "an in-flight round dropped a proposal"
    );
    assert_eq!(sync_drafter.retracted, 0);
    assert!(piped_rounds.drafted_rounds > 0);
}

/// A stop id and a callback stop inside a pipelined run unwind the forward
/// submitted past them.
#[test]
fn stop_and_callback_inside_a_pipelined_run_match_the_synchronous_rounds() {
    let stop = SamplingConfig {
        stop_token_ids: vec![5],
        ..SamplingConfig::greedy()
    };
    let greedy = SamplingConfig::greedy();
    for (sampling, deliveries) in [(&stop, None), (&greedy, Some(6))] {
        let mut piped_drafter = ScriptedDrafter::new(&[], true);
        let (piped, _, piped_fwd) =
            drafted(false, &mut piped_drafter, &[2], 20, sampling, deliveries);
        let mut sync_drafter = ScriptedDrafter::new(&[], true);
        let (sync, _, sync_fwd) = drafted(true, &mut sync_drafter, &[2], 20, sampling, deliveries);
        assert_eq!(piped, sync);
        assert_eq!(piped_fwd, sync_fwd + 1, "one unwound speculative forward");
    }
}

/// A drafter that never asks for pipelining keeps every round synchronous.
#[test]
fn a_drafter_that_does_not_pipeline_stays_synchronous() {
    let greedy = SamplingConfig::greedy();
    let mut drafter = ScriptedDrafter::new(&[], false);
    let (piped, _, piped_fwd) = drafted(false, &mut drafter, &[2], 20, &greedy, Some(6));
    let mut sync_drafter = ScriptedDrafter::new(&[], false);
    let (sync, _, sync_fwd) = drafted(true, &mut sync_drafter, &[2], 20, &greedy, Some(6));
    assert_eq!(piped, sync);
    assert_eq!(piped_fwd, sync_fwd);
}

/// A model whose verify forward returns fewer positions than it was given is
/// an error from the round loop, not an out-of-range index.
#[test]
fn a_short_verify_forward_is_an_error_not_a_panic() {
    let greedy = SamplingConfig::greedy();
    let stochastic = SamplingConfig {
        temperature: 0.8,
        seed: Some(7),
        ..SamplingConfig::greedy()
    };
    // Argmax-batched and per-position verification read the shape differently.
    for sampling in [&greedy, &stochastic] {
        let model = CycleModel {
            short_logits: true,
            ..CycleModel::default()
        };
        let mut client = DirectEngine::new(&model, 0).with_force_sync(true);
        let id = client.open_sequence().unwrap();
        let mut drafter = ScriptedDrafter::new(&[2], false);
        let err = client
            .speculate(
                id,
                &DirectRequest::text(&[0], 20, sampling),
                sampling,
                &mut drafter,
                4,
                |_| true,
            )
            .expect_err("the verify logits are one position short");
        assert!(err.to_string().contains("unexpected logits shape"), "{err}");
        client.close_sequence(id);
    }
}

/// The same for a scoring window: the logits cover fewer positions than the
/// window, so the gather would read past them.
#[test]
fn a_short_scoring_forward_is_an_error_not_a_panic() {
    let model = CycleModel {
        short_logits: true,
        ..CycleModel::default()
    };
    let mut client = DirectEngine::new(&model, 0);
    let err = client
        .loglikelihoods(&[0, 1, 2, 3])
        .expect_err("the scoring logits are short");
    assert!(err.to_string().contains("unexpected logits shape"), "{err}");
}

#[test]
fn logits_vocab_accepts_only_a_single_row_of_enough_positions() {
    use super::direct::logits_vocab;
    assert_eq!(logits_vocab(&[1, 4, 32], 4), Ok(32));
    assert_eq!(logits_vocab(&[1, 5, 32], 4), Ok(32));
    for bad in [
        &[1, 3, 32][..],
        &[2, 4, 32],
        &[1, 4],
        &[1, 4, 32, 1],
        &[1, 4, 0],
        &[],
    ] {
        assert!(logits_vocab(bad, 4).is_err(), "{bad:?}");
    }
}

/// An EOS inside a verify block leaves the pool offset where the synchronous
/// step does: the token that finishes the sequence is not committed.
#[test]
fn an_eos_inside_a_verify_block_leaves_the_offset_where_the_step_does() {
    let stop = SamplingConfig {
        stop_token_ids: vec![3],
        ..SamplingConfig::greedy()
    };
    // The model continues 1, 2, 3; the drafter proposes [2, 3] after the
    // first token, so the block ends on the stop id mid-verify.
    let mut proposing = ScriptedDrafter::new(&[2], false);
    let (verified, rounds, _) = drafted(true, &mut proposing, &[0], 20, &stop, None);
    assert!(rounds.drafted_rounds > 0, "a block was verified");
    let mut plain = ScriptedDrafter::new(&[], false);
    let (stepped, _, _) = drafted(true, &mut plain, &[0], 20, &stop, None);
    assert_eq!(verified, stepped);
}
