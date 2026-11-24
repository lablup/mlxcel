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

//! The Inkling 2 DSpark target on the tiny looped model of `tiny_tests`
//! (16 hidden, 6 layers, core 2..4 three passes: 10 visits): the captures,
//! the rollback, and the round loop's emitted tokens against the verify
//! argmax of every round.

use std::cell::RefCell;
use std::sync::atomic::AtomicBool;

use mlxcel_core::drafter::dflash::{DFlashGenerator, SpeculativeTarget};
use mlxcel_core::drafter::{Drafter, DrafterError, DrafterKind};
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::inkling_layer::InklingLayerCache;
use mlxcel_core::sampling::{LogprobsConfig, TokenBiasMap};
use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::super::Inkling2Model;
use super::super::tiny_tests::model;
use super::*;

const VISITS: usize = 10;
const HIDDEN: i32 = 16;
const CAPTURE: [usize; 2] = [3, VISITS];
const BLOCK: usize = 4;

fn ids(values: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_i32(values, &[1, values.len() as i32])
}

fn argmax_rows(logits: &MlxArray) -> Vec<i32> {
    let shape = mlxcel_core::array_shape(logits);
    let width = shape[2] as usize;
    array_to_vec_f32(&mlxcel_core::astype(logits, mlxcel_core::dtype::FLOAT32))
        .chunks(width)
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as i32)
                .unwrap()
        })
        .collect()
}

fn prompt() -> Vec<i32> {
    vec![3, 17, 9, 25, 4, 12, 7, 18, 2, 21]
}

/// Classic token-at-a-time greedy decode on fresh model state.
fn greedy_reference(model: &Inkling2Model, prompt: &[i32], n: usize) -> Vec<i32> {
    let mut caches = model.make_caches();
    let logits = model.forward(&ids(prompt), &mut caches, None);
    let mut next = *argmax_rows(&logits).last().unwrap();
    let mut out = vec![next];
    while out.len() < n {
        let logits = model.forward(&ids(&[next]), &mut caches, None);
        next = argmax_rows(&logits)[0];
        out.push(next);
    }
    out
}

#[test]
fn captures_follow_the_request_and_end_at_the_final_norm() {
    let model = model(3, false);
    assert_eq!(model.dspark_target_shape().visit_count, VISITS);
    let mut caches = model.make_speculative_caches();
    let out = model.forward_speculative(
        &ids(&prompt()),
        &mut caches,
        &CAPTURE,
        false,
        &TokenBiasMap::new(),
    );
    assert_eq!(out.hidden_states.len(), CAPTURE.len());
    let concat = concat_captures(&out);
    assert_eq!(
        mlxcel_core::array_shape(&concat),
        vec![1, prompt().len() as i32, CAPTURE.len() as i32 * HIDDEN]
    );
    // The two captures are different states of the stack.
    let first = array_to_vec_f32(&mlxcel_core::astype(
        &out.hidden_states[0],
        mlxcel_core::dtype::FLOAT32,
    ));
    let last = array_to_vec_f32(&mlxcel_core::astype(
        &out.hidden_states[1],
        mlxcel_core::dtype::FLOAT32,
    ));
    assert_ne!(first, last);
    // The prefill logits agree with the plain forward's.
    let mut plain = model.make_caches();
    let reference = model.forward(&ids(&prompt()), &mut plain, None);
    assert_eq!(argmax_rows(&out.logits), argmax_rows(&reference));
}

#[test]
fn suppressed_ids_never_win_a_verify_row() {
    let model = model(3, false);
    let mut caches = model.make_speculative_caches();
    let plain = model.forward_speculative(
        &ids(&prompt()),
        &mut caches,
        &[],
        false,
        &TokenBiasMap::new(),
    );
    let winners = argmax_rows(&plain.logits);
    let mut caches = model.make_speculative_caches();
    let suppressed = model.forward_speculative(
        &ids(&prompt()),
        &mut caches,
        &[],
        false,
        &suppression_map(&winners),
    );
    for (row, winner) in argmax_rows(&suppressed.logits).iter().zip(&winners) {
        assert!(
            !winners.contains(row),
            "a suppressed id {row} won (was {winner})"
        );
    }
}

#[test]
fn a_rolled_back_block_continues_like_the_accepted_prefix() {
    let model = model(3, false);
    let none = TokenBiasMap::new();
    let block = [5, 11, 6, 20];
    for accepted in 0..BLOCK as i32 {
        let keep = accepted as usize + 1;
        let mut rolled = model.make_speculative_caches();
        model.forward_speculative(&ids(&prompt()), &mut rolled, &[], false, &none);
        model.forward_speculative(&ids(&block), &mut rolled, &[], true, &none);
        model.rollback_speculative_cache(&mut rolled, accepted, BLOCK as i32);
        let mut direct = model.make_speculative_caches();
        model.forward_speculative(&ids(&prompt()), &mut direct, &[], false, &none);
        model.forward_speculative(&ids(&block[..keep]), &mut direct, &[], true, &none);
        let next = [9, 1, 13];
        let a = model.forward_speculative(&ids(&next), &mut rolled, &[], true, &none);
        let b = model.forward_speculative(&ids(&next), &mut direct, &[], true, &none);
        assert_eq!(
            argmax_rows(&a.logits),
            argmax_rows(&b.logits),
            "accepted {accepted}: rollback must leave the state of the kept prefix"
        );
    }
}

#[test]
fn the_probe_runs_and_refuses_degenerate_widths() {
    let model = model(3, false);
    assert!(matches!(
        model.probe_block_chain_exactness(1),
        BlockChainExactness::NotRun(_)
    ));
    let verdict = model.probe_block_chain_exactness(BLOCK);
    assert!(
        !matches!(verdict, BlockChainExactness::NotRun(_)),
        "{verdict:?}"
    );
}

// ── Round loop ─────────────────────────────────────────────────────────────

/// Delegates to the model and records every verify block's argmax rows.
struct Recorder<'a> {
    model: &'a Inkling2Model,
    verify_argmax: RefCell<Vec<Vec<i32>>>,
}

impl SpeculativeTarget for Recorder<'_> {
    type Cache = InklingLayerCache;
    type VerifyOut = Inkling2VerifyOutput;

    fn capture_layer_ids(&self) -> &[usize] {
        &[]
    }

    fn verify_forward(&self, input: &MlxArray, caches: &mut [Self::Cache]) -> Self::VerifyOut {
        self.verify_forward_with_capture_layers(input, caches, &[])
    }

    fn verify_forward_with_capture_layers(
        &self,
        input: &MlxArray,
        caches: &mut [Self::Cache],
        capture: &[usize],
    ) -> Self::VerifyOut {
        let out = self
            .model
            .verify_forward_with_capture_layers(input, caches, capture);
        self.verify_argmax
            .borrow_mut()
            .push(argmax_rows(self.model.verify_logits(&out)));
        out
    }

    fn rollback_partial(
        &self,
        caches: &mut [Self::Cache],
        out: &Self::VerifyOut,
        accepted: i32,
        block_size: i32,
    ) {
        self.model
            .rollback_partial(caches, out, accepted, block_size);
    }

    fn concat_hidden_for_drafter(&self, out: &Self::VerifyOut) -> UniquePtr<MlxArray> {
        self.model.concat_hidden_for_drafter(out)
    }

    fn verify_logits<'a>(&self, out: &'a Self::VerifyOut) -> &'a MlxArray {
        self.model.verify_logits(out)
    }
}

/// Proposes the reference continuation with one position per round
/// corrupted on a schedule, so rounds accept 0, 1, 2 and all proposals.
struct ScheduledDrafter {
    reference: Vec<i32>,
    schedule: Vec<usize>,
    round: usize,
    next: usize,
}

impl Drafter for ScheduledDrafter {
    fn bind(&mut self, _target: &dyn LanguageModel) -> Result<(), DrafterError> {
        Ok(())
    }

    fn dflash_target_layer_ids(&self) -> Option<&[usize]> {
        Some(&CAPTURE)
    }

    fn draft_block(
        &mut self,
        last_bonus: i32,
        hidden: Option<&MlxArray>,
        block_size: usize,
        _sampler: &SamplingConfig,
    ) -> Result<Vec<i32>, DrafterError> {
        let hidden = hidden.expect("captured hidden");
        assert_eq!(
            mlxcel_core::array_shape(hidden)[2],
            CAPTURE.len() as i32 * HIDDEN
        );
        // Re-sync on the bonus so a divergence from the reference does not
        // stall the schedule.
        if let Some(at) = self.reference[self.next..]
            .iter()
            .position(|&t| t == last_bonus)
        {
            self.next += at;
        }
        let mut proposals: Vec<i32> = (1..block_size)
            .map(|j| self.reference.get(self.next + j).copied().unwrap_or(0))
            .collect();
        let corrupt = self.schedule[self.round % self.schedule.len()];
        self.round += 1;
        if corrupt < proposals.len() {
            proposals[corrupt] = (proposals[corrupt] + 1) % 30;
        }
        self.next += corrupt.min(proposals.len()) + 1;
        Ok(proposals)
    }

    fn sanitize(&mut self, _weights: &mut WeightMap) -> Result<(), DrafterError> {
        Ok(())
    }

    fn kind(&self) -> DrafterKind {
        DrafterKind::Dflash
    }
}

#[test]
fn emitted_tokens_equal_the_verify_argmax_of_every_round() {
    let model = model(3, false);
    let n = 30;
    let prompt = prompt();
    let reference = greedy_reference(&model, &prompt, n + BLOCK);
    let recorder = Recorder {
        model: &model,
        verify_argmax: RefCell::new(Vec::new()),
    };
    let mut caches = model.make_speculative_caches();
    let prefill = model.prefill_forward_with_capture_layers(&ids(&prompt), &mut caches, &CAPTURE);
    let first_bonus = *argmax_rows(&prefill.logits).last().unwrap();
    let first_hidden = concat_captures(&prefill);
    drop(prefill);
    let drafter = ScheduledDrafter {
        reference: reference.clone(),
        schedule: vec![0, 2, 1, usize::MAX, 0, 1, 2],
        round: 0,
        next: 0,
    };
    let sampler = SamplingConfig {
        temperature: 0.0,
        ..SamplingConfig::default()
    };
    let mut generator = DFlashGenerator::new(Box::new(drafter), sampler, BLOCK as u32, 0);
    let out = generator
        .run(
            &recorder,
            &model as &dyn LanguageModel,
            &mut caches,
            first_bonus,
            first_hidden,
            &[],
            n,
            &AtomicBool::new(false),
            &LogprobsConfig::default(),
        )
        .expect("round loop runs");

    let rounds = recorder.verify_argmax.into_inner();
    assert_eq!(rounds.len(), out.accept_lens.len());
    let mut from_verify = Vec::new();
    for (argmax, accepted) in rounds.iter().zip(&out.accept_lens) {
        // The last round may narrow to the remaining token budget.
        assert!((2..=BLOCK).contains(&argmax.len()), "{}", argmax.len());
        assert!((*accepted as usize) < argmax.len());
        from_verify.extend_from_slice(&argmax[..=*accepted as usize]);
    }
    from_verify.truncate(out.tokens.len());
    assert_eq!(
        out.tokens, from_verify,
        "every emitted token is the target's verify argmax"
    );
    for accepted in [0, 1, 2, (BLOCK - 1) as u32] {
        assert!(
            out.accept_lens.contains(&accepted),
            "schedule should produce an accept of {accepted}: {:?}",
            out.accept_lens
        );
    }
    // On the tiny model the verify block is also byte-exact against the
    // chain, so the run reproduces classic decode.
    if model.probe_block_chain_exactness(BLOCK).is_equal() {
        let mut tokens = vec![first_bonus];
        tokens.extend(&out.tokens);
        assert_eq!(tokens, reference[..tokens.len()]);
    }
}
