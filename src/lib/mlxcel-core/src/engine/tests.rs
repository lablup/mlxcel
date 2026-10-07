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

//! Engine contract tests over a stub model: one `step` entry for every row
//! count, per-row errors, speculative unwinds and prefill trim.

use std::cell::Cell;

use super::*;
use crate::cache::SequenceId;
use crate::decode_finish::FinishHooks;
use crate::generate::{LanguageModel, SamplingConfig};
use crate::layers::KVCache;
use crate::sampling::LogprobsConfig;
use crate::sampling_row_step::{LogitMask, RowSampler};
use crate::utils::array_to_vec_f32;
use crate::{ffi, from_slice_i32};

/// Logits put the input token id at the top of a vocab of 8, so a sampled
/// or argmax token echoes the input and the stacked batch order is visible.
/// Every forward appends one position to each cache it is given.
struct EchoModel {
    single_calls: Cell<usize>,
    batched_calls: Cell<usize>,
    /// The batched forward returns one logits row fewer than it was given, the
    /// shape of a broken backend, so the fused draw reads back a short `[B]`.
    drop_last_batched_row: bool,
}

impl EchoModel {
    fn new() -> Self {
        Self {
            single_calls: Cell::new(0),
            batched_calls: Cell::new(0),
            drop_last_batched_row: false,
        }
    }

    fn with_short_batched_logits() -> Self {
        Self {
            drop_last_batched_row: true,
            ..Self::new()
        }
    }

    fn echo_logits(input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        ffi::eval(input_ids);
        let shape = ffi::array_shape(input_ids);
        let (b, l) = (shape[0] as usize, shape[1] as usize);
        let toks: Vec<i32> = array_to_vec_f32(&ffi::astype(input_ids, crate::dtype::FLOAT32))
            .into_iter()
            .map(|t| t as i32)
            .collect();
        let mut logits = vec![0.0f32; b * l * 8];
        for (i, &tok) in toks.iter().enumerate() {
            if (0..8).contains(&tok) {
                logits[i * 8 + tok as usize] = 10.0;
            }
        }
        ffi::from_slice_f32(&logits, &[b as i32, l as i32, 8])
    }

    fn append(caches: &mut [KVCache], n: i32) {
        for cache in caches.iter_mut() {
            let k = ffi::zeros(&[1, 1, n, 4], crate::dtype::FLOAT32);
            let v = ffi::zeros(&[1, 1, n, 4], crate::dtype::FLOAT32);
            cache.update(k, v);
        }
    }
}

impl LanguageModel for EchoModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.single_calls.set(self.single_calls.get() + 1);
        Self::append(caches, ffi::array_shape(input_ids)[1]);
        Self::echo_logits(input_ids)
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.batched_calls.set(self.batched_calls.get() + 1);
        let l = ffi::array_shape(input_ids)[1];
        for caches in batch_caches.iter_mut() {
            Self::append(caches, l);
        }
        let logits = Self::echo_logits(input_ids);
        if self.drop_last_batched_row {
            let rows = ffi::array_shape(&logits)[0];
            return ffi::slice(&logits, &[0, 0, 0], &[rows - 1, i32::MAX, i32::MAX]);
        }
        logits
    }

    fn make_caches(&self) -> Vec<KVCache> {
        vec![KVCache::new(), KVCache::new()]
    }

    fn num_layers(&self) -> usize {
        2
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![7]
    }

    fn supports_batching(&self) -> bool {
        true
    }
}

fn argmax_rows(logits: &MlxArray) -> Vec<i32> {
    ffi::eval(logits);
    let shape = ffi::array_shape(logits);
    let (b, v) = (shape[0] as usize, shape[2] as usize);
    let data = array_to_vec_f32(logits);
    (0..b)
        .map(|i| {
            let row = &data[i * v..(i + 1) * v];
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(j, _)| j as i32)
                .unwrap()
        })
        .collect()
}

fn engine() -> Engine<EchoModel> {
    Engine::with_capacity(EchoModel::new(), 4)
}

/// Test hooks: no stop strings or bounds, an optional mask that forces one
/// token, an optional override, and a matcher that can be told to fail.
#[derive(Default)]
struct Hooks {
    force: Option<i32>,
    override_to: Option<i32>,
    fail_consume: bool,
    resolved: Vec<(i32, i32)>,
}

struct ForceMask(i32);

impl LogitMask for ForceMask {
    fn apply(
        &mut self,
        logits: UniquePtr<MlxArray>,
        vocab_size: usize,
    ) -> Result<UniquePtr<MlxArray>, String> {
        if self.0 < 0 {
            return Err("mask rejected the row".to_string());
        }
        let shape = ffi::array_shape(&logits);
        let mut data = vec![-1.0e9f32; vocab_size];
        data[self.0 as usize] = 0.0;
        drop(logits);
        Ok(ffi::from_slice_f32(&data, &shape))
    }
}

impl FinishHooks for Hooks {
    fn stop_text(&mut self, _token: i32, _generated_len: usize) -> bool {
        false
    }
    fn bound_stopped(&self) -> bool {
        false
    }
    fn context_bound_due(&self, _generated_len: usize) -> bool {
        false
    }
}

impl StepRowHooks for Hooks {
    fn logit_mask(&mut self) -> Option<&mut dyn LogitMask> {
        // A leaked mask per call keeps the test simple; the engine takes it
        // by reference for the draw only.
        self.force
            .map(|t| Box::leak(Box::new(ForceMask(t))) as &mut dyn LogitMask)
    }
    fn override_token(&mut self, sampled: i32) -> i32 {
        self.override_to.unwrap_or(sampled)
    }
    fn consume_sampled(&mut self, _sampled: i32) -> Result<bool, String> {
        if self.fail_consume {
            Err("matcher failed".to_string())
        } else {
            Ok(false)
        }
    }
    fn resolved(&mut self, sampled: i32, token: i32) {
        self.resolved.push((sampled, token));
    }
}

/// One row's state the test owns; `row()` views it as a `StepRow`.
struct Row {
    seq_id: SequenceId,
    sampler: RowSampler,
    sampling: SamplingConfig,
    history: Vec<i32>,
    generated: Vec<i32>,
    eos: Vec<i32>,
    max_tokens: usize,
    logprobs: LogprobsConfig,
    hooks: Hooks,
}

impl Row {
    fn greedy(seq_id: SequenceId) -> Self {
        let sampling = SamplingConfig::greedy();
        Self {
            seq_id,
            sampler: RowSampler::new(&sampling),
            sampling,
            history: Vec::new(),
            generated: Vec::new(),
            eos: vec![7],
            max_tokens: 8,
            logprobs: LogprobsConfig::default(),
            hooks: Hooks::default(),
        }
    }

    fn row(&mut self) -> StepRow<'_, &mut Hooks> {
        StepRow {
            seq_id: self.seq_id,
            sampler: &mut self.sampler,
            sampling: &self.sampling,
            token_history: &mut self.history,
            generated: &mut self.generated,
            eos: &self.eos,
            max_tokens: self.max_tokens,
            logprobs: &self.logprobs,
            needs_mask: self.hooks.force.is_some(),
            needs_override: self.hooks.override_to.is_some(),
            hooks: &mut self.hooks,
        }
    }
}

impl FinishHooks for &mut Hooks {
    fn stop_text(&mut self, token: i32, generated_len: usize) -> bool {
        (**self).stop_text(token, generated_len)
    }
    fn bound_stopped(&self) -> bool {
        (**self).bound_stopped()
    }
    fn context_bound_due(&self, generated_len: usize) -> bool {
        (**self).context_bound_due(generated_len)
    }
}

impl StepRowHooks for &mut Hooks {
    fn logit_mask(&mut self) -> Option<&mut dyn LogitMask> {
        (**self).logit_mask()
    }
    fn override_token(&mut self, sampled: i32) -> i32 {
        (**self).override_token(sampled)
    }
    fn consume_sampled(&mut self, sampled: i32) -> Result<bool, String> {
        (**self).consume_sampled(sampled)
    }
    fn resolved(&mut self, sampled: i32, token: i32) {
        (**self).resolved(sampled, token)
    }
}

fn offset(engine: &Engine<EchoModel>, id: SequenceId) -> i32 {
    engine.pool().get(id).unwrap().current_offset
}

fn seq_len(engine: &Engine<EchoModel>, id: SequenceId) -> i32 {
    engine.pool().get(id).unwrap().caches[0].seq_len()
}

#[test]
fn open_allocates_and_close_releases() {
    let mut engine = engine();
    let id = engine.open(SequenceSpec::default()).unwrap();
    assert!(engine.is_open(id));
    assert_eq!(engine.pool().active_count(), 1);
    let closed = engine.close(id);
    assert_eq!(closed, ClosedSequence { id, was_open: true });
    assert!(!engine.is_open(id));
    assert!(!engine.close(id).was_open);
}

#[test]
fn step_of_one_runs_the_single_row_forward() {
    let mut engine = engine();
    let id = engine.open(SequenceSpec::default()).unwrap();
    let mut row = Row::greedy(id);
    let input = from_slice_i32(&[3], &[1, 1]);
    let out = engine
        .step(
            &StepBatch {
                seq_ids: &[id],
                input: &input,
            },
            &mut [row.row()],
        )
        .unwrap();
    assert_eq!(
        out.rows,
        vec![RowOutcome {
            seq_id: id,
            sampled: 3,
            token: 3,
            finish: None,
            error: None
        }]
    );
    assert_eq!(row.generated, vec![3]);
    assert_eq!(engine.model().single_calls.get(), 1);
    assert_eq!(engine.model().batched_calls.get(), 0);
    assert_eq!(seq_len(&engine, id), 1);
    assert_eq!(offset(&engine, id), 1);
}

/// A batch of one samples through the per-row chain even when its row is
/// fused-eligible, so B=1 keeps the per-row `try_eval` guard (#822) and the
/// sampler path the single-sequence decode always took. The chain is visible
/// through `resolved`, which the fused path never calls.
#[test]
fn a_batch_of_one_runs_the_per_row_chain() {
    let mut engine = engine();
    let id = engine.open(SequenceSpec::default()).unwrap();
    let mut row = Row::greedy(id);
    let input = from_slice_i32(&[3], &[1, 1]);
    let out = {
        let mut views = [row.row()];
        assert!(
            fused_params(&views).is_some(),
            "the row alone is fused-eligible"
        );
        engine
            .step(
                &StepBatch {
                    seq_ids: &[id],
                    input: &input,
                },
                &mut views,
            )
            .unwrap()
    };
    assert_eq!((out.rows[0].sampled, out.rows[0].token), (3, 3));
    assert_eq!(row.hooks.resolved, vec![(3, 3)]);
    assert_eq!(row.generated, vec![3]);
}

/// The shared eligibility rule: one ineligible or unresolved row, or rows
/// with different scalar parameters, keep the whole batch off the fused
/// draw; an empty batch has no shared parameters.
#[test]
fn shared_fused_params_needs_every_row() {
    let greedy = SamplingConfig::greedy();
    let gate = |sampling| FusedRowGate {
        sampling,
        needs_mask: false,
        needs_override: false,
        logprobs_enabled: false,
    };
    assert!(shared_fused_params([Some(gate(&greedy)), Some(gate(&greedy))]).is_some());
    assert!(shared_fused_params([Some(gate(&greedy)), None]).is_none());
    assert!(shared_fused_params(std::iter::empty::<Option<FusedRowGate<'_>>>()).is_none());
    for obligation in 0..3 {
        let mut g = gate(&greedy);
        match obligation {
            0 => g.needs_mask = true,
            1 => g.needs_override = true,
            _ => g.logprobs_enabled = true,
        }
        assert!(shared_fused_params([Some(gate(&greedy)), Some(g)]).is_none());
    }
    let warm = SamplingConfig {
        temperature: 0.7,
        ..SamplingConfig::greedy()
    };
    assert!(shared_fused_params([Some(gate(&greedy)), Some(gate(&warm))]).is_none());
}

#[test]
fn step_of_many_runs_the_batched_forward_in_row_order() {
    let mut engine = engine();
    let ids: Vec<SequenceId> = (0..3)
        .map(|_| engine.open(SequenceSpec::default()).unwrap())
        .collect();
    let mut rows: Vec<Row> = ids.iter().map(|&id| Row::greedy(id)).collect();
    // Row 2 draws the EOS id: it finishes, is not pushed, and its offset
    // stays put while the others advance.
    let input = from_slice_i32(&[5, 1, 7], &[3, 1]);
    let mut views: Vec<_> = rows.iter_mut().map(Row::row).collect();
    let out = engine
        .step(
            &StepBatch {
                seq_ids: &ids,
                input: &input,
            },
            &mut views,
        )
        .unwrap();
    drop(views);
    let tokens: Vec<i32> = out.rows.iter().map(|r| r.token).collect();
    assert_eq!(tokens, vec![5, 1, 7]);
    assert_eq!(out.rows[2].finish, Some(FinishCause::Eos));
    assert_eq!(out.rows[0].finish, None);
    assert_eq!(rows[0].generated, vec![5]);
    assert_eq!(rows[2].generated, Vec::<i32>::new());
    assert_eq!(engine.model().batched_calls.get(), 1);
    assert_eq!(engine.model().single_calls.get(), 0);
    for id in &ids {
        assert_eq!(seq_len(&engine, *id), 1);
    }
    assert_eq!(offset(&engine, ids[0]), 1);
    assert_eq!(offset(&engine, ids[2]), 0);
}

#[test]
fn fused_and_per_row_paths_agree_for_greedy_rows() {
    let mut engine = engine();
    let a = engine.open(SequenceSpec::default()).unwrap();
    let b = engine.open(SequenceSpec::default()).unwrap();
    let input = from_slice_i32(&[2, 6], &[2, 1]);
    let batch = StepBatch {
        seq_ids: &[a, b],
        input: &input,
    };
    let mut fused = [Row::greedy(a), Row::greedy(b)];
    {
        let mut views: Vec<_> = fused.iter_mut().map(Row::row).collect();
        assert!(fused_params(&views).is_some());
        engine.step(&batch, &mut views).unwrap();
    }
    // Logprobs keep a row off the fused path, so the batch takes the per-row
    // chain; the tokens are the same.
    let mut per_row = [Row::greedy(a), Row::greedy(b)];
    per_row[1].logprobs.enabled = true;
    {
        let mut views: Vec<_> = per_row.iter_mut().map(Row::row).collect();
        assert!(fused_params(&views).is_none());
        engine.step(&batch, &mut views).unwrap();
    }
    assert_eq!(fused[0].generated, per_row[0].generated);
    assert_eq!(fused[1].generated, per_row[1].generated);
    assert_eq!(per_row[0].generated, vec![2]);
    // The per-row chain reports the resolved token; the fused path has no
    // override and never calls it.
    assert_eq!(per_row[1].hooks.resolved, vec![(6, 6)]);
    assert!(fused[1].hooks.resolved.is_empty());
}

#[test]
fn mask_override_and_matcher_run_in_the_per_row_chain() {
    let mut engine = engine();
    let id = engine.open(SequenceSpec::default()).unwrap();
    let mut row = Row::greedy(id);
    row.hooks.force = Some(4);
    row.hooks.override_to = Some(5);
    let input = from_slice_i32(&[1], &[1, 1]);
    let batch = StepBatch {
        seq_ids: &[id],
        input: &input,
    };
    let out = engine.step(&batch, &mut [row.row()]).unwrap();
    // The mask forced token 4, the override emitted 5, and the matcher saw
    // the pre-override 4 through `resolved`.
    assert_eq!((out.rows[0].sampled, out.rows[0].token), (4, 5));
    assert_eq!(row.generated, vec![5]);
    assert_eq!(row.hooks.resolved, vec![(4, 5)]);
}

#[test]
fn a_failed_row_never_touches_its_neighbours() {
    let mut engine = engine();
    let a = engine.open(SequenceSpec::default()).unwrap();
    let b = engine.open(SequenceSpec::default()).unwrap();
    let c = engine.open(SequenceSpec::default()).unwrap();
    let mut rows = [Row::greedy(a), Row::greedy(b), Row::greedy(c)];
    rows[0].hooks.force = Some(-1); // mask failure
    rows[2].hooks.fail_consume = true; // matcher failure
    let input = from_slice_i32(&[1, 2, 3], &[3, 1]);
    let batch = StepBatch {
        seq_ids: &[a, b, c],
        input: &input,
    };
    let out = {
        let mut views: Vec<_> = rows.iter_mut().map(Row::row).collect();
        engine.step(&batch, &mut views).unwrap()
    };
    assert_eq!(
        out.rows[0].error,
        Some(RowError::Structured("mask rejected the row".to_string()))
    );
    assert_eq!(out.rows[1].error, None);
    assert_eq!(out.rows[1].token, 2);
    assert_eq!(
        out.rows[2].error,
        Some(RowError::Structured("matcher failed".to_string()))
    );
    assert!(out.rows.iter().all(RowOutcome::eval_ok));
    assert_eq!(rows[1].generated, vec![2]);
    assert!(rows[0].generated.is_empty());
    assert!(rows[2].generated.is_empty());
    // Only the healthy row advanced its offset.
    assert_eq!(offset(&engine, a), 0);
    assert_eq!(offset(&engine, b), 1);
    assert_eq!(offset(&engine, c), 0);
}

#[test]
fn step_names_the_missing_row() {
    let mut engine = engine();
    let a = engine.open(SequenceSpec::default()).unwrap();
    let gone = SequenceId::from_raw(u64::MAX);
    let input = from_slice_i32(&[1], &[1, 1]);
    let mut row = Row::greedy(gone);
    assert_eq!(
        engine
            .step(
                &StepBatch {
                    seq_ids: &[gone],
                    input: &input,
                },
                &mut [row.row()],
            )
            .err(),
        Some(EngineError::MissingSequence(gone))
    );
    let input = from_slice_i32(&[1, 2], &[2, 1]);
    let mut rows = [Row::greedy(a), Row::greedy(gone)];
    let mut views: Vec<_> = rows.iter_mut().map(Row::row).collect();
    assert!(matches!(
        engine.step(
            &StepBatch {
                seq_ids: &[a, gone],
                input: &input,
            },
            &mut views,
        ),
        Err(EngineError::Batch(_))
    ));
    let mut no_rows: [StepRow<'_, &mut Hooks>; 0] = [];
    assert!(matches!(
        engine.step(
            &StepBatch {
                seq_ids: &[],
                input: &input,
            },
            &mut no_rows,
        ),
        Err(EngineError::Batch(_))
    ));
    // The failed batch left the live row untouched.
    assert_eq!(seq_len(&engine, a), 0);
}

/// A row count that differs from the batch is refused before the forward, so
/// no cache gains a position the sampler would never finish or advance.
#[test]
fn step_refuses_a_row_count_that_differs_from_the_batch() {
    let mut engine = engine();
    let a = engine.open(SequenceSpec::default()).unwrap();
    let b = engine.open(SequenceSpec::default()).unwrap();
    let input = from_slice_i32(&[1, 2], &[2, 1]);
    let mut rows = [Row::greedy(a)];
    let mut views: Vec<_> = rows.iter_mut().map(Row::row).collect();
    assert!(matches!(
        engine.step(
            &StepBatch {
                seq_ids: &[a, b],
                input: &input,
            },
            &mut views,
        ),
        Err(EngineError::Batch(_))
    ));
    assert_eq!(engine.model().single_calls.get(), 0);
    assert_eq!(engine.model().batched_calls.get(), 0);
    for id in [a, b] {
        assert_eq!(offset(&engine, id), 0);
        assert_eq!(seq_len(&engine, id), 0);
    }
}

/// A fused draw whose readback holds fewer tokens than rows would let the
/// per-row zip drop a row whose KV was appended. It fails every row with one
/// shared `BatchEval` instead, finishes none and advances none.
#[test]
fn fused_readback_token_count_mismatch_fails_every_row() {
    let mut engine = Engine::with_capacity(EchoModel::with_short_batched_logits(), 4);
    let a = engine.open(SequenceSpec::default()).unwrap();
    let b = engine.open(SequenceSpec::default()).unwrap();
    let input = from_slice_i32(&[2, 6], &[2, 1]);
    let batch = StepBatch {
        seq_ids: &[a, b],
        input: &input,
    };
    let mut rows = [Row::greedy(a), Row::greedy(b)];
    let out = {
        let mut views: Vec<_> = rows.iter_mut().map(Row::row).collect();
        // Both rows are fused-eligible, so the step takes the fused draw.
        assert!(fused_params(&views).is_some());
        engine.step(&batch, &mut views).unwrap()
    };
    assert_eq!(engine.model().batched_calls.get(), 1);
    assert_eq!(out.rows.len(), 2);
    let messages: Vec<&str> = out
        .rows
        .iter()
        .map(|outcome| match &outcome.error {
            Some(RowError::BatchEval(msg)) => msg.as_str(),
            other => panic!("expected BatchEval, got {other:?}"),
        })
        .collect();
    assert_eq!(messages[0], messages[1]);
    assert!(
        messages[0].contains("1 tokens for 2 rows"),
        "{}",
        messages[0]
    );
    for (outcome, id) in out.rows.iter().zip([a, b]) {
        assert_eq!(outcome.seq_id, id);
        assert!(!outcome.eval_ok());
        assert_eq!(offset(&engine, id), 0);
    }
    assert!(rows.iter().all(|row| row.generated.is_empty()));
}

/// A pipelined collect whose token count differs from the rows fails every
/// row with one shared `BatchEval` and advances none.
#[test]
fn finish_rows_fails_every_row_on_a_token_count_mismatch() {
    let mut engine = engine();
    let a = engine.open(SequenceSpec::default()).unwrap();
    let b = engine.open(SequenceSpec::default()).unwrap();
    let mut rows = [Row::greedy(a), Row::greedy(b)];
    let outcomes = {
        let mut views: Vec<_> = rows.iter_mut().map(Row::row).collect();
        engine.finish_rows(&[2], &mut views)
    };
    assert_eq!(outcomes.len(), 2);
    for (outcome, id) in outcomes.iter().zip([a, b]) {
        assert_eq!(outcome.seq_id, id);
        assert!(matches!(outcome.error, Some(RowError::BatchEval(_))));
        assert!(!outcome.eval_ok());
        assert_eq!(offset(&engine, id), 0);
    }
    assert!(rows.iter().all(|row| row.generated.is_empty()));
}

#[test]
fn submit_then_finish_rows_is_one_pipelined_step() {
    let mut engine = engine();
    let a = engine.open(SequenceSpec::default()).unwrap();
    let b = engine.open(SequenceSpec::default()).unwrap();
    let mut rows = [Row::greedy(a), Row::greedy(b)];
    let input = from_slice_i32(&[2, 7], &[2, 1]);
    let batch = StepBatch {
        seq_ids: &[a, b],
        input: &input,
    };
    let params = crate::sampling::FusedSampleParams::from_config(&rows[0].sampling);
    let biases: Vec<&crate::sampling::TokenBiasMap> =
        rows.iter().map(|r| &r.sampling.token_bias).collect();
    let tokens = engine.submit(&batch, &params, &biases).unwrap();
    let host = tokens_to_host(&tokens);
    assert_eq!(host, vec![2, 7]);
    // The speculative appends are in place until the caller decides.
    assert_eq!(seq_len(&engine, a), 1);
    let outcomes = {
        let mut views: Vec<_> = rows.iter_mut().map(Row::row).collect();
        engine.finish_rows(&host, &mut views)
    };
    assert_eq!(outcomes[0].finish, None);
    assert_eq!(outcomes[1].finish, Some(FinishCause::Eos));
    assert_eq!(rows[0].generated, vec![2]);
    assert_eq!(offset(&engine, a), 1);
    assert_eq!(offset(&engine, b), 0);
}

#[test]
fn speculative_step_is_undone_by_unwind_appends() {
    let mut engine = engine();
    let a = engine.open(SequenceSpec::default()).unwrap();
    let b = engine.open(SequenceSpec::default()).unwrap();
    let input = from_slice_i32(&[1, 2], &[2, 1]);
    let batch = StepBatch {
        seq_ids: &[a, b],
        input: &input,
    };
    let params = crate::sampling::FusedSampleParams::from_config(&SamplingConfig::greedy());
    let bias = crate::sampling::TokenBiasMap::default();
    engine.submit(&batch, &params, &[&bias, &bias]).unwrap();
    engine.submit(&batch, &params, &[&bias, &bias]).unwrap();
    engine.submit(&batch, &params, &[&bias, &bias]).unwrap();
    assert_eq!(seq_len(&engine, a), 3);
    engine.unwind_appends(a, 2).unwrap();
    engine.unwind_appends(b, 1).unwrap();
    assert_eq!(seq_len(&engine, a), 1);
    assert_eq!(seq_len(&engine, b), 2);
    // Zero and a closed row are no-ops.
    engine.unwind_appends(a, 0).unwrap();
    engine
        .unwind_appends(SequenceId::from_raw(u64::MAX), 1)
        .unwrap();
}

#[test]
fn prefill_returns_last_real_logits_and_trims_padding() {
    let mut engine = engine();
    let id = engine.open(SequenceSpec::default()).unwrap();
    // Four real tokens padded to six.
    let input = from_slice_i32(&[2, 4, 6, 1, 0, 0], &[1, 6]);
    let outcome = engine
        .prefill(&PrefillStep {
            seq_id: id,
            input: &input,
            embeddings: None,
            mask: None,
            last_pos: 3,
            trim_excess: 2,
            eval: true,
        })
        .unwrap();
    assert!(outcome.eval.is_ok());
    assert!(outcome.trim.is_ok());
    assert_eq!(argmax_rows(&outcome.logits), vec![1]);
    assert_eq!(seq_len(&engine, id), 4);
    assert_eq!(
        engine
            .prefill(&PrefillStep {
                seq_id: SequenceId::from_raw(u64::MAX),
                input: &input,
                embeddings: None,
                mask: None,
                last_pos: 3,
                trim_excess: 0,
                eval: false,
            })
            .err(),
        Some(EngineError::MissingSequence(SequenceId::from_raw(u64::MAX)))
    );
}

#[test]
fn prefill_cohort_covers_every_row() {
    let mut engine = engine();
    let a = engine.open(SequenceSpec::default()).unwrap();
    let b = engine.open(SequenceSpec::default()).unwrap();
    let input = from_slice_i32(&[1, 2, 3, 4, 5, 0], &[2, 3]);
    let logits = engine.prefill_cohort(&[a, b], &input, None).unwrap();
    assert_eq!(ffi::array_shape(&logits), vec![2, 3, 8]);
    assert_eq!(seq_len(&engine, a), 3);
    assert_eq!(seq_len(&engine, b), 3);
    engine.trim_padding(b, 1).unwrap();
    assert_eq!(seq_len(&engine, b), 2);
    assert!(matches!(
        engine.trim_padding(SequenceId::from_raw(u64::MAX), 1),
        Err(EngineError::MissingSequence(_))
    ));
}

/// The raw-completion client (#2176) runs the same open, prefill, first
/// token, step and close sequence by hand-driven engine calls, over an
/// owned model and over a borrowed one, and closes its sequence afterwards.
#[test]
fn direct_engine_matches_a_hand_driven_sequence_and_closes_it() {
    let prompt = [3, 5, 1];
    let greedy = SamplingConfig::greedy();

    // Hand-driven: the echo model repeats its input, so greedy decode from
    // the last prompt token yields that token until the budget is spent.
    let mut engine = engine();
    let id = engine.open(SequenceSpec::default()).unwrap();
    let mut row = Row::greedy(id);
    row.max_tokens = 4;
    let input = from_slice_i32(&prompt, &[1, 3]);
    let outcome = engine
        .prefill(&PrefillStep {
            seq_id: id,
            input: &input,
            embeddings: None,
            mask: None,
            last_pos: 2,
            trim_excess: 0,
            eval: false,
        })
        .unwrap();
    engine.complete_prefill(&outcome.logits, &mut row.row());
    while row.generated.len() < 4 {
        let last = *row.generated.last().unwrap();
        let input = from_slice_i32(&[last], &[1, 1]);
        engine
            .step(
                &StepBatch {
                    seq_ids: &[id],
                    input: &input,
                },
                &mut [row.row()],
            )
            .unwrap();
    }
    let expected = row.generated.clone();
    assert_eq!(expected, vec![1, 1, 1, 1]);

    let mut owned = DirectEngine::new(EchoModel::new(), 0);
    let mut seen = Vec::new();
    let run = owned
        .generate(&DirectRequest::text(&prompt, 4, &greedy), |t| {
            seen.push(t);
            true
        })
        .unwrap();
    assert_eq!(run.tokens, expected);
    assert_eq!(seen, expected, "every appended token reaches the callback");
    assert_eq!(run.stats.generated_tokens, 4);
    assert_eq!(run.stats.prompt_tokens, 3);
    assert_eq!(
        owned.model().single_calls.get(),
        4,
        "one prefill plus three steps"
    );

    let model = EchoModel::new();
    let mut borrowed = DirectEngine::new(&model, 0);
    assert_eq!(borrowed.run(&prompt, 4, &greedy).unwrap(), expected);
    assert_eq!(model.single_calls.get(), 4);
    // A second run starts from a fresh sequence: same output, no leftover.
    assert_eq!(borrowed.run(&prompt, 4, &greedy).unwrap(), expected);
    assert_eq!(borrowed.into_model().single_calls.get(), 8);
}

/// The callback can stop a run early, and an EOS token is neither stored
/// nor delivered.
#[test]
fn direct_engine_honors_the_callback_and_withholds_eos() {
    let greedy = SamplingConfig::greedy();
    let mut client = DirectEngine::new(EchoModel::new(), 0);
    let mut seen = Vec::new();
    let run = client
        .generate(&DirectRequest::text(&[2, 4], 10, &greedy), |t| {
            seen.push(t);
            seen.len() < 2
        })
        .unwrap();
    assert_eq!(run.tokens, vec![4, 4]);
    assert_eq!(seen, vec![4, 4]);
    assert_eq!(client.model().single_calls.get(), 2);

    // The echo model's EOS is 7: a prompt ending in it finishes at once.
    let mut seen = Vec::new();
    let run = client
        .generate(&DirectRequest::text(&[7], 10, &greedy), |t| {
            seen.push(t);
            true
        })
        .unwrap();
    assert!(run.tokens.is_empty());
    assert!(seen.is_empty());
    assert_eq!(client.model().single_calls.get(), 3, "the prefill only");
}
