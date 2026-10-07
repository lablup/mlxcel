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
use crate::generate::LanguageModel;
use crate::layers::KVCache;
use crate::utils::array_to_vec_f32;
use crate::{ffi, from_slice_i32};

/// Logits put the input token id at the top of a vocab of 8, so a sampled
/// or argmax token echoes the input and the stacked batch order is visible.
/// Every forward appends one position to each cache it is given.
struct EchoModel {
    single_calls: Cell<usize>,
    batched_calls: Cell<usize>,
}

impl EchoModel {
    fn new() -> Self {
        Self {
            single_calls: Cell::new(0),
            batched_calls: Cell::new(0),
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
        for caches in batch_caches.iter_mut() {
            Self::append(caches, 1);
        }
        Self::echo_logits(input_ids)
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
    let input = from_slice_i32(&[3], &[1, 1]);
    let out = engine
        .step(&StepBatch {
            seq_ids: &[id],
            input: &input,
            context: None,
        })
        .unwrap();
    assert_eq!(argmax_rows(&out.logits), vec![3]);
    assert_eq!(engine.model().single_calls.get(), 1);
    assert_eq!(engine.model().batched_calls.get(), 0);
    assert_eq!(seq_len(&engine, id), 1);
}

#[test]
fn step_of_many_runs_the_batched_forward_in_row_order() {
    let mut engine = engine();
    let a = engine.open(SequenceSpec::default()).unwrap();
    let b = engine.open(SequenceSpec::default()).unwrap();
    let c = engine.open(SequenceSpec::default()).unwrap();
    let input = from_slice_i32(&[5, 1, 6], &[3, 1]);
    let out = engine
        .step(&StepBatch {
            seq_ids: &[a, b, c],
            input: &input,
            context: None,
        })
        .unwrap();
    assert_eq!(argmax_rows(&out.logits), vec![5, 1, 6]);
    assert_eq!(engine.model().batched_calls.get(), 1);
    assert_eq!(engine.model().single_calls.get(), 0);
    for id in [a, b, c] {
        assert_eq!(seq_len(&engine, id), 1);
    }
}

#[test]
fn step_names_the_missing_row() {
    let mut engine = engine();
    let a = engine.open(SequenceSpec::default()).unwrap();
    let gone = SequenceId::from_raw(u64::MAX);
    let input = from_slice_i32(&[1], &[1, 1]);
    assert_eq!(
        engine
            .step(&StepBatch {
                seq_ids: &[gone],
                input: &input,
                context: None,
            })
            .err(),
        Some(EngineError::MissingSequence(gone))
    );
    let input = from_slice_i32(&[1, 2], &[2, 1]);
    assert!(matches!(
        engine.step(&StepBatch {
            seq_ids: &[a, gone],
            input: &input,
            context: None,
        }),
        Err(EngineError::Batch(_))
    ));
    assert!(matches!(
        engine.step(&StepBatch {
            seq_ids: &[],
            input: &input,
            context: None,
        }),
        Err(EngineError::Batch(_))
    ));
    // The failed batch left the live row untouched.
    assert_eq!(seq_len(&engine, a), 0);
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
        context: None,
    };
    engine.step(&batch).unwrap();
    engine.step_speculative(&batch).unwrap();
    engine.step_speculative(&batch).unwrap();
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
