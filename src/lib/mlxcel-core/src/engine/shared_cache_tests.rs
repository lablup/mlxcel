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

//! A model that cannot batch and keeps one set of caches inside itself
//! (ExaOne 4, GPT-OSS, Ministral 3, Step 3.5) starts every raw completion on
//! fresh caches (#2494): the client resets them when it opens the sequence,
//! as the scheduler does at prefill time, so a warmup run does not leak its
//! KV into the timed run that follows it.

use std::cell::{Cell, RefCell};

use super::*;
use crate::ffi;
use crate::generate::{LanguageModel, SamplingConfig};
use crate::layers::KVCache;
use crate::utils::array_to_vec_f32;

const VOCAB: usize = 8;
const EOS: i32 = 7;

fn host_tokens(input_ids: &MlxArray) -> Vec<i32> {
    ffi::eval(input_ids);
    array_to_vec_f32(&ffi::astype(input_ids, crate::dtype::FLOAT32))
        .into_iter()
        .map(|t| t as i32)
        .collect()
}

/// Holds its KV length in a cell that only `make_caches` resets, like the
/// `RefCell` caches of the real wrappers, and ignores the caller's caches.
/// The top logit depends on that length, so a stream decoded on stale state
/// differs from one decoded on fresh state.
#[derive(Default)]
struct SharedCacheModel {
    offset: Cell<i32>,
    /// The internal offset each forward started from.
    starts: RefCell<Vec<i32>>,
    resets: Cell<usize>,
}

impl LanguageModel for SharedCacheModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = ffi::array_shape(input_ids);
        let tokens = host_tokens(input_ids);
        let start = self.offset.get();
        self.starts.borrow_mut().push(start);
        let mut logits = vec![0.0f32; tokens.len() * VOCAB];
        for (i, tok) in tokens.iter().enumerate() {
            let position = start + i as i32;
            let top = (tok + position + 1).rem_euclid(EOS);
            logits[i * VOCAB + top as usize] = 10.0;
        }
        self.offset.set(start + tokens.len() as i32);
        ffi::from_slice_f32(&logits, &[shape[0], shape[1], VOCAB as i32])
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.offset.set(0);
        self.resets.set(self.resets.get() + 1);
        vec![KVCache::new(), KVCache::new()]
    }

    fn num_layers(&self) -> usize {
        2
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![EOS]
    }

    fn supports_batching(&self) -> bool {
        false
    }
}

fn greedy() -> SamplingConfig {
    SamplingConfig {
        temperature: 0.0,
        ..SamplingConfig::default()
    }
}

/// `mlxcel generate`'s shape: a one-token warmup, then the timed run on the
/// same client. The timed run's prefill starts at offset zero and decodes
/// the stream a fresh client decodes; on main before #2494 it started at the
/// warmup's prompt length and decoded a different stream.
#[test]
fn warmup_does_not_leak_into_the_next_completion() {
    let prompt = [1, 2, 3];
    let sampling = greedy();

    let fresh = SharedCacheModel::default();
    let expected = DirectEngine::new(&fresh, 512)
        .run(&prompt, 6, &sampling)
        .expect("fresh run");
    assert_eq!(expected.len(), 6, "the stub decodes the whole budget");

    let model = SharedCacheModel::default();
    let mut client = DirectEngine::new(&model, 512);
    client.run(&prompt, 1, &sampling).expect("warmup");
    let forwards_before_timed = model.starts.borrow().len();
    let timed = client.run(&prompt, 6, &sampling).expect("timed run");

    assert_eq!(
        model.starts.borrow()[forwards_before_timed],
        0,
        "the timed run's prefill continued on the warmup's KV"
    );
    assert_eq!(timed, expected);
}

/// Every opened sequence resets the model's caches once, whether it
/// decodes, scores, or ends at its first token.
#[test]
fn every_open_resets_the_shared_caches() {
    let model = SharedCacheModel::default();
    let sampling = greedy();
    let mut client = DirectEngine::new(&model, 512);

    client.run(&[1, 2], 1, &sampling).expect("first");
    assert_eq!(model.resets.get(), 1);
    client.run(&[1, 2], 4, &sampling).expect("second");
    assert_eq!(model.resets.get(), 2);
    client.loglikelihoods(&[1, 2, 3]).expect("score");
    assert_eq!(model.resets.get(), 3);
    assert_eq!(
        model.starts.borrow().last().copied(),
        Some(0),
        "the scoring pass started on fresh caches"
    );
    // A zero budget opens nothing, so it resets nothing.
    client.run(&[1, 2], 0, &sampling).expect("zero budget");
    assert_eq!(model.resets.get(), 3);
}
