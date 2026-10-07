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

//! The raw-completion client's lookahead pipeline against its synchronous
//! loop (#2176): the same stream, the same callback deliveries and the same
//! sequence state at every kind of finish, and the synchronous fallback for
//! every request the eligibility rules turn away.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use super::direct_decode::Teardown;
use super::*;
use crate::cache::{SequenceId, SequenceStateLayout};
use crate::generate::{LanguageModel, SamplingConfig};
use crate::layers::KVCache;
use crate::utils::array_to_vec_f32;
use crate::{ffi, from_slice_i32};

const VOCAB: usize = 8;
const EOS: i32 = 7;

fn host_tokens(input_ids: &MlxArray) -> Vec<i32> {
    ffi::eval(input_ids);
    array_to_vec_f32(&ffi::astype(input_ids, crate::dtype::FLOAT32))
        .into_iter()
        .map(|t| t as i32)
        .collect()
}

/// `[B, L, 8]` logits that put `(token + 1) % 8` on top at every position,
/// so greedy decode counts up from the prompt's last token to the EOS (7).
fn count_logits(input_ids: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = ffi::array_shape(input_ids);
    let tokens = host_tokens(input_ids);
    let mut logits = vec![0.0f32; tokens.len() * VOCAB];
    for (i, tok) in tokens.into_iter().enumerate() {
        logits[i * VOCAB + ((tok + 1).rem_euclid(VOCAB as i32)) as usize] = 10.0;
    }
    ffi::from_slice_f32(&logits, &[shape[0], shape[1], VOCAB as i32])
}

/// A dense-cache counting model that records every forward's input.
#[derive(Default)]
struct CountModel {
    inputs: RefCell<Vec<Vec<i32>>>,
}

impl CountModel {
    fn forwards(&self) -> usize {
        self.inputs.borrow().len()
    }
}

impl LanguageModel for CountModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let n = ffi::array_shape(input_ids)[1];
        for cache in caches.iter_mut() {
            let k = ffi::zeros(&[1, 1, n, 4], crate::dtype::FLOAT32);
            let v = ffi::zeros(&[1, 1, n, 4], crate::dtype::FLOAT32);
            cache.update(k, v);
        }
        self.inputs.borrow_mut().push(host_tokens(input_ids));
        count_logits(input_ids)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        vec![KVCache::new(), KVCache::new()]
    }

    fn num_layers(&self) -> usize {
        2
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![EOS]
    }

    fn supports_batching(&self) -> bool {
        true
    }
}

/// A model-owned counting model: it keeps each sequence's length itself and
/// can be told whether it rewinds speculative appends (#2159).
struct OwnedModel {
    rewinds: bool,
    lengths: RefCell<HashMap<SequenceId, i32>>,
    released: RefCell<Vec<i32>>,
    forwards: Cell<usize>,
}

impl OwnedModel {
    fn new(rewinds: bool) -> Self {
        Self {
            rewinds,
            lengths: RefCell::new(HashMap::new()),
            released: RefCell::new(Vec::new()),
            forwards: Cell::new(0),
        }
    }
}

impl LanguageModel for OwnedModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        count_logits(input_ids)
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forwards.set(self.forwards.get() + 1);
        let id = seq_id.expect("the engine always passes the sequence id");
        *self.lengths.borrow_mut().entry(id).or_insert(0) += ffi::array_shape(input_ids)[1];
        count_logits(input_ids)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Vec::new()
    }

    fn num_layers(&self) -> usize {
        1
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![EOS]
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        SequenceStateLayout::model_owned(1)
    }

    fn supports_decode_lookahead_rewind(&self) -> bool {
        self.rewinds
    }

    fn rewind_decode_appends(&self, seq_id: SequenceId, n: i32) -> Result<(), String> {
        let mut lengths = self.lengths.borrow_mut();
        let len = lengths.get_mut(&seq_id).ok_or("unknown sequence")?;
        *len -= n;
        Ok(())
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        if let Some(len) = self.lengths.borrow_mut().remove(&seq_id) {
            self.released.borrow_mut().push(len);
        }
    }
}

/// What one run leaves behind: the stream, the callback deliveries, and the
/// sequence's pool offset and KV length right before it is closed.
#[derive(Debug, PartialEq, Eq)]
struct Trace {
    tokens: Vec<i32>,
    delivered: Vec<i32>,
    offset: i32,
    kv_len: Option<i32>,
}

/// Run one completion through the client's own open / run / close
/// sequence, stopping the callback after `deliveries` tokens when given.
fn trace<M: LanguageModel>(
    client: &mut DirectEngine<M>,
    prompt: &[i32],
    max_tokens: usize,
    sampling: &SamplingConfig,
    deliveries: Option<usize>,
) -> Trace {
    let id = client.open_sequence().unwrap();
    let mut delivered = Vec::new();
    let run = client
        .run_open_sequence(
            id,
            &DirectRequest::text(prompt, max_tokens, sampling),
            sampling,
            0,
            |t| {
                delivered.push(t);
                deliveries.is_none_or(|n| delivered.len() < n)
            },
        )
        .unwrap();
    let set = client.engine().pool().get(id).unwrap();
    let trace = Trace {
        tokens: run.tokens,
        delivered,
        offset: set.current_offset,
        kv_len: set.caches.first().map(KVCache::seq_len),
    };
    client.close_sequence(id);
    trace
}

/// Whether the client would take the pipeline for `sampling` on a fresh
/// sequence.
fn pipelines<M: LanguageModel>(client: &mut DirectEngine<M>, sampling: &SamplingConfig) -> bool {
    let id = client.open_sequence().unwrap();
    let eligible = client.lookahead_params(id, sampling).is_some();
    client.close_sequence(id);
    eligible
}

/// Pipelined and synchronous runs of one request on fresh dense clients,
/// with each run's forward count.
fn both(
    prompt: &[i32],
    max_tokens: usize,
    sampling: &SamplingConfig,
    deliveries: Option<usize>,
) -> ((Trace, usize), (Trace, usize)) {
    let run = |force_sync: bool| {
        let model = CountModel::default();
        let mut client = DirectEngine::new(&model, 0).with_force_sync(force_sync);
        assert_eq!(pipelines(&mut client, sampling), !force_sync);
        let trace = trace(&mut client, prompt, max_tokens, sampling, deliveries);
        (trace, model.forwards())
    };
    (run(false), run(true))
}

#[test]
fn eos_finish_matches_the_synchronous_loop() {
    let greedy = SamplingConfig::greedy();
    let ((piped, piped_fwd), (sync, sync_fwd)) = both(&[2, 3], 16, &greedy, None);
    assert_eq!(piped, sync);
    assert_eq!(
        sync.tokens,
        vec![4, 5, 6],
        "EOS is neither stored nor delivered"
    );
    assert_eq!(sync.delivered, vec![4, 5, 6]);
    // The EOS step's forward appended 6 (prompt 2 + 4, 5, 6); the offset is
    // `prompt_len + 1` after the first token, as the scheduler sets it, and
    // advanced only for the two steps that continued.
    assert_eq!(sync.kv_len, Some(5));
    assert_eq!(sync.offset, 5);
    // Prefill plus the steps fed 4, 5, 6; the pipeline also submitted the
    // forward fed by the EOS before it read it, and unwound it.
    assert_eq!(sync_fwd, 4);
    assert_eq!(piped_fwd, 5);
}

#[test]
fn stop_id_finish_matches_the_synchronous_loop() {
    let sampling = SamplingConfig {
        stop_token_ids: vec![5],
        ..SamplingConfig::greedy()
    };
    let ((piped, _), (sync, _)) = both(&[0, 3], 16, &sampling, None);
    assert_eq!(piped, sync);
    assert_eq!(sync.tokens, vec![4]);
}

#[test]
fn length_finish_matches_the_synchronous_loop_with_no_extra_forward() {
    let greedy = SamplingConfig::greedy();
    for max_tokens in [1, 2, 3] {
        let ((piped, piped_fwd), (sync, sync_fwd)) = both(&[0], max_tokens, &greedy, None);
        assert_eq!(piped, sync, "max_tokens {max_tokens}");
        assert_eq!(sync.tokens, (1..=max_tokens as i32).collect::<Vec<_>>());
        // The token that spends the budget never has a forward after it.
        assert_eq!(piped_fwd, sync_fwd, "max_tokens {max_tokens}");
        assert_eq!(sync_fwd, max_tokens);
    }
}

#[test]
fn callback_stop_matches_the_synchronous_loop() {
    let greedy = SamplingConfig::greedy();
    for deliveries in [1, 2, 3] {
        let ((piped, _), (sync, _)) = both(&[0], 16, &greedy, Some(deliveries));
        assert_eq!(piped, sync, "stop after {deliveries}");
        assert_eq!(sync.delivered.len(), deliveries);
    }
}

#[test]
fn token_bias_is_folded_into_the_pipelined_draw() {
    let mut bias = crate::sampling::TokenBiasMap::new();
    bias.insert(2, 20.0);
    let sampling = SamplingConfig {
        token_bias: bias,
        ..SamplingConfig::greedy()
    };
    let ((piped, _), (sync, _)) = both(&[0], 5, &sampling, None);
    assert_eq!(piped, sync);
    assert_eq!(sync.tokens, vec![2; 5]);
}

/// A second run on the same pipelined client starts from a clean sequence.
#[test]
fn a_pipelined_client_is_reusable() {
    let greedy = SamplingConfig::greedy();
    let model = CountModel::default();
    let mut client = DirectEngine::new(&model, 0).with_force_sync(false);
    let first = client.run(&[3], 16, &greedy).unwrap();
    let second = client.run(&[3], 16, &greedy).unwrap();
    assert_eq!(first, vec![4, 5, 6]);
    assert_eq!(first, second);
    assert_eq!(client.engine().pool().active_count(), 0);
}

#[test]
fn ineligible_sampling_falls_back_to_the_synchronous_loop() {
    let model = CountModel::default();
    let mut client = DirectEngine::new(&model, 0).with_force_sync(false);
    // A history penalty needs each host token before the next draw.
    let penalty = SamplingConfig {
        repetition_penalty: 1.3,
        ..SamplingConfig::greedy()
    };
    assert!(!pipelines(&mut client, &penalty));
    assert!(pipelines(&mut client, &SamplingConfig::greedy()));
    let before = model.forwards();
    let trace = trace(&mut client, &[3], 16, &penalty, None);
    assert_eq!(trace.tokens, vec![4, 5, 6]);
    assert_eq!(model.forwards() - before, 4, "no speculative forward");

    let forced = DirectEngine::new(&model, 0).with_force_sync(true);
    assert!(forced.force_sync());
}

/// Whether `decode` would take the pipeline for `sampling` on a fresh
/// sequence, and with which teardown.
fn decode_teardown<M: LanguageModel>(
    client: &mut DirectEngine<M>,
    sampling: &SamplingConfig,
) -> Option<Teardown> {
    let id = client.open_sequence().unwrap();
    let teardown = client.decode_lookahead(id, sampling).map(|(_, t)| t);
    client.close_sequence(id);
    teardown
}

/// A model-owned family that rewinds its own state pipelines with an exact
/// unwind and leaves that state where the synchronous loop does; one that
/// cannot rewind still pipelines `decode` (the retired CLI loop pipelined
/// every family), emitting the same stream and leaving the one append past
/// the finishing token for the close to release. The exact rule, which the
/// drafter loop and the scheduler use, still declines it.
#[test]
fn model_owned_families_pipeline_with_an_exact_or_a_discarding_teardown() {
    let greedy = SamplingConfig::greedy();
    let run = |rewinds: bool, force_sync: bool| {
        let model = OwnedModel::new(rewinds);
        let mut client = DirectEngine::new(&model, 0).with_force_sync(force_sync);
        let exact = pipelines(&mut client, &greedy);
        let teardown = decode_teardown(&mut client, &greedy);
        let tokens = client.run(&[2, 3], 16, &greedy).unwrap();
        let released = model.released.borrow().last().copied();
        (exact, teardown, tokens, released, model.forwards.get())
    };
    let (exact, teardown, sync_tokens, sync_len, sync_fwd) = run(true, true);
    assert!(!exact);
    assert_eq!(teardown, None);
    assert_eq!(sync_tokens, vec![4, 5, 6]);
    assert_eq!(sync_len, Some(5));

    let (exact, teardown, tokens, len, fwd) = run(true, false);
    assert!(exact);
    assert_eq!(teardown, Some(Teardown::Unwind));
    assert_eq!((tokens, len), (sync_tokens.clone(), sync_len));
    assert_eq!(fwd, sync_fwd + 1, "one unwound speculative forward");

    let (exact, teardown, tokens, len, fwd) = run(false, false);
    assert!(!exact, "no rewind, no exact unwind");
    assert_eq!(teardown, Some(Teardown::Discard));
    assert_eq!(tokens, sync_tokens);
    assert_eq!(fwd, sync_fwd + 1, "one discarded speculative forward");
    assert_eq!(
        len,
        sync_len.map(|l| l + 1),
        "the append past the EOS is released with the sequence"
    );

    let (_, teardown, tokens, len, fwd) = run(false, true);
    assert_eq!(teardown, None, "MLXCEL_FORCE_SYNC keeps it synchronous");
    assert_eq!((tokens, len, fwd), (sync_tokens, sync_len, sync_fwd));
}

/// The device-side feedback builds the `[B, 1]` int32 input a host-built
/// step input has.
#[test]
fn lookahead_feedback_input_matches_a_host_input() {
    let tokens = ffi::astype(&from_slice_i32(&[3, 5], &[2]), crate::dtype::UINT32);
    let input = lookahead_feedback_input(&tokens);
    ffi::eval(&input);
    assert_eq!(ffi::array_shape(&input), vec![2, 1]);
    assert_eq!(
        ffi::array_dtype(&input),
        ffi::array_dtype(&from_slice_i32(&[0], &[1, 1]))
    );
    assert_eq!(host_tokens(&input), vec![3, 5]);
}

/// A zero budget emits nothing and runs no forward, on both loops and on the
/// drafter loop, as the retired CLI generator did (`-n 0`).
#[test]
fn a_zero_budget_emits_nothing() {
    let greedy = SamplingConfig::greedy();
    for force_sync in [false, true] {
        let model = CountModel::default();
        let mut client = DirectEngine::new(&model, 0).with_force_sync(force_sync);
        let mut delivered = Vec::new();
        let run = client
            .generate(&DirectRequest::text(&[1, 2], 0, &greedy), |t| {
                delivered.push(t);
                true
            })
            .unwrap();
        assert!(run.tokens.is_empty());
        assert!(delivered.is_empty());
        assert_eq!(run.stats.prompt_tokens, 2);
        assert_eq!(run.stats.generated_tokens, 0);
        assert_eq!(model.forwards(), 0);

        let mut drafter = crate::speculative::prompt_lookup_drafter::PromptLookupDrafter::new(
            crate::speculative::prompt_lookup::PromptLookupConfig::default(),
        );
        let run = client
            .generate_with_drafter(
                &DirectRequest::text(&[1, 2], 0, &greedy),
                &mut drafter,
                4,
                |_| panic!("nothing is emitted"),
            )
            .unwrap();
        assert!(run.run.tokens.is_empty());
        assert_eq!(model.forwards(), 0);
    }
}
