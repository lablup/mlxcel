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

//! Gemma 3 on the decode lookahead pipeline (issue #2159).
//!
//! A real `Gemma3Wrapper` with one sliding (window 8) and one global layer
//! decodes well past its window twice, in lockstep: once with the lookahead
//! pipeline and once with `MLXCEL_FORCE_SYNC` semantics. A mid-generation
//! admission forces the one-position teardown and every finish forces the
//! two-position steady teardown, both after the ring wrapped. Whenever the
//! pipelined run holds no speculative step, every sequence's model-owned state
//! (offsets, write cursor, the whole sliding buffer, the live global K/V) must
//! equal the synchronous run's, and the decoded tokens must match.

use super::scheduler_model_owned_pad_trim_tests::{WINDOW, tiny_gemma3};
use super::*;
use mlxcel_core::cache::{SequenceStateBackend, SequenceStateLayout};
use mlxcel_core::generate::{LanguageModel, ModelStateSnapshot, SamplingConfig};
use mlxcel_core::layers::KVCache;
use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::{MlxArray, UniquePtr};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use crate::LoadedModel;
use crate::models::Gemma3Wrapper;
use crate::server::config::{DecodeStorageBackend, PreemptionPolicy, ReasoningBudgetOverride};
use crate::server::model_provider::GenerateEvent;
use crate::server::state::BatchMetrics;
use crate::tokenizer::MlxcelTokenizer;

fn scheduler(
    model: Gemma3Wrapper,
    backend: DecodeStorageBackend,
    force_sync: bool,
) -> BatchScheduler {
    let (_tx, rx) = mpsc::channel();
    let mut sched = BatchScheduler::with_config(
        LoadedModel::Gemma3(model),
        MlxcelTokenizer::stub(),
        Vec::new(),
        rx,
        4,
        8,
        Arc::new(BatchMetrics::new()),
        Arc::new(BatchObservability::new()),
        0,
        false,
        PreemptionPolicy::default(),
        1,
        backend,
    );
    sched.lookahead_force_sync = force_sync;
    install_thread_local_default_stream(sched.generation_stream.as_ref());
    sched
}

fn options(max_tokens: usize) -> ServerGenerateOptions {
    ServerGenerateOptions {
        n_indent: 0,
        t_max_predict_ms: None,
        reasoning_budget_message: None,
        retention: Default::default(),
        dry_breaker_strings: None,
        logit_bias: Vec::new(),
        logit_bias_texts: Vec::new(),
        post_sampling_probs: false,
        max_tokens,
        sampling: SamplingConfig::greedy(),
        stop_sequences: None,
        ignore_eos: true,
        priority: RequestPriority::Normal,
        lora_scales: None,
        logprobs: Default::default(),
        reasoning_budget: ReasoningBudgetOverride::InheritServerDefault,
        thinking_enter_block_on_start: false,
        reasoning_control: None,
        prompt_cache_ctx: None,
        structured: None,
        grammar: None,
        image_soft_tokens: None,
        pre_rendered_prompt_tokens: None,
    }
}

fn enqueue(
    sched: &mut BatchScheduler,
    prompt_len: usize,
    seed: usize,
    max_tokens: usize,
) -> mpsc::Receiver<GenerateEvent> {
    let prompt: Vec<i32> = (0..prompt_len)
        .map(|i| ((i * 5 + seed * 3 + 1) % 16) as i32)
        .collect();
    let (tx, rx) = mpsc::channel();
    sched.enqueue_request(
        "prompt".to_string(),
        Some(prompt),
        options(max_tokens),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        tx,
        Arc::new(AtomicBool::new(false)),
        true,
    );
    rx
}

/// One run-loop iteration without `finalize_completed` (so a sequence that
/// finished this tick can still be inspected). Returns false when idle.
fn tick(sched: &mut BatchScheduler) -> bool {
    match sched.decide_action() {
        BatchSchedulerAction::Prefill(id) => {
            // The run loop tears the lookahead down before any admission.
            sched.discard_lookahead();
            sched.execute_prefill(id);
        }
        BatchSchedulerAction::Decode(ids) => sched.execute_decode_step(&ids),
        BatchSchedulerAction::Idle => return false,
        other => panic!("unexpected scheduler action {other:?}"),
    }
    true
}

fn tensor_bits(
    snapshot: &ModelStateSnapshot,
    name: &str,
    len: Option<i32>,
) -> (Vec<i32>, Vec<u32>) {
    let array = snapshot
        .tensor(name)
        .unwrap_or_else(|| panic!("snapshot has no {name}"));
    let shape = mlxcel_core::array_shape(array);
    let cut = match len {
        Some(stop) => {
            mlxcel_core::slice(array, &[0, 0, 0, 0], &[shape[0], shape[1], stop, shape[3]])
        }
        None => mlxcel_core::array_handle_clone(array),
    };
    let bits = array_to_vec_f32(&cut).iter().map(|v| v.to_bits()).collect();
    (mlxcel_core::array_shape(&cut), bits)
}

fn scalar(snapshot: &ModelStateSnapshot, name: &str) -> i32 {
    let array = snapshot
        .tensor(name)
        .unwrap_or_else(|| panic!("snapshot has no {name}"));
    array_to_vec_f32(array)[0] as i32
}

/// Everything about one sequence's model-owned state a later step can read.
#[derive(Debug, PartialEq)]
struct SeqState {
    sliding_offset: i32,
    sliding_idx: i32,
    sliding_keys: (Vec<i32>, Vec<u32>),
    sliding_values: (Vec<i32>, Vec<u32>),
    global_offset: i32,
    global_keys: (Vec<i32>, Vec<u32>),
    global_values: (Vec<i32>, Vec<u32>),
}

fn seq_state(sched: &BatchScheduler, seq_id: SequenceId) -> SeqState {
    let snapshot = sched
        .model
        .snapshot_sequence_state(seq_id, 1)
        .expect("Gemma 3 snapshots a live sequence");
    let global_offset = scalar(&snapshot, "layer1.standard.offset");
    SeqState {
        sliding_offset: scalar(&snapshot, "layer0.rotating.offset"),
        sliding_idx: scalar(&snapshot, "layer0.rotating.idx"),
        sliding_keys: tensor_bits(&snapshot, "layer0.rotating.keys", None),
        sliding_values: tensor_bits(&snapshot, "layer0.rotating.values", None),
        global_offset,
        global_keys: tensor_bits(&snapshot, "layer1.standard.keys", Some(global_offset)),
        global_values: tensor_bits(&snapshot, "layer1.standard.values", Some(global_offset)),
    }
}

fn finish(rx: &mpsc::Receiver<GenerateEvent>) -> Vec<i32> {
    loop {
        match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(GenerateEvent::Done(result)) => return result.generated_token_ids,
            Ok(GenerateEvent::Error(err)) => panic!("unexpected generation error: {err}"),
            Ok(_) => {}
            Err(err) => panic!("generation did not finish: {err}"),
        }
    }
}

/// Requests `(prompt_len, max_tokens)` admitted at the start; `late` is
/// admitted once the first request has decoded `admit_after` tokens.
struct Scenario {
    initial: Vec<(usize, usize)>,
    late: (usize, usize),
    admit_after: usize,
}

/// Run `scenario` on a pipelined and a force-sync scheduler in lockstep and
/// compare them at every tick where the pipelined one holds no lookahead.
fn assert_lookahead_matches_force_sync(backend: DecodeStorageBackend, scenario: &Scenario) {
    let mut piped = scheduler(tiny_gemma3(), backend, false);
    let mut synced = scheduler(tiny_gemma3(), backend, true);
    assert!(piped.model.supports_decode_lookahead_rewind());

    let mut rxs: Vec<(mpsc::Receiver<GenerateEvent>, mpsc::Receiver<GenerateEvent>)> = Vec::new();
    for (seed, &(prompt_len, max_tokens)) in scenario.initial.iter().enumerate() {
        rxs.push((
            enqueue(&mut piped, prompt_len, seed, max_tokens),
            enqueue(&mut synced, prompt_len, seed, max_tokens),
        ));
    }

    let (mut pipelined_ticks, mut wrapped_checks, mut admitted) = (0, 0, false);
    for step in 0..256 {
        if !admitted {
            let first_generated = piped
                .active_batch
                .iter_sequences()
                .map(|seq| seq.generated_tokens.len())
                .max()
                .unwrap_or(0);
            if first_generated >= scenario.admit_after {
                let seed = scenario.initial.len();
                let (prompt_len, max_tokens) = scenario.late;
                rxs.push((
                    enqueue(&mut piped, prompt_len, seed, max_tokens),
                    enqueue(&mut synced, prompt_len, seed, max_tokens),
                ));
                admitted = true;
            }
        }
        let busy = tick(&mut piped);
        assert_eq!(tick(&mut synced), busy, "step {step}: both schedulers act");
        if !busy {
            break;
        }
        if piped.decode_lookahead.is_some() {
            pipelined_ticks += 1;
        } else {
            let mut ids = piped.active_batch.sequence_ids();
            let mut want_ids = synced.active_batch.sequence_ids();
            ids.sort_by_key(|id| id.as_u64());
            want_ids.sort_by_key(|id| id.as_u64());
            assert_eq!(ids, want_ids, "step {step}: batch");
            for seq_id in ids {
                let got = seq_state(&piped, seq_id);
                let want = seq_state(&synced, seq_id);
                if got.sliding_offset > WINDOW as i32 {
                    wrapped_checks += 1;
                }
                assert_eq!(got, want, "step {step}, {seq_id}: model-owned state");
                assert_eq!(
                    piped
                        .active_batch
                        .get(seq_id)
                        .map(|s| s.generated_tokens.clone()),
                    synced
                        .active_batch
                        .get(seq_id)
                        .map(|s| s.generated_tokens.clone()),
                    "step {step}, {seq_id}: tokens"
                );
            }
        }
        piped.finalize_completed();
        synced.finalize_completed();
    }
    assert!(admitted, "the late request was admitted mid-generation");
    assert!(
        pipelined_ticks >= 8,
        "the lookahead pipeline ran ({pipelined_ticks} ticks)"
    );
    assert!(
        wrapped_checks >= 3,
        "states were compared after the ring wrapped ({wrapped_checks} checks)"
    );
    for (i, (p, s)) in rxs.iter().enumerate() {
        let (got, want) = (finish(p), finish(s));
        assert_eq!(got, want, "request {i}: greedy tokens");
        let expected = if i < scenario.initial.len() {
            scenario.initial[i].1
        } else {
            scenario.late.1
        };
        assert_eq!(got.len(), expected, "request {i}: every token decoded");
    }
}

#[test]
fn single_sequence_lookahead_matches_force_sync_past_the_window() {
    // `--parallel 1`-shaped: no paged override, the sequence is allocated
    // `ModelOwned`. The late admission tears a wrapped pipeline down by one
    // position; each finish tears it down by two.
    assert_lookahead_matches_force_sync(
        DecodeStorageBackend::Dense,
        &Scenario {
            initial: vec![(5, 24)],
            late: (3, 6),
            admit_after: 10,
        },
    );
}

#[test]
fn batched_lookahead_matches_force_sync_past_the_window() {
    // Default server shape (paged override, shadow accounting) with up to
    // four concurrent sequences on the batched decode forward.
    assert_lookahead_matches_force_sync(
        DecodeStorageBackend::Paged,
        &Scenario {
            initial: vec![(5, 24), (11, 18), (2, 21)],
            late: (13, 7),
            admit_after: 9,
        },
    );
}

#[test]
fn lookahead_is_gated_on_the_rewind_capability() {
    // An FP16 Gemma 3 pipelines under both allocations.
    for backend in [DecodeStorageBackend::Dense, DecodeStorageBackend::Paged] {
        let mut sched = scheduler(tiny_gemma3(), backend, false);
        let _rx = enqueue(&mut sched, 4, 0, 8);
        assert!(tick(&mut sched), "prefill");
        let ids = sched.active_batch.sequence_ids();
        assert!(
            sched.lookahead_params(&ids).is_some(),
            "{backend:?}: FP16 Gemma 3 pipelines"
        );
    }

    // An INT8 sliding layer is outside what the undo log covers: the family
    // declines the capability and stays synchronous.
    let model = tiny_gemma3();
    model.set_kv_cache_layer_modes(vec![
        mlxcel_core::cache::KVCacheMode::Int8,
        mlxcel_core::cache::KVCacheMode::Fp16,
    ]);
    assert!(!model.supports_decode_lookahead_rewind());
    let mut sched = scheduler(model, DecodeStorageBackend::Paged, false);
    let _rx = enqueue(&mut sched, 4, 0, 8);
    assert!(tick(&mut sched), "prefill");
    let ids = sched.active_batch.sequence_ids();
    assert!(
        sched.lookahead_params(&ids).is_none(),
        "INT8 sliding layer stays synchronous"
    );
}

/// A model-owned family that does not implement the hooks.
struct ModelOwnedStub;

impl LanguageModel for ModelOwnedStub {
    fn forward(
        &self,
        _input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        mlxcel_core::zeros(&[1], 0)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Vec::new()
    }

    fn num_layers(&self) -> usize {
        1
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![0]
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        SequenceStateLayout::model_owned(1)
    }
}

#[test]
fn model_owned_default_declines_the_lookahead_rewind() {
    let stub = ModelOwnedStub;
    assert_eq!(
        stub.sequence_state_layout().backend,
        SequenceStateBackend::ModelOwned
    );
    assert!(!stub.supports_decode_lookahead_rewind());
    assert!(
        stub.rewind_decode_appends(SequenceId::from_raw(0), 1)
            .is_err()
    );
}

/// A failed rewind must finish the request with an error, never decode on or
/// donate from the desynchronized state.
#[test]
fn failed_rewind_fails_the_request() {
    let mut sched = scheduler(tiny_gemma3(), DecodeStorageBackend::Paged, false);
    let rx = enqueue(&mut sched, 4, 0, 16);
    while sched.decode_lookahead.is_none() {
        assert!(tick(&mut sched), "the request is still running");
        sched.finalize_completed();
    }
    let seq_id = sched.active_batch.sequence_ids()[0];
    // A teardown asking for more appends than the state holds is refused by
    // the model, the same outcome as any desynchronized rewind.
    let failed = sched.apply_lookahead_trim(&[seq_id], 64);
    assert_eq!(failed, vec![seq_id]);
    sched.decode_lookahead = None;
    assert!(matches!(
        sched.active_batch.get(seq_id).map(|s| &s.state),
        Some(SequenceState::Finished(FinishReason::Error(_)))
    ));
    sched.finalize_completed();
    assert!(sched.active_batch.is_empty());
    let mut saw_error = false;
    while let Ok(event) = rx.recv_timeout(Duration::from_secs(1)) {
        if matches!(event, GenerateEvent::Error(_)) {
            saw_error = true;
        }
    }
    assert!(saw_error, "the client sees the error");
}
