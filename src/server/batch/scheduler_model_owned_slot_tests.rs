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

//! The KV bound and the single-slot release for model-owned caches (#2766),
//! on the tiny NemotronNAS of `nemotron_nas_engine_tests`.
//!
//! NemotronNAS cannot batch and keeps one model-owned slot, so it is both a
//! `SequenceStateBackend::ModelOwned` layout (whose pool caches
//! `--max-kv-size` trimming cannot reach) and a single-slot family (whose KV
//! used to stay resident until the next prefill).

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use mlxcel_core::cache::SequenceStateBackend;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::weights::WeightMap;

use super::*;
use crate::LoadedModel;
use crate::models::NemotronNASModel;
use crate::models::nemotron_nas::engine_tests::{PROMPT, recompute_greedy, tiny_model};
use crate::server::batch::sequence::ContextRetentionPolicy;
use crate::server::config::{DecodeStorageBackend, PreemptionPolicy};
use crate::server::model_provider::{GenerateEvent, GenerationResult, StopKind};
use crate::server::state::BatchMetrics;
use crate::tokenizer::MlxcelTokenizer;

const BOUND: usize = 16;
const SHIFT_ON: ContextRetentionPolicy = ContextRetentionPolicy {
    context_shift: true,
    n_keep: 0,
};

fn scheduler_for(model: LoadedModel) -> BatchScheduler {
    let (_tx, rx) = mpsc::channel();
    let sched = BatchScheduler::with_config(
        model,
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
        DecodeStorageBackend::Paged,
    );
    install_thread_local_default_stream(sched.generation_stream.as_ref());
    sched
}

fn nas_scheduler() -> BatchScheduler {
    scheduler_for(LoadedModel::NemotronNAS(tiny_model()))
}

fn nas(sched: &BatchScheduler) -> &NemotronNASModel {
    match sched.engine.model() {
        LoadedModel::NemotronNAS(model) => model,
        _ => unreachable!("the fixture serves NemotronNAS"),
    }
}

fn enqueue(
    sched: &mut BatchScheduler,
    prompt: &[i32],
    max_tokens: usize,
) -> mpsc::Receiver<GenerateEvent> {
    let mut options = super::scheduler_model_owned_cache_tests::options();
    options.max_tokens = max_tokens;
    options.prompt_cache_ctx = None;
    let (tx, rx) = mpsc::channel();
    sched.enqueue_request(
        "prompt".to_string(),
        Some(prompt.to_vec()),
        options,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        tx,
        Arc::new(AtomicBool::new(false)),
        true,
    );
    rx
}

/// One run-loop tick: the action, then the completion sweep that sends
/// `Done` and releases finished sequences. Returns `false` once idle.
fn tick(sched: &mut BatchScheduler) -> bool {
    let busy = match sched.decide_action() {
        BatchSchedulerAction::Prefill(id) => {
            sched.discard_lookahead();
            sched.execute_prefill(id);
            true
        }
        BatchSchedulerAction::Decode(ids) => {
            sched.execute_decode_step(&ids);
            true
        }
        BatchSchedulerAction::Idle => false,
        other => panic!("unexpected scheduler action {other:?}"),
    };
    sched.finalize_completed();
    busy
}

/// Run scheduler ticks until it goes idle.
fn drain(sched: &mut BatchScheduler) {
    let mut guard = 0;
    while tick(sched) {
        guard += 1;
        assert!(guard < 1000, "the scheduler did not go idle");
    }
    assert!(sched.active_batch.is_empty() && sched.prefill_queue.is_empty());
}

fn result_of(rx: &mpsc::Receiver<GenerateEvent>) -> GenerationResult {
    loop {
        match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(GenerateEvent::Done(result)) => return result,
            Ok(GenerateEvent::Error(err)) => panic!("unexpected generation error: {err}"),
            Ok(GenerateEvent::RequestMemoryRejected(err)) => {
                panic!("unexpected memory rejection: {err}")
            }
            Ok(_) => {}
            Err(err) => panic!("generation did not finish: {err}"),
        }
    }
}

// ---------------------------------------------------------------------------
// `--max-kv-size` with `--context-shift`
// ---------------------------------------------------------------------------

/// Shifting trims pool caches, and a model-owned layout has none, so the
/// scheduler serves it with shifting off: the bound then stops the sequence
/// with `truncated: true` instead of letting its KV grow to `max_tokens`.
#[test]
fn model_owned_layout_stops_at_the_kv_bound_under_context_shift() {
    let mut sched = nas_scheduler()
        .with_max_kv_size(Some(BOUND))
        .with_context_retention(SHIFT_ON);
    assert_eq!(
        sched.engine.model().sequence_state_layout().backend,
        SequenceStateBackend::ModelOwned
    );

    let rx = enqueue(&mut sched, &PROMPT, 40);
    drain(&mut sched);
    let result = result_of(&rx);
    assert!(
        result.prompt_tokens + result.completion_tokens < BOUND,
        "the sequence passed the bound: {} + {}",
        result.prompt_tokens,
        result.completion_tokens
    );
    assert_eq!(
        result.stop_kind,
        StopKind::ContextExhausted,
        "{} prompt + {} generated tokens",
        result.prompt_tokens,
        result.completion_tokens
    );
    assert!(
        !sched.context_retention().context_shift,
        "the effective policy must not claim a shift nothing performs"
    );
    assert_eq!(sched.context_retention().n_keep, SHIFT_ON.n_keep);
}

/// The builder order does not matter: the bound may arrive after the policy.
#[test]
fn context_shift_fallback_holds_when_the_bound_is_set_last() {
    let sched = nas_scheduler()
        .with_context_retention(SHIFT_ON)
        .with_max_kv_size(Some(BOUND));
    assert!(!sched.context_retention().context_shift);
}

/// Without a bound there is nothing to trim, so the policy is kept as given.
#[test]
fn context_shift_is_kept_without_a_kv_bound() {
    let sched = nas_scheduler().with_context_retention(SHIFT_ON);
    assert!(sched.context_retention().context_shift);
}

/// A one-layer dense Llama with every projection zero: the residual stream
/// is the token's one-hot embedding and the head maps `t` to `t + 1`, so it
/// never emits a stop and generates exactly `max_tokens`.
fn cycle_llama() -> crate::models::Llama3Model {
    const VOCAB: usize = 16;
    let args: crate::models::llama3::ModelArgs = serde_json::from_value(serde_json::json!({
        "model_type": "llama", "hidden_size": VOCAB, "num_hidden_layers": 1,
        "intermediate_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 8, "rms_norm_eps": 1e-6, "vocab_size": VOCAB, "rope_theta": 10000.0,
        "attention_bias": false, "tie_word_embeddings": false,
    }))
    .expect("cycle llama args");
    let tensor = |values: Vec<f32>, shape: &[i32]| mlxcel_core::from_slice_f32(&values, shape);
    let mut embed = vec![0.0f32; VOCAB * VOCAB];
    let mut head = vec![0.0f32; VOCAB * VOCAB];
    for t in 0..VOCAB {
        embed[t * VOCAB + t] = 1.0;
        head[t * VOCAB + (t + VOCAB - 1) % VOCAB] = 1.0;
    }
    let v = VOCAB as i32;
    let mut w = WeightMap::new();
    w.insert("model.embed_tokens.weight".into(), tensor(embed, &[v, v]));
    w.insert("lm_head.weight".into(), tensor(head, &[v, v]));
    for (name, shape) in [
        ("self_attn.q_proj.weight", [16, v]),
        ("self_attn.k_proj.weight", [8, v]),
        ("self_attn.v_proj.weight", [8, v]),
        ("self_attn.o_proj.weight", [v, 16]),
        ("mlp.gate_proj.weight", [32, v]),
        ("mlp.up_proj.weight", [32, v]),
        ("mlp.down_proj.weight", [v, 32]),
    ] {
        let n = (shape[0] * shape[1]) as usize;
        w.insert(
            format!("model.layers.0.{name}"),
            tensor(vec![0.0; n], &shape),
        );
    }
    for norm in [
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
    ] {
        w.insert(norm.into(), tensor(vec![1.0; VOCAB], &[v]));
    }
    crate::models::Llama3Model::from_weights(&w, &args).expect("cycle llama builds")
}

/// A dense layout keeps shifting: its pool caches are what the trim walks,
/// so the same bound lets the request run to its full budget.
#[test]
fn dense_layout_keeps_context_shift_under_a_kv_bound() {
    let mut sched = scheduler_for(LoadedModel::Llama(cycle_llama()))
        .with_max_kv_size(Some(BOUND))
        .with_context_retention(SHIFT_ON);
    assert_ne!(
        sched.engine.model().sequence_state_layout().backend,
        SequenceStateBackend::ModelOwned
    );
    assert!(sched.context_retention().context_shift);

    let rx = enqueue(&mut sched, &[0, 1, 2, 3, 4, 5], 24);
    drain(&mut sched);
    let result = result_of(&rx);
    assert_eq!(result.stop_kind, StopKind::Limit);
    assert_eq!(result.completion_tokens, 24, "the shift kept generating");
}

// ---------------------------------------------------------------------------
// Releasing the single model-owned slot
// ---------------------------------------------------------------------------

/// Once the request that filled the slot completes, the slot is empty: an
/// idle server does not keep the last request's KV. The next request still
/// decodes exactly what a from-scratch recompute does.
#[test]
fn completing_the_slot_holder_frees_the_single_slot() {
    let mut sched = nas_scheduler();
    let reference = recompute_greedy(&tiny_model(), &PROMPT, 6);

    let first = enqueue(&mut sched, &PROMPT, 6);
    drain(&mut sched);
    let first = result_of(&first);
    assert_eq!(first.generated_token_ids, reference);
    assert_eq!(nas(&sched).slot_live_len(), 0, "the slot kept its KV");
    assert_eq!(sched.model_owned_slot_holder(), None);

    let second = enqueue(&mut sched, &PROMPT, 6);
    drain(&mut sched);
    assert_eq!(result_of(&second).generated_token_ids, reference);
    assert_eq!(nas(&sched).slot_live_len(), 0);
}

/// Releasing a queued request that never prefilled must not touch the slot,
/// which holds the in-flight sequence's KV; releasing the holder frees it.
#[test]
fn only_the_slot_holder_release_resets_the_slot() {
    let mut sched = nas_scheduler();
    let reference = recompute_greedy(&tiny_model(), &PROMPT, 6);

    let running = enqueue(&mut sched, &PROMPT, 6);
    assert!(tick(&mut sched), "the first prefill runs");
    let holder = sched
        .model_owned_slot_holder()
        .expect("the prefill took the slot");
    assert!(
        sched.active_batch.get(holder).is_some(),
        "holder is decoding"
    );
    let live = nas(&sched).slot_live_len();
    assert!(live > 0);

    let _queued_rx = enqueue(&mut sched, &[1, 2, 3], 4);
    let queued = sched
        .prefill_queue
        .dequeue()
        .expect("the second request is queued");
    assert_ne!(queued.seq_id, holder);
    sched.release_sequence_caches(queued.seq_id);
    assert_eq!(sched.model_owned_slot_holder(), Some(holder));
    assert_eq!(
        nas(&sched).slot_live_len(),
        live,
        "a queued release wiped the in-flight sequence's KV"
    );

    drain(&mut sched);
    assert_eq!(result_of(&running).generated_token_ids, reference);
    assert_eq!(sched.model_owned_slot_holder(), None);
    assert_eq!(nas(&sched).slot_live_len(), 0);
}
