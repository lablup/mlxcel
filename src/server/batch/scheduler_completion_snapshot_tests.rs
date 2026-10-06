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

//! Completion-origin prompt-cache snapshots claim exactly the tokens their
//! state holds (issue #1754).
//!
//! Decode forwards a sampled token on the step after it was sampled, so when
//! generation stops on anything but a merged EOS the last generated token was
//! never forwarded. Before #1754 the donation keyed the snapshot by the whole
//! `prompt ++ generated` vector, one token more than the per-layer offsets.
//! Gemma 3 (and every model-owned family allocated on the paged override) was
//! off in the other direction as well: the decode lookahead pipelined it while
//! its teardown trim could not reach the model's own caches, so the snapshot
//! held one or two speculative positions beyond the claimed length. Gemma 3
//! pipelines again since #2159, through a rewind of its own state, and these
//! tests run it on the default (lookahead) path.
//!
//! Each test reads the stored entry back and compares its token count with
//! the `offset` the snapshot itself carries.

use super::scheduler_model_owned_cache_tests::{cache_ctx, options, prompt, scheduler, test_store};
use super::*;
use mlxcel_core::generate::LanguageModel;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, mpsc};

use crate::server::prompt_cache::PromptCacheStore;

/// Run one request to its finish and return `prompt ++ generated` (every
/// token the request saw, whether or not the state consumed the last one).
fn run_request(
    sched: &mut BatchScheduler,
    first: &[i32],
    ignore_eos: bool,
    max_tokens: usize,
) -> Vec<i32> {
    let (tx, rx) = mpsc::channel();
    let mut opts = options();
    opts.ignore_eos = ignore_eos;
    opts.max_tokens = max_tokens;
    sched.enqueue_request(
        "prompt".to_string(),
        Some(first.to_vec()),
        opts,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        tx,
        Arc::new(AtomicBool::new(false)),
        true,
    );
    let result = drive(sched, &rx);
    let mut seen = first.to_vec();
    seen.extend_from_slice(&result.generated_token_ids);
    seen
}

/// Drive the scheduler until `rx` reports the request done. Unlike the
/// shared `run_to_completion` this keeps stepping through multi-token decode
/// and runs the completion pass that donates a decode-finished sequence.
fn drive(
    sched: &mut BatchScheduler,
    rx: &mpsc::Receiver<GenerateEvent>,
) -> crate::server::model_provider::GenerationResult {
    for _ in 0..64 {
        match rx.try_recv() {
            Ok(GenerateEvent::Done(result)) => return result,
            Ok(GenerateEvent::Error(err)) => panic!("unexpected generation error: {err}"),
            Ok(_) | Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => panic!("response channel closed"),
        }
        match sched.decide_action() {
            BatchSchedulerAction::Prefill(id) => sched.execute_prefill(id),
            BatchSchedulerAction::Decode(ids) => sched.execute_decode_step(&ids),
            BatchSchedulerAction::Idle => {}
            other => panic!("unexpected scheduler action {other:?}"),
        }
        // The run loop's per-tick completion pass: tears down any lookahead,
        // donates finished sequences and sends `Done`.
        sched.finalize_completed();
    }
    loop {
        match rx.recv_timeout(std::time::Duration::from_secs(5)) {
            Ok(GenerateEvent::Done(result)) => return result,
            Ok(GenerateEvent::Error(err)) => panic!("unexpected generation error: {err}"),
            Ok(_) => {}
            Err(err) => panic!("generation did not finish: {err}"),
        }
    }
}

/// The longest stored snapshot that prefixes `seen`, as
/// `(entry token count, per-layer offset the snapshot carries)`.
fn stored_len_and_offset(
    sched: &BatchScheduler,
    store: &PromptCacheStore,
    seen: &[i32],
) -> (usize, i32) {
    let ctx = cache_ctx();
    let key =
        BatchScheduler::compose_prompt_cache_key(&ctx, seen, sched.rope_regime_for(seen.len()));
    let (entry, matched) = store
        .lookup_snapshot_prefix(&key, seen)
        .expect("the finished request donated a snapshot that prefixes what it saw");
    let offset = entry.with_snapshot(|snapshot| {
        let tensor = snapshot
            .tensor("layer0.standard.offset")
            .expect("the tiny Gemma 3 layer is a standard KV cache");
        mlxcel_core::item_i32(tensor)
    });
    (matched, offset)
}

/// `length` finish after several decode steps: the last generated token was
/// sampled but never forwarded, so the entry must stop one short of what the
/// request saw, and must equal the offset the state holds.
#[test]
fn length_finish_snapshot_claims_only_forwarded_tokens() {
    let store = test_store();
    let mut sched = scheduler(store.clone());
    let first = prompt(40);
    let seen = run_request(&mut sched, &first, true, 4);
    assert_eq!(seen.len(), first.len() + 4, "four tokens generated");

    let (stored, offset) = stored_len_and_offset(&sched, &store, &seen);
    assert_eq!(
        stored as i32, offset,
        "the snapshot's token count must equal its per-layer offset"
    );
    assert_eq!(
        stored,
        seen.len() - 1,
        "the last sampled token was never forwarded"
    );
}

/// The one-decoded-token boundary: a request that finishes inside prefill
/// (`max_tokens == 1`) sampled its only token without forwarding it, so the
/// stored entry is the prompt alone.
#[test]
fn one_decoded_token_snapshot_is_the_prompt_alone() {
    let store = test_store();
    let mut sched = scheduler(store.clone());
    let first = prompt(40);
    let seen = run_request(&mut sched, &first, true, 1);
    assert_eq!(seen.len(), first.len() + 1, "one token generated");

    let (stored, offset) = stored_len_and_offset(&sched, &store, &seen);
    assert_eq!(stored as i32, offset);
    assert_eq!(stored, first.len());
}

/// A merged-EOS stop never pushes the EOS, so every pushed token was
/// forwarded and the entry is the whole prompt (here: EOS at prefill, the
/// tiny model's greedy first token).
#[test]
fn eos_stop_snapshot_keeps_every_pushed_token() {
    let store = test_store();
    let mut sched = scheduler(store.clone());
    let first = prompt(40);
    let seen = run_request(&mut sched, &first, false, 4);
    assert_eq!(
        seen.len(),
        first.len(),
        "the tiny model stops on EOS at once"
    );

    let (stored, offset) = stored_len_and_offset(&sched, &store, &seen);
    assert_eq!(stored as i32, offset);
    assert_eq!(stored, seen.len());
}

/// Gemma 3 rewinds its own model-owned state on a lookahead teardown (#2159),
/// so it pipelines even though the paged override allocates it on
/// `PagedKvCache` with an empty cache vector. A model-owned family without
/// that capability keeps the #1754 gate: see
/// `scheduler_model_owned_lookahead_tests::model_owned_default_declines_the_lookahead_rewind`
/// and `lookahead_is_gated_on_the_rewind_capability` (an INT8 sliding layer).
#[test]
fn model_owned_sequence_primes_the_decode_lookahead_only_with_a_rewind() {
    let mut sched = scheduler(test_store());
    let (tx, _rx) = mpsc::channel();
    let mut opts = options();
    opts.ignore_eos = true;
    opts.max_tokens = 8;
    sched.enqueue_request(
        "prompt".to_string(),
        Some(prompt(40)),
        opts,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        tx,
        Arc::new(AtomicBool::new(false)),
        true,
    );
    let seq_id = match sched.decide_action() {
        BatchSchedulerAction::Prefill(id) => {
            sched.execute_prefill(id);
            id
        }
        other => panic!("expected a prefill, got {other:?}"),
    };
    assert!(
        sched.active_batch.get(seq_id).is_some(),
        "the sequence is decoding"
    );
    assert_eq!(
        sched.cache_pool.get(seq_id).map(|set| set.backend),
        Some(SequenceStateBackend::PagedKvCache),
        "the allocated backend claims paged"
    );
    assert!(sched.model.supports_decode_lookahead_rewind());
    assert!(
        sched.lookahead_params(&[seq_id]).is_some(),
        "Gemma 3 rewinds its own state, so it pipelines"
    );
}
