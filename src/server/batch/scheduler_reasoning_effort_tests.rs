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

//! The scheduler hands each request's model-side reasoning effort to its
//! sequence (#3435), on the tiny random Raon-1 mini with one steered layer:
//! admission sets it before prefill, an unknown level fails only that
//! request, two concurrent requests at different efforts decode the same
//! greedy tokens as each run alone, and a preempted sequence keeps its level
//! under its new id.
//!
//! The dropped-queued-adoption carry (`drop_queued_adoption`) is not driven
//! here: it only picks requests holding paged blocks, and Raon-1 mini's
//! model-owned layout has no paged layout, so the reclaim never selects one
//! of its requests. The carry there guards a future paged steering model.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use super::*;
use crate::LoadedModel;
use crate::server::config::{DecodeStorageBackend, PreemptionPolicy};
use crate::server::model_provider::GenerateEvent;
use crate::server::state::BatchMetrics;
use crate::tokenizer::MlxcelTokenizer;

const EOS: i32 = 31;
const MAX_TOKENS: usize = 6;

fn scheduler() -> BatchScheduler {
    let (_tx, rx) = mpsc::channel();
    let sched = BatchScheduler::with_config(
        LoadedModel::Raon1Mini(crate::models::raon1_mini::steering_tests::steered_model()),
        MlxcelTokenizer::stub(),
        vec![EOS],
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

fn options(effort: Option<&str>) -> ServerGenerateOptions {
    let mut options = super::scheduler_model_owned_cache_tests::options();
    options.prompt_cache_ctx = None;
    options.max_tokens = MAX_TOKENS;
    options.ignore_eos = true;
    options.model_reasoning_effort = effort.map(str::to_string);
    options
}

fn enqueue(
    sched: &mut BatchScheduler,
    prompt: &[i32],
    effort: Option<&str>,
) -> mpsc::Receiver<GenerateEvent> {
    let (tx, rx) = mpsc::channel();
    sched.enqueue_request(
        "prompt".to_string(),
        Some(prompt.to_vec()),
        options(effort),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        tx,
        Arc::new(AtomicBool::new(false)),
        true,
    );
    rx
}

/// Tick as the run loop does until both queues drain, then collect each
/// request's greedy ids.
fn run(sched: &mut BatchScheduler, rxs: &[mpsc::Receiver<GenerateEvent>]) -> Vec<Vec<i32>> {
    for _ in 0..64 {
        match sched.decide_action() {
            BatchSchedulerAction::Prefill(id) => {
                sched.discard_lookahead();
                sched.execute_prefill(id);
            }
            BatchSchedulerAction::Decode(ids) => sched.execute_decode_step(&ids),
            BatchSchedulerAction::Idle => break,
            other => panic!("unexpected scheduler action {other:?}"),
        }
        sched.finalize_completed();
        if sched.active_batch.is_empty() && sched.prefill_queue.is_empty() {
            break;
        }
    }
    rxs.iter()
        .map(|rx| {
            loop {
                match rx.recv_timeout(Duration::from_secs(5)) {
                    Ok(GenerateEvent::Done(result)) => break result.generated_token_ids,
                    Ok(GenerateEvent::Error(err)) => panic!("unexpected generation error: {err}"),
                    Ok(_) => {}
                    Err(err) => panic!("generation did not finish: {err}"),
                }
            }
        })
        .collect()
}

const PROMPT_A: [i32; 5] = [1, 2, 3, 4, 5];
const PROMPT_B: [i32; 7] = [9, 8, 7, 6, 5, 4, 3];

#[test]
fn admission_sets_the_requests_effort_on_its_sequence() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let mut sched = scheduler();
    let _rx = enqueue(&mut sched, &PROMPT_A, Some("low"));
    let queued = sched.prefill_queue.peek().expect("request queued").seq_id;
    let model = sched.engine.model();
    assert_eq!(
        model.sequence_reasoning_effort(queued).as_deref(),
        Some("low")
    );
}

#[test]
fn an_unknown_level_fails_only_that_request() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let mut sched = scheduler();
    let bad = enqueue(&mut sched, &PROMPT_A, Some("minimal"));
    match bad.recv_timeout(Duration::from_secs(5)) {
        Ok(GenerateEvent::Error(err)) => {
            assert!(err.contains("none, low, medium, high, xhigh, max"), "{err}")
        }
        Ok(_) => panic!("expected an error event"),
        Err(err) => panic!("no event: {err}"),
    }
    assert!(sched.prefill_queue.is_empty());
    let good = enqueue(&mut sched, &PROMPT_B, Some("max"));
    assert_eq!(run(&mut sched, &[good])[0].len(), MAX_TOKENS);
}

#[test]
fn concurrent_requests_at_different_efforts_decode_as_each_alone() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let mut together = scheduler();
    let rx_a = enqueue(&mut together, &PROMPT_A, Some("low"));
    let rx_b = enqueue(&mut together, &PROMPT_B, Some("max"));
    let both = run(&mut together, &[rx_a, rx_b]);

    let mut alone_a = scheduler();
    let rx = enqueue(&mut alone_a, &PROMPT_A, Some("low"));
    let solo_a = run(&mut alone_a, &[rx]).remove(0);
    let mut alone_b = scheduler();
    let rx = enqueue(&mut alone_b, &PROMPT_B, Some("max"));
    let solo_b = run(&mut alone_b, &[rx]).remove(0);

    assert_eq!(both[0].len(), MAX_TOKENS);
    assert_eq!(both[0], solo_a, "the low-effort row");
    assert_eq!(both[1], solo_b, "the max-effort row");
}

#[test]
fn preemption_carries_the_effort_to_the_new_sequence_id() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let mut sched = scheduler();
    let rx = enqueue(&mut sched, &PROMPT_A, Some("low"));
    match sched.decide_action() {
        BatchSchedulerAction::Prefill(id) => {
            sched.discard_lookahead();
            sched.execute_prefill(id);
        }
        other => panic!("expected a prefill, got {other:?}"),
    }
    let old_id = sched
        .active_batch
        .iter_sequences()
        .next()
        .expect("the request is running")
        .seq_id;
    assert!(sched.preempt_sequence(old_id), "the row left the batch");

    let new_id = sched
        .prefill_queue
        .peek()
        .expect("the victim re-queued")
        .seq_id;
    assert_ne!(new_id, old_id);
    let model = sched.engine.model();
    assert_eq!(
        model.sequence_reasoning_effort(new_id).as_deref(),
        Some("low"),
        "the re-prefill is steered at the admitted level"
    );
    assert_eq!(model.sequence_reasoning_effort(old_id), None);
    assert_eq!(run(&mut sched, &[rx])[0].len(), MAX_TOKENS);
}
