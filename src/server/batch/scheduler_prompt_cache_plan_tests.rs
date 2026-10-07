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

//! Prompt-cache hit versus miss under one prefill plan (issue #2170).
//!
//! A hit adopts a stored prefix and forwards only what follows it. It
//! reproduces the cold run of the same prompt exactly when the adopted prefix
//! ends on a split point of the cold plan, because from there on both runs
//! forward the same pieces from the same state. These cases drive the real
//! scheduler over the tiny Gemma 3 fixture (a snapshot family whose chat
//! prompts split at the history boundary): a follow-up turn that adopts the
//! boundary snapshot of an earlier turn decodes the same tokens as a cold run
//! of that follow-up, and a hit from inside a piece is reported as such by the
//! plan. `docs/CONTINUOUS_BATCHING.md` records the real-checkpoint numbers.

use super::scheduler_model_owned_cache_tests::{options, prompt, test_store, tiny_gemma3};
use super::*;
use crate::LoadedModel;
use crate::server::config::{DecodeStorageBackend, PreemptionPolicy, PromptCacheRequestContext};
use crate::server::model_provider::{GenerateEvent, GenerationResult};
use crate::server::prompt_cache::{PromptCacheStore, key::MultimodalDigest};
use crate::server::state::BatchMetrics;
use crate::tokenizer::MlxcelTokenizer;
use mlxcel_core::streams::install_thread_local_default_stream;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, mpsc};
use std::time::Duration;

/// The model-owned fixture scheduler of `scheduler_model_owned_cache_tests`
/// with an explicit prefill chunk, so the plan's chunk pieces run across
/// ticks.
fn plan_scheduler(store: Arc<PromptCacheStore>, chunk: usize) -> BatchScheduler {
    let (_tx, rx) = mpsc::channel();
    let sched = BatchScheduler::with_config(
        LoadedModel::Gemma3(tiny_gemma3()),
        MlxcelTokenizer::stub(),
        vec![7],
        rx,
        4,
        8,
        Arc::new(BatchMetrics::new()),
        Arc::new(BatchObservability::new()),
        chunk,
        false,
        PreemptionPolicy::default(),
        1,
        DecodeStorageBackend::Paged,
    )
    .with_prompt_cache(Some(store));
    install_thread_local_default_stream(sched.generation_stream.as_ref());
    sched
}

/// A chat turn's cache context: `history` is the rendering without the
/// generation prompt, which is where the plan splits the prefill.
fn turn_ctx(history: Option<&[i32]>) -> PromptCacheRequestContext {
    PromptCacheRequestContext {
        model_id: "tiny-gemma3".to_string(),
        lora_id: None,
        template_sig: "tpl-sig-v1".to_string(),
        session_key: "session-plan".to_string(),
        mm_digest: MultimodalDigest::empty(),
        history_prompt: None,
        history_prefix_tokens: history.map(<[i32]>::to_vec),
    }
}

/// Enqueue `tokens` as a greedy request of `max_tokens`, then run the
/// scheduler to completion, chunked prefill ticks included, and return the
/// result plus the plan the scheduler built for the request and the prefill
/// tokens it forwarded.
fn run_turn(
    sched: &mut BatchScheduler,
    tokens: &[i32],
    history: Option<&[i32]>,
    max_tokens: usize,
) -> (
    GenerationResult,
    mlxcel_core::prefill_plan::PrefillPlan,
    u64,
) {
    let mut opts = options();
    opts.max_tokens = max_tokens;
    opts.ignore_eos = true;
    opts.prompt_cache_ctx = Some(turn_ctx(history));
    let (tx, rx) = mpsc::channel();
    sched.enqueue_request(
        "prompt".to_string(),
        Some(tokens.to_vec()),
        opts,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        tx,
        Arc::new(AtomicBool::new(false)),
        true,
    );
    // Read the plan off the admitted sequence, then put it back untouched.
    let queued = sched
        .prefill_queue
        .dequeue()
        .expect("the request is queued");
    let plan = sched.prefill_plan_for(&queued);
    sched
        .prefill_queue
        .enqueue_front(queued)
        .unwrap_or_else(|_| panic!("re-queue"));
    let forwarded_before = sched.batch_observability.snapshot().total_prefill_tokens;
    for _ in 0..32 {
        match sched.decide_action() {
            BatchSchedulerAction::Prefill(id) => sched.execute_prefill(id),
            BatchSchedulerAction::Decode(ids) => sched.execute_decode_step(&ids),
            BatchSchedulerAction::Idle => break,
            other => panic!("unexpected scheduler action {other:?}"),
        }
        sched.finalize_completed();
        if sched.active_batch.is_empty()
            && sched.prefill_queue.is_empty()
            && sched.chunked_prefill_seq.is_none()
        {
            break;
        }
    }
    let result = loop {
        match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(GenerateEvent::Done(result)) => break result,
            Ok(GenerateEvent::Error(err)) => panic!("unexpected generation error: {err}"),
            Ok(_) => {}
            Err(err) => panic!("generation did not finish: {err}"),
        }
    };
    let forwarded = sched.batch_observability.snapshot().total_prefill_tokens - forwarded_before;
    (result, plan, forwarded)
}

/// Turn 1 (`history ++ tail_a`) splits at the history boundary and snapshots
/// there. A follow-up turn (`history ++ tail_b`) adopts that snapshot and
/// forwards only `tail_b`, which is exactly the piece a cold prefill of the
/// follow-up forwards after its own boundary split: same pieces, same
/// state, same tokens. Chunked and unchunked alike.
#[test]
fn cache_hit_reproduces_miss_exactly_from_a_plan_split_point() {
    for chunk in [0usize, 8] {
        let history = prompt(24);
        let mut turn_a = history.clone();
        turn_a.extend((0..8).map(|i| (i + 3) % 6));
        let mut turn_b = history.clone();
        turn_b.extend((0..16).map(|i| (i + 1) % 6));

        // Cold run of the follow-up turn: its own scheduler, nothing stored.
        let cold_store = test_store();
        let mut cold = plan_scheduler(cold_store.clone(), chunk);
        let (miss, miss_plan, miss_forwarded) = run_turn(&mut cold, &turn_b, Some(&history), 4);
        assert_eq!(
            miss.cached_tokens, 0,
            "chunk {chunk}: the cold run adopts nothing"
        );
        assert_eq!(
            miss_plan.boundary(),
            Some(24),
            "chunk {chunk}: the plan splits at the boundary"
        );
        assert_eq!(
            miss_forwarded, 40,
            "chunk {chunk}: the cold run forwards the whole prompt"
        );
        let mut expected: Vec<std::ops::Range<usize>> = std::iter::once(0..24).collect();
        expected.extend(if chunk == 8 {
            vec![24..32, 32..40]
        } else {
            std::iter::once(24..40).collect()
        });
        assert_eq!(miss_plan.ranges(), expected, "chunk {chunk}");

        // Turn 1 in the served session stores the boundary snapshot...
        let store = test_store();
        let mut served = plan_scheduler(store.clone(), chunk);
        let (_, plan_a, _) = run_turn(&mut served, &turn_a, Some(&history), 1);
        assert_eq!(plan_a.boundary(), Some(24), "chunk {chunk}");
        assert!(
            store.stats().snapshot_entries >= 1,
            "chunk {chunk}: turn 1 donated its boundary snapshot"
        );
        // ...and the follow-up adopts it, forwarding only its own tail.
        let (hit, hit_plan, hit_forwarded) = run_turn(&mut served, &turn_b, Some(&history), 4);
        assert_eq!(
            hit.cached_tokens, 24,
            "chunk {chunk}: the hit adopts the boundary snapshot"
        );
        assert_eq!(
            hit_forwarded, 16,
            "chunk {chunk}: the hit forwards only the suffix"
        );
        assert_eq!(hit_plan.adopted(), 24, "chunk {chunk}");
        assert!(
            hit_plan.reproduces(&miss_plan),
            "chunk {chunk}: the hit's pieces {:?} are the tail of the miss's {:?}",
            hit_plan.ranges(),
            miss_plan.ranges()
        );
        assert_eq!(
            hit.generated_token_ids, miss.generated_token_ids,
            "chunk {chunk}: a hit from a plan split point decodes what the miss decoded"
        );
    }
}

/// A completion entry stored at a length that is not a split point of the
/// follow-up's cold plan wins the lookup (it is the longer prefix), and the
/// hit then forwards a piece the miss never ran. The plan reports that, and
/// the output may differ by the documented near-tie class; the request is
/// still served correctly from the adopted state.
#[test]
fn cache_hit_from_inside_a_piece_is_reported_by_the_plan() {
    let history = prompt(24);
    let mut turn_b = history.clone();
    turn_b.extend((0..16).map(|i| (i + 1) % 6));

    let store = test_store();
    let mut served = plan_scheduler(store.clone(), 0);
    // A raw completion of the first 30 tokens (no history boundary) stores a
    // completion snapshot keyed by those 30 tokens.
    let (primed, primed_plan, _) = run_turn(&mut served, &turn_b[..30], None, 1);
    assert_eq!(primed.cached_tokens, 0);
    assert_eq!(primed_plan.ranges(), vec![0..30]);
    assert_eq!(store.stats().snapshot_entries, 1);

    let cold_plan = {
        let cold = plan_scheduler(test_store(), 0);
        let mut opts = options();
        opts.prompt_cache_ctx = Some(turn_ctx(Some(&history)));
        let (tx, _rx) = mpsc::channel();
        let mut cold = cold;
        cold.enqueue_request(
            "prompt".to_string(),
            Some(turn_b.clone()),
            opts,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            tx,
            Arc::new(AtomicBool::new(false)),
            true,
        );
        let queued = cold.prefill_queue.dequeue().expect("queued");
        cold.prefill_plan_for(&queued)
    };
    assert_eq!(cold_plan.ranges(), vec![0..24, 24..40]);

    let (hit, hit_plan, hit_forwarded) = run_turn(&mut served, &turn_b, Some(&history), 4);
    assert_eq!(
        hit.cached_tokens, 30,
        "the longer completion entry wins the lookup"
    );
    assert_eq!(hit_forwarded, 10);
    assert_eq!(hit_plan.ranges(), vec![30..40]);
    assert!(
        !hit_plan.reproduces(&cold_plan),
        "30 is inside the cold plan's [24..40) piece, so the hit forwards a piece the miss never ran"
    );
    assert_eq!(
        hit.generated_token_ids.len(),
        4,
        "the hit still decodes to completion"
    );
}
