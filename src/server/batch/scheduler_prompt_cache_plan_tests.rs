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

use super::scheduler_model_owned_cache_tests::{
    options, prompt, test_store, tiny_gemma3, tiny_gemma3_args, tiny_gemma3_weights,
};
use super::*;
use crate::LoadedModel;
use crate::models::{Gemma3Model, Gemma3Wrapper};
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
    plan_scheduler_with_model(tiny_gemma3(), store, chunk)
}

fn plan_scheduler_with_model(
    model: Gemma3Wrapper,
    store: Arc<PromptCacheStore>,
    chunk: usize,
) -> BatchScheduler {
    let (_tx, rx) = mpsc::channel();
    let sched = BatchScheduler::with_config(
        LoadedModel::Gemma3(model),
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

/// Scheduler with the production adaptive chunk policy.
fn adaptive_plan_scheduler() -> BatchScheduler {
    let mut args = tiny_gemma3_args();
    args.max_position_embeddings = 8192;
    let model = Gemma3Wrapper::new(
        Gemma3Model::from_weights(&tiny_gemma3_weights(), &args).expect("tiny gemma3 loads"),
    );
    plan_scheduler_with_model(
        model,
        test_store(),
        mlxcel_core::prefill_plan::DEFAULT_PREFILL_CHUNK,
    )
    .with_prefill_chunk_policy(mlxcel_core::prefill_plan::PrefillChunkPolicy::resolve(None))
}

/// Admit a synthetic request and return its queued sequence and response
/// stream for execution tests.
fn queued_prompt(
    sched: &mut BatchScheduler,
    prompt_len: usize,
) -> (SequenceInfo, mpsc::Receiver<GenerateEvent>) {
    let mut opts = options();
    opts.max_tokens = 1;
    opts.ignore_eos = true;
    let tokens: Vec<i32> = (0..prompt_len).map(|index| (index % 6) as i32).collect();
    let (tx, rx) = mpsc::channel();
    sched.enqueue_request(
        "prompt".to_string(),
        Some(tokens),
        opts,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        tx,
        Arc::new(AtomicBool::new(false)),
        true,
    );
    (
        sched.prefill_queue.dequeue().expect("request is queued"),
        rx,
    )
}

/// Put a dequeued request back at the head and execute its first real prefill
/// forward through the scheduler and tiny Gemma test model.
fn start_recorded_prefill(sched: &mut BatchScheduler, seq: SequenceInfo) {
    let seq_id = seq.seq_id;
    sched
        .prefill_queue
        .enqueue_front(seq)
        .unwrap_or_else(|_| panic!("re-queue adaptive prefill request"));
    sched.execute_prefill(seq_id);
}

/// Execute every parked continuation, then require a successful Done frame.
fn finish_recorded_prefill(sched: &mut BatchScheduler, rx: &mpsc::Receiver<GenerateEvent>) {
    while sched.chunked_prefill_seq.is_some() {
        assert!(
            sched.continue_chunked_prefill(),
            "every parked continuation runs a model forward"
        );
    }
    loop {
        match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(GenerateEvent::Done(_)) => break,
            Ok(GenerateEvent::Error(err)) => panic!("adaptive prefill aborted: {err}"),
            Ok(_) => {}
            Err(err) => panic!("adaptive prefill did not complete: {err}"),
        }
    }
}

#[test]
fn adaptive_chunk_policy_executes_and_resumes_real_prefill_forwards() {
    // A live decode row selects 512 for every executed piece, including the
    // one-token terminal continuation.
    {
        let mut sched = adaptive_plan_scheduler();
        let (live, _live_rx) = queued_prompt(&mut sched, 8);
        sched.active_batch.add(live).expect("decode row fits");
        let (contended, rx) = queued_prompt(&mut sched, 1537);
        start_recorded_prefill(&mut sched, contended);
        finish_recorded_prefill(&mut sched, &rx);
        assert_eq!(sched.prefill_forward_lengths, vec![512, 512, 512, 1]);
    }

    // With no other row decoding, the same execution path selects 2048.
    {
        let mut sched = adaptive_plan_scheduler();
        let (alone, rx) = queued_prompt(&mut sched, 3000);
        start_recorded_prefill(&mut sched, alone);
        finish_recorded_prefill(&mut sched, &rx);
        assert_eq!(sched.prefill_forward_lengths, vec![2048, 952]);
    }

    // The decode row drains after the first forward. The parked request is
    // repartitioned at its live 512-token cursor and completes without the old
    // "no remaining tokens" abort.
    {
        let mut sched = adaptive_plan_scheduler();
        let (live, _live_rx) = queued_prompt(&mut sched, 8);
        let live_id = live.seq_id;
        sched.active_batch.add(live).expect("decode row fits");
        let (changing, rx) = queued_prompt(&mut sched, 5000);
        start_recorded_prefill(&mut sched, changing);
        assert_eq!(sched.prefill_forward_lengths, vec![512]);
        assert!(sched.chunked_prefill_seq.is_some());

        sched
            .active_batch
            .remove(live_id)
            .expect("decode row exists");
        finish_recorded_prefill(&mut sched, &rx);
        assert_eq!(sched.prefill_forward_lengths, vec![512, 2048, 2048, 392]);
    }
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
        root_prompt: None,
        root_prefix_tokens: None,
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
    let run = run_turn_observed(sched, tokens, history, max_tokens);
    (run.result, run.plan, run.forwarded)
}

/// What [`run_turn_observed`] saw of one request.
struct ObservedTurn {
    result: GenerationResult,
    plan: mlxcel_core::prefill_plan::PrefillPlan,
    forwarded: u64,
    /// `processed` of every mid-prefill `prompt_progress` frame, in order, the
    /// opening frame included.
    progress_frames: Vec<usize>,
    /// How many chunks the chunk counter gained.
    chunks_counted: u64,
}

/// [`run_turn`] that also reports the `prompt_progress` frames the request
/// emitted and the chunk counter's movement.
fn run_turn_observed(
    sched: &mut BatchScheduler,
    tokens: &[i32],
    history: Option<&[i32]>,
    max_tokens: usize,
) -> ObservedTurn {
    run_turn_with_ctx(sched, tokens, turn_ctx(history), max_tokens)
}

/// [`run_turn_observed`] with an explicit cache context.
fn run_turn_with_ctx(
    sched: &mut BatchScheduler,
    tokens: &[i32],
    ctx: PromptCacheRequestContext,
    max_tokens: usize,
) -> ObservedTurn {
    let mut opts = options();
    opts.max_tokens = max_tokens;
    opts.ignore_eos = true;
    opts.prompt_cache_ctx = Some(ctx);
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
    let before = sched.batch_observability.snapshot();
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
    let mut progress_frames = Vec::new();
    let result = loop {
        match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(GenerateEvent::Done(result)) => break result,
            Ok(GenerateEvent::Error(err)) => panic!("unexpected generation error: {err}"),
            Ok(GenerateEvent::Prefill(stats)) if !stats.first_token => {
                progress_frames.push(stats.processed);
            }
            Ok(_) => {}
            Err(err) => panic!("generation did not finish: {err}"),
        }
    };
    let after = sched.batch_observability.snapshot();
    ObservedTurn {
        result,
        plan,
        forwarded: after.total_prefill_tokens - before.total_prefill_tokens,
        progress_frames,
        chunks_counted: after.prefill_chunks_processed - before.prefill_chunks_processed,
    }
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

/// Only a real chunked prefill counts toward `mlxcel_batch_prefill_chunks_total`
/// (ADR 0005's mixed-step dispatch proof) and emits a `prompt_progress` frame
/// per chunk (b10621's cadence, #1477), after the one opening frame that
/// `begin_prefill` sends for every prefill. A prompt that fits one chunk, an
/// unchunked server and a history-boundary segment plus its suffix are one
/// unchunked prefill and neither count nor emit more, as before the plan
/// existed.
#[test]
fn only_real_chunks_count_and_emit_prompt_progress() {
    let history = prompt(24);
    let mut turn = history.clone();
    turn.extend((0..16).map(|i| (i + 1) % 6));
    let no_split: &[i32] = &turn[..40];

    // Fits one chunk: a single-pass prefill.
    let mut sched = plan_scheduler(test_store(), 64);
    let run = run_turn_observed(&mut sched, no_split, None, 2);
    assert_eq!(run.plan.ranges(), vec![0..40]);
    assert_eq!(run.chunks_counted, 0, "a single-piece prefill is no chunk");
    assert_eq!(run.progress_frames, vec![0]);

    // Chunking disabled.
    let mut sched = plan_scheduler(test_store(), 0);
    let run = run_turn_observed(&mut sched, no_split, None, 2);
    assert_eq!(run.chunks_counted, 0);
    assert_eq!(run.progress_frames, vec![0]);

    // History boundary without chunking: segment plus suffix, no chunk.
    let mut sched = plan_scheduler(test_store(), 0);
    let run = run_turn_observed(&mut sched, &turn, Some(&history), 2);
    assert_eq!(run.plan.ranges(), vec![0..24, 24..40]);
    assert_eq!(run.chunks_counted, 0, "the boundary segment is no chunk");
    assert_eq!(run.progress_frames, vec![0]);

    // Chunked: one count and one frame per chunk.
    let mut sched = plan_scheduler(test_store(), 8);
    let run = run_turn_observed(&mut sched, no_split, None, 2);
    assert_eq!(run.plan.ranges(), vec![0..8, 8..16, 16..24, 24..32, 32..40]);
    assert_eq!(run.chunks_counted, 5);
    assert_eq!(run.progress_frames, vec![0, 8, 16, 24, 32, 40]);

    // Chunked with a boundary: the segment is skipped, each chunk counts.
    let mut sched = plan_scheduler(test_store(), 8);
    let run = run_turn_observed(&mut sched, &turn, Some(&history), 2);
    assert_eq!(run.plan.ranges(), vec![0..24, 24..32, 32..40]);
    assert_eq!(run.chunks_counted, 2);
    assert_eq!(run.progress_frames, vec![0, 32, 40]);
}

/// #2885 Part B: with the system-prompt root on, chat A stores a root
/// snapshot at the end of the shared system block in addition to its history
/// boundary. Chat B, a new conversation that shares only that block, adopts
/// the root, forwards everything after it, and decodes what a cold run of
/// chat B (split at the same root) decodes.
#[test]
fn new_chat_adopts_the_system_root_and_reproduces_its_cold_run() {
    let _root_on = crate::server::prompt_cache::policy::override_system_root_for_test(Some(0));
    let ctx = |tokens: &[i32], history: usize, root: usize| {
        let mut ctx = turn_ctx(Some(&tokens[..history]));
        // The render ends at the user header; the scheduler drops that token.
        ctx.root_prefix_tokens = Some(tokens[..root + 1].to_vec());
        ctx
    };
    for chunk in [0usize, 8] {
        let system = prompt(12);
        let mut chat_a = system.clone();
        chat_a.extend((0..20).map(|i| (i + 3) % 6));
        let mut chat_b = system.clone();
        chat_b.extend((0..20).map(|i| (i + 1) % 5));

        // Cold run of chat B on its own scheduler.
        let mut cold = plan_scheduler(test_store(), chunk);
        let miss = run_turn_with_ctx(&mut cold, &chat_b, ctx(&chat_b, 24, 11), 4);
        assert_eq!(miss.result.cached_tokens, 0, "chunk {chunk}");
        assert_eq!(miss.plan.splits().as_slice(), &[11, 24], "chunk {chunk}");
        assert_eq!(miss.plan.ranges()[..2], [0..11, 11..24], "chunk {chunk}");

        // Chat A stores the root and its history boundary.
        let store = test_store();
        let mut served = plan_scheduler(store.clone(), chunk);
        let a = run_turn_with_ctx(&mut served, &chat_a, ctx(&chat_a, 24, 11), 1);
        assert_eq!(a.plan.splits().as_slice(), &[11, 24], "chunk {chunk}");
        let stats = store.stats();
        assert!(
            stats.snapshot_inserts >= 2,
            "chunk {chunk}: chat A stored the root and the boundary ({} inserts)",
            stats.snapshot_inserts
        );

        // Chat B adopts the root and reproduces its cold run.
        let hit = run_turn_with_ctx(&mut served, &chat_b, ctx(&chat_b, 24, 11), 4);
        assert_eq!(hit.result.cached_tokens, 11, "chunk {chunk}");
        assert_eq!(hit.forwarded, (chat_b.len() - 11) as u64, "chunk {chunk}");
        assert!(
            hit.plan.reproduces(&miss.plan),
            "chunk {chunk}: hit {:?} vs miss {:?}",
            hit.plan.ranges(),
            miss.plan.ranges()
        );
        assert_eq!(
            hit.result.generated_token_ids, miss.result.generated_token_ids,
            "chunk {chunk}"
        );
    }
}

/// With the switch off a root token vector on the context is ignored: the
/// plan splits only at the history boundary, exactly as before #2885.
#[test]
fn system_root_is_ignored_when_the_switch_is_off() {
    let _root_off = crate::server::prompt_cache::policy::override_system_root_for_test(None);
    let tokens: Vec<i32> = {
        let mut t = prompt(12);
        t.extend((0..20).map(|i| (i + 3) % 6));
        t
    };
    let mut ctx = turn_ctx(Some(&tokens[..24]));
    ctx.root_prefix_tokens = Some(tokens[..12].to_vec());
    let mut sched = plan_scheduler(test_store(), 0);
    let run = run_turn_with_ctx(&mut sched, &tokens, ctx, 1);
    assert_eq!(run.plan.splits().as_slice(), &[24]);
}

/// #2885 Part B: with periodic checkpoints on, a prefill also splits at every
/// multiple of the interval before its history boundary and stores a snapshot
/// at each. A later request that edits the history between two checkpoints
/// adopts the last one before the edit, forwards the rest, and decodes what a
/// cold run of the edited prompt (split at the same points) decodes.
#[test]
fn edited_history_adopts_the_last_checkpoint_and_reproduces_its_cold_run() {
    let _checkpoints_on =
        crate::server::prompt_cache::policy::override_checkpoint_interval_for_test(Some(8));
    for chunk in [0usize, 8] {
        let original: Vec<i32> = prompt(32);
        // Rewrite the conversation from token 19 on, as an agent compacting
        // its history would.
        let mut edited = original[..19].to_vec();
        edited.extend((0..13).map(|i| (i + 2) % 7));

        let mut cold = plan_scheduler(test_store(), chunk);
        let miss = run_turn_with_ctx(&mut cold, &edited, turn_ctx(Some(&edited[..28])), 4);
        assert_eq!(miss.result.cached_tokens, 0, "chunk {chunk}");
        assert_eq!(
            miss.plan.splits().as_slice(),
            &[8, 16, 24, 28],
            "chunk {chunk}"
        );

        let store = test_store();
        let mut served = plan_scheduler(store.clone(), chunk);
        let first = run_turn_with_ctx(&mut served, &original, turn_ctx(Some(&original[..28])), 1);
        assert_eq!(
            first.plan.splits().as_slice(),
            &[8, 16, 24, 28],
            "chunk {chunk}"
        );
        assert_eq!(
            store.stats().snapshot_supersedes,
            0,
            "chunk {chunk}: checkpoints of one conversation stay side by side"
        );

        let hit = run_turn_with_ctx(&mut served, &edited, turn_ctx(Some(&edited[..28])), 4);
        assert_eq!(hit.result.cached_tokens, 16, "chunk {chunk}");
        assert_eq!(hit.forwarded, (edited.len() - 16) as u64, "chunk {chunk}");
        assert!(
            hit.plan.reproduces(&miss.plan),
            "chunk {chunk}: hit {:?} vs miss {:?}",
            hit.plan.ranges(),
            miss.plan.ranges()
        );
        assert_eq!(
            hit.result.generated_token_ids, miss.result.generated_token_ids,
            "chunk {chunk}"
        );
    }
}

/// With the switch off the plan splits only at the history boundary.
#[test]
fn checkpoints_are_off_by_default() {
    let _checkpoints_off =
        crate::server::prompt_cache::policy::override_checkpoint_interval_for_test(None);
    let tokens = prompt(32);
    let mut sched = plan_scheduler(test_store(), 0);
    let run = run_turn_with_ctx(&mut sched, &tokens, turn_ctx(Some(&tokens[..28])), 1);
    assert_eq!(run.plan.splits().as_slice(), &[28]);
}
