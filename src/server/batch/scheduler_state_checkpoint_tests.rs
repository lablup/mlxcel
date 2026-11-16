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

//! Turn-boundary state checkpoints through the scheduler (issue #3047).
//!
//! A conversation of four turns on the two-layer Qwen 3.5 model leaves one
//! whole history-boundary snapshot (the newest) and a checkpoint per turn. A
//! request that edits the history after turn 2 must resume at turn 2's
//! boundary by rebuilding that checkpoint over the newest snapshot, and must
//! decode exactly what a server that still holds turn 2's whole snapshot
//! decodes.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use super::scheduler_model_owned_cache_tests::{cache_ctx, options};
use super::*;
use crate::server::prompt_cache::{PromptCacheConfig, PromptCacheStore};

const TURN: usize = 12;
const SCAFFOLD: [i32; 2] = [7, 6];

fn store(checkpoint_bytes: usize) -> Arc<PromptCacheStore> {
    Arc::new(PromptCacheStore::with_config(
        PromptCacheConfig::new(true, 1 << 20, 32, Duration::from_secs(600), 4)
            .with_state_checkpoint_capacity_bytes(checkpoint_bytes),
    ))
}

fn scheduler(store: Arc<PromptCacheStore>) -> BatchScheduler {
    let (_tx, rx) = mpsc::channel();
    let sched = BatchScheduler::with_config(
        LoadedModel::Qwen35(crate::models::qwen3_5_tests::two_layer_model()),
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
        DecodeStorageBackend::Dense,
    )
    .with_prompt_cache(Some(store));
    install_thread_local_default_stream(sched.generation_stream.as_ref());
    sched
}

/// The conversation's history through turn `n`.
fn history(n: usize) -> Vec<i32> {
    (0..n * TURN).map(|i| ((i * 5 + 1) % 6) as i32).collect()
}

/// History `keep` turns long followed by an edited turn and one more.
fn edited(keep: usize) -> Vec<i32> {
    let mut tokens = history(keep);
    tokens.extend((0..2 * TURN).map(|i| ((i * 3 + 2) % 5) as i32));
    tokens
}

/// Run one chat turn whose history boundary is `history`, returning the
/// generated tokens and how many prompt tokens it adopted.
fn run_turn(sched: &mut BatchScheduler, history: &[i32]) -> (Vec<i32>, usize) {
    let mut prompt = history.to_vec();
    prompt.extend(SCAFFOLD);
    let (tx, rx) = mpsc::channel();
    let mut opts = options();
    opts.ignore_eos = true;
    opts.max_tokens = 3;
    let mut ctx = cache_ctx();
    ctx.history_prefix_tokens = Some(history.to_vec());
    opts.prompt_cache_ctx = Some(ctx);
    sched.enqueue_request(
        "prompt".to_string(),
        Some(prompt),
        opts,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        tx,
        Arc::new(AtomicBool::new(false)),
        true,
    );
    for _ in 0..64 {
        match rx.try_recv() {
            Ok(GenerateEvent::Done(result)) => {
                return (result.generated_token_ids, result.cached_tokens);
            }
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
        sched.finalize_completed();
    }
    panic!("the turn did not finish");
}

#[test]
fn an_edit_two_turns_back_resumes_from_the_rebuilt_checkpoint() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let store = store(1 << 20);
    let mut sched = scheduler(store.clone());
    for n in 1..=4 {
        let (_, adopted) = run_turn(&mut sched, &history(n));
        if n > 1 {
            assert_eq!(
                adopted,
                (n - 1) * TURN,
                "turn {n} resumes at the last boundary"
            );
        }
    }
    let stats = store.stats();
    assert_eq!(
        stats.snapshot_entries, 1,
        "earlier boundaries were superseded"
    );
    assert_eq!(stats.state_checkpoint_entries, 4);

    let (served, adopted) = run_turn(&mut sched, &edited(2));
    assert_eq!(adopted, 2 * TURN, "the edit resumes at turn 2's boundary");
    assert_eq!(store.stats().state_checkpoint_hits, 1);

    // A server whose newest snapshot is still turn 2's whole one restores it
    // directly; the rebuilt state must decode exactly the same.
    let reference_store = store_without_checkpoints();
    let mut reference = scheduler(reference_store.clone());
    for n in 1..=2 {
        let _ = run_turn(&mut reference, &history(n));
    }
    let (expected, reference_adopted) = run_turn(&mut reference, &edited(2));
    assert_eq!(reference_adopted, 2 * TURN);
    assert_eq!(reference_store.stats().state_checkpoint_hits, 0);
    assert_eq!(
        served, expected,
        "the rebuilt state decodes like the whole one"
    );
}

#[test]
fn without_a_checkpoint_budget_the_edit_starts_over() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let store = store_without_checkpoints();
    let mut sched = scheduler(store.clone());
    for n in 1..=4 {
        let _ = run_turn(&mut sched, &history(n));
    }
    assert_eq!(store.stats().state_checkpoint_entries, 0);
    let (_, adopted) = run_turn(&mut sched, &edited(2));
    assert_eq!(
        adopted, 0,
        "only turn 4's boundary is stored, and the edit diverges before it"
    );
}

fn store_without_checkpoints() -> Arc<PromptCacheStore> {
    store(0)
}

#[test]
fn a_failed_rebuild_adopts_the_best_whole_snapshot() {
    // Issue #3058: a checkpoint the model cannot rebuild must not send the
    // request back to a cold prefill when a shorter whole snapshot matches.
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    // Periodic whole snapshots every 10 tokens leave one at 20, short of
    // turn 2's boundary at 24, that the edit can still adopt.
    let _interval =
        crate::server::prompt_cache::policy::override_checkpoint_interval_for_test(Some(10));
    let store = store(1 << 20);
    let mut sched = scheduler(store.clone());
    for n in 1..=4 {
        let _ = run_turn(&mut sched, &history(n));
    }
    assert_eq!(store.stats().state_checkpoint_entries, 4);

    // Replace turn 2's checkpoint with one the model refuses to restore. It
    // keeps its key and lineage, so the lookup still offers it over the
    // newest boundary snapshot.
    let ctx = cache_ctx();
    let mut newest = history(4);
    newest.extend(SCAFFOLD);
    let newest_key = BatchScheduler::compose_prompt_cache_key(
        &ctx,
        &newest,
        sched.rope_regime_for(newest.len()),
    );
    let (donor, _) = store
        .lookup_snapshot_prefix(&newest_key, &newest)
        .expect("turn 4's boundary snapshot");
    let boundary = history(2);
    let key = BatchScheduler::compose_prompt_cache_key(
        &ctx,
        &boundary,
        sched.rope_regime_for(boundary.len() + SCAFFOLD.len()),
    );
    let broken = crate::server::prompt_cache::ModelSnapshotEntry::new(
        boundary.clone(),
        mlxcel_core::generate::ModelStateSnapshot::new("not-qwen3_5", boundary.len()),
    )
    .with_origin(crate::server::prompt_cache::SnapshotOrigin::Boundary)
    .with_lineage(donor.lineage.clipped(boundary.len()));
    store
        .insert_state_checkpoint(&key, broken)
        .expect("replace turn 2's checkpoint");
    assert_eq!(store.stats().state_checkpoint_entries, 4);

    let before = store.stats();
    let (_, adopted) = run_turn(&mut sched, &edited(2));
    let after = store.stats();
    assert_eq!(
        adopted, 20,
        "the failed rebuild falls back to the periodic snapshot at 20, not a cold prefill"
    );
    assert_eq!(
        after.state_checkpoint_hits, before.state_checkpoint_hits,
        "a failed rebuild is not a checkpoint hit"
    );
    assert_eq!(
        after.snapshot_hits,
        before.snapshot_hits + 1,
        "the request counts one hit"
    );
    assert_eq!(
        after.state_checkpoint_evictions,
        before.state_checkpoint_evictions + 1,
        "the broken checkpoint was offered and then discarded"
    );
}
