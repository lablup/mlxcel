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

//! Prometheus prompt-cache counters with the store wired as in production
//! (issue #3246).
//!
//! The server builds its store with
//! `PromptCacheStore::with_metrics(.., BatchMetricsCacheAdapter::new(batch_metrics))`
//! and hands the same `Arc<BatchMetrics>` to the scheduler. Every other
//! scheduler test builds the store with `PromptCacheStore::with_config`, which
//! has no adapter, so none of them could see the store and the scheduler both
//! counting one lookup. These tests share one `BatchMetrics` between the two,
//! drive a cold turn and a warm turn through admission, and require the
//! `/metrics` counters to equal the store's own `/v1/cache/stats` counters and
//! the `cached_tokens` the responses reported.

use std::sync::atomic::Ordering;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use super::scheduler_model_owned_cache_tests::{
    EOS_ID, cache_ctx, enqueue, options, prompt, run_to_completion, tiny_gemma3,
};
use super::scheduler_prompt_lookup_tests::cycle_llama;
use super::*;
use crate::LoadedModel;
use crate::server::config::{DecodeStorageBackend, PreemptionPolicy};
use crate::server::model_provider::{GenerateEvent, GenerationResult};
use crate::server::prompt_cache::{PromptCacheConfig, PromptCacheStore};
use crate::server::state::{BatchMetrics, BatchMetricsCacheAdapter};
use crate::tokenizer::MlxcelTokenizer;

/// A store wired exactly as `startup.rs` wires it: its metrics sink is a
/// `BatchMetricsCacheAdapter` over the scheduler's own `BatchMetrics`.
fn production_store(metrics: &Arc<BatchMetrics>) -> Arc<PromptCacheStore> {
    Arc::new(PromptCacheStore::with_metrics(
        PromptCacheConfig::new(true, 1 << 20, 32, Duration::from_secs(600), 4),
        Arc::new(BatchMetricsCacheAdapter::new(Arc::clone(metrics))),
    ))
}

fn production_scheduler(
    model: LoadedModel,
    eos: Vec<i32>,
    storage: DecodeStorageBackend,
    metrics: &Arc<BatchMetrics>,
    store: Arc<PromptCacheStore>,
) -> BatchScheduler {
    let (_tx, rx) = mpsc::channel();
    let sched = BatchScheduler::with_config(
        model,
        MlxcelTokenizer::stub(),
        eos,
        rx,
        4,
        8,
        Arc::clone(metrics),
        Arc::new(BatchObservability::new()),
        0,
        false,
        PreemptionPolicy::default(),
        1,
        storage,
    )
    .with_prompt_cache(Some(store));
    install_thread_local_default_stream(sched.generation_stream.as_ref());
    sched
}

/// Run one request with the prompt cache context attached and `max_tokens`
/// generated tokens, ticking the way the run loop does (lookahead torn down
/// before an admission, finished rows finalized after every tick) until the
/// request has finished and been donated back.
fn run_turn(sched: &mut BatchScheduler, tokens: &[i32], max_tokens: usize) -> GenerationResult {
    let (tx, rx) = mpsc::channel();
    let mut opts = options();
    opts.max_tokens = max_tokens;
    sched.enqueue_request(
        "prompt".to_string(),
        Some(tokens.to_vec()),
        opts,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        tx,
        Arc::new(std::sync::atomic::AtomicBool::new(false)),
        true,
    );
    for _ in 0..64 {
        match sched.decide_action() {
            BatchSchedulerAction::Prefill(id) => {
                sched.discard_lookahead();
                sched.execute_prefill(id);
            }
            BatchSchedulerAction::Decode(ids) => sched.execute_decode_step(&ids),
            BatchSchedulerAction::Idle => {}
            other => panic!("unexpected scheduler action {other:?}"),
        }
        sched.finalize_completed();
        if sched.active_batch.is_empty() && sched.prefill_queue.is_empty() {
            break;
        }
    }
    let mut done = None;
    while let Ok(event) = rx.try_recv() {
        match event {
            GenerateEvent::Done(result) => done = Some(result),
            GenerateEvent::Error(err) => panic!("request failed: {err}"),
            _ => {}
        }
    }
    done.expect("the request finished")
}

fn load(counter: &std::sync::atomic::AtomicU64) -> u64 {
    counter.load(Ordering::Relaxed)
}

/// Dense KV path: one cold turn (miss), then a turn that extends the stored
/// conversation (hit). The run has no declined adoption, so the scheduler's
/// adoption outcomes and the store's lookups describe the same two events and
/// the Prometheus counters must match the store's exactly, once each.
#[test]
fn dense_lookup_counts_once_in_prometheus_with_production_wiring() {
    let metrics = Arc::new(BatchMetrics::new());
    let store = production_store(&metrics);
    let mut sched = production_scheduler(
        LoadedModel::Llama(cycle_llama()),
        Vec::new(),
        DecodeStorageBackend::Dense,
        &metrics,
        Arc::clone(&store),
    );
    assert!(
        !sched.engine.model().supports_snapshot_reuse(),
        "the dense fixture must take the KV adopt path"
    );

    let first: Vec<i32> = (0..8).collect();
    let cold = run_turn(&mut sched, &first, 4);
    assert_eq!(cold.cached_tokens, 0, "turn 1 starts cold");

    // The stored conversation is a prefix of turn 2's prompt.
    let mut second = first.clone();
    second.extend_from_slice(&cold.generated_token_ids);
    second.extend([12, 13]);
    let warm = run_turn(&mut sched, &second, 1);

    // Preconditions: the store saw exactly one miss and one hit, and the
    // scheduler adopted the hit without declining it.
    let stats = store.stats();
    assert_eq!(stats.lookups, 2, "two dense lookups");
    assert_eq!(stats.hits, 1, "turn 2 hits the donated entry");
    assert!(warm.cached_tokens > 0, "turn 2 reports reused tokens");
    let obs = sched.batch_observability.snapshot();
    assert_eq!(obs.prompt_cache_hits, 1, "the scheduler adopted the hit");

    assert_eq!(
        load(&metrics.prompt_cache_hits_total),
        stats.hits,
        "/metrics hits must equal /v1/cache/stats hits"
    );
    assert_eq!(
        load(&metrics.prompt_cache_misses_total),
        stats.lookups - stats.hits,
        "/metrics misses must equal /v1/cache/stats lookups - hits"
    );
    assert_eq!(
        load(&metrics.prompt_cache_prefix_tokens_reused_total),
        (cold.cached_tokens + warm.cached_tokens) as u64,
        "reused tokens must equal the summed cached_tokens"
    );
}

/// Snapshot path: the same shape on a snapshot-reuse family. The store counts
/// a snapshot hit when its lookup matches and the scheduler counts one when
/// the restore succeeds, so before #3246 one restore moved the Prometheus
/// snapshot hit counter by two and the token counter by the store's match
/// plus the restored length.
#[test]
fn snapshot_restore_counts_once_in_prometheus_with_production_wiring() {
    let metrics = Arc::new(BatchMetrics::new());
    let store = production_store(&metrics);
    let mut sched = production_scheduler(
        LoadedModel::Gemma3(tiny_gemma3()),
        vec![EOS_ID],
        DecodeStorageBackend::Paged,
        &metrics,
        Arc::clone(&store),
    );
    assert!(
        sched.engine.model().supports_snapshot_reuse(),
        "the fixture must take the snapshot restore path"
    );

    let first = prompt(40);
    let rx = enqueue(&mut sched, first.clone());
    let cold = run_to_completion(&mut sched, &rx);
    assert_eq!(cold.cached_tokens, 0, "turn 1 starts cold");

    // `prompt(n)` cycles, so a longer prompt extends the stored one.
    let rx = enqueue(&mut sched, prompt(48));
    let warm = run_to_completion(&mut sched, &rx);

    let stats = store.stats();
    assert_eq!(stats.snapshot_lookups, 2, "two snapshot lookups");
    assert_eq!(stats.snapshot_hits, 1, "turn 2 restores the snapshot");
    assert!(warm.cached_tokens > 0, "turn 2 reports reused tokens");
    let obs = sched.batch_observability.snapshot();
    assert_eq!(obs.prompt_cache_hits, 1, "the scheduler restored the hit");

    assert_eq!(
        load(&metrics.prompt_cache_snapshot_hits_total),
        stats.snapshot_hits,
        "/metrics snapshot hits must equal /v1/cache/stats snapshot_hits"
    );
    assert_eq!(
        load(&metrics.prompt_cache_snapshot_misses_total),
        stats.snapshot_lookups - stats.snapshot_hits,
        "/metrics snapshot misses must equal snapshot_lookups - snapshot_hits"
    );
    assert_eq!(
        load(&metrics.prompt_cache_snapshot_tokens_reused_total),
        (cold.cached_tokens + warm.cached_tokens) as u64,
        "snapshot reused tokens must equal the summed cached_tokens"
    );
    // A restore is not a dense KV adoption.
    assert_eq!(load(&metrics.prompt_cache_hits_total), 0);
}

/// Every prompt-cache counter the two surfaces expose, read at one instant:
/// the Prometheus atomics and the store's `/v1/cache/stats` counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Counters {
    hits_total: u64,
    misses_total: u64,
    snapshot_hits_total: u64,
    snapshot_misses_total: u64,
    lookups: u64,
    hits: u64,
    snapshot_lookups: u64,
    snapshot_hits: u64,
}

fn counters(metrics: &BatchMetrics, store: &PromptCacheStore) -> Counters {
    let stats = store.stats();
    Counters {
        hits_total: load(&metrics.prompt_cache_hits_total),
        misses_total: load(&metrics.prompt_cache_misses_total),
        snapshot_hits_total: load(&metrics.prompt_cache_snapshot_hits_total),
        snapshot_misses_total: load(&metrics.prompt_cache_snapshot_misses_total),
        lookups: stats.lookups,
        hits: stats.hits,
        snapshot_lookups: stats.snapshot_lookups,
        snapshot_hits: stats.snapshot_hits,
    }
}

/// The production-wired Gemma 3 harness after one cold turn on `prompt(40)`,
/// which leaves one snapshot of that conversation in the store.
fn gemma3_after_cold_turn() -> (Arc<BatchMetrics>, Arc<PromptCacheStore>, BatchScheduler) {
    let metrics = Arc::new(BatchMetrics::new());
    let store = production_store(&metrics);
    let mut sched = production_scheduler(
        LoadedModel::Gemma3(tiny_gemma3()),
        vec![EOS_ID],
        DecodeStorageBackend::Paged,
        &metrics,
        Arc::clone(&store),
    );
    let rx = enqueue(&mut sched, prompt(40));
    let cold = run_to_completion(&mut sched, &rx);
    assert_eq!(cold.cached_tokens, 0, "turn 1 starts cold");
    assert!(
        store.stats().snapshot_entries >= 1,
        "turn 1 donated a snapshot"
    );
    (metrics, store, sched)
}

/// Snapshot family whose state lives outside the KV slice (issue #3294):
/// Gemma 3 never runs a dense KV
/// lookup, so a cold turn is not a dense miss. Before the fix admission
/// recorded `prompt_cache_misses_total` for it while `lookups` stayed at 0.
#[test]
fn snapshot_family_cold_turn_records_no_dense_miss() {
    let (metrics, store, mut sched) = gemma3_after_cold_turn();
    let rx = enqueue(&mut sched, prompt(48));
    let warm = run_to_completion(&mut sched, &rx);
    assert!(warm.cached_tokens > 0, "turn 2 restores the snapshot");

    let c = counters(&metrics, &store);
    assert_eq!(c.lookups, 0, "a snapshot family runs no dense lookup");
    assert_eq!(c.misses_total, 0, "and so records no dense miss");
    assert_eq!(c.misses_total + c.hits_total, c.lookups);
    assert_eq!(c.snapshot_hits_total, c.snapshot_hits);
    assert_eq!(
        c.snapshot_misses_total,
        c.snapshot_lookups - c.snapshot_hits
    );
}

/// The `!kv_cache_captures_sequence_state()` arm of the same bail, which
/// Gemma 3 does not reach: Baichuan-M1 keeps convolution state outside its KV
/// cache, so it neither donates nor looks up a dense entry (#2567).
#[test]
fn kv_state_incomplete_family_records_no_dense_miss() {
    let metrics = Arc::new(BatchMetrics::new());
    let store = production_store(&metrics);
    let mut sched = production_scheduler(
        LoadedModel::Baichuan(crate::models::baichuan::baichuan_tests::tiny_baichuan_model()),
        Vec::new(),
        DecodeStorageBackend::Dense,
        &metrics,
        Arc::clone(&store),
    );
    let model = sched.engine.model();
    assert!(
        !model.kv_cache_captures_sequence_state(),
        "the fixture must take the second arm"
    );
    assert!(
        !model
            .sequence_state_layout()
            .backend
            .keeps_state_outside_kv_slice(),
        "the fixture must not take the first arm"
    );
    assert!(!model.supports_snapshot_reuse());

    let first: Vec<i32> = (0..8).collect();
    let cold = run_turn(&mut sched, &first, 2);
    let mut second = first.clone();
    second.extend_from_slice(&cold.generated_token_ids);
    second.extend([20, 21]);
    run_turn(&mut sched, &second, 1);

    let c = counters(&metrics, &store);
    assert_eq!(c.lookups, 0, "no dense lookup ran");
    assert_eq!(c.misses_total, 0, "so no dense miss is recorded");
    assert_eq!(c.hits_total, 0);
}

/// A dense hit the scheduler declines (here `PrefixTooShort`, an exact replay
/// of an entry exactly `min_prefix_tokens` long, which the whole-prompt cap
/// leaves one token short) keeps the #3268 rule: the store counts a hit, the
/// scheduler counts a miss, and `/metrics` hits mean adoptions. This pins the
/// rule; it passes before and after #3294.
#[test]
fn declined_dense_hit_counts_as_a_dense_miss() {
    let first: Vec<i32> = (0..8).collect();
    let metrics = Arc::new(BatchMetrics::new());
    let store = Arc::new(PromptCacheStore::with_metrics(
        PromptCacheConfig::new(true, 1 << 20, 32, Duration::from_secs(600), first.len()),
        Arc::new(BatchMetricsCacheAdapter::new(Arc::clone(&metrics))),
    ));
    let mut sched = production_scheduler(
        LoadedModel::Llama(cycle_llama()),
        Vec::new(),
        DecodeStorageBackend::Dense,
        &metrics,
        Arc::clone(&store),
    );
    let cold = run_turn(&mut sched, &first, 1);
    assert_eq!(cold.cached_tokens, 0, "turn 1 starts cold");
    let too_short = || {
        sched
            .batch_observability
            .prompt_cache_reject_reasons
            .count(PromptCacheRejectReason::PrefixTooShort)
    };
    let declines_before = too_short();
    let hits_before = store.stats().hits;

    let replay = run_turn(&mut sched, &first, 1);
    assert_eq!(replay.cached_tokens, 0, "the declined hit prefills cold");
    // Preconditions: the store matched and the scheduler declined it.
    assert_eq!(store.stats().hits, hits_before + 1, "the store saw a hit");
    let declines_after = sched
        .batch_observability
        .prompt_cache_reject_reasons
        .count(PromptCacheRejectReason::PrefixTooShort);
    assert_eq!(
        declines_after,
        declines_before + 1,
        "the scheduler declined it"
    );

    let c = counters(&metrics, &store);
    assert_eq!(c.hits_total, 0, "/metrics hits count adoptions only");
    assert_eq!(c.misses_total, 2, "the cold turn and the declined hit");
    assert_eq!(c.hits_total + c.misses_total, c.lookups);
}

/// A background warm-up whose base-snapshot lookup misses is not a request
/// lookup (issue #3294): it must move neither surface's snapshot counters.
#[test]
fn warmup_snapshot_miss_is_not_counted() {
    assert!(!crate::server::prompt_cache::cache_warmup_disabled());
    let (metrics, store, mut sched) = gemma3_after_cold_turn();
    let before = counters(&metrics, &store);
    let skipped = sched
        .batch_observability
        .snapshot()
        .prompt_cache_warmups_skipped;

    sched.enqueue_prompt_cache_warmup(vec![5, 4, 3, 2, 1, 0], cache_ctx());
    assert_eq!(
        sched.prompt_cache_warmups.len(),
        1,
        "the warm-up was queued"
    );
    sched.run_next_prompt_cache_warmup();
    assert_eq!(
        sched
            .batch_observability
            .snapshot()
            .prompt_cache_warmups_skipped,
        skipped + 1,
        "the warm-up ran and found no base snapshot"
    );

    assert_eq!(
        counters(&metrics, &store),
        before,
        "a warm-up miss is not a request lookup"
    );
}

/// A warm-up whose base-snapshot lookup hits: not counted either, and the
/// conversation really is warmed afterwards (regression guard).
#[test]
fn warmup_snapshot_hit_is_not_counted_and_still_warms() {
    assert!(!crate::server::prompt_cache::cache_warmup_disabled());
    let (metrics, store, mut sched) = gemma3_after_cold_turn();
    let before = counters(&metrics, &store);
    let run = sched
        .batch_observability
        .snapshot()
        .prompt_cache_warmups_run;

    let target = prompt(44);
    sched.enqueue_prompt_cache_warmup(target.clone(), cache_ctx());
    assert_eq!(
        sched.prompt_cache_warmups.len(),
        1,
        "the warm-up was queued"
    );
    sched.run_next_prompt_cache_warmup();
    assert_eq!(
        sched
            .batch_observability
            .snapshot()
            .prompt_cache_warmups_run,
        run + 1,
        "the warm-up restored a base snapshot and ran"
    );
    // Read before the probe below, which counts itself.
    let after = counters(&metrics, &store);
    assert_eq!(after, before, "a warm-up hit is not a request lookup");

    let ctx = cache_ctx();
    let key = BatchScheduler::compose_prompt_cache_key(
        &ctx,
        &target,
        sched.rope_regime_for(target.len()),
    );
    let (_, matched_len) = store
        .lookup_snapshot_prefix(&key, &target)
        .expect("the warm-up stored a snapshot of its target");
    assert_eq!(
        matched_len,
        target.len(),
        "the warm-up target is fully warm"
    );
}
