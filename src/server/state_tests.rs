// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

use std::sync::atomic::Ordering;

use super::{BatchMetrics, Metrics};

#[test]
fn metrics_record_request_accumulates_counters() {
    let metrics = Metrics::new();

    metrics.record_request(10, 3, 120);
    metrics.record_request(4, 2, 30);

    assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 2);
    assert_eq!(metrics.prompt_tokens_total.load(Ordering::Relaxed), 14);
    assert_eq!(metrics.completion_tokens_total.load(Ordering::Relaxed), 5);
    assert_eq!(
        metrics.generation_time_ms_total.load(Ordering::Relaxed),
        150
    );
}

// ------------------------------------------------------------------
// BatchMetrics
// ------------------------------------------------------------------

#[test]
fn batch_metrics_initializes_to_zero() {
    let m = BatchMetrics::new();
    assert_eq!(m.active_count(), 0);
    assert_eq!(m.queue_depth(), 0);
    assert_eq!(m.total_sequences_processed.load(Ordering::Relaxed), 0);
    assert_eq!(m.total_tokens_generated.load(Ordering::Relaxed), 0);
}

#[test]
fn batch_metrics_set_active_count_is_reflected_by_getter() {
    let m = BatchMetrics::new();
    m.set_active_count(4);
    assert_eq!(m.active_count(), 4);
    m.set_active_count(0);
    assert_eq!(m.active_count(), 0);
}

#[test]
fn batch_metrics_set_queue_depth_is_reflected_by_getter() {
    let m = BatchMetrics::new();
    m.set_queue_depth(7);
    assert_eq!(m.queue_depth(), 7);
    m.set_queue_depth(0);
    assert_eq!(m.queue_depth(), 0);
}

#[test]
fn batch_metrics_record_sequence_completed_accumulates() {
    let m = BatchMetrics::new();
    m.record_sequence_completed(10);
    m.record_sequence_completed(25);
    assert_eq!(m.total_sequences_processed.load(Ordering::Relaxed), 2);
    assert_eq!(m.total_tokens_generated.load(Ordering::Relaxed), 35);
}

#[test]
fn batch_metrics_default_equals_new() {
    let a = BatchMetrics::new();
    let b = BatchMetrics::default();
    assert_eq!(a.active_count(), b.active_count());
    assert_eq!(a.queue_depth(), b.queue_depth());
}

// ------------------------------------------------------------------
// can_accept_request logic (tested via BatchMetrics + ServerConfig)
// ------------------------------------------------------------------

/// The admission-control predicate is:
///   queue_depth < max_queue_depth
/// Test this boundary without constructing a full AppState.
#[test]
fn admission_control_accepts_when_queue_below_limit() {
    let m = BatchMetrics::new();
    let max = 4usize;

    m.set_queue_depth(0);
    assert!(m.queue_depth() < max, "empty queue should accept");

    m.set_queue_depth(3);
    assert!(m.queue_depth() < max, "queue below limit should accept");
}

#[test]
fn admission_control_rejects_when_queue_at_or_above_limit() {
    let m = BatchMetrics::new();
    let max = 4usize;

    m.set_queue_depth(4);
    assert!((m.queue_depth() >= max), "queue at limit should reject");

    m.set_queue_depth(10);
    assert!((m.queue_depth() >= max), "queue above limit should reject");
}

// ------------------------------------------------------------------
// Prompt-prefix cache integration
// ------------------------------------------------------------------

/// `PromptCacheConfig` is wired onto `ServerConfig` with a sensible default
/// (enabled, 2 GiB budget, 1024 entries, 1 h TTL, min-prefix 32 tokens).
#[test]
fn server_config_carries_prompt_cache_defaults() {
    use super::super::prompt_cache::PromptCacheConfig;
    let cfg = super::super::ServerConfig::default();
    assert!(cfg.prompt_cache.enabled);
    assert_eq!(
        cfg.prompt_cache.capacity_bytes,
        PromptCacheConfig::DEFAULT_CAPACITY_BYTES
    );
    assert_eq!(
        cfg.prompt_cache.max_entries,
        PromptCacheConfig::DEFAULT_MAX_ENTRIES
    );
    assert_eq!(
        cfg.prompt_cache.ttl.as_secs(),
        PromptCacheConfig::DEFAULT_TTL_SECONDS
    );
    assert_eq!(
        cfg.prompt_cache.min_prefix_tokens,
        PromptCacheConfig::DEFAULT_MIN_PREFIX_TOKENS
    );
}

/// A `PromptCacheStore` behaves as `Send + Sync`, which is what lets us
/// plumb it into both `AppState` and `ModelProvider` via `Arc`.
#[test]
fn prompt_cache_store_is_send_sync_via_arc() {
    use std::sync::Arc;
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Arc<super::super::prompt_cache::PromptCacheStore>>();
}

// ------------------------------------------------------------------
// Prompt-cache BatchMetrics counters
// ------------------------------------------------------------------

#[test]
fn batch_metrics_prompt_cache_hit_increments_counters() {
    let m = BatchMetrics::new();

    m.record_prompt_cache_hit(32);
    m.record_prompt_cache_hit(48);

    assert_eq!(
        m.prompt_cache_hits_total.load(Ordering::Relaxed),
        2,
        "two hits should register as 2"
    );
    assert_eq!(
        m.prompt_cache_prefix_tokens_reused_total
            .load(Ordering::Relaxed),
        80,
        "token counts 32+48=80 should be accumulated"
    );
    assert_eq!(
        m.prompt_cache_misses_total.load(Ordering::Relaxed),
        0,
        "no misses should have been recorded"
    );
}

#[test]
fn batch_metrics_prompt_cache_miss_increments_counter() {
    let m = BatchMetrics::new();

    m.record_prompt_cache_miss();
    m.record_prompt_cache_miss();

    assert_eq!(
        m.prompt_cache_misses_total.load(Ordering::Relaxed),
        2,
        "two misses should register as 2"
    );
    assert_eq!(
        m.prompt_cache_hits_total.load(Ordering::Relaxed),
        0,
        "no hits should have been recorded"
    );
}

#[test]
fn batch_metrics_prompt_cache_evictions_labeled_by_reason() {
    let m = BatchMetrics::new();

    m.record_prompt_cache_eviction_lru();
    m.record_prompt_cache_eviction_lru();
    m.record_prompt_cache_eviction_ttl();
    m.record_prompt_cache_eviction_capacity();

    assert_eq!(
        m.prompt_cache_evictions_lru_total.load(Ordering::Relaxed),
        2,
        "should have 2 LRU evictions"
    );
    assert_eq!(
        m.prompt_cache_evictions_ttl_total.load(Ordering::Relaxed),
        1,
        "should have 1 TTL eviction"
    );
    assert_eq!(
        m.prompt_cache_evictions_capacity_total
            .load(Ordering::Relaxed),
        1,
        "should have 1 capacity eviction"
    );
}

#[test]
fn batch_metrics_prompt_cache_gauges_update() {
    let m = BatchMetrics::new();

    m.update_prompt_cache_gauges(1024 * 1024, 10);

    assert_eq!(
        m.prompt_cache_bytes.load(Ordering::Relaxed),
        1024 * 1024,
        "byte gauge should reflect insert"
    );
    assert_eq!(
        m.prompt_cache_entries.load(Ordering::Relaxed),
        10,
        "entry gauge should reflect insert"
    );

    // Simulate eviction
    m.update_prompt_cache_gauges(512 * 1024, 5);

    assert_eq!(
        m.prompt_cache_bytes.load(Ordering::Relaxed),
        512 * 1024,
        "byte gauge should reflect eviction"
    );
    assert_eq!(
        m.prompt_cache_entries.load(Ordering::Relaxed),
        5,
        "entry gauge should reflect eviction"
    );
}

#[test]
fn batch_metrics_cache_adapter_routes_lookups_to_hit_and_miss_counters() {
    use super::super::prompt_cache::metrics::PromptCacheMetrics;
    use super::BatchMetricsCacheAdapter;
    use std::sync::Arc;

    let m = Arc::new(BatchMetrics::new());
    let adapter = BatchMetricsCacheAdapter::new(m.clone());

    // Hit
    adapter.record_lookup(true, 40);
    // Miss
    adapter.record_lookup(false, 0);
    // Another hit
    adapter.record_lookup(true, 20);

    assert_eq!(m.prompt_cache_hits_total.load(Ordering::Relaxed), 2);
    assert_eq!(
        m.prompt_cache_prefix_tokens_reused_total
            .load(Ordering::Relaxed),
        60
    );
    assert_eq!(m.prompt_cache_misses_total.load(Ordering::Relaxed), 1);
}

#[test]
fn batch_metrics_cache_adapter_routes_evictions() {
    use super::super::prompt_cache::metrics::PromptCacheMetrics;
    use super::BatchMetricsCacheAdapter;
    use std::sync::Arc;

    let m = Arc::new(BatchMetrics::new());
    let adapter = BatchMetricsCacheAdapter::new(m.clone());

    adapter.record_evict_lru(100);
    adapter.record_evict_ttl(200);

    assert_eq!(
        m.prompt_cache_evictions_lru_total.load(Ordering::Relaxed),
        1
    );
    assert_eq!(
        m.prompt_cache_evictions_ttl_total.load(Ordering::Relaxed),
        1
    );
    assert_eq!(
        m.prompt_cache_evictions_capacity_total
            .load(Ordering::Relaxed),
        0
    );
}
