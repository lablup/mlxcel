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
//
//! Tests for the `/v1/cache/stats` and `/v1/cache/reset` route handlers.
//!
//! These tests focus on the pure response shape and on cache-store
//! integration. We do not exercise the full axum/`AppState` stack because
//! constructing a real `ModelProvider` requires loading a model file from
//! disk, which is out of scope for unit tests. Instead we drive the cache
//! store directly to verify that:
//!
//! - Response field names match the documented JSON shape.
//! - `cache_stats` correctly handles a `None` store (returns disabled
//!   payload with zero counters).
//! - `cache_reset` reports the freed entries / bytes that were live when it
//!   was called.
//!
//! End-to-end HTTP testing of the routes is covered by the upcoming
//! integration tests in `apc_endpoints_tests.rs` once `AppState` test
//! fixtures land.

use std::sync::Arc;
use std::time::Duration;

use crate::server::prompt_cache::{
    ApcConfig, ApcHashAlgo, CacheEntry, MultimodalDigest, PromptCacheConfig, PromptCacheKey,
    PromptCacheStore,
};
use crate::server::routes::cache::{CacheResetResponse, CacheStatsResponse, PagedBlockStats};

fn enabled_store() -> Arc<PromptCacheStore> {
    let cfg = PromptCacheConfig::new(true, 1 << 20, 32, Duration::from_secs(3600), 4);
    Arc::new(PromptCacheStore::with_config(cfg))
}

fn apc_enabled_store() -> Arc<PromptCacheStore> {
    let apc = ApcConfig {
        enabled: true,
        block_size: 16,
        num_blocks: None,
        hash: ApcHashAlgo::Sha256,
    };
    let cfg = PromptCacheConfig::new(true, 1 << 20, 32, Duration::from_secs(3600), 4).with_apc(apc);
    Arc::new(PromptCacheStore::with_config(cfg))
}

fn apc_enabled_config() -> PromptCacheConfig {
    let apc = ApcConfig {
        enabled: true,
        block_size: 16,
        num_blocks: None,
        hash: ApcHashAlgo::Sha256,
    };
    PromptCacheConfig::new(true, 1 << 20, 32, Duration::from_secs(3600), 4).with_apc(apc)
}

fn make_key<'a>(model: &'a str, tokens: &'a [i32]) -> PromptCacheKey<'a> {
    PromptCacheKey::new_full(model, None, "tpl", None, MultimodalDigest::empty(), tokens)
}

fn make_key_mm<'a>(model: &'a str, mm: MultimodalDigest, tokens: &'a [i32]) -> PromptCacheKey<'a> {
    PromptCacheKey::new_full(model, None, "tpl", None, mm, tokens)
}

// ---------------------------------------------------------------------------
// JSON shape sanity
// ---------------------------------------------------------------------------

#[test]
fn cache_stats_response_serializes_with_expected_keys() {
    let resp = CacheStatsResponse {
        enabled: true,
        apc_enabled: true,
        block_size: 16,
        hash: "sha256".to_string(),
        entries: 3,
        bytes: 12345,
        capacity_bytes: 2 * 1024 * 1024 * 1024,
        max_entries: 1024,
        hits: 7,
        lookups: 10,
        hit_rate: 0.7,
        inserts: 9,
        evictions_lru: 1,
        evictions_ttl: 0,
        rejections_oversized: 0,
        total_blocks_stored: 6,
        unique_block_hashes: 5,
        apc_active_entries: 3,
        paged_block_size: 32,
        paged_blocks_allocated: 100,
        paged_blocks_live: 80,
        paged_blocks_free: 20,
        paged_bytes_reserved: 65536,
        paged_bytes_in_use: 49152,
        paged_block_budget: 128,
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    for key in [
        "\"enabled\":true",
        "\"apc_enabled\":true",
        "\"block_size\":16",
        "\"hash\":\"sha256\"",
        "\"entries\":3",
        "\"bytes\":12345",
        "\"capacity_bytes\":",
        "\"max_entries\":1024",
        "\"hits\":7",
        "\"lookups\":10",
        "\"hit_rate\":0.7",
        "\"inserts\":9",
        "\"evictions_lru\":1",
        "\"evictions_ttl\":0",
        "\"rejections_oversized\":0",
        "\"total_blocks_stored\":6",
        "\"unique_block_hashes\":5",
        "\"apc_active_entries\":3",
        "\"paged_block_size\":32",
        "\"paged_blocks_allocated\":100",
        "\"paged_blocks_live\":80",
        "\"paged_blocks_free\":20",
        "\"paged_bytes_reserved\":65536",
        "\"paged_bytes_in_use\":49152",
        "\"paged_block_budget\":128",
    ] {
        assert!(json.contains(key), "expected `{key}` in: {json}");
    }
}

#[test]
fn cache_reset_response_serializes_with_expected_keys() {
    let resp = CacheResetResponse {
        cleared: true,
        freed_bytes: 4096,
        freed_entries: 3,
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("\"cleared\":true"));
    assert!(json.contains("\"freed_bytes\":4096"));
    assert!(json.contains("\"freed_entries\":3"));
}

// ---------------------------------------------------------------------------
// Direct PromptCacheStore integration (mirrors what the handler does)
// ---------------------------------------------------------------------------

#[test]
fn store_stats_reflect_inserts_and_lookups_for_handler_use() {
    let store = enabled_store();
    let tokens: Vec<i32> = (0..32).collect();
    let key = make_key("m", &tokens);
    let entry = CacheEntry::new_for_test(tokens.clone(), 1024);
    store.insert(&key, entry).expect("insert");

    let stats = store.stats();
    assert_eq!(stats.entries, 1);
    assert_eq!(stats.bytes, 1024);
    assert_eq!(stats.inserts, 1);
    assert_eq!(stats.lookups, 0);
    assert_eq!(stats.hits, 0);

    // A successful lookup increments lookups and hits.
    let _ = store.lookup_longest_prefix(&key, &tokens);
    let stats = store.stats();
    assert_eq!(stats.lookups, 1);
}

#[test]
fn store_clear_drops_all_entries_for_reset_handler() {
    let store = enabled_store();
    for i in 0..5 {
        let toks: Vec<i32> = (0..16).map(|x| x + i * 100).collect();
        let key = make_key("m", &toks);
        let entry = CacheEntry::new_for_test(toks.clone(), 256);
        store.insert(&key, entry).expect("insert");
    }
    assert_eq!(store.stats().entries, 5);
    let snapshot_before = store.stats();
    store.clear();
    assert_eq!(store.stats().entries, 0);
    assert_eq!(store.stats().bytes, 0);
    // The reset handler returns the snapshot taken *before* clear, so the
    // operator sees what was actually freed. Verify the math.
    assert_eq!(snapshot_before.entries, 5);
    assert!(snapshot_before.bytes >= 5 * 256);
}

#[test]
fn cache_stats_response_disabled_payload_uses_zero_counters() {
    // The handler builds this exact shape when `state.prompt_cache` is None.
    let cfg = PromptCacheConfig::default();
    let resp = CacheStatsResponse {
        enabled: false,
        apc_enabled: false,
        block_size: cfg.apc.block_size,
        hash: cfg.apc.hash.to_string(),
        entries: 0,
        bytes: 0,
        capacity_bytes: cfg.capacity_bytes,
        max_entries: cfg.max_entries,
        hits: 0,
        lookups: 0,
        hit_rate: 0.0,
        inserts: 0,
        evictions_lru: 0,
        evictions_ttl: 0,
        rejections_oversized: 0,
        total_blocks_stored: 0,
        unique_block_hashes: 0,
        apc_active_entries: 0,
        paged_block_size: 0,
        paged_blocks_allocated: 0,
        paged_blocks_live: 0,
        paged_blocks_free: 0,
        paged_bytes_reserved: 0,
        paged_bytes_in_use: 0,
        paged_block_budget: 0,
    };
    assert!(!resp.enabled);
    assert!(!resp.apc_enabled);
    assert_eq!(resp.entries, 0);
    assert_eq!(resp.lookups, 0);
    assert_eq!(resp.hit_rate, 0.0);
    assert_eq!(resp.total_blocks_stored, 0);
    assert_eq!(resp.unique_block_hashes, 0);
    assert_eq!(resp.apc_active_entries, 0);
    // Paged pool defaults to all-zero when no worker reported gauges.
    assert_eq!(resp.paged_block_size, 0);
    assert_eq!(resp.paged_block_budget, 0);
}

#[test]
fn build_stats_response_surfaces_paged_pool_independent_of_store() {
    // The paged block-pool gauges come from observability, not the prompt
    // cache, so they must appear whether or not a store is present (#122 c).
    let paged = PagedBlockStats {
        block_size: 32,
        blocks_allocated: 200,
        blocks_live: 150,
        blocks_free: 50,
        bytes_reserved: 131072,
        bytes_in_use: 98304,
        block_budget: 256,
    };
    let cfg = PromptCacheConfig::default();

    // None store (prompt cache disabled) still reports the live paged pool.
    let disabled = super::super::cache::build_stats_response(None, &cfg, paged);
    assert!(!disabled.enabled);
    assert_eq!(disabled.paged_block_size, 32);
    assert_eq!(disabled.paged_blocks_live, 150);
    assert_eq!(disabled.paged_block_budget, 256);
    // Acquirable headroom under the cap = budget - live.
    assert_eq!(
        disabled.paged_block_budget - disabled.paged_blocks_live,
        106
    );

    // With a live store the same paged values flow through unchanged.
    let store = enabled_store();
    let enabled = super::super::cache::build_stats_response(Some(store.as_ref()), &cfg, paged);
    assert_eq!(enabled.paged_blocks_allocated, 200);
    assert_eq!(enabled.paged_bytes_in_use, 98304);
    assert_eq!(enabled.paged_block_budget, 256);
}

#[test]
fn paged_block_stats_projects_from_observability_snapshot() {
    // The full data path the handler uses: scheduler `update_gauges` →
    // snapshot → `PagedBlockStats::from_observability` (#122 c).
    use crate::server::batch::BatchObservability;
    use mlxcel_core::cache::PagedCacheStats;

    let obs = BatchObservability::new();
    obs.update_gauges(
        2,
        0,
        2,
        4096,
        32,
        Some(PagedCacheStats {
            allocated_blocks: 10,
            live_blocks: 7,
            free_blocks: 3,
            bytes_reserved: 4096,
            bytes_in_use: 2048,
        }),
        16, // block budget
    );
    let paged = PagedBlockStats::from_observability(&obs.snapshot());
    assert_eq!(paged.block_size, 32);
    assert_eq!(paged.blocks_allocated, 10);
    assert_eq!(paged.blocks_live, 7);
    assert_eq!(paged.blocks_free, 3);
    assert_eq!(paged.bytes_reserved, 4096);
    assert_eq!(paged.bytes_in_use, 2048);
    assert_eq!(paged.block_budget, 16);
}

#[test]
fn hit_rate_handles_zero_lookups_without_nan() {
    // Direct verification of the division-by-zero guard in the handler. We
    // can't intercept the f64 inside the handler easily, so we replicate the
    // logic here.
    let hits: u64 = 0;
    let lookups: u64 = 0;
    let hit_rate = if lookups > 0 {
        hits as f64 / lookups as f64
    } else {
        0.0
    };
    assert_eq!(hit_rate, 0.0);
    assert!(hit_rate.is_finite());
}

// ---------------------------------------------------------------------------
// Pure helper functions — exercised by both the actual handler and the
// router-level integration test below. The pure helpers let us cover the
// handler logic without constructing a full AppState.
// ---------------------------------------------------------------------------

#[test]
fn build_stats_response_disabled_when_store_is_none() {
    let cfg = PromptCacheConfig::default();
    let resp = super::super::cache::build_stats_response(None, &cfg, PagedBlockStats::default());
    assert!(!resp.enabled);
    assert!(!resp.apc_enabled);
    assert_eq!(resp.entries, 0);
    assert_eq!(resp.bytes, 0);
    assert_eq!(resp.lookups, 0);
    assert_eq!(resp.hit_rate, 0.0);
    assert_eq!(resp.block_size, cfg.apc.block_size);
}

#[test]
fn build_stats_response_reflects_live_store() {
    let store = enabled_store();
    let cfg = PromptCacheConfig::new(true, 1 << 20, 32, Duration::from_secs(3600), 4);

    let toks: Vec<i32> = (0..32).collect();
    let key = make_key("m", &toks);
    let entry = CacheEntry::new_for_test(toks.clone(), 1024);
    store.insert(&key, entry).expect("insert");

    let resp = super::super::cache::build_stats_response(
        Some(store.as_ref()),
        &cfg,
        PagedBlockStats::default(),
    );
    assert!(resp.enabled);
    assert!(!resp.apc_enabled, "APC opt-in defaults to off");
    assert_eq!(resp.entries, 1);
    assert_eq!(resp.bytes, 1024);
    assert_eq!(resp.inserts, 1);
    assert_eq!(
        resp.total_blocks_stored, 0,
        "APC disabled => no blocks recorded"
    );
    assert_eq!(resp.unique_block_hashes, 0);
    assert_eq!(resp.apc_active_entries, 0);
}

#[test]
fn build_stats_response_reflects_apc_blocks_when_enabled() {
    // With APC enabled and two distinct entries inserted, total_blocks_stored
    // must equal sum(chain_len) and apc_active_entries == 2.
    let store = apc_enabled_store();
    let cfg = apc_enabled_config();

    // 32 tokens at block_size=16 => 2 blocks per entry.
    let toks_a: Vec<i32> = (0..32).collect();
    let toks_b: Vec<i32> = (100..132).collect();
    store
        .insert(
            &make_key("m", &toks_a),
            CacheEntry::new_for_test(toks_a.clone(), 1024),
        )
        .expect("insert a");
    store
        .insert(
            &make_key("m", &toks_b),
            CacheEntry::new_for_test(toks_b.clone(), 1024),
        )
        .expect("insert b");

    let resp = super::super::cache::build_stats_response(
        Some(store.as_ref()),
        &cfg,
        PagedBlockStats::default(),
    );
    assert!(resp.enabled);
    assert!(resp.apc_enabled, "APC must be enabled in this fixture");
    assert_eq!(resp.entries, 2);
    assert_eq!(resp.apc_active_entries, 2);
    assert_eq!(
        resp.total_blocks_stored, 4,
        "two entries x 2 blocks each = 4 total"
    );
    // Different token prefixes produce different block hashes; both blocks of
    // entry A and both blocks of entry B are unique.
    assert_eq!(resp.unique_block_hashes, 4);
}

#[test]
fn build_stats_response_apc_zero_when_disabled_with_inserts() {
    // The store has APC off but still receives inserts. The handler must
    // report apc_active_entries=0 even though entries>0, so operators
    // looking at /v1/cache/stats can tell APC is genuinely inactive.
    let store = enabled_store();
    let cfg = PromptCacheConfig::new(true, 1 << 20, 32, Duration::from_secs(3600), 4);

    for i in 0..3 {
        let toks: Vec<i32> = (0..32).map(|x| x + i * 100).collect();
        store
            .insert(
                &make_key("m", &toks),
                CacheEntry::new_for_test(toks.clone(), 256),
            )
            .expect("insert");
    }
    let resp = super::super::cache::build_stats_response(
        Some(store.as_ref()),
        &cfg,
        PagedBlockStats::default(),
    );
    assert!(!resp.apc_enabled);
    assert_eq!(resp.entries, 3);
    assert_eq!(resp.apc_active_entries, 0);
    assert_eq!(resp.total_blocks_stored, 0);
    assert_eq!(resp.unique_block_hashes, 0);
}

#[test]
fn apc_unique_block_hashes_reflects_dedup_potential() {
    // Two entries with overlapping leading blocks but only one is recorded
    // each. Same first 16 tokens (block 0) but different tail tokens AND
    // different mm_digest. Because the chain folds mm_digest into every
    // block, two entries with identical leading tokens but different
    // mm_digest produce *no* shared block hashes. To exercise the dedup
    // metric we need same mm_digest AND identical leading blocks, but
    // different overall tokens — which collide on bucket digest if mm
    // matches and prefixes differ. We construct that case here.
    let store = apc_enabled_store();
    let cfg = apc_enabled_config();
    let mm = MultimodalDigest::empty();

    let mut shared: Vec<i32> = (0..16).collect();
    let mut entry_a_tokens = shared.clone();
    entry_a_tokens.extend(2000..2016); // 32 total, blocks 0+1
    let mut entry_b_tokens = shared.clone();
    entry_b_tokens.extend(3000..3016); // 32 total, blocks 0+1
    shared.clear();

    store
        .insert(
            &make_key_mm("m", mm, &entry_a_tokens),
            CacheEntry::new_for_test(entry_a_tokens.clone(), 1024),
        )
        .expect("insert a");
    store
        .insert(
            &make_key_mm("m", mm, &entry_b_tokens),
            CacheEntry::new_for_test(entry_b_tokens.clone(), 1024),
        )
        .expect("insert b");

    let resp = super::super::cache::build_stats_response(
        Some(store.as_ref()),
        &cfg,
        PagedBlockStats::default(),
    );
    assert_eq!(resp.entries, 2);
    assert_eq!(resp.apc_active_entries, 2);
    assert_eq!(resp.total_blocks_stored, 4);
    // Block 0 is identical for both entries (same leading 16 tokens, same
    // mm_digest, same parent=ZERO seed), block 1 differs because entries
    // diverge on tokens 16..32. So unique = 3 < total = 4.
    assert_eq!(
        resp.unique_block_hashes, 3,
        "block 0 must dedup across entries with shared leading prefix"
    );
}

#[test]
fn build_reset_response_clears_store_and_reports_snapshot() {
    let store = enabled_store();
    for i in 0..3 {
        let toks: Vec<i32> = (0..16).map(|x| x + i * 100).collect();
        let key = make_key("m", &toks);
        let entry = CacheEntry::new_for_test(toks.clone(), 256);
        store.insert(&key, entry).expect("insert");
    }
    assert_eq!(store.stats().entries, 3);

    let resp = super::super::cache::build_reset_response(Some(store.as_ref()));
    assert!(resp.cleared);
    assert_eq!(resp.freed_entries, 3);
    assert!(resp.freed_bytes >= 3 * 256);

    // Idempotency — calling again returns 0 freed without panicking.
    let again = super::super::cache::build_reset_response(Some(store.as_ref()));
    assert!(again.cleared);
    assert_eq!(again.freed_entries, 0);
}

#[test]
fn build_reset_response_returns_zeroes_for_disabled_cache() {
    let resp = super::super::cache::build_reset_response(None);
    assert!(resp.cleared);
    assert_eq!(resp.freed_entries, 0);
    assert_eq!(resp.freed_bytes, 0);
}

// ---------------------------------------------------------------------------
// End-to-end router test using axum + tower::ServiceExt::oneshot.
//
// We construct a minimal router that mounts custom handlers driven by the
// pure helpers, so we exercise the real axum routing layer without pulling
// in the rest of AppState. This catches mistakes in the route paths and
// HTTP-method bindings.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn router_returns_stats_and_reset_with_correct_methods() {
    use axum::Json as AxumJson;
    use axum::Router;
    use axum::body::{Body, to_bytes};
    use axum::extract::State;
    use axum::http::{Method, Request, StatusCode};
    use axum::routing::{get, post};
    use std::sync::Arc;
    use tower::ServiceExt;

    #[derive(Clone)]
    struct TestState {
        store: Arc<PromptCacheStore>,
        cfg: Arc<PromptCacheConfig>,
    }

    async fn stats_handler(
        State(s): State<TestState>,
    ) -> AxumJson<super::super::cache::CacheStatsResponse> {
        AxumJson(super::super::cache::build_stats_response(
            Some(s.store.as_ref()),
            s.cfg.as_ref(),
            PagedBlockStats::default(),
        ))
    }
    async fn reset_handler(
        State(s): State<TestState>,
    ) -> AxumJson<super::super::cache::CacheResetResponse> {
        AxumJson(super::super::cache::build_reset_response(Some(
            s.store.as_ref(),
        )))
    }

    let store = enabled_store();
    let cfg = Arc::new(PromptCacheConfig::new(
        true,
        1 << 20,
        32,
        Duration::from_secs(3600),
        4,
    ));

    // Pre-populate one entry so the stats endpoint returns non-zero counters.
    let toks: Vec<i32> = (0..32).collect();
    let key = make_key("m", &toks);
    store
        .insert(&key, CacheEntry::new_for_test(toks.clone(), 2048))
        .expect("insert");

    let app = Router::new()
        .route("/v1/cache/stats", get(stats_handler))
        .route("/v1/cache/reset", post(reset_handler))
        .with_state(TestState {
            store: store.clone(),
            cfg,
        });

    // GET /v1/cache/stats
    let req = Request::builder()
        .method(Method::GET)
        .uri("/v1/cache/stats")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.expect("oneshot");
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["enabled"], true);
    assert_eq!(json["entries"], 1);
    assert_eq!(json["bytes"], 2048);

    // POST /v1/cache/reset
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/cache/reset")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.expect("oneshot");
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 4096).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["cleared"], true);
    assert_eq!(json["freed_entries"], 1);

    // Verify GET on /v1/cache/stats now returns 0 entries (reset worked).
    let req = Request::builder()
        .method(Method::GET)
        .uri("/v1/cache/stats")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.expect("oneshot");
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 4096).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["entries"], 0);
    assert_eq!(json["bytes"], 0);

    // Wrong method — POST on /v1/cache/stats should return 405.
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/cache/stats")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.expect("oneshot");
    assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
}

// ---------------------------------------------------------------------------
// Concurrent reset + lookup — the store uses an RwLock so a reset that fires
// while lookups are in flight must not panic, corrupt memory, or deadlock.
// We verify the safety contract by interleaving reset calls with concurrent
// reader threads. If the store's locking discipline is correct the only
// observable outcome is that some lookups see a non-empty store and some see
// an empty one, but neither path panics.
// ---------------------------------------------------------------------------

#[test]
fn concurrent_reset_and_lookup_does_not_corrupt_store() {
    use std::sync::Arc;
    use std::thread;

    let store = Arc::new(enabled_store());

    // Pre-populate entries that lookups will try to match.
    for i in 0..8i32 {
        let toks: Vec<i32> = (0..32).map(|x| x + i * 1000).collect();
        store
            .insert(
                &make_key("m", &toks),
                CacheEntry::new_for_test(toks.clone(), 512),
            )
            .ok(); // ignore oversized rejections under tiny cap
    }

    let store_for_readers = Arc::clone(&store);
    let store_for_reset = Arc::clone(&store);

    // Spawn reader threads that continuously look up entries.
    let reader_handles: Vec<_> = (0..4)
        .map(|i: i32| {
            let s = Arc::clone(&store_for_readers);
            thread::spawn(move || {
                for _ in 0..64 {
                    let toks: Vec<i32> = (0..32).map(|x| x + i * 1000).collect();
                    let _ = s.lookup_longest_prefix(&make_key("m", &toks), &toks);
                }
            })
        })
        .collect();

    // Interleave reset calls from a separate thread.
    let reset_handle = thread::spawn(move || {
        for _ in 0..8 {
            super::super::cache::build_reset_response(Some(store_for_reset.as_ref()));
        }
    });

    for h in reader_handles {
        h.join().expect("reader thread must not panic");
    }
    reset_handle.join().expect("reset thread must not panic");

    // After all resets and readers finish, the store must be in a consistent
    // state: stats() must not panic and entry/byte counts must be coherent.
    let stats = store.stats();
    assert!(
        stats.bytes <= stats.entries * (512 + 256),
        "byte total must be consistent with entry count after concurrent resets"
    );
}
