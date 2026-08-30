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

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use mlxcel_core::generate::ModelStateSnapshot;

use super::super::entry::{CacheEntry, ModelSnapshotEntry, SnapshotOrigin};
use super::super::key::{MultimodalDigest, PromptCacheKey};
use super::super::metrics::AtomicPromptCacheMetrics;
use super::super::policy::PromptCacheConfig;
use super::{InsertError, PromptCacheStore};

fn cfg(capacity_bytes: usize, max_entries: usize, min_prefix_tokens: usize) -> PromptCacheConfig {
    PromptCacheConfig::new(
        true,
        capacity_bytes,
        max_entries,
        Duration::from_secs(3600),
        min_prefix_tokens,
    )
}

fn tokens(base: i32, n: usize) -> Vec<i32> {
    (0..n as i32).map(|i| i + base).collect()
}

fn key_for<'a>(model: &'a str, tokens: &'a [i32]) -> PromptCacheKey<'a> {
    PromptCacheKey::new_full(model, None, "tpl", None, MultimodalDigest::empty(), tokens)
}

fn key_for_session<'a>(
    model: &'a str,
    session_key: Option<&'a str>,
    tokens: &'a [i32],
) -> PromptCacheKey<'a> {
    PromptCacheKey::new_full(
        model,
        None,
        "tpl",
        session_key,
        MultimodalDigest::empty(),
        tokens,
    )
}

fn key_for_mm<'a>(
    model: &'a str,
    mm_digest: MultimodalDigest,
    tokens: &'a [i32],
) -> PromptCacheKey<'a> {
    PromptCacheKey::new_full(model, None, "tpl", None, mm_digest, tokens)
}

fn snapshot_entry_for_test(tokens: Vec<i32>, family: &str) -> ModelSnapshotEntry {
    let snapshot = ModelStateSnapshot::new(family, tokens.len());
    ModelSnapshotEntry::new(tokens, snapshot)
}

#[test]
fn insert_then_lookup_returns_entry_and_matched_len() {
    let store = PromptCacheStore::with_config(cfg(1 << 20, 64, 4));
    let toks = tokens(0, 16);
    store
        .insert(
            &key_for("m", &toks),
            CacheEntry::new_for_test(toks.clone(), 1024),
        )
        .expect("insert succeeds");

    let (entry, matched) = store
        .lookup_longest_prefix(&key_for("m", &toks), &toks)
        .expect("lookup returns entry");
    assert_eq!(matched, toks.len());
    assert_eq!(entry.tokens, toks);
}

#[test]
fn lookup_misses_when_prefix_is_shorter_than_min_prefix() {
    // Store an entry with 16 tokens but set `min_prefix_tokens = 32`, so
    // matches shorter than 32 should not be returned.
    let store = PromptCacheStore::with_config(cfg(1 << 20, 64, 32));
    let stored = tokens(0, 16);
    let err = store
        .insert(
            &key_for("m", &stored),
            CacheEntry::new_for_test(stored.clone(), 1024),
        )
        .unwrap_err();
    match err {
        InsertError::PrefixTooShort { got, min_required } => {
            assert_eq!(got, 16);
            assert_eq!(min_required, 32);
        }
        other => panic!("expected PrefixTooShort, got {other:?}"),
    }
}

#[test]
fn insert_rejects_when_single_entry_exceeds_capacity() {
    let store = PromptCacheStore::with_config(cfg(1024, 64, 4));
    let toks = tokens(0, 16);
    let err = store
        .insert(
            &key_for("m", &toks),
            CacheEntry::new_for_test(toks.clone(), 2048),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        InsertError::OversizedEntry {
            entry_bytes: 2048,
            ..
        }
    ));
    assert_eq!(store.len(), 0);
    assert_eq!(store.bytes(), 0);
}

#[test]
fn lru_evicts_oldest_under_entry_cap() {
    let store = PromptCacheStore::with_config(cfg(1 << 20, 2, 4));
    let a = tokens(0, 16);
    let b = tokens(100, 16);
    let c = tokens(200, 16);

    store
        .insert(&key_for("m", &a), CacheEntry::new_for_test(a.clone(), 1024))
        .unwrap();
    thread::sleep(Duration::from_millis(5));
    store
        .insert(&key_for("m", &b), CacheEntry::new_for_test(b.clone(), 1024))
        .unwrap();
    thread::sleep(Duration::from_millis(5));

    // Touch `a` so it becomes the most-recent, leaving `b` as the LRU victim.
    let _ = store.lookup_longest_prefix(&key_for("m", &a), &a);
    thread::sleep(Duration::from_millis(5));

    store
        .insert(&key_for("m", &c), CacheEntry::new_for_test(c.clone(), 1024))
        .unwrap();

    assert_eq!(store.len(), 2);
    assert!(
        store.lookup_longest_prefix(&key_for("m", &a), &a).is_some(),
        "a is most-recent and should survive"
    );
    assert!(
        store.lookup_longest_prefix(&key_for("m", &c), &c).is_some(),
        "c was just inserted and should be present"
    );
    assert!(
        store.lookup_longest_prefix(&key_for("m", &b), &b).is_none(),
        "b should have been LRU-evicted"
    );
}

#[test]
fn lru_evicts_under_byte_cap() {
    // 3 entries each 512 bytes but byte cap is 1024 → at most 2 can live.
    let store = PromptCacheStore::with_config(cfg(1024, 64, 4));
    let a = tokens(0, 16);
    let b = tokens(100, 16);
    let c = tokens(200, 16);

    store
        .insert(&key_for("m", &a), CacheEntry::new_for_test(a.clone(), 512))
        .unwrap();
    thread::sleep(Duration::from_millis(5));
    store
        .insert(&key_for("m", &b), CacheEntry::new_for_test(b.clone(), 512))
        .unwrap();
    thread::sleep(Duration::from_millis(5));
    store
        .insert(&key_for("m", &c), CacheEntry::new_for_test(c.clone(), 512))
        .unwrap();

    assert!(store.bytes() <= 1024);
    assert_eq!(store.len(), 2);
}

#[test]
fn ttl_expiry_drops_idle_entries_on_lookup() {
    let cfg = PromptCacheConfig::new(true, 1 << 20, 64, Duration::from_millis(25), 4);
    let store = PromptCacheStore::with_config(cfg);
    let toks = tokens(0, 16);
    store
        .insert(
            &key_for("m", &toks),
            CacheEntry::new_for_test(toks.clone(), 1024),
        )
        .unwrap();

    // Within TTL: present.
    assert!(
        store
            .lookup_longest_prefix(&key_for("m", &toks), &toks)
            .is_some()
    );

    thread::sleep(Duration::from_millis(60));
    // After TTL: TTL sweep runs on lookup.
    assert!(
        store
            .lookup_longest_prefix(&key_for("m", &toks), &toks)
            .is_none()
    );
    assert_eq!(store.len(), 0);
    let stats = store.stats();
    assert!(stats.evictions_ttl >= 1);
}

#[test]
fn concurrent_access_many_threads() {
    let store = Arc::new(PromptCacheStore::with_config(cfg(8 * 1024 * 1024, 512, 4)));
    let threads: i32 = 16;
    let per_thread: i32 = 32;
    let total_inserts = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..threads)
        .map(|t: i32| {
            let store = Arc::clone(&store);
            let counter = Arc::clone(&total_inserts);
            thread::spawn(move || {
                for i in 0..per_thread {
                    let base = (t * per_thread + i) * 1000;
                    let toks = tokens(base, 16);
                    if store
                        .insert(
                            &key_for("m", &toks),
                            CacheEntry::new_for_test(toks.clone(), 2048),
                        )
                        .is_ok()
                    {
                        counter.fetch_add(1, Ordering::Relaxed);
                    }
                    let _ = store.lookup_longest_prefix(&key_for("m", &toks), &toks);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // At least many inserts succeeded; caps are respected.
    let inserted = total_inserts.load(Ordering::Relaxed);
    assert!(inserted > 0);
    assert!(store.len() <= 512);
    assert!(store.bytes() <= 8 * 1024 * 1024);
}

#[test]
fn fuzz_stress_no_unbounded_growth() {
    // Hammer the store with inserts from a deterministic rotating seed and
    // confirm caps hold.
    let cap_bytes = 64 * 1024;
    let max_entries = 32;
    let store = PromptCacheStore::with_config(cfg(cap_bytes, max_entries, 4));
    for i in 0..5_000i32 {
        let base = (i * 7) % 10_000;
        let toks = tokens(base, 16);
        let size = 1 + ((i as usize) % 2048);
        let _ = store.insert(
            &key_for("m", &toks),
            CacheEntry::new_for_test(toks.clone(), size),
        );
        if i % 50 == 0 {
            store.evict_if_needed();
        }
    }
    assert!(store.len() <= max_entries);
    assert!(store.bytes() <= cap_bytes);
}

#[test]
fn different_buckets_do_not_cross_contaminate() {
    let store = PromptCacheStore::with_config(cfg(1 << 20, 64, 4));
    let toks = tokens(0, 16);
    store
        .insert(
            &key_for("model-a", &toks),
            CacheEntry::new_for_test(toks.clone(), 1024),
        )
        .unwrap();

    // A bucket-mismatching lookup must return None even though the tokens
    // and digest shape are otherwise identical.
    let result = store.lookup_longest_prefix(&key_for("model-b", &toks), &toks);
    assert!(result.is_none());
    // But the original bucket still hits.
    let result = store.lookup_longest_prefix(&key_for("model-a", &toks), &toks);
    assert!(result.is_some());
}

#[test]
fn longest_prefix_across_multiple_entries_same_bucket() {
    // Two entries share the same bucket (same model) but differ in token
    // prefix length. The longer overlapping prefix must win.
    let store = PromptCacheStore::with_config(cfg(1 << 20, 64, 4));
    let short = tokens(0, 8);
    let longer = tokens(0, 24);

    store
        .insert(
            &key_for("m", &short),
            CacheEntry::new_for_test(short.clone(), 512),
        )
        .unwrap();
    store
        .insert(
            &key_for("m", &longer),
            CacheEntry::new_for_test(longer.clone(), 512),
        )
        .unwrap();

    // Incoming request has 24 tokens matching `longer` exactly.
    let incoming = longer.clone();
    let (entry, matched) = store
        .lookup_longest_prefix(&key_for("m", &incoming), &incoming)
        .expect("lookup hit");
    assert_eq!(matched, 24);
    assert_eq!(entry.tokens.len(), 24);

    // Incoming request has 10 tokens. The 24-token entry shares those 10
    // tokens, but adopting it would import KV state for tokens that are not
    // present in this request. It must therefore be ignored; the shorter
    // fully-contained 8-token entry is the only safe hit.
    let incoming = tokens(0, 10);
    let (entry, matched) = store
        .lookup_longest_prefix(&key_for("m", &incoming), &incoming)
        .expect("lookup hit");
    assert_eq!(matched, 8);
    assert_eq!(entry.tokens.len(), 8);

    // Now exercise the "divergence past a certain index" case explicitly:
    // an incoming request whose tokens diverge at index 8 from the longer
    // entry but still fully match the shorter entry for 8 tokens.
    let mut diverging = tokens(0, 8);
    diverging.extend([999, 999, 999]);
    let (_, matched) = store
        .lookup_longest_prefix(&key_for("m", &diverging), &diverging)
        .expect("lookup hit");
    assert_eq!(matched, 8);
}

#[test]
fn lookup_ignores_stored_prefix_longer_than_request_when_no_safe_entry_exists() {
    let store = PromptCacheStore::with_config(cfg(1 << 20, 64, 4));
    let stored = tokens(0, 24);
    store
        .insert(
            &key_for("m", &stored),
            CacheEntry::new_for_test(stored.clone(), 512),
        )
        .unwrap();

    let incoming = tokens(0, 10);
    assert!(
        store
            .lookup_longest_prefix(&key_for("m", &incoming), &incoming)
            .is_none(),
        "a longer stored KV prefix must not be adopted for a shorter request"
    );
}

#[test]
fn multimodal_digest_isolates_prefix_lookup_buckets() {
    let store = PromptCacheStore::with_config(cfg(1 << 20, 64, 4));
    let toks = tokens(0, 16);
    let image_a = MultimodalDigest([1; 32]);
    let image_b = MultimodalDigest([2; 32]);

    store
        .insert(
            &key_for_mm("m", image_a, &toks),
            CacheEntry::new_for_test(toks.clone(), 1024),
        )
        .unwrap();

    assert!(
        store
            .lookup_longest_prefix(&key_for_mm("m", image_b, &toks), &toks)
            .is_none(),
        "same text with different multimodal digest must not share KV entries"
    );
    assert!(
        store
            .lookup_longest_prefix(&key_for_mm("m", image_a, &toks), &toks)
            .is_some(),
        "same text with same multimodal digest should still hit"
    );
}

#[test]
fn consumed_entries_are_swept_from_accounting_on_next_store_touch() {
    let store = PromptCacheStore::with_config(cfg(1 << 20, 64, 4));
    let toks = tokens(0, 16);
    store
        .insert(
            &key_for("m", &toks),
            CacheEntry::new_for_test(toks.clone(), 4096),
        )
        .unwrap();

    let (entry, _) = store
        .lookup_longest_prefix(&key_for("m", &toks), &toks)
        .expect("first lookup hits");
    assert!(entry.take_detached().is_some());
    assert_eq!(store.len(), 1);
    assert_eq!(store.bytes(), 4096);

    assert!(
        store
            .lookup_longest_prefix(&key_for("m", &toks), &toks)
            .is_none(),
        "next lookup should sweep the drained shell before matching"
    );
    assert_eq!(store.len(), 0);
    assert_eq!(store.bytes(), 0);
}

#[test]
fn disabled_config_rejects_inserts_and_lookups() {
    let store = PromptCacheStore::with_config(PromptCacheConfig::disabled());
    let toks = tokens(0, 64);
    assert!(!store.is_enabled());
    let err = store
        .insert(
            &key_for("m", &toks),
            CacheEntry::new_for_test(toks.clone(), 1024),
        )
        .unwrap_err();
    assert_eq!(err, InsertError::Disabled);
    assert!(
        store
            .lookup_longest_prefix(&key_for("m", &toks), &toks)
            .is_none()
    );
}

#[test]
fn metrics_hooks_fire_on_insert_and_lookup() {
    let metrics = AtomicPromptCacheMetrics::shared();
    let store = PromptCacheStore::with_metrics(cfg(1 << 20, 64, 4), metrics.clone());
    let toks = tokens(0, 16);
    store
        .insert(
            &key_for("m", &toks),
            CacheEntry::new_for_test(toks.clone(), 1024),
        )
        .unwrap();
    let _ = store.lookup_longest_prefix(&key_for("m", &toks), &toks);
    let miss = tokens(999, 16);
    let _ = store.lookup_longest_prefix(&key_for("m", &miss), &miss);

    assert_eq!(metrics.inserts.load(Ordering::Relaxed), 1);
    assert_eq!(metrics.insert_bytes.load(Ordering::Relaxed), 1024);
    assert_eq!(metrics.lookups.load(Ordering::Relaxed), 2);
    assert_eq!(metrics.hits.load(Ordering::Relaxed), 1);
}

#[test]
fn metrics_hooks_fire_reject_reasons_for_each_insert_decline() {
    // issue #774: each of the three store-detectable `InsertError` variants
    // must classify into its own fixed `PromptCacheRejectReason` counter.
    use super::super::metrics::PromptCacheRejectReason;

    // Oversized: single entry too large for the byte budget.
    let metrics = AtomicPromptCacheMetrics::shared();
    let store = PromptCacheStore::with_metrics(cfg(1024, 64, 4), metrics.clone());
    let toks = tokens(0, 16);
    let err = store
        .insert(
            &key_for("m", &toks),
            CacheEntry::new_for_test(toks.clone(), 2048),
        )
        .unwrap_err();
    assert!(matches!(err, InsertError::OversizedEntry { .. }));
    assert_eq!(
        metrics
            .reject_reasons
            .count(PromptCacheRejectReason::Oversized),
        1
    );

    // Disabled: the store itself is disabled.
    let metrics = AtomicPromptCacheMetrics::shared();
    let store = PromptCacheStore::with_metrics(PromptCacheConfig::disabled(), metrics.clone());
    let err = store
        .insert(
            &key_for("m", &toks),
            CacheEntry::new_for_test(toks.clone(), 1024),
        )
        .unwrap_err();
    assert_eq!(err, InsertError::Disabled);
    assert_eq!(
        metrics
            .reject_reasons
            .count(PromptCacheRejectReason::Disabled),
        1
    );

    // PrefixTooShort: entry token length below `min_prefix_tokens`.
    let metrics = AtomicPromptCacheMetrics::shared();
    let store = PromptCacheStore::with_metrics(cfg(1 << 20, 64, 32), metrics.clone());
    let short = tokens(0, 16);
    let err = store
        .insert(
            &key_for("m", &short),
            CacheEntry::new_for_test(short.clone(), 1024),
        )
        .unwrap_err();
    assert!(matches!(err, InsertError::PrefixTooShort { .. }));
    assert_eq!(
        metrics
            .reject_reasons
            .count(PromptCacheRejectReason::PrefixTooShort),
        1
    );
}

#[test]
fn snapshot_metrics_hooks_fire_on_insert_and_lookup() {
    let metrics = AtomicPromptCacheMetrics::shared();
    let store = PromptCacheStore::with_metrics(
        cfg(1 << 20, 64, 4).with_snapshot_limits(1 << 20, 64, Duration::from_secs(3600)),
        metrics.clone(),
    );
    let toks = tokens(0, 16);
    store
        .insert_snapshot(
            &key_for("m", &toks),
            snapshot_entry_for_test(toks.clone(), "mamba"),
        )
        .unwrap();
    let _ = store.lookup_snapshot_prefix(&key_for("m", &toks), &toks);
    let miss = tokens(999, 16);
    let _ = store.lookup_snapshot_prefix(&key_for("m", &miss), &miss);

    assert_eq!(metrics.snapshot_inserts.load(Ordering::Relaxed), 1);
    assert_eq!(metrics.snapshot_lookups.load(Ordering::Relaxed), 2);
    assert_eq!(metrics.snapshot_hits.load(Ordering::Relaxed), 1);
    assert_eq!(
        metrics.snapshot_hit_tokens_total.load(Ordering::Relaxed),
        toks.len() as u64
    );
}

#[test]
fn evict_if_needed_returns_freed_bytes() {
    let cfg = PromptCacheConfig::new(true, 1 << 20, 64, Duration::from_millis(25), 4);
    let store = PromptCacheStore::with_config(cfg);
    for i in 0..4 {
        let toks = tokens(i * 100, 16);
        store
            .insert(
                &key_for("m", &toks),
                CacheEntry::new_for_test(toks.clone(), 1024),
            )
            .unwrap();
    }
    thread::sleep(Duration::from_millis(60));
    let freed = store.evict_if_needed();
    assert_eq!(freed, 4 * 1024);
    assert_eq!(store.len(), 0);
}

#[test]
fn stats_reflect_mutations() {
    let store = PromptCacheStore::with_config(cfg(1 << 20, 64, 4));
    let toks = tokens(0, 16);
    store
        .insert(
            &key_for("m", &toks),
            CacheEntry::new_for_test(toks.clone(), 2048),
        )
        .unwrap();
    let _ = store.lookup_longest_prefix(&key_for("m", &toks), &toks);

    let stats = store.stats();
    assert_eq!(stats.entries, 1);
    assert_eq!(stats.bytes, 2048);
    assert_eq!(stats.inserts, 1);
    assert_eq!(stats.lookups, 1);
    assert_eq!(stats.hits, 1);
}

#[test]
fn snapshot_lookup_requires_whole_stored_prefix_and_same_session() {
    let cfg = cfg(1 << 20, 64, 4).with_snapshot_limits(1 << 20, 64, Duration::from_secs(3600));
    let store = PromptCacheStore::with_config(cfg);
    let stored = tokens(0, 16);
    store
        .insert_snapshot(
            &key_for_session("m", Some("chat-a"), &stored),
            snapshot_entry_for_test(stored.clone(), "mamba"),
        )
        .unwrap();

    let mut extension = stored.clone();
    extension.extend([999, 1000]);
    let (entry, matched) = store
        .lookup_snapshot_prefix(
            &key_for_session("m", Some("chat-a"), &extension),
            &extension,
        )
        .expect("exact stored prefix should restore");
    assert_eq!(matched, stored.len());
    assert_eq!(entry.tokens, stored);
    entry.with_snapshot(|snapshot| {
        assert_eq!(snapshot.family(), "mamba");
        assert_eq!(snapshot.token_len(), stored.len());
    });

    let mut diverging = tokens(0, 15);
    diverging.push(42_424);
    assert!(
        store
            .lookup_snapshot_prefix(
                &key_for_session("m", Some("chat-a"), &diverging),
                &diverging
            )
            .is_none(),
        "snapshot lookup must not truncate to a partial recurrent state"
    );
    assert!(
        store
            .lookup_snapshot_prefix(
                &key_for_session("m", Some("chat-b"), &extension),
                &extension
            )
            .is_none(),
        "snapshots are exact-session only"
    );
    assert!(
        store
            .lookup_snapshot_prefix(&key_for("m", &extension), &extension)
            .is_none(),
        "sessionless lookup must not adopt a session-bound snapshot"
    );
}

#[test]
fn snapshot_lru_cap_is_independent_from_kv_entries() {
    let cfg = cfg(1 << 20, 64, 4).with_snapshot_limits(1 << 20, 2, Duration::from_secs(3600));
    let store = PromptCacheStore::with_config(cfg);
    let kv_tokens = tokens(10_000, 16);
    store
        .insert(
            &key_for("m", &kv_tokens),
            CacheEntry::new_for_test(kv_tokens.clone(), 1024),
        )
        .unwrap();

    let a = tokens(0, 16);
    let b = tokens(100, 16);
    let c = tokens(200, 16);
    store
        .insert_snapshot(
            &key_for("m", &a),
            snapshot_entry_for_test(a.clone(), "mamba"),
        )
        .unwrap();
    thread::sleep(Duration::from_millis(5));
    store
        .insert_snapshot(
            &key_for("m", &b),
            snapshot_entry_for_test(b.clone(), "mamba"),
        )
        .unwrap();
    thread::sleep(Duration::from_millis(5));
    store
        .insert_snapshot(
            &key_for("m", &c),
            snapshot_entry_for_test(c.clone(), "mamba"),
        )
        .unwrap();

    let stats = store.stats();
    assert_eq!(stats.snapshot_entries, 2);
    assert_eq!(stats.entries, 3, "KV entries and snapshots share reporting");
    assert!(stats.snapshot_evictions_lru >= 1);
    assert!(
        store
            .lookup_longest_prefix(&key_for("m", &kv_tokens), &kv_tokens)
            .is_some(),
        "snapshot cap must not evict detached KV entries"
    );
    assert!(
        store
            .lookup_snapshot_prefix(&key_for("m", &a), &a)
            .is_none(),
        "oldest snapshot should be evicted by the snapshot entry cap"
    );
    assert!(
        store
            .lookup_snapshot_prefix(&key_for("m", &b), &b)
            .is_some()
    );
    assert!(
        store
            .lookup_snapshot_prefix(&key_for("m", &c), &c)
            .is_some()
    );
}

// ── session-chain supersede (issue #1146) ───────────────────────────────────
//
// A conversation stores one snapshot per turn and every turn's token vector
// extends the previous turn's, so the store would otherwise accumulate the
// whole chain and let LRU pick a victim under byte pressure. These tests pin
// the four boundaries of the supersede rule (fires on a same-session strict
// extension, and only there) plus the byte accounting that lets a freed
// ancestor admit its successor.

/// Snapshot config with a caller-chosen byte budget, entry cap fixed high
/// enough to stay out of the way.
fn snapshot_cfg(snapshot_capacity_bytes: usize) -> PromptCacheConfig {
    cfg(1 << 20, 64, 4).with_snapshot_limits(snapshot_capacity_bytes, 64, Duration::from_secs(3600))
}

#[test]
fn session_chain_supersede_keeps_only_the_longest_snapshot() {
    let store = PromptCacheStore::with_config(snapshot_cfg(1 << 20));
    let turn1 = tokens(0, 16);
    let turn2 = tokens(0, 24);
    let turn3 = tokens(0, 32);

    store
        .insert_snapshot(
            &key_for_session("m", Some("chat-a"), &turn1),
            ModelSnapshotEntry::new_for_test(turn1.clone(), "mamba", 4096),
        )
        .unwrap();
    store
        .insert_snapshot(
            &key_for_session("m", Some("chat-a"), &turn2),
            ModelSnapshotEntry::new_for_test(turn2.clone(), "mamba", 6144),
        )
        .unwrap();
    store
        .insert_snapshot(
            &key_for_session("m", Some("chat-a"), &turn3),
            ModelSnapshotEntry::new_for_test(turn3.clone(), "mamba", 8192),
        )
        .unwrap();

    let stats = store.stats();
    assert_eq!(stats.snapshot_inserts, 3);
    assert_eq!(
        stats.snapshot_entries, 1,
        "three turns of one conversation must collapse to one resident snapshot"
    );
    assert_eq!(
        stats.snapshot_bytes, 8192,
        "only the longest turn's bytes stay accounted"
    );
    assert_eq!(stats.snapshot_supersedes, 2);
    assert_eq!(
        stats.snapshot_evictions_lru, 0,
        "collapsing a chain is a supersede, not capacity pressure"
    );

    let (entry, matched) = store
        .lookup_snapshot_prefix(&key_for_session("m", Some("chat-a"), &turn3), &turn3)
        .expect("the surviving snapshot is the longest one");
    assert_eq!(matched, turn3.len());
    assert_eq!(entry.tokens, turn3);
}

#[test]
fn supersede_does_not_reach_across_sessions() {
    let store = PromptCacheStore::with_config(snapshot_cfg(1 << 20));
    let short = tokens(0, 16);
    let long = tokens(0, 24);

    store
        .insert_snapshot(
            &key_for_session("m", Some("chat-a"), &short),
            ModelSnapshotEntry::new_for_test(short.clone(), "mamba", 4096),
        )
        .unwrap();
    // Same model/lora/template bucket and a strict token extension, but a
    // different conversation. Session "chat-a" must be left alone.
    store
        .insert_snapshot(
            &key_for_session("m", Some("chat-b"), &long),
            ModelSnapshotEntry::new_for_test(long.clone(), "mamba", 6144),
        )
        .unwrap();

    let stats = store.stats();
    assert_eq!(
        stats.snapshot_entries, 2,
        "two concurrent conversations hold snapshots simultaneously"
    );
    assert_eq!(stats.snapshot_bytes, 4096 + 6144);
    assert_eq!(stats.snapshot_supersedes, 0);
    assert_eq!(stats.snapshot_evictions_lru, 0);
    assert!(
        store
            .lookup_snapshot_prefix(&key_for_session("m", Some("chat-a"), &short), &short)
            .is_some(),
        "the other session's snapshot must survive untouched"
    );
    assert!(
        store
            .lookup_snapshot_prefix(&key_for_session("m", Some("chat-b"), &long), &long)
            .is_some()
    );
}

#[test]
fn supersede_does_not_fire_without_a_session_key() {
    let store = PromptCacheStore::with_config(snapshot_cfg(1 << 20));
    let short = tokens(0, 16);
    let long = tokens(0, 24);

    // A `None` session carries no conversation identity, so a longer prefix is
    // not evidence that it came from the same chat and may not evict the
    // shorter one.
    store
        .insert_snapshot(
            &key_for("m", &short),
            ModelSnapshotEntry::new_for_test(short.clone(), "mamba", 4096),
        )
        .unwrap();
    store
        .insert_snapshot(
            &key_for("m", &long),
            ModelSnapshotEntry::new_for_test(long.clone(), "mamba", 6144),
        )
        .unwrap();

    let stats = store.stats();
    assert_eq!(stats.snapshot_entries, 2);
    assert_eq!(stats.snapshot_supersedes, 0);
    assert!(
        store
            .lookup_snapshot_prefix(&key_for("m", &short), &short)
            .is_some()
    );
}

#[test]
fn supersede_requires_a_strict_token_extension() {
    let store = PromptCacheStore::with_config(snapshot_cfg(1 << 20));
    let original = tokens(0, 16);
    // Same session, longer, sharing the first 8 tokens but diverging after
    // that. The stored vector is not a prefix of the new one, so the stored
    // snapshot is still the right match for its own branch and must survive.
    let mut forked = tokens(0, 8);
    forked.extend(tokens(500, 16));

    store
        .insert_snapshot(
            &key_for_session("m", Some("chat-a"), &original),
            ModelSnapshotEntry::new_for_test(original.clone(), "mamba", 4096),
        )
        .unwrap();
    store
        .insert_snapshot(
            &key_for_session("m", Some("chat-a"), &forked),
            ModelSnapshotEntry::new_for_test(forked.clone(), "mamba", 6144),
        )
        .unwrap();

    let stats = store.stats();
    assert_eq!(
        stats.snapshot_entries, 2,
        "same session is not enough; the new vector must extend the stored one"
    );
    assert_eq!(stats.snapshot_supersedes, 0);
    assert!(
        store
            .lookup_snapshot_prefix(&key_for_session("m", Some("chat-a"), &original), &original)
            .is_some()
    );
}

#[test]
fn supersede_frees_bytes_before_the_capacity_check() {
    // Budget fits either turn alone but not both at once. Supersede runs
    // before the new entry's bytes are accounted, so the extension is admitted
    // outright rather than being admitted and then fought over by LRU.
    let store = PromptCacheStore::with_config(snapshot_cfg(10_000));
    let turn1 = tokens(0, 16);
    let turn2 = tokens(0, 24);

    store
        .insert_snapshot(
            &key_for_session("m", Some("chat-a"), &turn1),
            ModelSnapshotEntry::new_for_test(turn1.clone(), "mamba", 8_000),
        )
        .unwrap();
    store
        .insert_snapshot(
            &key_for_session("m", Some("chat-a"), &turn2),
            ModelSnapshotEntry::new_for_test(turn2.clone(), "mamba", 9_000),
        )
        .expect("the extension fits once its own ancestor's bytes are freed");

    let stats = store.stats();
    assert_eq!(stats.snapshot_entries, 1);
    assert_eq!(stats.snapshot_bytes, 9_000);
    assert_eq!(stats.snapshot_supersedes, 1);
    assert_eq!(
        stats.snapshot_evictions_lru, 0,
        "no LRU eviction should have been needed to make room"
    );
    let (_, matched) = store
        .lookup_snapshot_prefix(&key_for_session("m", Some("chat-a"), &turn2), &turn2)
        .expect("the newest turn is the resident snapshot");
    assert_eq!(matched, turn2.len());
}

#[test]
fn snapshot_capacity_self_eviction_is_counted_and_warned_once_per_session() {
    let store = PromptCacheStore::with_config(snapshot_cfg(12_000));
    let first = tokens(0, 16);
    let second = tokens(100, 16);
    let third = tokens(200, 16);

    store
        .insert_snapshot(
            &key_for_session("m", Some("chat-a"), &first),
            ModelSnapshotEntry::new_for_test(first.clone(), "qwen3-next", 8_000)
                .with_origin(SnapshotOrigin::Completion),
        )
        .unwrap();
    thread::sleep(Duration::from_millis(2));
    store
        .insert_snapshot(
            &key_for_session("m", Some("chat-a"), &second),
            ModelSnapshotEntry::new_for_test(second.clone(), "qwen3-next", 8_000)
                .with_origin(SnapshotOrigin::Boundary),
        )
        .unwrap();

    let stats = store.stats();
    assert_eq!(stats.snapshot_entries, 1);
    assert_eq!(stats.snapshot_evictions_lru, 1);
    assert_eq!(stats.snapshot_self_evictions, 1);

    thread::sleep(Duration::from_millis(2));
    store
        .insert_snapshot(
            &key_for_session("m", Some("chat-a"), &third),
            ModelSnapshotEntry::new_for_test(third.clone(), "qwen3-next", 8_000)
                .with_origin(SnapshotOrigin::Completion),
        )
        .unwrap();

    let stats = store.stats();
    assert_eq!(stats.snapshot_evictions_lru, 2);
    assert_eq!(stats.snapshot_self_evictions, 2);
    assert_eq!(
        store
            .inner
            .read()
            .expect("prompt cache inner lock")
            .warned_snapshot_self_evictions
            .len(),
        1,
        "operator WARN should be emitted once for the affected session"
    );
}

#[test]
fn idempotent_insert_replaces_existing_entry() {
    let store = PromptCacheStore::with_config(cfg(1 << 20, 64, 4));
    let toks = tokens(0, 16);
    store
        .insert(
            &key_for("m", &toks),
            CacheEntry::new_for_test(toks.clone(), 1024),
        )
        .unwrap();
    // Re-insert with the same key but a larger entry; the new entry must
    // replace the old one without exceeding the count cap.
    store
        .insert(
            &key_for("m", &toks),
            CacheEntry::new_for_test(toks.clone(), 2048),
        )
        .unwrap();
    assert_eq!(store.len(), 1);
    assert_eq!(store.bytes(), 2048);
}

#[test]
fn clear_drops_everything() {
    let store = PromptCacheStore::with_config(cfg(1 << 20, 64, 4));
    for i in 0..4 {
        let toks = tokens(i * 100, 16);
        store
            .insert(
                &key_for("m", &toks),
                CacheEntry::new_for_test(toks.clone(), 1024),
            )
            .unwrap();
    }
    store.clear();
    assert_eq!(store.len(), 0);
    assert_eq!(store.bytes(), 0);
}

// specific two-tier matcher tests live in
// `super::prefix_matcher_tests`; keeping them in a sibling test module
// keeps this file focused on store-level invariants (insert/evict/lookup
// mechanics) and cleanly below the 500-line code-file limit.

// ---------------------------------------------------------------------------
// History-boundary snapshot alongside the end-of-generation one (issue #1143)
// ---------------------------------------------------------------------------

#[test]
fn history_boundary_snapshot_hits_where_the_end_of_generation_one_cannot() {
    // Reproduces the epic #1148 shape at the store level.
    //
    // Turn N stores two snapshots for the same session:
    //   * the boundary snapshot, keyed by the tokenization of the history
    //     render (everything up to the last user message);
    //   * the end-of-generation snapshot, keyed by prompt + generated, where
    //     the prompt tail is a generation-prompt scaffold and the generated
    //     tail is a sampled (non-canonical) token sequence.
    //
    // Turn N+1's prompt continues from the history boundary with re-rendered
    // history, so it shares the boundary vector and diverges from the
    // end-of-generation vector at the scaffold. Only the boundary snapshot may
    // be returned, and it must be returned rather than reported as a miss.
    let store = PromptCacheStore::with_config(cfg(1 << 20, 64, 4));

    let history: Vec<i32> = tokens(100, 24);
    // Prompt = history + generation-prompt scaffold; the scaffold ids are
    // deliberately outside the history range.
    let mut prompt = history.clone();
    prompt.extend([9001, 9002]);
    // End-of-generation vector = prompt + sampled completion ids.
    let mut generated_vec = prompt.clone();
    generated_vec.extend([9101, 9102, 9103]);

    // Tag each entry with its real producer: the session-chain supersede rule
    // (#1146) is scoped per producer, and an untagged pair would collapse.
    store
        .insert_snapshot(
            &key_for_session("m", Some("sess"), &history),
            snapshot_entry_for_test(history.clone(), "fam").with_origin(SnapshotOrigin::Boundary),
        )
        .expect("boundary snapshot insert succeeds");
    store
        .insert_snapshot(
            &key_for_session("m", Some("sess"), &generated_vec),
            snapshot_entry_for_test(generated_vec.clone(), "fam")
                .with_origin(SnapshotOrigin::Completion),
        )
        .expect("end-of-generation snapshot insert succeeds");
    assert_eq!(store.stats().snapshot_entries, 2);

    // Turn N+1: history + re-rendered assistant reply + new user turn +
    // scaffold. The re-rendered reply retokenizes differently from the sampled
    // ids, which is divergence class (c).
    let mut next_prompt = history.clone();
    next_prompt.extend([8201, 8202, 8203, 8204, 8205]);

    let (entry, matched) = store
        .lookup_snapshot_prefix(
            &key_for_session("m", Some("sess"), &next_prompt),
            &next_prompt,
        )
        .expect("the boundary snapshot must be reachable on the next turn");
    assert_eq!(matched, history.len());
    assert_eq!(entry.tokens, history);

    // Counter-based confirmation, mirroring the /v1/cache/stats check the
    // issue's acceptance criteria call for.
    assert_eq!(store.stats().snapshot_hits, 1);

    // Control: without the boundary entry the same lookup is a miss, which is
    // the behavior this issue exists to change.
    let bare = PromptCacheStore::with_config(cfg(1 << 20, 64, 4));
    bare.insert_snapshot(
        &key_for_session("m", Some("sess"), &generated_vec),
        snapshot_entry_for_test(generated_vec.clone(), "fam")
            .with_origin(SnapshotOrigin::Completion),
    )
    .expect("insert succeeds");
    assert!(
        bare.lookup_snapshot_prefix(
            &key_for_session("m", Some("sess"), &next_prompt),
            &next_prompt
        )
        .is_none()
    );
    assert_eq!(bare.stats().snapshot_hits, 0);
}

#[test]
fn completion_snapshot_does_not_supersede_the_boundary_snapshot_of_its_own_turn() {
    // Cross-issue regression guard for #1143 against #1146's session-chain
    // supersede rule.
    //
    // A turn's history-boundary vector is ALWAYS a strict prefix of that same
    // turn's completion vector, because the completion vector is
    // `history + generation-prompt scaffold + sampled reply`. An unscoped
    // supersede rule therefore deletes the boundary entry the moment the turn
    // finishes, and the boundary entry is the only one that can match the next
    // turn. That combination was measured live: turn 2 `cached_tokens` fell
    // back to 0 of 189 with both features present and no LRU eviction
    // involved. Scoping the chain per producer is what keeps both alive.
    let store = PromptCacheStore::with_config(cfg(1 << 20, 64, 4));

    let history: Vec<i32> = tokens(100, 24);
    let mut completion = history.clone();
    completion.extend([9001, 9002, 9101, 9102, 9103]);

    store
        .insert_snapshot(
            &key_for_session("m", Some("sess"), &history),
            snapshot_entry_for_test(history.clone(), "fam").with_origin(SnapshotOrigin::Boundary),
        )
        .expect("boundary snapshot insert succeeds");
    store
        .insert_snapshot(
            &key_for_session("m", Some("sess"), &completion),
            snapshot_entry_for_test(completion.clone(), "fam")
                .with_origin(SnapshotOrigin::Completion),
        )
        .expect("completion snapshot insert succeeds");

    // Both survive: the completion entry did not chain over the boundary one.
    assert_eq!(store.stats().snapshot_entries, 2);
    assert_eq!(store.stats().snapshot_supersedes, 0);

    // And the boundary entry is still reachable from the next turn, which is
    // the whole point of keeping it.
    let mut next_prompt = history.clone();
    next_prompt.extend([8201, 8202, 8203, 8204, 8205]);
    let (entry, matched) = store
        .lookup_snapshot_prefix(
            &key_for_session("m", Some("sess"), &next_prompt),
            &next_prompt,
        )
        .expect("the boundary snapshot must survive its own turn's completion donate");
    assert_eq!(matched, history.len());
    assert_eq!(entry.tokens, history);
}

#[test]
fn boundary_snapshots_still_chain_against_each_other() {
    // The other half of the scoping rule: per-producer chains must still
    // collapse, or #1146's bounded steady-state footprint is lost and a long
    // conversation accumulates one boundary entry per turn.
    let store = PromptCacheStore::with_config(cfg(1 << 20, 64, 4));

    let turn1: Vec<i32> = tokens(100, 24);
    let mut turn2 = turn1.clone();
    turn2.extend([700, 701, 702, 703]);

    store
        .insert_snapshot(
            &key_for_session("m", Some("sess"), &turn1),
            snapshot_entry_for_test(turn1.clone(), "fam").with_origin(SnapshotOrigin::Boundary),
        )
        .expect("turn 1 boundary insert succeeds");
    store
        .insert_snapshot(
            &key_for_session("m", Some("sess"), &turn2),
            snapshot_entry_for_test(turn2.clone(), "fam").with_origin(SnapshotOrigin::Boundary),
        )
        .expect("turn 2 boundary insert succeeds");

    assert_eq!(store.stats().snapshot_entries, 1);
    assert_eq!(store.stats().snapshot_supersedes, 1);
}

// ---------------------------------------------------------------------------
// #1346: a paged set with no per-layer handles carries no reusable KV.
// ---------------------------------------------------------------------------

#[test]
fn paged_set_without_handles_is_empty() {
    use super::super::entry::paged_set_is_empty;

    // A real paged donation: visible tokens, pinned blocks, one handle per
    // layer. Reusable.
    assert!(!paged_set_is_empty(64, 4, 2));

    // The #1346 shape. The block table looks healthy because
    // `sync_paged_state_with_lengths` mirrored the model-owned lengths into
    // it, but zero per-layer handles means nothing ever wrote K/V behind those
    // pages. Adopting it would skip prefill for 64 tokens that do not exist.
    assert!(paged_set_is_empty(64, 4, 0));

    // The two pre-existing terms are unchanged.
    assert!(paged_set_is_empty(0, 4, 2));
    assert!(paged_set_is_empty(64, 0, 2));
}
