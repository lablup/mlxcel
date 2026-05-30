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

use super::super::entry::CacheEntry;
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

fn key_for_mm<'a>(
    model: &'a str,
    mm_digest: MultimodalDigest,
    tokens: &'a [i32],
) -> PromptCacheKey<'a> {
    PromptCacheKey::new_full(model, None, "tpl", None, mm_digest, tokens)
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
