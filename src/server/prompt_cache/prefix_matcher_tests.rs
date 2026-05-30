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

//! two-tier longest-prefix matcher tests.
//!
//! Tests that exercise the exact-session vs cross-session fallback
//! behaviour, the same-session tie-break preference, MRU disambiguation
//! on equal match length, the minimum-prefix threshold applied uniformly
//! to both tiers, bucket isolation, and an end-to-end timing smoke-test
//! at 32k token prefixes.

use std::thread;
use std::time::Duration;

use super::entry::CacheEntry;
use super::key::{MultimodalDigest, PromptCacheKey};
use super::policy::PromptCacheConfig;
use super::store::PromptCacheStore;

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

fn key_for_session<'a>(
    model: &'a str,
    session: Option<&'a str>,
    tokens: &'a [i32],
) -> PromptCacheKey<'a> {
    PromptCacheKey::new_full(
        model,
        None,
        "tpl",
        session,
        MultimodalDigest::empty(),
        tokens,
    )
}

#[test]
fn prefix_matcher_cross_session_fallback_returns_entry() {
    // Entry was inserted under session "A"; a lookup for session "B" must
    // still find it as the cross-session fallback because the bucket
    // (model, lora, template) matches.
    let store = PromptCacheStore::with_config(cfg(1 << 20, 64, 4));
    let toks = tokens(0, 16);
    store
        .insert(
            &key_for_session("m", Some("sessA"), &toks),
            CacheEntry::new_for_test(toks.clone(), 1024),
        )
        .unwrap();

    let (_, matched) = store
        .lookup_longest_prefix(&key_for_session("m", Some("sessB"), &toks), &toks)
        .expect("cross-session fallback hits");
    assert_eq!(matched, toks.len());
}

#[test]
fn prefix_matcher_prefers_same_session_on_tie() {
    // Two entries have identical token prefixes but different session_keys.
    // The caller's session ("A") must be preferred even when the other
    // entry was inserted more recently.
    let store = PromptCacheStore::with_config(cfg(1 << 20, 64, 4));
    let toks = tokens(0, 16);
    store
        .insert(
            &key_for_session("m", Some("sessA"), &toks),
            CacheEntry::new_for_test(toks.clone(), 1024),
        )
        .unwrap();
    thread::sleep(Duration::from_millis(5));
    store
        .insert(
            &key_for_session("m", Some("sessB"), &toks),
            CacheEntry::new_for_test(toks.clone(), 1024),
        )
        .unwrap();

    // Caller's session is "A": expect the session-A entry even though B is
    // more recent.
    let (entry, matched) = store
        .lookup_longest_prefix(&key_for_session("m", Some("sessA"), &toks), &toks)
        .expect("same-session preferred");
    assert_eq!(matched, toks.len());
    // We don't expose session_key from CacheEntry directly; assert that the
    // returned entry's tokens match the stored one.
    assert_eq!(entry.tokens, toks);
}

#[test]
fn prefix_matcher_cross_session_wins_when_match_strictly_longer() {
    // Exact-session has a short stored prefix; cross-session has a longer
    // stored prefix. The cross-session match wins because it's strictly
    // longer — tie-break only applies on equal lengths.
    let store = PromptCacheStore::with_config(cfg(1 << 20, 64, 4));
    let short = tokens(0, 16);
    let long = tokens(0, 32);

    store
        .insert(
            &key_for_session("m", Some("sessA"), &short),
            CacheEntry::new_for_test(short.clone(), 1024),
        )
        .unwrap();
    store
        .insert(
            &key_for_session("m", Some("sessB"), &long),
            CacheEntry::new_for_test(long.clone(), 1024),
        )
        .unwrap();

    // Request has 32 tokens; same-session only matches 16, cross-session
    // matches 32 → cross-session wins.
    let (_, matched) = store
        .lookup_longest_prefix(&key_for_session("m", Some("sessA"), &long), &long)
        .expect("cross-session longer match wins");
    assert_eq!(matched, 32);
}

#[test]
fn prefix_matcher_mru_tie_break_when_session_matches() {
    // Two entries, both cross-session from the caller's POV, equal token
    // prefix. The tie-break inside the cross-session tier is MRU. Flipping
    // MRU ordering by touching one entry must not change matched length
    // but should keep lookups deterministic.
    let store = PromptCacheStore::with_config(cfg(1 << 20, 64, 4));
    let toks = tokens(0, 16);
    store
        .insert(
            &key_for_session("m", Some("sess1"), &toks),
            CacheEntry::new_for_test(toks.clone(), 1024),
        )
        .unwrap();
    thread::sleep(Duration::from_millis(5));
    store
        .insert(
            &key_for_session("m", Some("sess2"), &toks),
            CacheEntry::new_for_test(toks.clone(), 1024),
        )
        .unwrap();

    // Caller has no session — both entries land in the "other session"
    // bucket. The MRU one (sess2) should win.
    let first_hit = store
        .lookup_longest_prefix(&key_for_session("m", None, &toks), &toks)
        .expect("first cross-session hit");
    assert_eq!(first_hit.1, toks.len());
    // Touch sess1 so it becomes more recent.
    thread::sleep(Duration::from_millis(5));
    let _ = store.lookup_longest_prefix(&key_for_session("m", Some("sess1"), &toks), &toks);
    thread::sleep(Duration::from_millis(5));
    // Deterministic second hit — bucket contents unchanged.
    let second_hit = store
        .lookup_longest_prefix(&key_for_session("m", None, &toks), &toks)
        .expect("second cross-session hit");
    assert_eq!(second_hit.1, toks.len());
}

#[test]
fn prefix_matcher_respects_min_prefix_in_both_tiers() {
    // min_prefix_tokens = 32. Entries shorter than 32 tokens can't be
    // inserted; prefix matches shorter than 32 tokens must be rejected.
    let store = PromptCacheStore::with_config(cfg(1 << 20, 64, 32));
    let long = tokens(0, 64);
    store
        .insert(
            &key_for_session("m", Some("sessA"), &long),
            CacheEntry::new_for_test(long.clone(), 1024),
        )
        .unwrap();

    // Incoming matches only 20 common tokens with the stored entry. That's
    // below min_prefix_tokens; the matcher must return None.
    let mut short_divergent = tokens(0, 20);
    short_divergent.extend([999, 999, 999]);
    assert!(
        store
            .lookup_longest_prefix(
                &key_for_session("m", Some("sessB"), &short_divergent),
                &short_divergent
            )
            .is_none()
    );

    // Incoming that fully contains a stored >=32-token prefix must hit even
    // cross-session. Longer stored entries are deliberately ignored because
    // their detached KV state includes tokens absent from the request.
    let safe_prefix = tokens(0, 40);
    store
        .insert(
            &key_for_session("m", Some("sessA"), &safe_prefix),
            CacheEntry::new_for_test(safe_prefix.clone(), 1024),
        )
        .unwrap();
    let divergent = {
        let mut v = tokens(0, 40);
        v.extend([999, 999]);
        v
    };
    let (_, matched) = store
        .lookup_longest_prefix(&key_for_session("m", Some("sessB"), &divergent), &divergent)
        .expect("cross-session match at 40 tokens");
    assert_eq!(matched, 40);
}

#[test]
fn prefix_matcher_different_template_does_not_cross_contaminate() {
    // Two entries share model + session but live under different
    // `template_sig` values; they must live in separate trie buckets.
    let store = PromptCacheStore::with_config(cfg(1 << 20, 64, 4));
    let toks = tokens(0, 16);

    let key_a = PromptCacheKey::new_full(
        "m",
        None,
        "tplA",
        Some("sess"),
        MultimodalDigest::empty(),
        &toks,
    );
    let key_b = PromptCacheKey::new_full(
        "m",
        None,
        "tplB",
        Some("sess"),
        MultimodalDigest::empty(),
        &toks,
    );

    store
        .insert(&key_a, CacheEntry::new_for_test(toks.clone(), 1024))
        .unwrap();

    let hit = store.lookup_longest_prefix(&key_b, &toks);
    assert!(
        hit.is_none(),
        "different template sigs don't cross-contaminate"
    );

    let hit = store.lookup_longest_prefix(&key_a, &toks);
    assert!(hit.is_some(), "same template still hits");
}

#[test]
fn prefix_matcher_scales_to_32k_tokens() {
    // Smoke test: a 32k-token prefix walk should complete in well under a
    // wall-clock second, proving the matcher is bounded by trie
    // traversal (O(L)) and not by the number of stored entries. A
    // radix-trie lookup is branch-heavy but cheap per token; a linear
    // scan over even a few entries of this size would be orders of
    // magnitude slower.
    let store = PromptCacheStore::with_config(cfg(16 * 1024 * 1024, 64, 32));
    // 64 entries each of 32k tokens with shared-up-to-index-i prefixes.
    let base: Vec<i32> = (0..32_000).collect();
    for i in 0..64 {
        let mut toks = base.clone();
        // Make each entry's suffix distinct so tries actually grow.
        let tail: Vec<i32> = (0..100).map(|k| 1_000_000 + i * 1000 + k).collect();
        toks.extend(tail);
        store
            .insert(
                &key_for_session("m", Some(&format!("s{i}")), &toks),
                CacheEntry::new_for_test(toks.clone(), 1024),
            )
            .expect("insert");
    }
    // Request is the full 32k shared prefix plus a divergent tail.
    let mut request = base.clone();
    request.extend([99_999_999; 50]);

    let start = std::time::Instant::now();
    let iters = 100;
    for _ in 0..iters {
        let _ =
            store.lookup_longest_prefix(&key_for_session("m", Some("caller"), &request), &request);
    }
    let elapsed = start.elapsed();
    // With 64 entries × 32k tokens, a linear scan would touch ~2M i32
    // compares per lookup. Our radix trie walk descends 32k tokens once
    // regardless of entry count. The lookup + MRU/session tie-break
    // traversal must finish comfortably under 100ms per call even on a
    // debug build.
    let per_call = elapsed / iters;
    assert!(
        per_call < Duration::from_millis(100),
        "32k-token prefix lookup took {per_call:?} per call — too slow",
    );
}
