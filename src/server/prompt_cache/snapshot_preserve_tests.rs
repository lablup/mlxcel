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

//! Which entry `PromptCacheStore::insert_snapshot_preserving` keeps, and what
//! the kept entry costs (issue #3467).
//!
//! The sequence the issue describes (a warm-up restores the turn's boundary
//! snapshot and extends it) is pinned in `store_tests.rs`. These tests cover
//! the restore sources that are not that boundary snapshot, the inputs at the
//! edges of the rule, the byte and entry budgets with the extra entry, and
//! the cost of the rule on a long chain.

use std::sync::Arc;
use std::time::{Duration, Instant};

use super::entry::{ModelSnapshotEntry, SnapshotOrigin};
use super::key::{MultimodalDigest, PromptCacheKey};
use super::policy::PromptCacheConfig;
use super::store::PromptCacheStore;

const SESSION: &str = "chat-1";
const HOUR: Duration = Duration::from_secs(3600);

fn config(snapshot_bytes: usize, snapshot_entries: usize) -> PromptCacheConfig {
    PromptCacheConfig::new(true, 1 << 20, 64, HOUR, 4).with_snapshot_limits(
        snapshot_bytes,
        snapshot_entries,
        HOUR,
    )
}

fn store() -> PromptCacheStore {
    PromptCacheStore::with_config(config(1 << 20, 64))
}

fn key<'a>(session: Option<&'a str>, tokens: &'a [i32]) -> PromptCacheKey<'a> {
    PromptCacheKey::new_full("m", None, "tpl", session, MultimodalDigest::empty(), tokens)
}

/// Synthetic token run. Runs with one `base` are prefixes of each other;
/// distinct `base` values are disjoint.
fn run(base: i32, n: usize) -> Vec<i32> {
    (0..n as i32).map(|i| base + i).collect()
}

/// `head` followed by a tail no other vector of these tests shares.
fn fork(head: &[i32], tail_base: i32, n: usize) -> Vec<i32> {
    let mut tokens = head.to_vec();
    tokens.extend(run(tail_base, n));
    tokens
}

fn entry(tokens: &[i32], origin: SnapshotOrigin, bytes: usize) -> ModelSnapshotEntry {
    ModelSnapshotEntry::new_for_test(tokens.to_vec(), "fam", bytes).with_origin(origin)
}

/// Plain insert of a one-byte entry in [`SESSION`].
fn insert(store: &PromptCacheStore, tokens: &[i32], origin: SnapshotOrigin) {
    store
        .insert_snapshot(&key(Some(SESSION), tokens), entry(tokens, origin, 1))
        .expect("snapshot insert succeeds");
}

/// What a warm-up for `target` restores from: the longest stored prefix.
fn restore_source(store: &PromptCacheStore, target: &[i32]) -> Arc<ModelSnapshotEntry> {
    store
        .peek_snapshot_prefix(&key(Some(SESSION), target), target)
        .expect("a stored prefix of the warm-up target")
        .0
}

/// The warm-up's insert: `tokens` as a one-byte Boundary entry in
/// [`SESSION`], preserving `restored`.
fn insert_warmed(store: &PromptCacheStore, tokens: &[i32], restored: &Arc<ModelSnapshotEntry>) {
    store
        .insert_snapshot_preserving(
            &key(Some(SESSION), tokens),
            entry(tokens, SnapshotOrigin::Boundary, 1),
            Some(restored),
        )
        .expect("warmed snapshot insert succeeds");
}

/// Whether [`SESSION`] holds an entry for exactly `tokens`.
fn holds(store: &PromptCacheStore, tokens: &[i32]) -> bool {
    store
        .peek_snapshot_prefix(&key(Some(SESSION), tokens), tokens)
        .is_some_and(|(entry, _)| entry.tokens == tokens)
}

// ── which entry is kept ─────────────────────────────────────────────────────

/// The warm-up restores the longest stored prefix of its target. When the
/// turn stored a completion snapshot (`MLXCEL_PROMPT_CACHE_DECODED_TAIL`) and
/// the template's history render extends the sampled reply, that is the
/// Completion entry, not the Boundary one. The Completion entry is in another
/// chain, so exempting it by identity keeps nothing; the boundary snapshot it
/// extends is the one a retry needs.
#[test]
fn a_completion_restore_source_keeps_the_boundary_it_extends() {
    let store = store();
    let boundary = run(0, 24);
    let completion = run(0, 32);
    let warmed = run(0, 40);
    insert(&store, &boundary, SnapshotOrigin::Boundary);
    insert(&store, &completion, SnapshotOrigin::Completion);

    let restored = restore_source(&store, &warmed);
    assert_eq!(restored.origin, SnapshotOrigin::Completion);
    insert_warmed(&store, &warmed, &restored);

    let stats = store.stats();
    assert_eq!(stats.snapshot_entries, 3);
    assert_eq!(stats.snapshot_supersedes, 0);
    // The retried prompt: the boundary plus a generation prompt, shorter than
    // the completion vector.
    let retry = run(0, 27);
    let (hit, matched) = store
        .lookup_snapshot_prefix(&key(Some(SESSION), &retry), &retry)
        .expect("a retry restores the boundary snapshot");
    assert_eq!((matched, &hit.tokens), (boundary.len(), &boundary));
}

/// One entry is kept, the newest the restore point covers. Older ones are
/// superseded as before, so the chain does not grow by one per warm-up.
#[test]
fn only_the_newest_covered_entry_is_kept() {
    let store = store();
    let (older, newer, completion, warmed) = (run(0, 16), run(0, 24), run(0, 32), run(0, 40));
    // Longest first: a shorter insert supersedes nothing.
    insert(&store, &newer, SnapshotOrigin::Boundary);
    insert(&store, &older, SnapshotOrigin::Boundary);
    insert(&store, &completion, SnapshotOrigin::Completion);

    let restored = restore_source(&store, &warmed);
    insert_warmed(&store, &warmed, &restored);

    assert_eq!(store.stats().snapshot_supersedes, 1);
    assert!(holds(&store, &newer));
    assert!(!holds(&store, &older));
    assert!(holds(&store, &completion) && holds(&store, &warmed));
}

/// A boundary entry past the restore point is not covered by it and is
/// superseded as before. This is the warm-up whose base is a system-prompt
/// root or a periodic checkpoint: both end before the history boundary.
#[test]
fn an_entry_past_the_restore_point_is_still_superseded() {
    for base_origin in [SnapshotOrigin::SystemRoot, SnapshotOrigin::Checkpoint] {
        let store = store();
        let (base, boundary, warmed) = (run(0, 8), run(0, 24), run(0, 32));
        insert(&store, &base, base_origin);
        insert(&store, &boundary, SnapshotOrigin::Boundary);

        // Not the store's own choice (that is the longer boundary entry):
        // the base a warm-up is left with once the boundary entry is gone.
        let restored = restore_source(&store, &base);
        assert_eq!(restored.origin, base_origin);
        insert_warmed(&store, &warmed, &restored);

        assert_eq!(store.stats().snapshot_supersedes, 1, "{base_origin:?}");
        assert!(!holds(&store, &boundary), "{base_origin:?}");
        assert!(
            holds(&store, &base) && holds(&store, &warmed),
            "{base_origin:?}: the base is in another chain and stays"
        );
    }
}

/// The entry is chosen by token prefix, not by identity: a restore source
/// that was replaced since the lookup still names the boundary to keep, and
/// one that is gone with nothing under it keeps nothing.
#[test]
fn a_restore_source_that_left_the_store_keeps_what_its_tokens_cover() {
    let store = store();
    let (boundary, warmed) = (run(0, 24), run(0, 32));
    insert(&store, &boundary, SnapshotOrigin::Boundary);
    let restored = restore_source(&store, &warmed);
    // The same vector stored again replaces the entry the lookup returned.
    insert(&store, &boundary, SnapshotOrigin::Boundary);
    insert_warmed(&store, &warmed, &restored);
    assert_eq!(store.stats().snapshot_supersedes, 0);
    assert!(holds(&store, &boundary) && holds(&store, &warmed));

    let store = self::store();
    insert(&store, &boundary, SnapshotOrigin::Boundary);
    let restored = restore_source(&store, &warmed);
    store.clear();
    insert_warmed(&store, &warmed, &restored);
    let stats = store.stats();
    assert_eq!((stats.snapshot_entries, stats.snapshot_supersedes), (1, 0));
    assert!(holds(&store, &warmed));
}

/// Without a session key no supersede runs at all, preserving or not.
#[test]
fn a_session_less_insert_supersedes_nothing_either_way() {
    let store = store();
    let (boundary, warmed) = (run(0, 24), run(0, 32));
    store
        .insert_snapshot(
            &key(None, &boundary),
            entry(&boundary, SnapshotOrigin::Boundary, 1),
        )
        .unwrap();
    let restored = store
        .peek_snapshot_prefix(&key(None, &warmed), &warmed)
        .expect("the session-less boundary")
        .0;
    store
        .insert_snapshot_preserving(
            &key(None, &warmed),
            entry(&warmed, SnapshotOrigin::Boundary, 1),
            Some(&restored),
        )
        .unwrap();
    let stats = store.stats();
    assert_eq!((stats.snapshot_entries, stats.snapshot_supersedes), (2, 0));
}

/// Another conversation's entries are neither kept nor superseded by this
/// one's warm-up, even when its vectors are prefixes of the warmed one.
#[test]
fn another_session_is_left_alone() {
    let store = store();
    let (boundary, warmed) = (run(0, 24), run(0, 32));
    insert(&store, &boundary, SnapshotOrigin::Boundary);
    store
        .insert_snapshot(
            &key(Some("chat-2"), &boundary),
            entry(&boundary, SnapshotOrigin::Boundary, 1),
        )
        .unwrap();

    let restored = restore_source(&store, &warmed);
    insert_warmed(&store, &warmed, &restored);

    let stats = store.stats();
    assert_eq!((stats.snapshot_entries, stats.snapshot_supersedes), (3, 0));
    assert!(
        store
            .peek_snapshot_prefix(&key(Some("chat-2"), &boundary), &boundary)
            .is_some()
    );
}

/// A Checkpoint insert never superseded anything (#2885 Part B) and still
/// does not when it names a restore source.
#[test]
fn a_checkpoint_insert_supersedes_nothing_either_way() {
    let store = store();
    let (first, second) = (run(0, 16), run(0, 32));
    insert(&store, &first, SnapshotOrigin::Checkpoint);
    let restored = restore_source(&store, &second);
    store
        .insert_snapshot_preserving(
            &key(Some(SESSION), &second),
            entry(&second, SnapshotOrigin::Checkpoint, 1),
            Some(&restored),
        )
        .unwrap();
    let stats = store.stats();
    assert_eq!((stats.snapshot_entries, stats.snapshot_supersedes), (2, 0));
}

/// The edges of "covers": a restore source with no tokens or on another
/// branch covers no entry and the insert is a plain one; one exactly as long
/// as an entry covers it.
#[test]
fn a_restore_source_that_covers_nothing_keeps_nothing() {
    let (boundary, warmed) = (run(0, 24), run(0, 32));
    let uncovering: [Vec<i32>; 4] = [
        Vec::new(),
        run(0, 1),
        run(0, 23),
        fork(&run(0, 23), 9_000, 16),
    ];
    for tokens in uncovering {
        let store = store();
        insert(&store, &boundary, SnapshotOrigin::Boundary);
        let restored = Arc::new(entry(&tokens, SnapshotOrigin::Completion, 1));
        insert_warmed(&store, &warmed, &restored);
        let stats = store.stats();
        assert_eq!(
            (stats.snapshot_entries, stats.snapshot_supersedes),
            (1, 1),
            "a {}-token restore source keeps nothing",
            tokens.len()
        );
    }

    let store = store();
    insert(&store, &boundary, SnapshotOrigin::Boundary);
    let restored = Arc::new(entry(&boundary, SnapshotOrigin::Completion, 1));
    insert_warmed(&store, &warmed, &restored);
    assert_eq!(store.stats().snapshot_supersedes, 0);
    assert!(holds(&store, &boundary));

    // An empty chain with a restore source is a plain insert too.
    let store = self::store();
    let restored = Arc::new(entry(&boundary, SnapshotOrigin::Boundary, 1));
    insert_warmed(&store, &warmed, &restored);
    let stats = store.stats();
    assert_eq!((stats.snapshot_entries, stats.snapshot_supersedes), (1, 0));
}

/// `keep_previous_turn` (#3035) exempts the session's restore source. A
/// warm-up's lookup records its base as that source, so when the base is the
/// boundary entry both exemptions name the same entry and keep one, and the
/// entry the switch kept through the turn is superseded.
#[test]
fn keep_previous_turn_and_the_kept_boundary_do_not_stack() {
    let store = PromptCacheStore::with_config(config(1 << 20, 64).with_keep_previous_turn(true));
    let (previous, turn, warmed) = (run(0, 16), run(0, 24), run(0, 32));
    insert(&store, &previous, SnapshotOrigin::Boundary);
    // The turn restores `previous`, which makes it the kept restore source.
    store
        .lookup_snapshot_prefix(&key(Some(SESSION), &turn), &turn)
        .expect("the turn restores the previous boundary");
    insert(&store, &turn, SnapshotOrigin::Boundary);
    assert_eq!(store.stats().snapshot_entries, 2);

    let restored = restore_source(&store, &warmed);
    insert_warmed(&store, &warmed, &restored);

    let stats = store.stats();
    assert_eq!((stats.snapshot_entries, stats.snapshot_supersedes), (2, 1));
    assert!(holds(&store, &turn) && holds(&store, &warmed));
    assert!(!holds(&store, &previous));
}

// ── what the kept entry costs ───────────────────────────────────────────────

/// The kept entry's bytes stay accounted until the next turn's insert
/// supersedes it, and are released with it.
#[test]
fn the_kept_entry_is_accounted_and_released_with_the_next_turn() {
    let store = store();
    let (boundary, warmed, next) = (run(0, 24), run(0, 32), run(0, 40));
    store
        .insert_snapshot(
            &key(Some(SESSION), &boundary),
            entry(&boundary, SnapshotOrigin::Boundary, 4_096),
        )
        .unwrap();
    let restored = restore_source(&store, &warmed);
    store
        .insert_snapshot_preserving(
            &key(Some(SESSION), &warmed),
            entry(&warmed, SnapshotOrigin::Boundary, 6_144),
            Some(&restored),
        )
        .unwrap();
    let stats = store.stats();
    assert_eq!(stats.snapshot_entries, 2);
    assert_eq!(stats.snapshot_bytes, 4_096 + 6_144);
    assert_eq!(store.snapshot_growth_bytes(1 << 30), (1 << 20) - 10_240);

    store
        .insert_snapshot(
            &key(Some(SESSION), &next),
            entry(&next, SnapshotOrigin::Boundary, 8_192),
        )
        .unwrap();
    let stats = store.stats();
    assert_eq!(stats.snapshot_entries, 1);
    assert_eq!(stats.snapshot_bytes, 8_192);
    assert_eq!(stats.snapshot_supersedes, 2);
}

/// Entries of one conversation at a byte budget or entry cap that does not
/// hold all of them. Each is built a moment after the one before, so their
/// LRU order is the order given.
fn pressured(
    budget: PromptCacheConfig,
    stored: &[(&[i32], SnapshotOrigin, usize)],
) -> PromptCacheStore {
    let store = PromptCacheStore::with_config(budget);
    for (tokens, origin, bytes) in stored {
        store
            .insert_snapshot(&key(Some(SESSION), tokens), entry(tokens, *origin, *bytes))
            .unwrap();
        std::thread::sleep(Duration::from_millis(2));
    }
    store
}

fn insert_warmed_sized(
    store: &PromptCacheStore,
    tokens: &[i32],
    bytes: usize,
    restored: &Arc<ModelSnapshotEntry>,
) {
    std::thread::sleep(Duration::from_millis(2));
    store
        .insert_snapshot_preserving(
            &key(Some(SESSION), tokens),
            entry(tokens, SnapshotOrigin::Boundary, bytes),
            Some(restored),
        )
        .unwrap();
}

/// A budget that held the conversation's two entries does not hold three.
/// The kept boundary is not superseded, so LRU makes the room instead, and
/// the oldest entry goes: the completion snapshot when the warm-up restored
/// from the boundary (its lookup touched that one).
#[test]
fn under_byte_pressure_lru_drops_the_completion_before_a_restored_boundary() {
    let boundary = run(0, 24);
    // The reply as sampled: no prefix of its history render.
    let completion = fork(&boundary, 5_000, 8);
    let warmed = run(0, 32);
    let store = pressured(
        config(10_000, 64),
        &[
            (completion.as_slice(), SnapshotOrigin::Completion, 3_000),
            (boundary.as_slice(), SnapshotOrigin::Boundary, 4_000),
        ],
    );
    let restored = restore_source(&store, &warmed);
    assert_eq!(restored.origin, SnapshotOrigin::Boundary);
    insert_warmed_sized(&store, &warmed, 5_000, &restored);

    let stats = store.stats();
    assert_eq!(stats.snapshot_bytes, 9_000);
    assert_eq!(stats.snapshot_supersedes, 0);
    assert_eq!(stats.snapshot_evictions_lru, 1);
    assert_eq!(
        stats.snapshot_self_evictions, 1,
        "the eviction is reported as this conversation's own"
    );
    assert!(holds(&store, &boundary) && holds(&store, &warmed));
    assert!(!holds(&store, &completion));
}

/// When the warm-up restored from the completion snapshot, the boundary is
/// the least recently used of the three and is the one LRU drops: under
/// pressure the retry misses as it did before #3467, and nothing else is lost.
#[test]
fn under_byte_pressure_lru_drops_a_boundary_the_warmup_did_not_touch() {
    let (boundary, completion, warmed) = (run(0, 24), run(0, 32), run(0, 40));
    let store = pressured(
        config(10_000, 64),
        &[
            (boundary.as_slice(), SnapshotOrigin::Boundary, 4_000),
            (completion.as_slice(), SnapshotOrigin::Completion, 3_000),
        ],
    );
    let restored = restore_source(&store, &warmed);
    assert_eq!(restored.origin, SnapshotOrigin::Completion);
    insert_warmed_sized(&store, &warmed, 5_000, &restored);

    let stats = store.stats();
    assert_eq!(stats.snapshot_bytes, 8_000);
    assert_eq!(
        (stats.snapshot_supersedes, stats.snapshot_evictions_lru),
        (0, 1)
    );
    assert!(holds(&store, &completion) && holds(&store, &warmed));
    assert!(!holds(&store, &boundary));
}

/// A budget that holds the warmed entry alone ends where a superseding
/// insert ended, with the warmed entry resident, through LRU.
#[test]
fn a_budget_for_one_entry_keeps_the_warmed_one() {
    let (boundary, warmed) = (run(0, 24), run(0, 32));
    let store = pressured(
        config(8_000, 64),
        &[(boundary.as_slice(), SnapshotOrigin::Boundary, 4_000)],
    );
    let restored = restore_source(&store, &warmed);
    insert_warmed_sized(&store, &warmed, 7_000, &restored);

    let stats = store.stats();
    assert_eq!((stats.snapshot_entries, stats.snapshot_bytes), (1, 7_000));
    assert_eq!(
        (stats.snapshot_supersedes, stats.snapshot_evictions_lru),
        (0, 1)
    );
    assert!(holds(&store, &warmed));
}

/// The entry cap bounds the extra entry the same way.
#[test]
fn the_entry_cap_drops_the_oldest_entry_for_the_kept_one() {
    let boundary = run(0, 24);
    let completion = fork(&boundary, 5_000, 8);
    let warmed = run(0, 32);
    let store = pressured(
        config(1 << 20, 2),
        &[
            (completion.as_slice(), SnapshotOrigin::Completion, 1),
            (boundary.as_slice(), SnapshotOrigin::Boundary, 1),
        ],
    );
    let restored = restore_source(&store, &warmed);
    insert_warmed_sized(&store, &warmed, 1, &restored);

    let stats = store.stats();
    assert_eq!(
        (stats.snapshot_entries, stats.snapshot_evictions_lru),
        (2, 1)
    );
    assert!(holds(&store, &boundary) && holds(&store, &warmed));
}

/// Choosing the kept entry compares only the entries about to be
/// superseded, once each, so a preserving insert over a chain of 2,000
/// prefixes of 8,192 tokens costs what the supersede scan itself costs.
/// Comparing each against every other one would be 2,000 times that.
#[test]
fn a_preserving_insert_over_a_long_chain_takes_linear_time() {
    const CHAIN: usize = 2_000;
    const LEN: usize = 8_192;
    let store = PromptCacheStore::with_config(config(1 << 30, CHAIN + 8));
    // Longest first, so every prefix stays: a shorter insert supersedes
    // nothing.
    for cut in 0..CHAIN {
        insert(&store, &run(0, LEN - cut), SnapshotOrigin::Boundary);
    }
    assert_eq!(store.stats().snapshot_entries, CHAIN);
    let warmed = run(0, LEN + 1);
    let restored = restore_source(&store, &warmed);
    assert_eq!(restored.tokens.len(), LEN);

    let started = Instant::now();
    insert_warmed(&store, &warmed, &restored);
    let elapsed = started.elapsed();

    let stats = store.stats();
    assert_eq!(stats.snapshot_entries, 2);
    assert_eq!(stats.snapshot_supersedes, CHAIN as u64 - 1);
    assert!(holds(&store, &run(0, LEN)));
    assert!(
        elapsed < Duration::from_secs(5),
        "a preserving insert over {CHAIN} prefixes of {LEN} tokens took {elapsed:?}"
    );
}
