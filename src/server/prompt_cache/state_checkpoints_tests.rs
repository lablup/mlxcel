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

//! Store-level tests for turn-boundary state checkpoints (issue #3047).

use std::time::Duration;

use super::super::super::entry::{ModelSnapshotEntry, SnapshotOrigin};
use super::super::super::key::{MultimodalDigest, PromptCacheKey};
use super::super::super::lineage::SnapshotLineage;
use super::super::super::policy::PromptCacheConfig;
use super::super::{InsertError, PromptCacheStore, SnapshotLookupOutcome};

const SESSION: &str = "chat-a";

fn config(checkpoint_bytes: usize) -> PromptCacheConfig {
    PromptCacheConfig::new(true, 1 << 20, 64, Duration::from_secs(3600), 4)
        .with_snapshot_limits(1 << 20, 64, Duration::from_secs(3600))
        .with_state_checkpoint_capacity_bytes(checkpoint_bytes)
}

fn key<'a>(session: &'a str, tokens: &'a [i32]) -> PromptCacheKey<'a> {
    PromptCacheKey::new_full(
        "m",
        None,
        "tpl",
        Some(session),
        MultimodalDigest::empty(),
        tokens,
    )
}

/// A conversation's prompt: turn `n` ends at `10 * n` tokens.
fn history(len: usize) -> Vec<i32> {
    (0..len as i32).collect()
}

/// The prompt of a request that edits the history after `keep` tokens.
fn edited_after(keep: usize, len: usize) -> Vec<i32> {
    let mut tokens = history(len);
    for t in &mut tokens[keep + 2..] {
        *t += 1000;
    }
    tokens
}

/// Lineage of a chain where turn `n` (sequence id `n`) computed `[10(n-1), 10n)`.
fn chain_lineage(turns: usize) -> SnapshotLineage {
    (1..=turns).fold(SnapshotLineage::default(), |l, n| {
        l.extended(10 * n, n as u64)
    })
}

fn whole(tokens: &[i32], lineage: SnapshotLineage) -> ModelSnapshotEntry {
    ModelSnapshotEntry::new_for_test(tokens.to_vec(), "qwen3_5", 10_000)
        .with_origin(SnapshotOrigin::Boundary)
        .with_lineage(lineage)
}

fn checkpoint(tokens: &[i32], lineage: SnapshotLineage) -> ModelSnapshotEntry {
    ModelSnapshotEntry::new_for_test(tokens.to_vec(), "qwen3_5", 100)
        .with_origin(SnapshotOrigin::Boundary)
        .with_lineage(lineage)
}

/// Store turns `1..=turns` the way the scheduler does: each turn's whole
/// history-boundary snapshot, then its checkpoint.
fn run_turns(store: &PromptCacheStore, session: &str, turns: usize) {
    for n in 1..=turns {
        let tokens = history(10 * n);
        let lineage = chain_lineage(n);
        store
            .insert_snapshot(&key(session, &tokens), whole(&tokens, lineage.clone()))
            .expect("whole snapshot");
        store
            .insert_state_checkpoint(&key(session, &tokens), checkpoint(&tokens, lineage))
            .expect("checkpoint");
    }
}

fn lookup(store: &PromptCacheStore, session: &str, tokens: &[i32]) -> SnapshotLookupOutcome {
    store.lookup_snapshot_outcome(&key(session, tokens), tokens, |_, _| false)
}

#[test]
fn an_edit_a_few_turns_back_rebuilds_the_checkpoint_over_the_newest_snapshot() {
    let store = PromptCacheStore::with_config(config(1 << 20));
    run_turns(&store, SESSION, 8);
    let stats = store.stats();
    assert_eq!(stats.snapshot_entries, 1, "turns supersede each other");
    assert_eq!(stats.state_checkpoint_entries, 8);

    for back in 1..=3 {
        let keep = 10 * (8 - back);
        match lookup(&store, SESSION, &edited_after(keep, 95)) {
            SnapshotLookupOutcome::CheckpointHit {
                checkpoint,
                donor,
                matched_len,
            } => {
                assert_eq!(matched_len, keep, "{back} turns back");
                assert_eq!(checkpoint.tokens, history(keep));
                assert_eq!(donor.tokens, history(80));
            }
            other => panic!("{back} turns back: expected a checkpoint hit, got {other:?}"),
        }
    }
    assert_eq!(store.stats().state_checkpoint_hits, 3);
    assert_eq!(store.stats().snapshot_hits, 3);
}

#[test]
fn a_whole_snapshot_that_reaches_further_wins() {
    let store = PromptCacheStore::with_config(config(1 << 20));
    run_turns(&store, SESSION, 4);
    match lookup(&store, SESSION, &history(55)) {
        SnapshotLookupOutcome::Hit { matched_len, .. } => assert_eq!(matched_len, 40),
        other => panic!("expected the whole snapshot, got {other:?}"),
    }
    // The warm-up path asks for whole snapshots only.
    let edited = edited_after(20, 55);
    assert!(
        store
            .lookup_snapshot_prefix(&key(SESSION, &edited), &edited)
            .is_none()
    );
}

#[test]
fn a_donor_computed_by_another_prefill_is_not_used() {
    let store = PromptCacheStore::with_config(config(1 << 20));
    run_turns(&store, SESSION, 3);
    // A cold run of turn 4 recomputed every row itself.
    let tokens = history(40);
    store
        .insert_snapshot(
            &key(SESSION, &tokens),
            whole(&tokens, SnapshotLineage::default().extended(40, 99)),
        )
        .unwrap();
    assert!(!matches!(
        lookup(&store, SESSION, &edited_after(20, 55)),
        SnapshotLookupOutcome::CheckpointHit { .. }
    ));
}

#[test]
fn checkpoints_stay_in_their_session() {
    let store = PromptCacheStore::with_config(config(1 << 20));
    run_turns(&store, SESSION, 3);
    assert!(!matches!(
        lookup(&store, "chat-b", &edited_after(20, 55)),
        SnapshotLookupOutcome::CheckpointHit { .. }
    ));
}

#[test]
fn a_checkpoint_covering_the_whole_request_is_not_offered() {
    // The prefill must re-run the last prompt token, and a recurrent state
    // cannot be rewound by one.
    let store = PromptCacheStore::with_config(config(1 << 20));
    run_turns(&store, SESSION, 3);
    // Turn 2's checkpoint is exactly this request; turn 1's is the longest
    // one that leaves a token to prefill.
    match lookup(&store, SESSION, &history(20)) {
        SnapshotLookupOutcome::CheckpointHit { matched_len, .. } => assert_eq!(matched_len, 10),
        other => panic!("expected turn 1's checkpoint, got {other:?}"),
    }
}

#[test]
fn checkpoints_have_their_own_lru_budget() {
    // Room for three 100-byte checkpoints.
    let store = PromptCacheStore::with_config(config(300));
    run_turns(&store, SESSION, 6);
    let stats = store.stats();
    assert_eq!(stats.state_checkpoint_entries, 3);
    assert_eq!(stats.state_checkpoint_bytes, 300);
    assert_eq!(stats.state_checkpoint_evictions, 3);
    assert_eq!(stats.snapshot_entries, 1, "whole snapshots are not touched");
    // The newest turns survive: an edit two turns back still hits, five back
    // does not.
    assert!(matches!(
        lookup(&store, SESSION, &edited_after(40, 75)),
        SnapshotLookupOutcome::CheckpointHit {
            matched_len: 40,
            ..
        }
    ));
    assert!(!matches!(
        lookup(&store, SESSION, &edited_after(10, 75)),
        SnapshotLookupOutcome::CheckpointHit { .. }
    ));
}

#[test]
fn checkpoints_without_a_donor_are_dropped() {
    // Room for one whole snapshot, so a second conversation evicts the first.
    let store = PromptCacheStore::with_config(config(1 << 20).with_snapshot_limits(
        10_000,
        64,
        Duration::from_secs(3600),
    ));
    run_turns(&store, "chat-b", 2);
    assert_eq!(store.stats().state_checkpoint_entries, 2);
    // Chat A's first turn pushes chat B's snapshot out; chat B's checkpoints
    // can no longer be rebuilt and go with the next checkpoint insert.
    run_turns(&store, SESSION, 1);
    let stats = store.stats();
    assert_eq!(stats.snapshot_entries, 1);
    assert_eq!(stats.state_checkpoint_entries, 1);
    assert_eq!(stats.state_checkpoint_evictions, 2);

    let lonely = history(10);
    let fresh = PromptCacheStore::with_config(config(1 << 20));
    fresh
        .insert_state_checkpoint(
            &key(SESSION, &lonely),
            checkpoint(&lonely, chain_lineage(1)),
        )
        .unwrap();
    assert_eq!(fresh.stats().state_checkpoint_entries, 0);
    assert_eq!(fresh.stats().state_checkpoint_evictions, 1);
}

#[test]
fn checkpoints_are_off_without_a_budget_or_a_lineage() {
    let store = PromptCacheStore::with_config(config(0));
    let tokens = history(10);
    assert!(matches!(
        store.insert_state_checkpoint(
            &key(SESSION, &tokens),
            checkpoint(&tokens, chain_lineage(1))
        ),
        Err(InsertError::Disabled)
    ));
    let store = PromptCacheStore::with_config(config(1 << 20));
    assert!(matches!(
        store.insert_state_checkpoint(
            &key(SESSION, &tokens),
            checkpoint(&tokens, SnapshotLineage::default())
        ),
        Err(InsertError::Disabled)
    ));
    let oversized = ModelSnapshotEntry::new_for_test(tokens.clone(), "qwen3_5", 2 << 20)
        .with_lineage(chain_lineage(1));
    assert!(matches!(
        store.insert_state_checkpoint(&key(SESSION, &tokens), oversized),
        Err(InsertError::OversizedEntry { .. })
    ));
}

#[test]
fn clear_drops_checkpoints() {
    let store = PromptCacheStore::with_config(config(1 << 20));
    run_turns(&store, SESSION, 3);
    store.clear();
    let stats = store.stats();
    assert_eq!(stats.state_checkpoint_entries, 0);
    assert_eq!(stats.state_checkpoint_bytes, 0);
}
