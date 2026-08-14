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

//! Classified snapshot-divergence rejects (issue #1147).
//!
//! Fixtures reproduce the three divergence classes epic #1148 verified live
//! against real checkpoints, at the token-vector level the store actually
//! compares:
//!
//! * **(a) generation-prompt-only scaffold tokens** — gemma-4-31b-it-4bit.
//!   The template appends an empty thought scaffold to the generation prompt
//!   but not when the assistant turn is re-rendered as history. Measured:
//!   stored entry 139 tokens (94 prompt + 45 generated), turn-2 prompt shares
//!   90 of them, divergence tokens `[100, 45518, 107, 101]`. Qwen 3.5 with
//!   `enable_thinking=false` reaches the same shape through its empty
//!   `<think>\n\n</think>\n\n` injection.
//! * **(b) thinking stripped from history** — qwen3.5-4b-4bit, default
//!   thinking mode. History renders the assistant turn without its `<think>`
//!   block while the stored vector holds the primed `<think>\n` plus the
//!   reasoning body, so the two diverge immediately after the assistant
//!   header.
//! * **(c) retokenization drift** — falcon-h1-tiny-90m-instruct-4bit. No
//!   thinking scaffold at all: the stored vector holds 120 sampled completion
//!   tokens and the same reply text re-tokenizes to 118, so a plain template
//!   still diverges near the end of an otherwise matching prefix.
//!
//! Each class must land on the same `SnapshotDiverged` classification, and
//! the negative cases (cold store, foreign session, foreign model bucket,
//! genuine hit) must not.

use std::time::Duration;

use mlxcel_core::generate::ModelStateSnapshot;

use super::entry::ModelSnapshotEntry;
use super::key::{MultimodalDigest, PromptCacheKey};
use super::policy::PromptCacheConfig;
use super::store::{PromptCacheStore, SnapshotDivergence, SnapshotLookupOutcome};

const SESSION: &str = "chat-1";

fn store_with_snapshots() -> PromptCacheStore {
    let cfg = PromptCacheConfig::new(true, 1 << 20, 64, Duration::from_secs(3600), 4)
        .with_snapshot_limits(1 << 20, 64, Duration::from_secs(3600));
    PromptCacheStore::with_config(cfg)
}

fn key<'a>(model: &'a str, session: Option<&'a str>, tokens: &'a [i32]) -> PromptCacheKey<'a> {
    PromptCacheKey::new_full(
        model,
        None,
        "tpl",
        session,
        MultimodalDigest::empty(),
        tokens,
    )
}

fn snapshot(tokens: Vec<i32>, family: &str) -> ModelSnapshotEntry {
    let snapshot = ModelStateSnapshot::new(family, tokens.len());
    ModelSnapshotEntry::new(tokens, snapshot)
}

/// Synthetic token run standing in for a stretch of chat text. Distinct
/// `base` values produce disjoint runs, so a divergence is unambiguous.
fn run(base: i32, n: usize) -> Vec<i32> {
    (0..n as i32).map(|i| base + i).collect()
}

fn insert(store: &PromptCacheStore, family: &str, tokens: &[i32]) {
    store
        .insert_snapshot(
            &key("m", Some(SESSION), tokens),
            snapshot(tokens.to_vec(), family),
        )
        .expect("snapshot insert succeeds");
}

/// Exact-prefix lookup: the truncation capability is refused, which is what
/// every family whose recurrent state cannot be rewound reports (issue #1145).
/// The partial-adoption path has its own tests below.
fn outcome(
    store: &PromptCacheStore,
    k: &PromptCacheKey<'_>,
    tokens: &[i32],
) -> SnapshotLookupOutcome {
    store.lookup_snapshot_outcome(k, tokens, |_, _| false)
}

fn diverged(outcome: SnapshotLookupOutcome) -> SnapshotDivergence {
    match outcome {
        SnapshotLookupOutcome::Diverged(d) => d,
        SnapshotLookupOutcome::Hit { matched_len, .. } => {
            panic!("expected a divergence report, got a hit at {matched_len} tokens")
        }
        SnapshotLookupOutcome::NoCandidate => {
            panic!(
                "expected a divergence report, got NoCandidate (indistinguishable from a cold store)"
            )
        }
    }
}

// -- class (a): generation-prompt-only scaffold tokens --------------------

#[test]
fn generation_prompt_scaffold_divergence_is_classified_with_its_geometry() {
    // Gemma 4 shape, at the measured lengths: 94 prompt tokens + 45 generated
    // tokens stored, of which the turn-2 prompt reproduces only the first 90
    // before the template's thought scaffold appears.
    let store = store_with_snapshots();
    let shared = run(1_000, 90);
    let mut stored = shared.clone();
    // The 4 scaffold tokens the generation prompt carried, then the reply.
    stored.extend([100, 45_518, 107, 101]);
    stored.extend(run(5_000, 45));
    assert_eq!(stored.len(), 139);
    insert(&store, "gemma4", &stored);

    // Turn 2 re-renders the assistant turn as history: no scaffold, so the
    // reply text follows the shared prefix directly, then the new user turn.
    let mut request = shared.clone();
    request.extend(run(5_000, 45));
    request.extend(run(7_000, 30));

    let report = diverged(outcome(
        &store,
        &key("m", Some(SESSION), &request),
        &request,
    ));
    assert_eq!(
        report.common_prefix_len, 90,
        "the classification must carry how far the vectors agreed"
    );
    assert_eq!(
        report.stored_len, 139,
        "the classification must carry the stored entry length, not just a counter"
    );
    // The counter is not a proxy for the hit counter: this is still a miss.
    assert!(
        store
            .lookup_snapshot_prefix(&key("m", Some(SESSION), &request), &request)
            .is_none()
    );
}

// -- class (b): thinking stripped from the history re-render --------------

#[test]
fn thinking_stripped_from_history_divergence_is_classified() {
    // Qwen 3.5 default thinking mode: the vectors part company right after
    // the assistant header, because the stored vector continues into the
    // primed `<think>\n` while the history re-render goes straight to the
    // visible answer.
    let store = store_with_snapshots();
    let header = run(2_000, 61);
    let mut stored = header.clone();
    stored.extend(run(3_100, 62)); // `<think>\n` + reasoning body + answer
    insert(&store, "qwen3_5", &stored);

    let mut request = header.clone();
    request.extend(run(4_100, 40)); // answer only, no reasoning
    request.extend(run(8_000, 22)); // new user turn

    let report = diverged(outcome(
        &store,
        &key("m", Some(SESSION), &request),
        &request,
    ));
    assert_eq!(report.common_prefix_len, header.len());
    assert_eq!(report.stored_len, stored.len());
}

// -- class (c): retokenization drift --------------------------------------

#[test]
fn retokenization_drift_divergence_is_classified() {
    // Falcon-H1 shape: plain ChatML, no thinking scaffold, and the miss is
    // still structural. 120 sampled completion tokens are stored; the same
    // reply text re-tokenizes to 118, so the vectors agree through the prompt
    // and the first stretch of the reply and then drift.
    let store = store_with_snapshots();
    let prompt = run(500, 130);
    let mut stored = prompt.clone();
    stored.extend(run(6_000, 100)); // identically-tokenizing head of the reply
    stored.extend(run(9_000, 20)); // 20 sampled tokens
    assert_eq!(stored.len() - prompt.len(), 120);
    insert(&store, "falcon_h1", &stored);

    let mut request = prompt.clone();
    request.extend(run(6_000, 100));
    request.extend(run(9_500, 18)); // same text, 18 tokens after re-tokenizing
    request.extend(run(11_000, 25));

    let report = diverged(outcome(
        &store,
        &key("m", Some(SESSION), &request),
        &request,
    ));
    assert_eq!(report.common_prefix_len, 230);
    assert_eq!(report.stored_len, 250);
    assert!(
        report.common_prefix_len < report.stored_len,
        "a divergence report always means the stored tail is unusable"
    );
}

// -- no false classification ----------------------------------------------

#[test]
fn cold_store_reports_no_candidate_not_a_divergence() {
    let store = store_with_snapshots();
    let request = run(1_000, 40);
    assert!(matches!(
        outcome(&store, &key("m", Some(SESSION), &request), &request),
        SnapshotLookupOutcome::NoCandidate
    ));
}

#[test]
fn foreign_session_bucket_reports_no_candidate_not_a_divergence() {
    let store = store_with_snapshots();
    let stored = run(1_000, 40);
    insert(&store, "gemma4", &stored);

    let mut request = run(1_000, 30);
    request.extend(run(20_000, 12));
    // Same tokens, different session: the candidate is not this caller's to
    // diverge from, so classifying it would be a false positive.
    assert!(matches!(
        outcome(&store, &key("m", Some("chat-2"), &request), &request),
        SnapshotLookupOutcome::NoCandidate
    ));
    // A sessionless caller must not see the session-bound entry either.
    assert!(matches!(
        outcome(&store, &key("m", None, &request), &request),
        SnapshotLookupOutcome::NoCandidate
    ));
}

#[test]
fn foreign_model_bucket_reports_no_candidate_not_a_divergence() {
    let store = store_with_snapshots();
    let stored = run(1_000, 40);
    insert(&store, "gemma4", &stored);

    let mut request = run(1_000, 30);
    request.extend(run(20_000, 12));
    assert!(matches!(
        outcome(
            &store,
            &key("other-model", Some(SESSION), &request),
            &request
        ),
        SnapshotLookupOutcome::NoCandidate
    ));
}

#[test]
fn exact_prefix_still_hits_and_reports_no_divergence() {
    let store = store_with_snapshots();
    let stored = run(1_000, 40);
    insert(&store, "gemma4", &stored);

    let mut request = stored.clone();
    request.extend(run(20_000, 12));
    match outcome(&store, &key("m", Some(SESSION), &request), &request) {
        SnapshotLookupOutcome::Hit { matched_len, entry } => {
            assert_eq!(matched_len, stored.len());
            assert_eq!(entry.tokens, stored);
        }
        other => panic!("exact prefix must still hit, got {other:?}"),
    }
}

#[test]
fn a_hit_wins_over_a_sibling_divergence() {
    // With both a usable and an unusable candidate in the bucket, the lookup
    // must adopt rather than report a reject: a divergence classification is
    // only ever a description of a miss.
    let store = store_with_snapshots();
    let usable = run(1_000, 40);
    insert(&store, "gemma4", &usable);
    let mut unusable = run(1_000, 20);
    unusable.extend(run(30_000, 15));
    insert(&store, "gemma4", &unusable);

    let mut request = usable.clone();
    request.extend(run(20_000, 12));
    assert!(matches!(
        outcome(&store, &key("m", Some(SESSION), &request), &request),
        SnapshotLookupOutcome::Hit { .. }
    ));
}

#[test]
fn the_longest_common_prefix_candidate_is_the_one_reported() {
    let store = store_with_snapshots();
    let shared = run(1_000, 40);
    let mut shallow = run(1_000, 10);
    shallow.extend(run(30_000, 15));
    insert(&store, "gemma4", &shallow);
    let mut deep = shared.clone();
    deep.extend(run(31_000, 9));
    insert(&store, "gemma4", &deep);

    let mut request = shared.clone();
    request.extend(run(32_000, 20));
    let report = diverged(outcome(
        &store,
        &key("m", Some(SESSION), &request),
        &request,
    ));
    assert_eq!(report.common_prefix_len, 40, "best candidate is reported");
    assert_eq!(report.stored_len, deep.len());
}

#[test]
fn a_request_shorter_than_the_stored_entry_is_a_divergence() {
    // The stored vector extends past the request, so it is not a prefix of
    // it and cannot be restored. The two lengths say exactly that.
    let store = store_with_snapshots();
    let stored = run(1_000, 40);
    insert(&store, "gemma4", &stored);

    let request = run(1_000, 25);
    let report = diverged(outcome(
        &store,
        &key("m", Some(SESSION), &request),
        &request,
    ));
    assert_eq!(report.common_prefix_len, 25);
    assert_eq!(report.stored_len, 40);
}

#[test]
fn a_disabled_store_reports_no_candidate() {
    let cfg = PromptCacheConfig::new(false, 1 << 20, 64, Duration::from_secs(3600), 4)
        .with_snapshot_limits(1 << 20, 64, Duration::from_secs(3600));
    let store = PromptCacheStore::with_config(cfg);
    let request = run(1_000, 40);
    assert!(matches!(
        outcome(&store, &key("m", Some(SESSION), &request), &request),
        SnapshotLookupOutcome::NoCandidate
    ));
}

#[test]
fn divergence_does_not_inflate_the_snapshot_hit_counters() {
    let store = store_with_snapshots();
    let mut stored = run(1_000, 20);
    stored.extend(run(30_000, 20));
    insert(&store, "gemma4", &stored);

    let mut request = run(1_000, 20);
    request.extend(run(40_000, 25));
    let _ = outcome(&store, &key("m", Some(SESSION), &request), &request);

    let stats = store.stats();
    assert_eq!(stats.snapshot_lookups, 1, "the lookup is still counted");
    assert_eq!(stats.snapshot_hits, 0, "a divergence is a miss, not a hit");
}

// -- partial adoption at the longest common prefix (issue #1145) ----------
//
// The store never decides truncatability itself; it asks the model. These
// tests drive that seam directly with a stub capability, so they pin the
// store's half of the contract: which candidate is offered, at what length,
// and what happens when the answer is no. The rotating-cache rule the real
// capability implements lives in `mlxcel_core::cache::rotating_truncation_tests`.

/// Capability stub standing in for a rotating-attention model that can
/// truncate anywhere (Gemma 4 with every sliding layer unwrapped).
fn truncatable(
    store: &PromptCacheStore,
    k: &PromptCacheKey<'_>,
    t: &[i32],
) -> SnapshotLookupOutcome {
    store.lookup_snapshot_outcome(k, t, |_, _| true)
}

#[test]
fn a_diverging_candidate_is_adopted_at_the_longest_common_prefix() {
    let store = store_with_snapshots();
    let shared = run(1_000, 90);
    let mut stored = shared.clone();
    stored.extend([100, 45_518, 107, 101]);
    stored.extend(run(5_000, 45));
    insert(&store, "gemma4", &stored);

    let mut request = shared.clone();
    request.extend(run(5_000, 45));
    request.extend(run(7_000, 30));

    match truncatable(&store, &key("m", Some(SESSION), &request), &request) {
        SnapshotLookupOutcome::Hit { matched_len, entry } => {
            assert_eq!(matched_len, 90, "adopts exactly the common prefix");
            assert_eq!(
                entry.tokens.len(),
                139,
                "the stored entry is longer than what was adopted, which is how \
                 the caller knows to use the truncating restore"
            );
        }
        other => panic!("expected a partial adopt, got {other:?}"),
    }

    let stats = store.stats();
    assert_eq!(stats.snapshot_hits, 1, "a partial adopt counts as a hit");
    assert_eq!(stats.snapshot_lookups, 1);
}

#[test]
fn recurrent_families_keep_exact_prefix_semantics() {
    // Same store, same request, capability refused: this is the qwen3.5 /
    // falcon-h1 side of the class, where the recurrent state cannot be
    // rewound at all. It must stay a classified miss, never a partial adopt.
    let store = store_with_snapshots();
    let shared = run(1_000, 90);
    let mut stored = shared.clone();
    stored.extend(run(5_000, 49));
    insert(&store, "qwen3_5", &stored);

    let mut request = shared.clone();
    request.extend(run(7_000, 30));

    let report = diverged(outcome(
        &store,
        &key("m", Some(SESSION), &request),
        &request,
    ));
    assert_eq!(report.common_prefix_len, 90);
    assert_eq!(store.stats().snapshot_hits, 0);
}

#[test]
fn a_declining_model_leaves_the_wrapped_case_as_a_classified_reject() {
    // The capability is what a wrapped sliding layer reports. The decline has
    // to surface as the divergence reject rather than a silent None, so the
    // stats still say a candidate was there and was structurally unusable.
    let store = store_with_snapshots();
    let mut stored = run(1_000, 60);
    stored.extend(run(30_000, 40));
    insert(&store, "gemma4", &stored);

    let mut request = run(1_000, 60);
    request.extend(run(40_000, 25));

    let report = diverged(store.lookup_snapshot_outcome(
        &key("m", Some(SESSION), &request),
        &request,
        |_, _| false,
    ));
    assert_eq!(report.common_prefix_len, 60);
    assert_eq!(report.stored_len, 100);
    assert_eq!(store.stats().snapshot_hits, 0);
}

#[test]
fn an_exact_prefix_candidate_wins_over_a_truncating_one() {
    // Adopting whole is always better than adopting truncated: it covers more
    // tokens and needs no trim. A usable exact candidate must therefore
    // suppress the partial path even when the model would allow it.
    let store = store_with_snapshots();
    let usable = run(1_000, 40);
    insert(&store, "gemma4", &usable);
    let mut longer_but_diverging = run(1_000, 35);
    longer_but_diverging.extend(run(30_000, 60));
    insert(&store, "gemma4", &longer_but_diverging);

    let mut request = usable.clone();
    request.extend(run(20_000, 12));
    match truncatable(&store, &key("m", Some(SESSION), &request), &request) {
        SnapshotLookupOutcome::Hit { matched_len, entry } => {
            assert_eq!(matched_len, 40);
            assert_eq!(
                entry.tokens.len(),
                40,
                "the whole-entry candidate was adopted, not the truncating one"
            );
        }
        other => panic!("expected the exact-prefix hit, got {other:?}"),
    }
}

#[test]
fn a_common_prefix_below_min_prefix_tokens_is_not_adopted() {
    // min_prefix_tokens is 4 in this fixture. A 2-token agreement is not
    // worth a truncating restore, and the store must not ask the model to
    // perform one.
    let cfg = PromptCacheConfig::new(true, 1 << 20, 64, Duration::from_secs(3600), 8)
        .with_snapshot_limits(1 << 20, 64, Duration::from_secs(3600));
    let store = PromptCacheStore::with_config(cfg);
    let mut stored = run(1_000, 5);
    stored.extend(run(30_000, 40));
    store
        .insert_snapshot(
            &key("m", Some(SESSION), &stored),
            snapshot(stored.clone(), "gemma4"),
        )
        .expect("insert");

    let mut request = run(1_000, 5);
    request.extend(run(40_000, 30));

    let report = diverged(truncatable(
        &store,
        &key("m", Some(SESSION), &request),
        &request,
    ));
    assert_eq!(report.common_prefix_len, 5, "below the 8-token floor");
    assert_eq!(store.stats().snapshot_hits, 0);
}

#[test]
fn the_capability_is_asked_about_the_adopted_length_not_the_stored_length() {
    // The model must be able to answer "can you truncate to N", because that
    // is the question that decides correctness. Passing the stored length
    // instead would make the check vacuous.
    use std::sync::Mutex;
    let store = store_with_snapshots();
    let mut stored = run(1_000, 70);
    stored.extend(run(30_000, 30));
    insert(&store, "gemma4", &stored);

    let mut request = run(1_000, 70);
    request.extend(run(40_000, 25));

    let asked: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    let _ = store.lookup_snapshot_outcome(&key("m", Some(SESSION), &request), &request, |_, n| {
        asked.lock().expect("lock").push(n);
        true
    });
    assert_eq!(
        asked.into_inner().expect("lock"),
        vec![70],
        "asked about the common prefix, not the 100-token stored entry"
    );
}
