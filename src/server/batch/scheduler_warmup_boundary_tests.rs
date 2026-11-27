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

//! The boundary snapshot a background warm-up extends, past the sequence the
//! issue spells out (issue #3467; that one is
//! `a_retried_turn_restores_the_boundary_snapshot_its_warmup_extended`).
//!
//! Two things vary here. The model: the plain tiny Gemma 3 can restore a
//! longer snapshot truncated to its common prefix with the prompt (#1145), so
//! before the fix its retried turn still adopted something (the warmed entry
//! cut back to the boundary, or the completion snapshot cut back to all but
//! the last prompt token); the sliding one refuses that once its ring has
//! wrapped, like a real Gemma 3 past its window or a recurrent family, and
//! its retried turn prefilled from cold. The warm-up's restore source: a chat
//! turn stores no completion snapshot by default (#3040), so the warm-up
//! restores from the boundary snapshot itself. With
//! `MLXCEL_PROMPT_CACHE_DECODED_TAIL` on the turn also stores
//! `prompt + reply`, and on a template whose history render extends the
//! sampled reply that entry is the longest stored prefix of the warm-up
//! target, so the warm-up restores from it instead.

use std::time::Duration;

use super::scheduler_model_owned_cache_tests::{
    TURN_BOUNDARY, TURN_MAX_TOKENS, WarmTarget, cache_ctx, enqueue_turn, finish_turn,
    history_then_reply, queued, retry_after_warmup_restores_the_boundary, run_warmup,
    scheduler_with_model, test_store, tiny_gemma3, tiny_gemma3_args, tiny_gemma3_weights, turn,
};
use super::*;
use crate::models::{Gemma3Model, Gemma3Wrapper};
use crate::server::prompt_cache::decoded_tail::override_decoded_tail_for_test;
use crate::server::prompt_cache::{PromptCacheConfig, PromptCacheStore};

/// The generation prompt every turn ends with.
const GENERATION_PROMPT: [i32; 4] = [0, 1, 2, 3];
/// The next user message of the conversation.
const USER: [i32; 8] = [5, 4, 3, 2, 1, 0, 5, 4];
/// The end-of-turn token a history render puts after the reply. No prompt of
/// these tests holds it.
const TURN_END: i32 = 6;
/// Turns of the conversations whose entry counts are pinned.
const TURNS: usize = 8;

/// The fixture with its only layer made a sliding one (window 8): any prompt
/// longer than the window wraps the ring, and the model then refuses every
/// truncating restore, so only an exact-prefix snapshot can hit.
fn tiny_sliding_gemma3() -> Gemma3Wrapper {
    let mut args = tiny_gemma3_args();
    args.sliding_window_pattern = 2;
    Gemma3Wrapper::new(
        Gemma3Model::from_weights(&tiny_gemma3_weights(), &args).expect("tiny gemma3 loads"),
    )
}

/// The warm-up target of a template whose history render extends the sampled
/// reply: the whole prompt, the reply, and the end-of-turn token.
fn prompt_then_reply(tokens: &[i32], _boundary: usize, reply: &[i32]) -> Vec<i32> {
    let mut target = tokens.to_vec();
    target.extend_from_slice(reply);
    target.push(TURN_END);
    target
}

/// The stored completion vector of a turn: the prompt and the reply tokens
/// its state holds (decode never forwards the last sampled one, #1754).
fn completion_vector(tokens: &[i32], reply: &[i32]) -> Vec<i32> {
    let mut stored = tokens.to_vec();
    stored.extend_from_slice(&reply[..reply.len() - 1]);
    stored
}

/// Whether the store holds the turn's completion snapshot `stored`, which
/// then is the longest stored prefix of any target that extends it.
fn holds_completion_snapshot(
    sched: &BatchScheduler,
    store: &PromptCacheStore,
    stored: &[i32],
) -> bool {
    let ctx = cache_ctx();
    let key =
        BatchScheduler::compose_prompt_cache_key(&ctx, stored, sched.rope_regime_for(stored.len()));
    store
        .peek_snapshot_prefix(&key, stored)
        .is_some_and(|(entry, matched)| {
            matched == stored.len() && entry.origin == SnapshotOrigin::Completion
        })
}

#[test]
fn a_retried_turn_restores_its_boundary_on_a_model_that_cannot_truncate() {
    let (store, _sched, _) =
        retry_after_warmup_restores_the_boundary(tiny_sliding_gemma3, history_then_reply);
    assert_eq!(store.stats().snapshot_entries, 2);
}

/// With the end-of-generation snapshot on, the warm-up's restore source is
/// the completion snapshot, which is in another supersede chain than the
/// entry the warm-up stores. The boundary snapshot to keep is the one that
/// completion snapshot extends.
#[test]
fn a_warmup_restored_from_the_completion_snapshot_keeps_the_boundary() {
    let _tail = override_decoded_tail_for_test(true);
    for model in [tiny_gemma3 as fn() -> Gemma3Wrapper, tiny_sliding_gemma3] {
        let (store, sched, reply) =
            retry_after_warmup_restores_the_boundary(model, prompt_then_reply);
        let stored = completion_vector(&turn(), &reply);
        assert!(
            stored.len() > TURN_BOUNDARY
                && prompt_then_reply(&turn(), 0, &reply).starts_with(&stored),
            "the warm-up target extends the completion vector past the boundary"
        );
        assert!(
            holds_completion_snapshot(&sched, &store, &stored),
            "so the warm-up restored from the completion snapshot, not the boundary one"
        );
        assert_eq!(
            store.stats().snapshot_entries,
            3,
            "the boundary snapshot, the completion snapshot and the warmed prefix"
        );
    }
}

/// The next real turn restores the warmed prefix, and its own boundary insert
/// supersedes both boundary entries of the turn before, so the conversation
/// is back at one boundary snapshot.
#[test]
fn the_next_turn_restores_the_warmed_prefix_and_supersedes_both_boundaries() {
    // (end-of-generation snapshot, warm-up target, entries after turn 2)
    let cases: [(bool, WarmTarget, usize); 2] = [
        (false, history_then_reply, 1),
        // Turn 2's boundary and its completion snapshot, which superseded
        // turn 1's.
        (true, prompt_then_reply, 2),
    ];
    for (decoded_tail, warm_target, entries) in cases {
        let _tail = override_decoded_tail_for_test(decoded_tail);
        let (store, mut sched, reply) =
            retry_after_warmup_restores_the_boundary(tiny_sliding_gemma3, warm_target);
        let warmed = warm_target(&turn(), TURN_BOUNDARY, &reply);
        let supersedes = store.stats().snapshot_supersedes;

        let mut tokens = warmed.clone();
        tokens.extend(USER);
        let boundary = tokens.len();
        tokens.extend(GENERATION_PROMPT);
        let rx = enqueue_turn(&mut sched, &tokens, boundary);
        assert_eq!(
            queued(&sched).1,
            warmed.len(),
            "turn 2 restores the prefix the warm-up stored"
        );
        let _ = finish_turn(&mut sched, &rx);
        let stats = store.stats();
        assert_eq!(
            stats.snapshot_entries, entries,
            "one boundary snapshot is left after turn 2"
        );
        assert!(
            stats.snapshot_supersedes >= supersedes + 2,
            "turn 2's boundary insert superseded turn 1's boundary and its warmed prefix"
        );
    }
}

/// What a conversation of [`TURNS`] turns holds and restores.
#[derive(Debug, PartialEq)]
struct Conversation {
    /// Whole snapshots in the store after each turn.
    after_turn: Vec<usize>,
    /// Whole snapshots in the store after each turn's warm-up.
    after_warmup: Vec<usize>,
    /// Whether a retry of each turn, sent after that warm-up, restored the
    /// turn's history boundary.
    retry_restores_boundary: Vec<bool>,
}

impl Conversation {
    fn holding(after_turn: [usize; TURNS], after_warmup: [usize; TURNS]) -> Self {
        Self {
            after_turn: after_turn.to_vec(),
            after_warmup: after_warmup.to_vec(),
            retry_restores_boundary: vec![true; TURNS],
        }
    }
}

/// Run [`TURNS`] turns of one conversation on the sliding fixture, each
/// followed by its warm-up and then by a retry of the same turn. Every turn
/// after the first must restore the prefix the warm-up before it stored.
fn converse(store: &Arc<PromptCacheStore>, warm_target: WarmTarget) -> Conversation {
    let mut sched = scheduler_with_model(tiny_sliding_gemma3(), store.clone());
    let mut history = turn()[..TURN_BOUNDARY].to_vec();
    let mut warmed = 0;
    let mut seen = Conversation {
        after_turn: Vec::new(),
        after_warmup: Vec::new(),
        retry_restores_boundary: Vec::new(),
    };
    for n in 1..=TURNS {
        let mut tokens = history.clone();
        tokens.extend(GENERATION_PROMPT);
        let rx = enqueue_turn(&mut sched, &tokens, history.len());
        assert_eq!(
            queued(&sched).1,
            warmed,
            "turn {n} restores the prefix the warm-up before it stored"
        );
        let reply = finish_turn(&mut sched, &rx).generated_token_ids;
        assert_eq!(reply.len(), TURN_MAX_TOKENS);
        seen.after_turn.push(store.stats().snapshot_entries);

        let target = warm_target(&tokens, history.len(), &reply);
        run_warmup(&mut sched, &target);
        seen.after_warmup.push(store.stats().snapshot_entries);

        let rx = enqueue_turn(&mut sched, &tokens, history.len());
        seen.retry_restores_boundary
            .push(queued(&sched).1 == history.len());
        let _ = finish_turn(&mut sched, &rx);

        warmed = target.len();
        history = target;
        history.extend(USER);
    }
    seen
}

/// A long conversation holds one entry more than before #3467, and only
/// between a warm-up and the next turn: its boundary snapshot through each
/// turn, and that plus the warmed prefix after each warm-up. The count does
/// not grow with the number of turns.
#[test]
fn a_long_conversation_holds_one_more_entry_between_a_warmup_and_the_next_turn() {
    for warm_target in [history_then_reply as WarmTarget, prompt_then_reply] {
        assert_eq!(
            converse(&test_store(), warm_target),
            Conversation::holding([1; TURNS], [2; TURNS])
        );
    }
}

/// The same with the end-of-generation snapshot on, which adds the
/// completion snapshot to both counts. Where the completion vector prefixes
/// the next turn's (the first case) the completion chain supersedes too and
/// the counts are constant. Where it does not (the second case) that chain
/// never superseded, before or after #3467, and grows by one entry a turn
/// until the store's LRU bounds it (`the_entry_cap_bounds_the_kept_boundary`).
#[test]
fn a_long_conversation_with_completion_snapshots_holds_one_more_entry_too() {
    let _tail = override_decoded_tail_for_test(true);
    assert_eq!(
        converse(&test_store(), prompt_then_reply),
        Conversation::holding([2; TURNS], [3; TURNS])
    );
    assert_eq!(
        converse(&test_store(), history_then_reply),
        Conversation::holding([2, 3, 4, 5, 6, 7, 8, 9], [3, 4, 5, 6, 7, 8, 9, 10])
    );
}

/// With `keep_previous_turn` on (#3035) the conversation already held a
/// second entry through each turn, the previous turn's restore source, and
/// already kept the boundary its warm-up restored from (the warm-up's lookup
/// records it as the restore source). The kept boundary adds no third entry.
/// This passes before #3467 as well.
#[test]
fn keep_previous_turn_holds_two_entries_with_the_kept_boundary() {
    let store = Arc::new(PromptCacheStore::with_config(
        PromptCacheConfig::new(true, 1 << 20, 32, Duration::from_secs(600), 4)
            .with_keep_previous_turn(true),
    ));
    assert_eq!(
        converse(&store, history_then_reply),
        Conversation::holding([1, 2, 2, 2, 2, 2, 2, 2], [2; TURNS])
    );
}

/// The store's entry cap bounds the extra entry like any other: with room for
/// four snapshots the conversation never holds more, LRU drops the oldest
/// completion snapshots first, and every turn still restores its warmed
/// prefix and every retry its boundary.
#[test]
fn the_entry_cap_bounds_the_kept_boundary() {
    let _tail = override_decoded_tail_for_test(true);
    let store = Arc::new(PromptCacheStore::with_config(
        PromptCacheConfig::new(true, 1 << 20, 32, Duration::from_secs(600), 4)
            .with_snapshot_limits(1 << 20, 4, Duration::from_secs(600)),
    ));
    assert_eq!(
        converse(&store, history_then_reply),
        Conversation::holding([2, 3, 4, 4, 4, 4, 4, 4], [3, 4, 4, 4, 4, 4, 4, 4])
    );
}
