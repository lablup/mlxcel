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

//! Whole-prompt prompt-cache hits (issue #1760).
//!
//! A client that replays an identical prompt gets a cache hit covering every
//! prompt token. Prefill still has to forward the last one to produce a
//! sampling logit, so the adopt must install `len - 1` tokens and leave the
//! last for prefill. Before #1760 the adopt installed all `len` and admission
//! backed only the prefill cursor off, so the last token landed in the cache
//! twice and the warm reply diverged from the cold one.
//!
//! These tests read the restored per-layer cache offset itself, through the
//! model's own snapshot, rather than only the offset the scheduler reports.

use super::scheduler_model_owned_cache_tests::{
    cache_ctx, enqueue, options, prompt, run_to_completion, scheduler, scheduler_with_model,
    test_store, tiny_gemma3_args, tiny_gemma3_weights,
};
use super::*;
use mlxcel_core::generate::LanguageModel;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, mpsc};

use crate::models::{Gemma3Model, Gemma3Wrapper};
use crate::server::prompt_cache::PromptCacheRejectReason;

/// Per-layer cache offset of `seq_id`'s live model-owned state, read from the
/// snapshot the model itself would donate. The tiny fixture has one layer.
fn restored_offset(sched: &BatchScheduler, seq_id: SequenceId, kind: &str) -> i32 {
    let snapshot = sched
        .model
        .snapshot_sequence_state(seq_id, 0)
        .expect("restored sequence has model-owned state");
    let name = format!("layer0.{kind}.offset");
    let offset = snapshot
        .tensor(&name)
        .unwrap_or_else(|| panic!("snapshot carries {name}"));
    mlxcel_core::item_i32(offset)
}

/// The fixture with its only layer made a sliding one (window 8), so any
/// prompt longer than the window wraps the ring and the model refuses every
/// truncating restore.
fn tiny_sliding_gemma3() -> Gemma3Wrapper {
    let mut args = tiny_gemma3_args();
    args.sliding_window_pattern = 2;
    Gemma3Wrapper::new(
        Gemma3Model::from_weights(&tiny_gemma3_weights(), &args).expect("tiny gemma3 loads"),
    )
}

/// Turn 1 on `first`, returning the full stored conversation
/// (`first ++ generated`), which is the completion snapshot's token vector.
///
/// The tiny fixture samples EOS straight away, so `ignore_eos` with
/// `max_tokens == 1` is what makes the stored entry one token longer than the
/// prompt; without it the stored entry is the prompt itself.
fn donate_turn(sched: &mut BatchScheduler, first: &[i32], ignore_eos: bool) -> Vec<i32> {
    let (tx, rx) = mpsc::channel();
    let mut opts = options();
    opts.ignore_eos = ignore_eos;
    sched.enqueue_request(
        "prompt".to_string(),
        Some(first.to_vec()),
        opts,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        tx,
        Arc::new(AtomicBool::new(false)),
        true,
    );
    let result = run_to_completion(sched, &rx);
    let mut stored = first.to_vec();
    stored.extend_from_slice(&result.generated_token_ids);
    stored
}

/// An identical replay of turn 1's prompt: the stored completion entry is
/// longer than the prompt, so the store adopts by truncating to the prompt
/// length. The adopt must go one further and leave the last prompt token for
/// prefill, and the restored cache must hold exactly that many tokens.
#[test]
fn identical_prompt_replay_restores_all_but_the_last_token() {
    let mut sched = scheduler(test_store());
    let first = prompt(40);
    let stored = donate_turn(&mut sched, &first, true);
    assert_eq!(
        stored.len(),
        first.len() + 1,
        "turn 1 stored one generated token"
    );

    let (seq_id, adopted) = sched
        .try_adopt_cached_prefix(&cache_ctx(), &first, false)
        .expect("an identical replay hits the cache");
    assert_eq!(adopted, first.len() - 1, "adopt stops one token short");
    assert_eq!(
        restored_offset(&sched, seq_id, "standard"),
        (first.len() - 1) as i32,
        "the restored cache must not already hold the last prompt token"
    );
}

/// The exact-entry shape: the request is the stored conversation itself, so
/// the store matches the whole entry without truncating. The cap must turn
/// that into a truncating restore of `len - 1`, and the admission path must
/// then prefill exactly the one remaining token.
#[test]
fn whole_entry_hit_restores_len_minus_one_and_prefills_one_token() {
    let store = test_store();
    let mut sched = scheduler(store.clone());
    let replay = donate_turn(&mut sched, &prompt(40), false);

    let (seq_id, adopted) = sched
        .try_adopt_cached_prefix(&cache_ctx(), &replay, false)
        .expect("the stored conversation hits its own entry");
    assert_eq!(adopted, replay.len() - 1);
    assert_eq!(
        restored_offset(&sched, seq_id, "standard"),
        (replay.len() - 1) as i32
    );
    sched.release_sequence_caches(seq_id);

    // Through admission: the queued sequence starts prefill at `len - 1` and
    // reports the same count as cached, so `usage.cached_tokens` never claims
    // the token prefill re-ran.
    let _rx = enqueue(&mut sched, replay.clone());
    let queued = sched
        .prefill_queue
        .dequeue()
        .expect("the replay is queued for prefill");
    assert_eq!(queued.prefill_start_offset, replay.len() - 1);
    assert_eq!(queued.already_cached_tokens, replay.len() - 1);
    assert_eq!(
        restored_offset(&sched, queued.seq_id, "standard"),
        (replay.len() - 1) as i32
    );
}

/// A family that cannot truncate (here a wrapped sliding ring) must fall back
/// to a cold prefill on a whole-prompt hit, record the decline, and never
/// install the full entry for prefill to append the last token onto.
#[test]
fn untruncatable_whole_entry_hit_falls_back_to_cold_prefill() {
    let store = test_store();
    let mut sched = scheduler_with_model(tiny_sliding_gemma3(), store.clone());
    let replay = donate_turn(&mut sched, &prompt(40), false);
    assert_eq!(store.stats().snapshot_entries, 1, "turn 1 donated");

    let before = sched
        .batch_observability
        .prompt_cache_reject_reasons
        .count(PromptCacheRejectReason::LayoutConstraints);
    let adopted = sched.try_adopt_cached_prefix(&cache_ctx(), &replay, false);
    assert!(
        adopted.is_none(),
        "a wrapped ring cannot drop the last token, so the hit must decline"
    );
    assert_eq!(
        sched
            .batch_observability
            .prompt_cache_reject_reasons
            .count(PromptCacheRejectReason::LayoutConstraints),
        before + 1,
        "the decline is recorded"
    );

    // Admission then prefills the whole prompt cold.
    let _rx = enqueue(&mut sched, replay.clone());
    let queued = sched
        .prefill_queue
        .dequeue()
        .expect("the replay is queued for prefill");
    assert_eq!(queued.prefill_start_offset, 0);
    assert_eq!(queued.already_cached_tokens, 0);
}
