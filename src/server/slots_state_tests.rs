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

//! Unit tests for the b10621 slot registry (issue #1440).

use std::sync::Arc;

use super::{SlotCache, SlotRegistry};

fn registry(total: usize, retain: bool) -> Arc<SlotRegistry> {
    Arc::new(SlotRegistry::new(total, retain))
}

#[test]
fn slots_bind_lowest_free_id_and_release_on_drop() {
    let reg = registry(2, false);
    let a = reg.begin("p1", serde_json::json!({}), Some(8));
    let b = reg.begin("p2", serde_json::json!({}), Some(8));
    // Since #1440 a handle binds on its first progress signal, not at
    // creation, so a request still waiting in the scheduler queue does not
    // hold a slot away from one that is being served.
    assert_eq!(a.id_slot(), -1);
    a.on_prefill(4, 0);
    b.on_prefill(4, 0);
    assert_eq!(a.id_slot(), 0);
    assert_eq!(b.id_slot(), 1);
    assert_eq!(reg.idle_count(), 0);
    drop(a);
    assert_eq!(reg.idle_count(), 1);
    // The freed slot 0 is reused before any higher id.
    let c = reg.begin("p3", serde_json::json!({}), Some(8));
    c.on_prefill(4, 0);
    assert_eq!(c.id_slot(), 0);
}

#[test]
fn oversubscribed_request_is_unbound_then_binds_on_progress() {
    let reg = registry(1, false);
    let first = reg.begin("p1", serde_json::json!({}), None);
    // Nothing is bound until a request makes progress (#1440).
    assert_eq!(first.id_slot(), -1);
    first.on_token("a");
    assert_eq!(first.id_slot(), 0);
    let second = reg.begin("p2", serde_json::json!({}), None);
    // Every slot busy: the handle reports b10621's no-slot sentinel.
    assert_eq!(second.id_slot(), -1);
    drop(first);
    // Progress re-tries the bind, the analogue of b10621 assigning a slot to
    // a deferred task once one frees.
    second.on_token("x");
    assert_eq!(second.id_slot(), 0);
}

/// The regression #1440 fixed: a queued request must not hold a slot away
/// from one the worker is actually serving.
///
/// Measured against the pre-fix binary with `--parallel 2` and five
/// concurrent streams, one request emitted every frame with `id_slot: -1`
/// because a request still sitting in the scheduler queue had taken the slot
/// at route entry. b10621 never emits a frame without a slot: its task waits
/// until one is assigned. Binding on first progress restores that.
#[test]
fn a_queued_request_does_not_hold_a_slot_away_from_a_serving_one() {
    let reg = registry(1, false);
    let queued = reg.begin(
        "waiting in the scheduler queue",
        serde_json::json!({}),
        None,
    );
    let serving = reg.begin("being served right now", serde_json::json!({}), None);

    // The one that reaches the worker first gets the slot, whichever order
    // the routes ran in.
    serving.on_prefill(6, 0);
    assert_eq!(serving.id_slot(), 0);
    assert_eq!(queued.id_slot(), -1);
    assert_eq!(reg.idle_count(), 0);

    let slots = reg.slots_json(2048, false, false);
    assert_eq!(slots.len(), 1);
    assert_eq!(slots[0]["is_processing"], true);
    assert_eq!(slots[0]["n_prompt_tokens"], 6);
}

#[test]
fn task_counters_survive_release_like_task_prev() {
    let reg = registry(1, false);
    let handle = reg.begin("prompt", serde_json::json!({"n_predict": 4}), Some(4));
    handle.on_prefill(10, 3);
    handle.on_token("hello");
    handle.on_token("\n");
    handle.finish(10, 3, 2, "hello\nout");
    drop(handle);

    let slots = reg.slots_json(2048, false, false);
    assert_eq!(slots.len(), 1);
    let slot = &slots[0];
    assert_eq!(slot["id"], 0);
    assert_eq!(slot["is_processing"], false);
    assert_eq!(slot["n_prompt_tokens"], 10);
    assert_eq!(slot["n_prompt_tokens_cache"], 3);
    assert_eq!(slot["n_prompt_tokens_processed"], 7);
    let next = &slot["next_token"][0];
    assert_eq!(next["has_next_token"], false);
    assert_eq!(next["has_new_line"], true);
    assert_eq!(next["n_decoded"], 2);
    assert_eq!(next["n_remain"], 2);
}

#[test]
fn slots_json_matches_b10621_shape_for_idle_and_processing() {
    let reg = registry(2, false);
    let handle = reg.begin("p", serde_json::json!({"temperature": 0.5}), None);
    // The handle binds on its first progress signal (#1440), which is what
    // makes the slot report `is_processing`.
    handle.on_prefill(3, 0);
    let slots = reg.slots_json(4096, true, false);

    // Busy slot: b10621 field set, params echoed, no prompt/generated
    // without the debug switch.
    let busy = &slots[0];
    assert_eq!(busy["is_processing"], true);
    assert_eq!(busy["n_ctx"], 4096);
    assert_eq!(busy["speculative"], true);
    assert_eq!(busy["params"]["temperature"], 0.5);
    assert!(busy.get("prompt").is_none(), "prompt must be redacted");
    assert!(
        busy.get("generated").is_none(),
        "generated must be redacted"
    );
    // An unlimited budget reports n_remain -1 like b10621.
    assert_eq!(busy["next_token"][0]["n_remain"], -1);

    // Fresh idle slot: no task block at all, matching a b10621 slot that has
    // never carried a task.
    let idle = &slots[1];
    assert_eq!(idle["is_processing"], false);
    assert!(idle.get("id_task").is_none());
    drop(handle);
}

#[test]
fn debug_switch_exposes_prompt_and_generated_only_when_retained() {
    let reg = registry(1, true);
    let handle = reg.begin("secret prompt", serde_json::json!({}), None);
    handle.on_token("out");
    drop(handle);
    let slots = reg.slots_json(1024, false, true);
    assert_eq!(slots[0]["prompt"], "secret prompt");
    assert_eq!(slots[0]["generated"], "out");
}

#[test]
fn text_is_not_retained_without_the_debug_or_save_gate() {
    let reg = registry(1, false);
    let handle = reg.begin("secret prompt", serde_json::json!({}), None);
    handle.on_token("out");
    drop(handle);
    // Even a debug-view request cannot recover text that was never stored.
    let slots = reg.slots_json(1024, false, true);
    assert_eq!(slots[0]["prompt"], "");
    assert_eq!(slots[0]["generated"], "");
}

#[test]
fn id_task_increments_across_requests() {
    let reg = registry(1, false);
    let a = reg.begin("p", serde_json::json!({}), None);
    a.on_token("x");
    drop(a);
    let b = reg.begin("p", serde_json::json!({}), None);
    b.on_token("y");
    drop(b);
    let slots = reg.slots_json(1024, false, false);
    assert_eq!(slots[0]["id_task"], 1);
}

#[test]
fn restore_and_erase_manage_the_slot_cache() {
    let reg = registry(1, true);
    assert!(matches!(
        reg.install_restored(0, vec![1, 2, 3]),
        Some(Ok(()))
    ));
    let (cache, _, _) = reg.cache_for_save(0).expect("slot exists").expect("idle");
    assert_eq!(cache, SlotCache::Restored(vec![1, 2, 3]));
    let (erased, _, _) = reg.erase(0).expect("slot exists").expect("idle");
    assert_eq!(erased, SlotCache::Restored(vec![1, 2, 3]));
    let (after, _, _) = reg.cache_for_save(0).expect("slot exists").expect("idle");
    assert_eq!(after, SlotCache::Empty);
}

#[test]
fn busy_slot_refuses_cache_actions() {
    let reg = registry(1, true);
    let handle = reg.begin("p", serde_json::json!({}), None);
    // A slot is busy once the request it serves has made progress (#1440);
    // before that the request is still queued and holds no slot.
    handle.on_prefill(2, 0);
    assert!(matches!(reg.cache_for_save(0), Some(Err(()))));
    assert!(matches!(reg.install_restored(0, vec![1]), Some(Err(()))));
    assert!(matches!(reg.erase(0), Some(Err(()))));
    drop(handle);
    assert!(matches!(reg.cache_for_save(0), Some(Ok(_))));
}

#[test]
fn out_of_range_slot_reports_none() {
    let reg = registry(2, false);
    assert!(reg.cache_for_save(2).is_none());
    assert!(reg.install_restored(9, vec![]).is_none());
    assert!(reg.erase(9).is_none());
}
