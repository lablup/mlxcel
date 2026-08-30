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

//! Unit tests for the b10621 slot-state and checkpoint flag group (#1473).
//!
//! These are the diagnostic tests the manifest's five `not_applicable`
//! entries name: each asserts that the inert value is accepted and that a
//! request for the behavior is refused with a message that names the missing
//! structure rather than with clap's unknown-argument error.

use super::SlotCompatArgs;

fn args() -> SlotCompatArgs {
    SlotCompatArgs::default()
}

#[test]
fn the_default_group_is_accepted() {
    assert!(args().resolve().is_ok());
}

#[test]
fn cache_idle_slots_is_refused_when_asked_for_and_inert_when_disabled() {
    let mut on = args();
    on.cache_idle_slots = Some(true);
    let message = on.resolve().expect_err("an idle-slot save must be refused");
    assert!(message.contains("--cache-idle-slots"));
    assert!(message.contains("per-slot prompt"));

    let mut off = args();
    off.cache_idle_slots = Some(false);
    assert!(off.resolve().is_ok());

    let mut negated = args();
    negated.cache_idle_slots = Some(true);
    negated.no_cache_idle_slots = true;
    assert!(
        negated.resolve().is_ok(),
        "--no-cache-idle-slots must beat an environment-supplied true"
    );
}

#[test]
fn slot_prompt_similarity_refuses_upstreams_own_default() {
    let mut upstream_default = args();
    upstream_default.slot_prompt_similarity = Some(0.10);
    let message = upstream_default
        .resolve()
        .expect_err("a similarity threshold must be refused");
    assert!(message.contains("--slot-prompt-similarity"));
    assert!(message.contains("prefix trie"));

    let mut disabled = args();
    disabled.slot_prompt_similarity = Some(0.0);
    assert!(disabled.resolve().is_ok());
}

#[test]
fn kv_unified_is_refused_and_its_negation_is_inert() {
    let mut on = args();
    on.kv_unified = Some(true);
    let message = on
        .resolve()
        .expect_err("a unified KV buffer must be refused");
    assert!(message.contains("--kv-unified"));
    assert!(message.contains("per sequence"));

    let mut off = args();
    off.kv_unified = Some(false);
    assert!(off.resolve().is_ok());
}

#[test]
fn ctx_checkpoints_refuses_a_ring_and_accepts_zero() {
    let mut ring = args();
    ring.ctx_checkpoints = Some(32);
    let message = ring
        .resolve()
        .expect_err("a checkpoint ring must be refused");
    assert!(message.contains("--ctx-checkpoints"));
    assert!(message.contains("one snapshot per"));

    let mut none = args();
    none.ctx_checkpoints = Some(0);
    assert!(none.resolve().is_ok());
}

#[test]
fn checkpoint_min_step_refuses_a_spacing_and_accepts_zero() {
    let mut spacing = args();
    spacing.checkpoint_min_step = Some(8192);
    let message = spacing
        .resolve()
        .expect_err("a checkpoint spacing must be refused");
    assert!(message.contains("--checkpoint-min-step"));
    assert!(message.contains("--ctx-checkpoints"));

    let mut none = args();
    none.checkpoint_min_step = Some(0);
    assert!(none.resolve().is_ok());
}
