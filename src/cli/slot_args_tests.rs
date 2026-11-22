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

//! Unit tests for the b10621 slot-state and checkpoint flag group (#1473).
//!
//! These are the diagnostic tests the manifest's slot-state compatibility
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
fn ctx_checkpoints_accepts_a_count_and_zero_and_refuses_a_negative() {
    let mut ring = args();
    ring.ctx_checkpoints = Some(3);
    let flags = ring
        .resolve()
        .expect("a checkpoint count is accepted (#3255)");
    assert_eq!(flags.ctx_checkpoints, Some(3));

    let mut upstream_default = args();
    upstream_default.ctx_checkpoints = Some(32);
    assert!(upstream_default.resolve().is_ok());

    let mut none = args();
    none.ctx_checkpoints = Some(0);
    assert_eq!(none.resolve().unwrap().ctx_checkpoints, Some(0));

    let mut negative = args();
    negative.ctx_checkpoints = Some(-1);
    let message = negative.resolve().expect_err("a negative count is refused");
    assert!(message.contains("--ctx-checkpoints -1"), "{message}");
}

#[test]
fn kv_unified_resolves_to_b10621_auto_default() {
    assert!(
        args().resolve_kv_unified(true, false),
        "auto --parallel enables unified shared-budget context by default"
    );
    assert!(
        !args().resolve_kv_unified(false, false),
        "explicit --parallel keeps split context windows by default"
    );
    assert!(
        !args().resolve_kv_unified(true, true),
        "explicit --max-batch-size keeps split context windows by default"
    );
}

#[test]
fn kv_unified_explicit_flags_override_the_auto_default() {
    let mut explicit_on = args();
    explicit_on.kv_unified = Some(true);
    assert!(explicit_on.resolve().is_ok());
    assert!(explicit_on.resolve_kv_unified(false, true));

    let mut explicit_off = args();
    explicit_off.kv_unified = Some(false);
    assert!(explicit_off.resolve().is_ok());
    assert!(!explicit_off.resolve_kv_unified(true, false));

    let mut negated = args();
    negated.kv_unified = Some(true);
    negated.no_kv_unified = true;
    assert!(negated.resolve().is_ok());
    assert!(
        !negated.resolve_kv_unified(true, false),
        "--no-kv-unified must beat an environment-supplied true"
    );
}

#[test]
fn checkpoint_min_step_accepts_a_spacing_from_the_floor_and_zero() {
    let mut spacing = args();
    spacing.checkpoint_min_step = Some(8192);
    let flags = spacing
        .resolve()
        .expect("upstream's default spacing is accepted (#3255)");
    assert_eq!(flags.checkpoint_min_step, Some(8192));

    let mut floor = args();
    floor.checkpoint_min_step = Some(256);
    assert!(floor.resolve().is_ok());

    let mut none = args();
    none.checkpoint_min_step = Some(0);
    assert_eq!(none.resolve().unwrap().checkpoint_min_step, Some(0));

    let mut below = args();
    below.checkpoint_min_step = Some(100);
    let message = below
        .resolve()
        .expect_err("a spacing below the floor is refused, not clamped");
    assert!(message.contains("floor of 256"), "{message}");

    let mut negative = args();
    negative.checkpoint_min_step = Some(-8);
    assert!(negative.resolve().is_err());
}

#[test]
fn the_flags_override_the_environment_switches() {
    use crate::server::prompt_cache::{
        CtxCheckpointEnv, CtxCheckpointFlags, resolve_ctx_checkpoint_policy,
    };
    let env = CtxCheckpointEnv {
        boundary_disabled: false,
        keep_previous_turn: true,
        checkpoint_interval: Some(16_384),
    };
    let mut off = args();
    off.ctx_checkpoints = Some(0);
    off.checkpoint_min_step = Some(4096);
    let (policy, _) = resolve_ctx_checkpoint_policy(off.resolve().unwrap(), env, None);
    assert!(
        !policy.keep_previous_turn,
        "--ctx-checkpoints 0 beats MLXCEL_PROMPT_CACHE_KEEP_PREVIOUS_TURN"
    );
    assert_eq!(
        policy.checkpoint_interval,
        Some(4096),
        "--checkpoint-min-step beats MLXCEL_PROMPT_CACHE_CHECKPOINT_INTERVAL"
    );

    let (policy, _) = resolve_ctx_checkpoint_policy(CtxCheckpointFlags::default(), env, None);
    assert!(policy.keep_previous_turn);
    assert_eq!(
        policy.checkpoint_interval,
        Some(16_384),
        "unset flags leave the switches"
    );
}

#[test]
fn the_help_and_module_doc_no_longer_name_the_removed_capture_function() {
    let source = include_str!("slot_args.rs");
    assert!(!source.contains("capture_history_boundary_snapshot"));
    assert!(!source.contains("one snapshot per sequence"));
}
