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

//! Precedence tests for `--ctx-checkpoints` / `--checkpoint-min-step` over
//! the prompt-cache environment switches (issue #3255).

use super::*;

fn flags(ctx: Option<usize>, step: Option<usize>) -> CtxCheckpointFlags {
    CtxCheckpointFlags {
        ctx_checkpoints: ctx,
        checkpoint_min_step: step,
    }
}

const ENV_ON: CtxCheckpointEnv = CtxCheckpointEnv {
    boundary_disabled: false,
    keep_previous_turn: true,
    checkpoint_interval: Some(16_384),
};

#[test]
fn unset_flags_leave_the_environment() {
    let (policy, notices) =
        resolve_ctx_checkpoint_policy(CtxCheckpointFlags::default(), ENV_ON, None);
    assert!(policy.keep_previous_turn);
    assert_eq!(policy.checkpoint_interval, Some(16_384));
    assert!(!policy.state_checkpoints_off && !policy.ring_checkpoints);
    assert_eq!(policy.per_conversation, None);
    assert_eq!(policy.min_spacing_tokens, 0);
    assert!(notices.is_empty());

    let (policy, _) = resolve_ctx_checkpoint_policy(
        CtxCheckpointFlags::default(),
        CtxCheckpointEnv::default(),
        None,
    );
    assert_eq!(policy, CtxCheckpointPolicy::default());
}

#[test]
fn a_positive_count_turns_everything_on_and_caps_the_conversation() {
    let (policy, notices) =
        resolve_ctx_checkpoint_policy(flags(Some(3), None), CtxCheckpointEnv::default(), None);
    assert!(policy.keep_previous_turn, "N > 0 keeps the previous turn");
    assert!(policy.ring_checkpoints && !policy.state_checkpoints_off);
    assert_eq!(policy.per_conversation, Some(3));
    assert_eq!(
        policy.checkpoint_interval, None,
        "the count does not turn on periodic checkpoints"
    );
    assert!(notices.is_empty());
}

#[test]
fn zero_overrides_the_keep_previous_turn_switch_and_says_so() {
    let (policy, notices) = resolve_ctx_checkpoint_policy(flags(Some(0), None), ENV_ON, Some(2));
    assert!(!policy.keep_previous_turn);
    assert!(policy.state_checkpoints_off);
    assert!(!policy.ring_checkpoints);
    assert_eq!(
        policy.checkpoint_interval,
        Some(16_384),
        "--ctx-checkpoints leaves the interval switch alone"
    );
    assert!(
        matches!(notices.as_slice(), [CtxCheckpointNotice::Info(msg)] if msg.contains("state_checkpoint_capacity_bytes = 0"))
    );
}

#[test]
fn the_step_overrides_the_interval_switch() {
    let (policy, _) = resolve_ctx_checkpoint_policy(flags(None, Some(4096)), ENV_ON, None);
    assert_eq!(policy.checkpoint_interval, Some(4096));
    assert_eq!(policy.min_spacing_tokens, 4096);
    assert!(
        policy.keep_previous_turn,
        "--checkpoint-min-step leaves keep-previous-turn alone"
    );

    let (policy, _) = resolve_ctx_checkpoint_policy(flags(None, Some(0)), ENV_ON, None);
    assert_eq!(
        policy.checkpoint_interval, None,
        "0 means no periodic checkpoints"
    );
    assert_eq!(policy.min_spacing_tokens, 0, "and no spacing rule");
}

#[test]
fn the_family_default_applies_only_without_a_count() {
    let env = CtxCheckpointEnv::default();
    let (policy, _) = resolve_ctx_checkpoint_policy(CtxCheckpointFlags::default(), env, Some(2));
    assert!(policy.keep_previous_turn && policy.ring_checkpoints);
    assert_eq!(policy.per_conversation, Some(2));

    let (policy, _) = resolve_ctx_checkpoint_policy(flags(Some(5), None), env, Some(2));
    assert_eq!(policy.per_conversation, Some(5));
}

#[test]
fn the_boundary_kill_switch_wins_and_the_flags_warn() {
    let env = CtxCheckpointEnv {
        boundary_disabled: true,
        ..ENV_ON
    };
    let (policy, notices) = resolve_ctx_checkpoint_policy(flags(Some(3), Some(1024)), env, None);
    assert!(!policy.keep_previous_turn);
    assert!(policy.state_checkpoints_off);
    assert_eq!(policy.checkpoint_interval, None);
    assert!(
        matches!(notices.as_slice(), [CtxCheckpointNotice::Warn(msg)] if msg.contains("MLXCEL_DISABLE_BOUNDARY_SNAPSHOT"))
    );

    let (_, notices) = resolve_ctx_checkpoint_policy(CtxCheckpointFlags::default(), env, None);
    assert!(notices.is_empty(), "no flag, nothing to warn about");
}

#[test]
fn raw_values_are_validated() {
    assert_eq!(
        CtxCheckpointFlags::from_raw(Some(3), Some(256)),
        Ok(flags(Some(3), Some(256)))
    );
    assert_eq!(
        CtxCheckpointFlags::from_raw(None, None),
        Ok(flags(None, None))
    );
    assert_eq!(
        CtxCheckpointFlags::from_raw(Some(0), Some(0)),
        Ok(flags(Some(0), Some(0)))
    );
    let err = CtxCheckpointFlags::from_raw(Some(-1), None).unwrap_err();
    assert!(
        err.contains("--ctx-checkpoints -1") && err.contains("negative"),
        "{err}"
    );
    let err = CtxCheckpointFlags::from_raw(None, Some(-8)).unwrap_err();
    assert!(err.contains("--checkpoint-min-step -8"), "{err}");
    let err = CtxCheckpointFlags::from_raw(None, Some(255)).unwrap_err();
    assert!(err.contains("floor of 256"), "{err}");
}
