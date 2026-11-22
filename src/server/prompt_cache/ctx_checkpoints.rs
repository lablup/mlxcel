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

//! `--ctx-checkpoints` and `--checkpoint-min-step` (llama-server b10621,
//! issue #3255), resolved against the prompt-cache environment switches.
//!
//! llama-server keeps a ring of per-slot context checkpoints. mlxcel's
//! equivalents are already in the prompt-cache store, behind environment
//! switches: keeping the previous turn's restore point (#3035), turn-boundary
//! state checkpoints (#3047, ring-only for the sliding-window families since
//! #3255), and periodic prefill checkpoints at absolute multiples of an
//! interval (#2973). The two flags map onto them:
//!
//! * `--ctx-checkpoints N` with `N > 0` turns on keep-previous-turn and the
//!   state checkpoints the model supports, and keeps at most `N` state
//!   checkpoints per conversation. `N == 0` turns both off. Unset leaves the
//!   environment and model defaults.
//! * `--checkpoint-min-step M` with `M > 0` sets the periodic checkpoint
//!   interval and keeps no two state checkpoints of one conversation closer
//!   than `M` tokens (the newer replaces the older). `M == 0` turns periodic
//!   checkpoints off and sets no spacing. Unset leaves
//!   `MLXCEL_PROMPT_CACHE_CHECKPOINT_INTERVAL`.
//!
//! A flag overrides `MLXCEL_PROMPT_CACHE_KEEP_PREVIOUS_TURN` and
//! `MLXCEL_PROMPT_CACHE_CHECKPOINT_INTERVAL`.
//! `MLXCEL_PROMPT_CACHE_STATE_CHECKPOINT_CAPACITY_BYTES` still sets the byte
//! budget, and `MLXCEL_DISABLE_BOUNDARY_SNAPSHOT=1` still turns all of it off.
//!
//! Used by: `server::startup::resolve_prompt_cache_store`,
//! `cli::slot_args::SlotCompatArgs::resolve`.

use std::path::Path;

/// State checkpoints per conversation the sliding-window families (Gemma 3,
/// Gemma 4, AFMoE) get with `--ctx-checkpoints` unset. `None` keeps them on
/// today's default (no ring checkpoints): see
/// `docs/benchmark_results/gemma4-ctx-checkpoints-replay-gb10-2026-11-23.md`
/// for the measurement this rests on.
pub const RING_FAMILY_DEFAULT_CTX_CHECKPOINTS: Option<usize> = None;

/// The `--ctx-checkpoints` default for the model at `model_dir`:
/// [`RING_FAMILY_DEFAULT_CTX_CHECKPOINTS`] for a ring-checkpoint family,
/// `None` otherwise.
pub fn family_default_ctx_checkpoints(model_dir: &Path) -> Option<usize> {
    RING_FAMILY_DEFAULT_CTX_CHECKPOINTS
        .filter(|_| super::snapshot_sizing::model_stores_ring_checkpoints(model_dir))
}

/// Smallest `--checkpoint-min-step` accepted above zero: the floor
/// `MLXCEL_PROMPT_CACHE_CHECKPOINT_INTERVAL` applies, since the value is also
/// the periodic checkpoint interval and each point is one more prefill split.
pub const CHECKPOINT_MIN_STEP_FLOOR: usize = 256;

/// The two flags as the operator gave them, validated (issue #3255).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CtxCheckpointFlags {
    /// `--ctx-checkpoints`, or `LLAMA_ARG_CTX_CHECKPOINTS`.
    pub ctx_checkpoints: Option<usize>,
    /// `--checkpoint-min-step`, or `LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT`.
    pub checkpoint_min_step: Option<usize>,
}

impl CtxCheckpointFlags {
    /// Validate the raw values clap parsed.
    ///
    /// # Errors
    ///
    /// A negative value, or a `--checkpoint-min-step` between 1 and 255.
    pub fn from_raw(
        ctx_checkpoints: Option<i64>,
        checkpoint_min_step: Option<i64>,
    ) -> Result<Self, String> {
        let ctx_checkpoints = ctx_checkpoints
            .map(|n| {
                usize::try_from(n).map_err(|_| {
                    format!(
                        "--ctx-checkpoints {n} is negative. Pass the number of state checkpoints to keep per conversation, or 0 to turn them off"
                    )
                })
            })
            .transpose()?;
        let checkpoint_min_step = checkpoint_min_step
            .map(|m| {
                let step = usize::try_from(m).map_err(|_| {
                    format!(
                        "--checkpoint-min-step {m} is negative. Pass a token spacing of at least {CHECKPOINT_MIN_STEP_FLOOR}, or 0 for none"
                    )
                })?;
                if step > 0 && step < CHECKPOINT_MIN_STEP_FLOOR {
                    return Err(format!(
                        "--checkpoint-min-step {step} is below the floor of {CHECKPOINT_MIN_STEP_FLOOR} tokens. It also sets the periodic prefill checkpoint interval, and each checkpoint is one more prefill split, so the same floor as MLXCEL_PROMPT_CACHE_CHECKPOINT_INTERVAL applies. Pass at least {CHECKPOINT_MIN_STEP_FLOOR}, or 0 for none"
                    ));
                }
                Ok(step)
            })
            .transpose()?;
        Ok(Self {
            ctx_checkpoints,
            checkpoint_min_step,
        })
    }

    /// Whether either flag was given.
    pub fn any(&self) -> bool {
        self.ctx_checkpoints.is_some() || self.checkpoint_min_step.is_some()
    }
}

/// The environment switches the flags override, read by the caller.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CtxCheckpointEnv {
    /// `MLXCEL_DISABLE_BOUNDARY_SNAPSHOT=1`.
    pub boundary_disabled: bool,
    /// `MLXCEL_PROMPT_CACHE_KEEP_PREVIOUS_TURN`.
    pub keep_previous_turn: bool,
    /// `MLXCEL_PROMPT_CACHE_CHECKPOINT_INTERVAL`.
    pub checkpoint_interval: Option<usize>,
}

/// What the store and scheduler run with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CtxCheckpointPolicy {
    /// [`super::PromptCacheConfig::keep_previous_turn`].
    pub keep_previous_turn: bool,
    /// State checkpoints are off: the budget is zero whatever the model or
    /// the byte override says.
    pub state_checkpoints_off: bool,
    /// Size a budget for the ring-only checkpoints of a sliding-window
    /// family. The hybrid families get theirs either way.
    pub ring_checkpoints: bool,
    /// [`super::PromptCacheConfig::state_checkpoints_per_conversation`].
    pub per_conversation: Option<usize>,
    /// [`super::PromptCacheConfig::state_checkpoint_min_spacing_tokens`].
    pub min_spacing_tokens: usize,
    /// The periodic prefill checkpoint interval.
    pub checkpoint_interval: Option<usize>,
}

/// A startup log line the resolution asks for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CtxCheckpointNotice {
    /// Logged at `info`.
    Info(String),
    /// Logged at `warn`.
    Warn(String),
}

/// Resolve the flags against the environment.
///
/// `family_default` is the per-conversation state checkpoint count the
/// model's family gets with both flags unset, if any.
pub fn resolve_ctx_checkpoint_policy(
    flags: CtxCheckpointFlags,
    env: CtxCheckpointEnv,
    family_default: Option<usize>,
) -> (CtxCheckpointPolicy, Vec<CtxCheckpointNotice>) {
    let mut notices = Vec::new();
    if env.boundary_disabled {
        if flags.any() {
            notices.push(CtxCheckpointNotice::Warn(
                "--ctx-checkpoints and --checkpoint-min-step have no effect: MLXCEL_DISABLE_BOUNDARY_SNAPSHOT=1 turns off every prompt-cache checkpoint".to_string(),
            ));
        }
        return (
            CtxCheckpointPolicy {
                state_checkpoints_off: true,
                ..CtxCheckpointPolicy::default()
            },
            notices,
        );
    }
    let count = flags.ctx_checkpoints.or(family_default);
    let (keep_previous_turn, state_checkpoints_off, ring_checkpoints, per_conversation) =
        match count {
            Some(0) => {
                notices.push(CtxCheckpointNotice::Info(
                    "--ctx-checkpoints 0: turn-boundary state checkpoints are off (state_checkpoint_capacity_bytes = 0), including for the hybrid families that store them by default, and the previous turn's restore point is not kept".to_string(),
                ));
                (false, true, false, None)
            }
            Some(n) => (true, false, true, Some(n)),
            None => (env.keep_previous_turn, false, false, None),
        };
    let (checkpoint_interval, min_spacing_tokens) = match flags.checkpoint_min_step {
        Some(0) => (None, 0),
        Some(step) => (Some(step), step),
        None => (env.checkpoint_interval, 0),
    };
    (
        CtxCheckpointPolicy {
            keep_previous_turn,
            state_checkpoints_off,
            ring_checkpoints,
            per_conversation,
            min_spacing_tokens,
            checkpoint_interval,
        },
        notices,
    )
}

#[cfg(test)]
#[path = "ctx_checkpoints_tests.rs"]
mod tests;
