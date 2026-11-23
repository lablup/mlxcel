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

//! Scheduler entry points of the generic snapshot path (issue #3254).
//!
//! The prompt cache calls these instead of the model's snapshot hooks. A
//! scheduler-owned family takes [`StatePolicy`] over the state its pool entry
//! holds; every other family reaches the hook its wrapper implements, exactly
//! as before.

use mlxcel_core::cache::{CachePool, LayerState, SequenceId};
use mlxcel_core::generate::{LanguageModel, ModelStateSnapshot};

use super::StatePolicy;

/// How `model`'s snapshots are handled: `Ok(Some)` for a scheduler-owned
/// family that donates snapshots, `Ok(None)` for a family that keeps its own
/// hooks, and `Err` for a scheduler-owned family with no snapshot vocabulary,
/// whose state no hook can reach either.
fn policy_of<M: LanguageModel + ?Sized>(model: &M) -> Result<Option<StatePolicy>, String> {
    match (model.layer_state_kinds(), model.layer_state_snapshot_spec()) {
        (Some(kinds), Some(spec)) => Ok(Some(StatePolicy::new(spec, kinds))),
        (Some(_), None) => {
            Err("the model's scheduler-owned state has no snapshot encoding".to_string())
        }
        (None, _) => Ok(None),
    }
}

/// Fresh state for a restore, and the error when the model builds none.
fn fresh_state<M: LanguageModel + ?Sized>(model: &M) -> Result<Vec<LayerState>, String> {
    model
        .make_layer_state()
        .ok_or_else(|| "the model built no scheduler-owned layer state".to_string())
}

/// Install a restored state vector into `seq_id`'s pool entry.
fn install(pool: &mut CachePool, seq_id: SequenceId, state: Vec<LayerState>) -> Result<(), String> {
    pool.get_mut(seq_id)
        .ok_or_else(|| format!("sequence {seq_id} is not open"))?
        .install_layer_state(state)
}

/// [`LanguageModel::snapshot_truncatable_to`] for any family.
pub(crate) fn snapshot_truncatable_to<M: LanguageModel + ?Sized>(
    model: &M,
    snapshot: &ModelStateSnapshot,
    target_len: usize,
) -> bool {
    match policy_of(model) {
        Ok(Some(policy)) => policy.truncatable_to(snapshot, target_len),
        Ok(None) => model.snapshot_truncatable_to(snapshot, target_len),
        Err(_) => false,
    }
}

/// [`LanguageModel::snapshot_sequence_state`] for any family. A snapshot the
/// encoders refuse is logged and skipped, as the per-family hooks did.
pub(crate) fn snapshot_sequence_state<M: LanguageModel + ?Sized>(
    model: &M,
    pool: &CachePool,
    seq_id: SequenceId,
    token_len: usize,
) -> Option<ModelStateSnapshot> {
    let policy = match policy_of(model) {
        Ok(Some(policy)) => policy,
        Ok(None) => return model.snapshot_sequence_state(seq_id, token_len),
        Err(_) => return None,
    };
    match policy.snapshot(pool.layer_state(seq_id)?, token_len) {
        Ok(snapshot) => snapshot,
        Err(error) => {
            tracing::warn!(
                error,
                seq_id = %seq_id,
                family = policy.spec.family,
                "prompt-cache snapshot donation skipped"
            );
            None
        }
    }
}

/// [`LanguageModel::restore_sequence_state`] for any family.
pub(crate) fn restore_sequence_state<M: LanguageModel + ?Sized>(
    model: &M,
    pool: &mut CachePool,
    seq_id: SequenceId,
    snapshot: &ModelStateSnapshot,
) -> Result<(), String> {
    match policy_of(model)? {
        Some(policy) => {
            let state = policy.restore(fresh_state(model)?, snapshot)?;
            install(pool, seq_id, state)
        }
        None => model.restore_sequence_state(seq_id, snapshot),
    }
}

/// [`LanguageModel::restore_sequence_state_truncated`] for any family.
pub(crate) fn restore_sequence_state_truncated<M: LanguageModel + ?Sized>(
    model: &M,
    pool: &mut CachePool,
    seq_id: SequenceId,
    snapshot: &ModelStateSnapshot,
    target_len: usize,
) -> Result<(), String> {
    match policy_of(model)? {
        Some(policy) => {
            let state = policy.restore_truncated(fresh_state(model)?, snapshot, target_len)?;
            install(pool, seq_id, state)
        }
        None => model.restore_sequence_state_truncated(seq_id, snapshot, target_len),
    }
}

/// [`LanguageModel::state_checkpoint_from_snapshot`] for any family.
pub(crate) fn state_checkpoint_from_snapshot<M: LanguageModel + ?Sized>(
    model: &M,
    snapshot: &ModelStateSnapshot,
) -> Option<ModelStateSnapshot> {
    match policy_of(model) {
        Ok(Some(policy)) => policy.state_checkpoint(snapshot),
        Ok(None) => model.state_checkpoint_from_snapshot(snapshot),
        Err(_) => None,
    }
}

/// [`LanguageModel::restore_sequence_state_from_checkpoint`] for any family.
pub(crate) fn restore_sequence_state_from_checkpoint<M: LanguageModel + ?Sized>(
    model: &M,
    pool: &mut CachePool,
    seq_id: SequenceId,
    checkpoint: &ModelStateSnapshot,
    donor: &ModelStateSnapshot,
) -> Result<(), String> {
    match policy_of(model)? {
        Some(policy) => {
            let state = policy.restore_from_checkpoint(fresh_state(model)?, checkpoint, donor)?;
            install(pool, seq_id, state)
        }
        None => model.restore_sequence_state_from_checkpoint(seq_id, checkpoint, donor),
    }
}
