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

//! Per-sequence reasoning effort of Raon-1 mini (#3435).
//!
//! The vendor runtimes keep the effort per request (vLLM) or per cache
//! (`raon_mlx_model.py`'s `EffortCache`), so batched decode steers each row
//! with its own alpha. A sequence with no setting of its own runs at the
//! model default: the config's `default` level, or the level a caller set
//! for the model as a whole (`mlxcel generate --reasoning-effort`). The
//! server sets every admitted sequence explicitly, so the model default
//! never leaks from one request into another there.

use super::super::reasoning_effort_steering::ReasoningEffortSteering;
use super::{Ffn, Raon1MiniModel};
use mlxcel_core::cache::SequenceId;
use mlxcel_core::{MlxArray, UniquePtr};
use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

/// A resolved setting: the normalized level and its alpha.
#[derive(Debug, Clone, PartialEq)]
struct Setting {
    level: String,
    alpha: f32,
}

#[derive(Debug)]
struct Inner {
    default: Setting,
    per_sequence: HashMap<SequenceId, Setting>,
}

/// The steering config plus the effort each sequence runs at.
#[derive(Debug)]
pub(crate) struct EffortState {
    steering: Option<ReasoningEffortSteering>,
    inner: Mutex<Inner>,
}

impl EffortState {
    /// # Errors
    /// Returns the message for a steering block whose default does not resolve.
    pub(crate) fn new(steering: Option<ReasoningEffortSteering>) -> Result<Self, String> {
        let default = match &steering {
            Some(s) => Setting {
                level: s.default.clone(),
                alpha: s.alpha_for(None)?,
            },
            None => Setting {
                level: String::new(),
                alpha: 0.0,
            },
        };
        Ok(Self {
            steering,
            inner: Mutex::new(Inner {
                default,
                per_sequence: HashMap::new(),
            }),
        })
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(crate) fn steering(&self) -> Option<&ReasoningEffortSteering> {
        self.steering.as_ref()
    }

    /// Set `seq`'s effort (`None`: the model default for sequences without
    /// their own). `level == None` selects the config default for the model
    /// and clears a sequence's own setting. Without a steering block every
    /// level is ignored, as for any model that does not steer.
    ///
    /// # Errors
    /// Returns the client-facing message for an unknown level.
    pub(crate) fn set(&self, seq: Option<SequenceId>, level: Option<&str>) -> Result<(), String> {
        let Some(steering) = &self.steering else {
            return Ok(());
        };
        let setting = Setting {
            level: match level {
                Some(raw) => steering.normalize_level(raw)?,
                None => steering.default.clone(),
            },
            alpha: steering.alpha_for(level)?,
        };
        let mut inner = self.lock();
        match (seq, level) {
            (None, _) => inner.default = setting,
            (Some(id), None) => {
                inner.per_sequence.remove(&id);
            }
            (Some(id), Some(_)) => {
                inner.per_sequence.insert(id, setting);
            }
        }
        Ok(())
    }

    /// The level `seq` set for itself, if any.
    pub(crate) fn level(&self, seq: SequenceId) -> Option<String> {
        self.lock()
            .per_sequence
            .get(&seq)
            .map(|setting| setting.level.clone())
    }

    /// Drop `seq`'s own setting (open and close of a sequence id).
    pub(crate) fn forget(&self, seq: SequenceId) {
        self.lock().per_sequence.remove(&seq);
    }

    /// The alpha `seq` runs at (`None`: the model default).
    pub(crate) fn alpha(&self, seq: Option<SequenceId>) -> f32 {
        let inner = self.lock();
        seq.and_then(|id| inner.per_sequence.get(&id))
            .unwrap_or(&inner.default)
            .alpha
    }
}

impl Raon1MiniModel {
    /// `[B, 1, 1]` float32 alphas for rows at `alphas`, or `None` when the
    /// model carries no steering or every row is at 0 (the add is skipped).
    pub(crate) fn steer_rows(&self, alphas: &[f32]) -> Option<UniquePtr<MlxArray>> {
        if self.effort.steering().is_none() || alphas.iter().all(|&a| a == 0.0) {
            return None;
        }
        Some(mlxcel_core::from_slice_f32(
            alphas,
            &[alphas.len() as i32, 1, 1],
        ))
    }

    /// The reasoning-effort steering block this model was loaded with.
    pub fn reasoning_effort_config(&self) -> Option<&ReasoningEffortSteering> {
        self.effort.steering()
    }

    /// Steering vectors of the steered layers, `(layer, [hidden] float32)`.
    pub fn reasoning_effort_steering(&self) -> Vec<(usize, &MlxArray)> {
        self.layers
            .iter()
            .filter_map(|layer| match &layer.ffn {
                Ffn::Moe(moe) => moe.steering_vector().map(|v| (layer.layer_idx, v)),
                Ffn::Dense(_) => None,
            })
            .collect()
    }

    /// Set `seq`'s reasoning effort, or the model default (`seq == None`);
    /// see [`EffortState::set`].
    ///
    /// # Errors
    /// Returns the client-facing message for an unknown level.
    pub fn set_reasoning_effort(
        &self,
        seq: Option<SequenceId>,
        level: Option<&str>,
    ) -> Result<(), String> {
        self.effort.set(seq, level)
    }

    /// The alpha `seq` runs at (`None`: the model default).
    pub fn reasoning_effort_alpha(&self, seq: Option<SequenceId>) -> f32 {
        self.effort.alpha(seq)
    }

    /// The level `seq` set for itself, if any.
    pub fn reasoning_effort_level(&self, seq: SequenceId) -> Option<String> {
        self.effort.level(seq)
    }

    pub(crate) fn forget_reasoning_effort(&self, seq: SequenceId) {
        self.effort.forget(seq);
    }

    /// `[B, 1, 1]` alphas for rows owned by `ids`.
    pub(crate) fn steer_sequences(&self, ids: &[SequenceId]) -> Option<UniquePtr<MlxArray>> {
        let alphas: Vec<f32> = ids.iter().map(|&id| self.effort.alpha(Some(id))).collect();
        self.steer_rows(&alphas)
    }
}
