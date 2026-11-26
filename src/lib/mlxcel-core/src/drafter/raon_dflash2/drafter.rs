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

//! [`RaonDFlash2Drafter`]: the [`Drafter`] adapter around
//! [`RaonDFlash2DraftModel`].
//!
//! Lifecycle matches the other DFlash-family adapters: `load` reads the
//! checkpoint, `bind` borrows the target's `embed_tokens` and `lm_head`,
//! `draft_block` appends the newly committed target features to the
//! context and proposes the next block, `reset` clears the context between
//! generations. The draft is the selector's greedy chain, so it is
//! deterministic and needs no sampler; under sampling the round loop
//! samples the verify rows instead (`dflash::sampled_verify`).

use crate::drafter::{Drafter, DrafterError, DrafterKind};
use crate::ffi::{self, MlxArray};
use crate::generate::{LanguageModel, SamplingConfig};
use crate::weights::WeightMap;
use cxx::UniquePtr;
use std::path::Path;

use super::config::RaonDFlash2Config;
use super::model::{DFlash2State, RaonDFlash2DraftModel};
use super::sanitize::sanitize_weights;

/// Boxed [`Drafter`] for the Raon-1 mini DFlash2 speculator (B = 1).
pub struct RaonDFlash2Drafter {
    pub model: RaonDFlash2DraftModel,
    state: DFlash2State,
}

impl RaonDFlash2Drafter {
    /// Load the drafter checkpoint at `path`.
    ///
    /// # Errors
    /// [`DrafterError::ConfigIo`] / [`DrafterError::ConfigParse`] for an
    /// unreadable config, [`DrafterError::LoadFailed`] for an invalid config
    /// or weight inventory.
    pub fn load(path: &Path) -> Result<Self, DrafterError> {
        let config_path = path.join("config.json");
        let bytes = std::fs::read(&config_path).map_err(|e| DrafterError::ConfigIo {
            path: config_path.display().to_string(),
            source: e,
        })?;
        let json: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| DrafterError::ConfigParse {
                path: config_path.display().to_string(),
                source: e,
            })?;
        let config = RaonDFlash2Config::from_json(&json)
            .map_err(|reason| DrafterError::LoadFailed { reason })?;
        let mut weights = crate::weights::load_weights_from_dir(path)
            .map_err(|reason| DrafterError::LoadFailed { reason })?;
        sanitize_weights(&mut weights, &config)
            .map_err(|reason| DrafterError::LoadFailed { reason })?;
        Self::from_weights(&weights, config)
    }

    /// Build from an already sanitized weight map.
    ///
    /// # Errors
    /// [`DrafterError::LoadFailed`] naming a missing tensor.
    pub fn from_weights(
        weights: &WeightMap,
        config: RaonDFlash2Config,
    ) -> Result<Self, DrafterError> {
        let model = RaonDFlash2DraftModel::from_weights(weights, config)
            .map_err(|reason| DrafterError::LoadFailed { reason })?;
        let state = model.make_state();
        Ok(Self { model, state })
    }

    /// Absolute target position of the next context row.
    pub fn context_position(&self) -> i32 {
        self.state.ctx_pos
    }

    /// Context positions each layer holds.
    pub fn context_len(&self) -> i32 {
        self.state.windows.first().map_or(0, |w| w.len())
    }

    fn check_args<'a>(
        &self,
        hidden: Option<&'a MlxArray>,
        block_size: usize,
    ) -> Result<&'a MlxArray, DrafterError> {
        let trained = self.model.config.block_size;
        if !(2..=trained).contains(&block_size) {
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "Raon DFlash2 drafter serves blocks of 2..={trained} rows, its trained \
                     block (got {block_size}); block size 1 drafts nothing"
                ),
            });
        }
        if self.model.embed_tokens.is_none() {
            return Err(DrafterError::DraftFailed {
                reason: "Raon DFlash2 drafter is not bound to a target embedding table".to_string(),
            });
        }
        hidden.ok_or_else(|| DrafterError::DraftFailed {
            reason: "Raon DFlash2 drafter requires the target's captured mHC streams; got \
                     hidden = None"
                .to_string(),
        })
    }

    fn draft(
        &mut self,
        last_bonus: i32,
        hidden: &MlxArray,
        block_size: usize,
    ) -> Result<UniquePtr<MlxArray>, DrafterError> {
        self.model
            .add_context(hidden, &mut self.state)
            .map_err(|reason| DrafterError::DraftFailed { reason })?;
        let bonus = ffi::from_slice_i32(&[last_bonus], &[1]);
        Ok(self.model.propose(&bonus, block_size, &self.state))
    }
}

impl Drafter for RaonDFlash2Drafter {
    fn bind(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        let embed = target
            .embed_tokens_module()
            .ok_or_else(|| DrafterError::BindFailed {
                reason: "Raon DFlash2 drafter borrows the target's embedding table, but the \
                         target does not expose embed_tokens_module()"
                    .to_string(),
            })?;
        self.model.embed_tokens = Some(embed);
        self.model.lm_head = target.lm_head_module();
        Ok(())
    }

    fn validate_target_compat(&self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        let embed = target
            .embed_tokens_module()
            .ok_or_else(|| DrafterError::BindFailed {
                reason: "Raon DFlash2 drafter requires a target that exposes its embedding \
                         table"
                    .to_string(),
            })?;
        let shape = ffi::array_shape(embed.weight());
        self.model
            .config
            .validate_target(
                target.num_layers(),
                target.residual_stream_count(),
                shape[1] as usize,
                shape[0] as usize,
            )
            .map_err(DrafterError::Config)
    }

    fn reset(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        self.bind(target)?;
        self.state = self.model.make_state();
        Ok(())
    }

    fn dflash_target_layer_ids(&self) -> Option<&[usize]> {
        Some(&self.model.config.target_layer_ids)
    }

    fn configured_block_size(&self) -> Option<usize> {
        Some(self.model.config.block_size)
    }

    /// The vendor runtime drafts at 5 or 7 tokens; the requested width is
    /// the one the exactness gate probed, so the adaptive controller stays
    /// off.
    fn prefer_requested_block_size(&self) -> bool {
        true
    }

    /// The draft is the selector's greedy chain, a function of the context
    /// alone, so a sampling request is served by sampling each verify row
    /// and comparing, which keeps plain sampling's distribution
    /// (`dflash::sampled_verify`).
    fn supports_sampled_verify(&self) -> bool {
        true
    }

    fn is_raon_dflash2(&self) -> bool {
        true
    }

    /// The context window (`swa_window_size`): older prompt rows never
    /// reach `fc`, so the target captures only the newest this many.
    fn context_window(&self) -> Option<usize> {
        Some(self.model.config.window)
    }

    fn skip_context(&mut self, positions: usize) {
        let positions = i32::try_from(positions).unwrap_or(i32::MAX);
        self.model.skip_context(positions, &mut self.state);
    }

    fn draft_block(
        &mut self,
        last_bonus: i32,
        hidden: Option<&MlxArray>,
        block_size: usize,
        _sampler: &SamplingConfig,
    ) -> Result<Vec<i32>, DrafterError> {
        let hidden = self.check_args(hidden, block_size)?;
        let tokens = self.draft(last_bonus, hidden, block_size)?;
        crate::try_eval(&tokens).map_err(|e| DrafterError::DraftFailed {
            reason: e.to_string(),
        })?;
        Ok(crate::drafter::dflash::materialize_argmax_i32_vec(
            &tokens,
            block_size - 1,
        ))
    }

    fn draft_block_array(
        &mut self,
        last_bonus: i32,
        hidden: Option<&MlxArray>,
        block_size: usize,
        _sampler: &SamplingConfig,
    ) -> Result<UniquePtr<MlxArray>, DrafterError> {
        let hidden = self.check_args(hidden, block_size)?;
        self.draft(last_bonus, hidden, block_size)
    }

    fn sanitize(&mut self, weights: &mut WeightMap) -> Result<(), DrafterError> {
        sanitize_weights(weights, &self.model.config)
            .map_err(|reason| DrafterError::LoadFailed { reason })
    }

    fn kind(&self) -> DrafterKind {
        DrafterKind::Dflash
    }
}
