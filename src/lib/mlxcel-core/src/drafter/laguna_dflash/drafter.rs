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

//! [`LagunaDFlashDrafter`]: the [`Drafter`] adapter around
//! [`LagunaDFlashDraftModel`] and its per-layer context caches.
//!
//! Lifecycle matches the Qwen 3.5 DFlash adapter: `load` reads the
//! checkpoint, `bind` borrows the target's `embed_tokens` and `lm_head`,
//! `draft_block` runs one masked forward, `reset` clears the context caches
//! between generations.

use crate::drafter::dflash::drafter::{sample_block_per_position, sample_block_per_position_array};
use crate::drafter::{Drafter, DrafterError, DrafterKind};
use crate::ffi::{self, MlxArray};
use crate::generate::{LanguageModel, SamplingConfig};
use crate::weights::WeightMap;
use cxx::UniquePtr;
use std::path::Path;

use super::cache::LagunaDFlashContextCache;
use super::config::LagunaDFlashConfig;
use super::model::LagunaDFlashDraftModel;
use super::sanitize::sanitize_weights;

/// Boxed [`Drafter`] for the Laguna DFlash drafter (B = 1).
pub struct LagunaDFlashDrafter {
    pub model: LagunaDFlashDraftModel,
    caches: Vec<LagunaDFlashContextCache>,
    bound: bool,
}

impl LagunaDFlashDrafter {
    /// Load the drafter checkpoint at `path`.
    pub fn load(path: &Path) -> Result<Self, DrafterError> {
        let config_path = path.join("config.json");
        let config_bytes = std::fs::read(&config_path).map_err(|e| DrafterError::ConfigIo {
            path: config_path.display().to_string(),
            source: e,
        })?;
        let config_json: serde_json::Value =
            serde_json::from_slice(&config_bytes).map_err(|e| DrafterError::ConfigParse {
                path: config_path.display().to_string(),
                source: e,
            })?;
        let config = LagunaDFlashConfig::from_json(&config_json)
            .map_err(|reason| DrafterError::LoadFailed { reason })?;
        let mut weights = crate::weights::load_weights_from_dir(path)
            .map_err(|reason| DrafterError::LoadFailed { reason })?;
        sanitize_weights(&mut weights, &config)
            .map_err(|reason| DrafterError::LoadFailed { reason })?;
        if drafter_weights_prefer_f16() {
            crate::drafter::dflash::drafter::convert_bf16_to_f16_non_quantized(&mut weights);
        }
        Self::from_weights(&weights, config)
    }

    /// Build from an already sanitized weight map.
    pub fn from_weights(
        weights: &WeightMap,
        config: LagunaDFlashConfig,
    ) -> Result<Self, DrafterError> {
        let model = LagunaDFlashDraftModel::from_weights(weights, config)
            .map_err(|reason| DrafterError::LoadFailed { reason })?;
        let caches = model.make_cache();
        Ok(Self {
            model,
            caches,
            bound: false,
        })
    }

    pub fn is_bound(&self) -> bool {
        self.bound
    }

    /// Per-layer context caches.
    pub fn context_caches(&self) -> &[LagunaDFlashContextCache] {
        &self.caches
    }

    fn build_block(&self, last_bonus: i32, block_size: usize) -> UniquePtr<MlxArray> {
        let mask_id = self.model.config.mask_token_id;
        let mut block = Vec::with_capacity(block_size);
        block.push(last_bonus);
        block.resize(block_size, mask_id);
        ffi::from_slice_i32(&block, &[1, block_size as i32])
    }

    fn check_draft_args(hidden: Option<&MlxArray>, block_size: usize) -> Result<(), DrafterError> {
        if hidden.is_none() {
            return Err(DrafterError::DraftFailed {
                reason: "Laguna DFlash drafter requires the target's captured hidden states; \
                         got hidden = None"
                    .to_string(),
            });
        }
        if !(2..=super::config::MAX_BLOCK_SIZE).contains(&block_size) {
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "Laguna DFlash drafter requires 2 <= block_size <= {} (got {block_size}); \
                     block size 1 has no masked position to sample and a wider block would \
                     size the verify logits and the cache slack by an unbounded value",
                    super::config::MAX_BLOCK_SIZE
                ),
            });
        }
        Ok(())
    }
}

/// Whether the drafter weights should be normalized bf16 -> f16 at load.
///
/// Follows the binary crate's `bf16_to_f16_at_load` for a non-quantized text
/// checkpoint so the drafter runs in the same activation dtype as the target
/// it is paired with: f16 on Apple Silicon and pre-Ampere CUDA, bf16 on
/// Ampere-and-later CUDA unless `MLXCEL_CUDA_F16_NORMALIZE` opts in;
/// `MLXCEL_KEEP_BF16` keeps bf16 everywhere.
pub(crate) fn drafter_weights_prefer_f16() -> bool {
    if std::env::var_os("MLXCEL_KEEP_BF16").is_some() {
        return false;
    }
    match crate::hardware::cuda_compute_capability() {
        Some((major, _)) if major < 8 => true,
        Some(_) => std::env::var("MLXCEL_CUDA_F16_NORMALIZE")
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                !(v.is_empty() || v == "0" || v == "false" || v == "no" || v == "off")
            })
            .unwrap_or(false),
        None => {
            crate::hardware::get_hardware().silicon_gen != crate::hardware::AppleSiliconGen::Unknown
        }
    }
}

impl Drafter for LagunaDFlashDrafter {
    fn bind(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        let embed = target
            .embed_tokens_module()
            .ok_or_else(|| DrafterError::BindFailed {
                reason: "Laguna DFlash drafter borrows the target's embedding table, but the \
                         target does not expose embed_tokens_module()"
                    .to_string(),
            })?;
        self.model.bind_target_embedding(embed);
        self.model.bind_target_lm_head(target.lm_head_module());
        self.bound = true;
        Ok(())
    }

    fn validate_target_compat(&self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        let embed = target
            .embed_tokens_module()
            .ok_or_else(|| DrafterError::BindFailed {
                reason: "Laguna DFlash drafter requires a target that exposes its embedding \
                         table"
                    .to_string(),
            })?;
        let vocab = ffi::array_shape(embed.weight())[0] as usize;
        self.model
            .config
            .validate_target(target.num_layers(), vocab)
            .map_err(DrafterError::Config)
    }

    fn reset(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        self.bind(target)?;
        self.caches = self.model.make_cache();
        Ok(())
    }

    fn dflash_target_layer_ids(&self) -> Option<&[usize]> {
        Some(&self.model.config.target_layer_ids)
    }

    fn configured_block_size(&self) -> Option<usize> {
        Some(self.model.config.block_size)
    }

    /// The checkpoint has one trained block width; a wider request must not
    /// engage the adaptive width controller, whose alternate width the
    /// exactness gate never probed.
    fn prefer_requested_block_size(&self) -> bool {
        true
    }

    fn draft_block(
        &mut self,
        last_bonus: i32,
        hidden: Option<&MlxArray>,
        block_size: usize,
        sampler: &SamplingConfig,
    ) -> Result<Vec<i32>, DrafterError> {
        Self::check_draft_args(hidden, block_size)?;
        let inputs = self.build_block(last_bonus, block_size);
        let logits = self
            .model
            .forward(&inputs, hidden.expect("checked"), &mut self.caches);
        sample_block_per_position(&logits, block_size, sampler)
    }

    fn draft_block_array(
        &mut self,
        last_bonus: i32,
        hidden: Option<&MlxArray>,
        block_size: usize,
        sampler: &SamplingConfig,
    ) -> Result<UniquePtr<MlxArray>, DrafterError> {
        Self::check_draft_args(hidden, block_size)?;
        let inputs = self.build_block(last_bonus, block_size);
        let logits = self
            .model
            .forward(&inputs, hidden.expect("checked"), &mut self.caches);
        sample_block_per_position_array(&logits, block_size, sampler)
    }

    fn sanitize(&mut self, weights: &mut WeightMap) -> Result<(), DrafterError> {
        sanitize_weights(weights, &self.model.config)
            .map_err(|reason| DrafterError::LoadFailed { reason })
    }

    fn is_laguna_dflash(&self) -> bool {
        true
    }

    /// The DFlash round loop verifies with a per-position argmax and has no
    /// stochastic acceptance rule; a sampling request is served classically.
    fn greedy_only(&self) -> bool {
        true
    }

    fn kind(&self) -> DrafterKind {
        DrafterKind::Dflash
    }
}
