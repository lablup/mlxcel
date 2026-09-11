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

//! [`MuseAssistantDrafter`]: the [`Drafter`] adapter of the Muse Glimmer
//! assistant model (issue #1343), the object `load_drafter` hands the DFlash
//! round loop for a `muse_glimmer_assistant` checkpoint.

use std::path::Path;

use crate::drafter::{Drafter, DrafterError, DrafterKind};
use crate::ffi::{self, MlxArray};
use crate::generate::{LanguageModel, SamplingConfig};
use crate::weights::WeightMap;
use cxx::UniquePtr;

use super::cache::MuseAssistantContextCache;
use super::config::{MUSE_ASSISTANT_INITIAL_BLOCK_SIZE, MuseAssistantConfig};
use super::model::{MuseAssistantModel, block_input};
use crate::drafter::dflash::drafter::{
    apply_drafter_load_dtype_policy, sample_block_per_position_array,
};

/// Boxed [`Drafter`] for the Muse Glimmer assistant. Owns the model and its
/// per-layer context windows; `reset` rebuilds the windows between runs.
pub struct MuseAssistantDrafter {
    pub model: MuseAssistantModel,
    caches: Vec<MuseAssistantContextCache>,
    bound: bool,
}

impl MuseAssistantDrafter {
    /// Load the checkpoint at `path`: `config.json`, every safetensors
    /// shard, the `model.` strip, bf16 to f16 on the dense tensors, then
    /// the model with empty context windows.
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
        let config = MuseAssistantConfig::from_json(&config_json).map_err(|e| {
            DrafterError::ConfigParse {
                path: config_path.display().to_string(),
                source: serde::de::Error::custom(e),
            }
        })?;

        let mut weights = crate::weights::load_weights_from_dir(path)
            .map_err(|msg| DrafterError::LoadFailed { reason: msg })?;
        MuseAssistantModel::sanitize(&mut weights);
        apply_drafter_load_dtype_policy(&mut weights);
        let model = MuseAssistantModel::from_weights(&weights, config)
            .map_err(|msg| DrafterError::LoadFailed { reason: msg })?;
        Ok(Self::from_model(model))
    }

    /// Wrap an already-built model (tests, and any in-process caller).
    pub fn from_model(model: MuseAssistantModel) -> Self {
        let caches = model.make_caches();
        Self {
            model,
            caches,
            bound: false,
        }
    }

    pub fn is_bound(&self) -> bool {
        self.bound
    }

    /// The per-layer context windows, for tests pinning the append-only
    /// invariant.
    pub fn caches(&self) -> &[MuseAssistantContextCache] {
        &self.caches
    }

    fn ensure_block_size(block_size: usize) -> Result<(), DrafterError> {
        if block_size < 2 {
            return Err(DrafterError::DraftFailed {
                reason: format!(
                    "Muse Glimmer assistant requires block_size >= 2 (got {block_size}); block \
                     size 1 has no masked slot to propose"
                ),
            });
        }
        Ok(())
    }

    fn logits_for(
        &mut self,
        last_bonus: i32,
        hidden: Option<&MlxArray>,
        block_size: usize,
    ) -> Result<UniquePtr<MlxArray>, DrafterError> {
        let target_hidden = hidden.ok_or_else(|| DrafterError::DraftFailed {
            reason: "Muse Glimmer assistant requires the target's captured residual streams \
                     (target_layer_ids concatenation); got hidden = None"
                .to_string(),
        })?;
        Self::ensure_block_size(block_size)?;
        if !self.bound {
            return Err(DrafterError::DraftFailed {
                reason: "Muse Glimmer assistant draft_block called before bind".to_string(),
            });
        }
        let block = block_input(last_bonus, self.model.config.mask_token_id, block_size);
        Ok(self.model.forward(&block, target_hidden, &mut self.caches))
    }
}

/// The target-side facts the pairing check reads.
struct TargetShape {
    hidden: usize,
    layers: usize,
    vocab: Option<usize>,
}

/// Compare the drafter config against the measured target. `None` when the
/// pairing is admissible; the operator-facing reason otherwise. A free
/// function so the messages are testable without a checkpoint.
///
/// Used by: [`Drafter::validate_target_compat`] on [`MuseAssistantDrafter`].
fn target_pairing_error(config: &MuseAssistantConfig, target: &TargetShape) -> Option<String> {
    if target.hidden != config.hidden_size {
        return Some(format!(
            "Muse Glimmer assistant is incompatible with this target: drafter hidden_size = {} \
             but the target's hidden size = {}. The drafter's encoder.fc reads the target's \
             residual streams, so these must be equal (pair Muse-Glimmer-30B with \
             Muse-Glimmer-30B-assistant).",
            config.hidden_size, target.hidden
        ));
    }
    if let Some(expected) = config.num_target_layers
        && expected != target.layers
    {
        return Some(format!(
            "Muse Glimmer assistant is incompatible with this target: drafter num_target_layers = \
             {expected} but the target has {} layers.",
            target.layers
        ));
    }
    if config
        .target_layer_ids
        .last()
        .is_some_and(|&last| last >= target.layers)
    {
        return Some(format!(
            "Muse Glimmer assistant target_layer_ids {:?} reach past the target's {} layers.",
            config.target_layer_ids, target.layers
        ));
    }
    if let Some(vocab) = target.vocab {
        if let Some(expected) = config.vocab_size
            && expected != vocab
        {
            return Some(format!(
                "Muse Glimmer assistant vocabulary is incompatible with this target: drafter \
                 vocab_size = {expected} but the target's lm_head emits {vocab} logits."
            ));
        }
        if usize::try_from(config.mask_token_id).is_ok_and(|id| id >= vocab) {
            return Some(format!(
                "Muse Glimmer assistant mask_token_id {} is outside the target vocabulary [0, \
                 {vocab}); the mask id indexes the target's embedding table, which the drafter \
                 borrows.",
                config.mask_token_id
            ));
        }
    }
    None
}

impl Drafter for MuseAssistantDrafter {
    /// Pairing gate, run by the server before the round loop binds. The
    /// target is seen as a [`LanguageModel`] with no architecture string, so
    /// the measured hidden width, layer count and head vocabulary stand in
    /// for the `model_type` check; a target that hands out no raw embedding
    /// table or no untied head cannot be bound and is refused here.
    fn validate_target_compat(&self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        let sentinel = ffi::from_slice_i32(&[0_i32], &[1, 1]);
        let embedded =
            target
                .embed_tokens(&sentinel)
                .ok_or(DrafterError::TargetMissingFeature {
                    feature: "embed_tokens",
                })?;
        let hidden = ffi::array_shape(&embedded).last().copied().unwrap_or(0) as usize;
        if target.embed_tokens_module().is_none() {
            return Err(DrafterError::BindFailed {
                reason: "Muse Glimmer assistant borrows the target's raw embedding table, and \
                         this target hands out no embed_tokens_module(); pair it with a Muse \
                         Glimmer target"
                    .to_string(),
            });
        }
        let Some(head) = target.lm_head_module() else {
            return Err(DrafterError::BindFailed {
                reason: "Muse Glimmer assistant borrows the target's untied lm_head, and this \
                         target hands out no lm_head_module(); pair it with a Muse Glimmer target"
                    .to_string(),
            });
        };
        let vocab = if hidden > 0 {
            let zero_hidden = ffi::zeros(&[1, 1, hidden as i32], crate::dtype::FLOAT32);
            let logits = head.forward(&zero_hidden);
            Some(ffi::array_shape(&logits).last().copied().unwrap_or(0) as usize)
        } else {
            None
        };
        let shape = TargetShape {
            hidden,
            layers: target.num_layers(),
            vocab,
        };
        match target_pairing_error(&self.model.config, &shape) {
            Some(reason) => Err(DrafterError::BindFailed { reason }),
            None => Ok(()),
        }
    }

    fn bind(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        if self.model.needs_embed_binding() || self.model.needs_lm_head_binding() {
            let embed = target
                .embed_tokens_module()
                .ok_or_else(|| DrafterError::BindFailed {
                    reason: "Muse Glimmer assistant checkpoint omits embed_tokens.weight and the \
                             target does not expose embed_tokens_module(); the drafter needs the \
                             target's raw embedding table (the Muse Glimmer family hands it out)"
                        .to_string(),
                })?;
            let lm_head = target
                .lm_head_module()
                .ok_or_else(|| DrafterError::BindFailed {
                    reason:
                        "Muse Glimmer assistant checkpoint omits lm_head.weight and the target \
                             does not expose lm_head_module(); the drafter needs the target's \
                             untied head (the Muse Glimmer family hands it out)"
                            .to_string(),
                })?;
            self.model.bind_target(embed, lm_head);
        } else {
            let dummy = ffi::from_slice_i32(&[0_i32], &[1, 1]);
            if target.embed_tokens(&dummy).is_none() {
                return Err(DrafterError::BindFailed {
                    reason: "target model does not expose embed_tokens".to_string(),
                });
            }
        }
        self.bound = true;
        Ok(())
    }

    fn reset(&mut self, target: &dyn LanguageModel) -> Result<(), DrafterError> {
        self.bind(target)?;
        self.caches = self.model.make_caches();
        Ok(())
    }

    fn dflash_target_layer_ids(&self) -> Option<&[usize]> {
        Some(&self.model.config.target_layer_ids)
    }

    /// The depth the round loop starts at,
    /// `min(MUSE_ASSISTANT_INITIAL_BLOCK_SIZE, runtime verify width)`. The
    /// requested width (`--draft-block-size`, defaulting to the checkpoint's
    /// `runtime_verify_width()` through `peek_muse_assistant_configured_block_size`)
    /// is the ceiling the loop's throughput comparator widens to; with
    /// [`Self::prefer_requested_block_size`] left at the default `false`
    /// that comparator is what decides the width per measurement window.
    fn configured_block_size(&self) -> Option<usize> {
        Some(MUSE_ASSISTANT_INITIAL_BLOCK_SIZE.min(self.model.config.runtime_verify_width()))
    }

    fn is_muse_assistant(&self) -> bool {
        true
    }

    fn draft_block(
        &mut self,
        last_bonus: i32,
        hidden: Option<&MlxArray>,
        block_size: usize,
        sampler: &SamplingConfig,
    ) -> Result<Vec<i32>, DrafterError> {
        let logits = self.logits_for(last_bonus, hidden, block_size)?;
        let tokens = sample_block_per_position_array(&logits, block_size, sampler)?;
        let tokens = ffi::astype(&tokens, crate::dtype::INT32);
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
        sampler: &SamplingConfig,
    ) -> Result<UniquePtr<MlxArray>, DrafterError> {
        let logits = self.logits_for(last_bonus, hidden, block_size)?;
        sample_block_per_position_array(&logits, block_size, sampler)
    }

    fn sanitize(&mut self, weights: &mut WeightMap) -> Result<(), DrafterError> {
        MuseAssistantModel::sanitize(weights);
        Ok(())
    }

    fn kind(&self) -> DrafterKind {
        DrafterKind::Dflash
    }
}

#[cfg(test)]
mod pairing_tests {
    use super::*;

    fn config() -> MuseAssistantConfig {
        MuseAssistantConfig::from_json(&serde_json::json!({
            "model_type": "muse_glimmer_assistant",
            "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
            "mask_token_id": 6, "target_layer_ids": [1, 3],
        }))
        .unwrap()
    }

    #[test]
    fn pairing_error_names_the_measured_mismatch() {
        let config = config();
        let ok = TargetShape {
            hidden: 8,
            layers: 4,
            vocab: Some(8),
        };
        assert!(target_pairing_error(&config, &ok).is_none());

        let hidden = target_pairing_error(&config, &TargetShape { hidden: 6656, ..ok })
            .expect("hidden mismatch must decline");
        assert!(hidden.contains("hidden_size = 8"), "{hidden}");

        let layers = target_pairing_error(
            &config,
            &TargetShape {
                hidden: 8,
                layers: 3,
                vocab: Some(8),
            },
        )
        .expect("target_layer_ids past the stack must decline");
        assert!(layers.contains("reach past"), "{layers}");

        let vocab = target_pairing_error(
            &config,
            &TargetShape {
                hidden: 8,
                layers: 4,
                vocab: Some(6),
            },
        )
        .expect("a mask id past the vocabulary must decline");
        assert!(vocab.contains("mask_token_id 6"), "{vocab}");
    }
}
