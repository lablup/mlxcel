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

//! Mixtral MoE stage-local executor for pipeline-parallel inference.
//!
//! Mixtral uses a standard Llama-style transformer block shell (pre-norm
//! attention + pre-norm MLP with residuals) but replaces the dense MLP with a
//! `SparseMoeBlock`. The routing path is token-local: every token picks
//! `num_experts_per_tok` experts independently, and the expert weights live on
//! the same stage as the enclosing layer. Because Mixtral has no per-layer
//! state beyond the KV cache, splitting between any two adjacent decoder
//! layers is safe — stage boundaries behave identically to the Llama family.
//!
//! Expert-balance considerations: the auto-partitioner
//! can treat each Mixtral layer as a uniform cost block because every layer
//! carries the same expert count. MoE-specific weight accounting is not
//! required here; it is surfaced via the `ModelProfile` layer cost, not via
//! the stage executor.

use std::path::Path;

use anyhow::{Result, anyhow, ensure};
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, copy};

use crate::distributed::pipeline::partial_loading::filter_weight_map;
use crate::models;
use crate::models::sanitize_config_json;

use super::common::{TransformerStageLayer, TransformerStageModel};
use super::{FamilyStageExecutor, LayerFilter, StageExecutionInput, StageExecutionOutput};

pub struct MixtralStageExecutor {
    model: TransformerStageModel<MixtralStageLayer>,
}

impl MixtralStageExecutor {
    pub fn load(model_dir: &Path, filter: &LayerFilter, stage_index: usize) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|err| anyhow!("failed to read {}: {}", config_path.display(), err))?;
        let config_str = sanitize_config_json(&config_str);
        let args: models::mixtral::ModelArgs = serde_json::from_str(&config_str)
            .map_err(|err| anyhow!("failed to parse {}: {}", config_path.display(), err))?;

        let mut weights = models::load_text_weights(model_dir, None).map_err(anyhow::Error::msg)?;

        let mut effective_filter = filter.clone();
        if args.tie_word_embeddings && filter.has_lm_head {
            effective_filter.has_embedding = true;
        }
        filter_weight_map(&mut weights, &effective_filter);

        let group_size = args.group_size();
        let bits = args.bits();

        let load_embeddings =
            filter.has_embedding || (args.tie_word_embeddings && filter.has_lm_head);
        let embed_tokens = if load_embeddings {
            Some(
                UnifiedEmbedding::from_weights(&weights, "model.embed_tokens", group_size, bits)
                    .map_err(anyhow::Error::msg)?,
            )
        } else {
            None
        };

        let mut layers = Vec::with_capacity(filter.num_layers());
        for layer_idx in filter.layer_range.clone() {
            let layer = models::mixtral::DecoderLayer::from_weights(&weights, &args, layer_idx)
                .map_err(anyhow::Error::msg)?;
            layers.push(MixtralStageLayer(layer));
        }

        let (norm, lm_head) = if filter.has_lm_head {
            let norm_weight = get_weight_copy(&weights, "model.norm.weight")?;
            let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);
            let lm_head = if args.tie_word_embeddings {
                UnifiedLinear::from_weights(&weights, "model.embed_tokens", group_size, bits)
                    .map_err(anyhow::Error::msg)?
            } else {
                UnifiedLinear::from_weights(&weights, "lm_head", group_size, bits)
                    .map_err(anyhow::Error::msg)?
            };
            (Some(norm), Some(lm_head))
        } else {
            (None, None)
        };

        ensure!(
            !layers.is_empty(),
            "stage {} did not load any layers from range {}..{}",
            stage_index,
            filter.layer_range.start,
            filter.layer_range.end
        );

        Ok(Self {
            model: TransformerStageModel::new(filter.clone(), embed_tokens, layers, norm, lm_head)?,
        })
    }
}

impl FamilyStageExecutor for MixtralStageExecutor {
    fn make_caches(&self) -> Vec<KVCache> {
        self.model.make_caches()
    }

    fn execute(
        &self,
        input: StageExecutionInput<'_>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> Result<StageExecutionOutput> {
        self.model.execute(input, caches, mask)
    }
}

struct MixtralStageLayer(models::mixtral::DecoderLayer);

impl TransformerStageLayer for MixtralStageLayer {
    fn forward(
        &self,
        hidden: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.0.forward(hidden, cache, mask)
    }
}

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>> {
    weights
        .get(name)
        .map(|weight| copy(weight))
        .ok_or_else(|| anyhow!("weight not found: {}", name))
}
