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

//! DeepSeek V3 stage-local executor for pipeline-parallel inference.
//!
//! DeepSeek V3 combines three features that matter for stage partitioning:
//!
//! - **Multi-Latent Attention (MLA)** — every decoder layer carries a
//!   compressed-KV attention with per-layer `q_a_proj` / `kv_a_layernorm`
//!   weights, and a `kv_b_proj` split into `embed_q` + `unembed_out` during
//!   `sanitize_weights`. The split has to run **before** partial weight
//!   filtering, otherwise stages past the last MoE layer lose the decomposed
//!   keys. This executor performs full-model sanitation on load, then trims
//!   the weight map to the stage's layer range.
//! - **Mixed dense / MoE MLPs** — `first_k_dense_replace` dense layers sit at
//!   the start of the stack, followed by routed-MoE layers. Expert weights
//!   are stacked via the `sanitize_weights` helper on the full model weight
//!   map. Auto-partitioner callers should note that expert-heavy layers carry
//!   a larger parameter footprint than the dense head; that bookkeeping lives
//!   on the partitioner (feeds sub-issue 7), not on the stage executor.
//! - **Multi-token-prediction (MTP) trailer layer**: checkpoints that ship
//!   the MTP head store it at layer index `num_hidden_layers` (one past the
//!   decoder stack, e.g. `model.layers.61` for genuine DeepSeek-V3);
//!   `sanitize_weights` strips it. All `num_hidden_layers` in-range layers
//!   are real decoder layers, so the pipeline partitioner sees exactly
//!   `num_hidden_layers` transformer blocks.
//!
//! Stage boundaries are safe at any layer index in
//! `0..num_hidden_layers` because DeepSeek V3 does not share layer
//! state across blocks beyond the standard KV cache.

use std::path::Path;

use anyhow::{Result, anyhow, ensure};
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, copy};

use crate::distributed::pipeline::partial_loading::filter_weight_map;
use crate::models;

use super::common::{TransformerStageLayer, TransformerStageModel};
use super::{FamilyStageExecutor, LayerFilter, StageExecutionInput, StageExecutionOutput};

pub struct DeepSeekV3StageExecutor {
    model: TransformerStageModel<DeepSeekV3StageLayer>,
}

impl DeepSeekV3StageExecutor {
    pub fn load(model_dir: &Path, filter: &LayerFilter, stage_index: usize) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|err| anyhow!("failed to read {}: {}", config_path.display(), err))?;
        let config: models::deepseek_v3::DeepSeekV3Config = serde_json::from_str(&config_str)
            .map_err(|err| anyhow!("failed to parse {}: {}", config_path.display(), err))?;

        let num_transformer_layers = config.num_hidden_layers;
        ensure!(
            filter.layer_range.end <= num_transformer_layers,
            "stage {} layer range {}..{} exceeds DeepSeek V3 transformer depth {}",
            stage_index,
            filter.layer_range.start,
            filter.layer_range.end,
            num_transformer_layers
        );

        // Full-model sanitation (MoE expert stacking, kv_b_proj split) must
        // run before we trim the weight map, otherwise `kv_b_proj.weight`
        // for layers outside the current stage would be missing and the
        // split for layers inside the stage would still look for a
        // not-yet-decomposed key.
        let weights = models::load_text_weights(model_dir, None).map_err(anyhow::Error::msg)?;
        let mut weights = models::DeepSeekV3Model::sanitize_weights_with_args(weights, &config);

        // DeepSeek V3 does not tie word embeddings, so no tied-head adjustment
        // like the one in `LlamaStageExecutor::load` is needed here.
        filter_weight_map(&mut weights, filter);

        let group_size = config.group_size();
        let bits = config.bits();

        let embed_tokens = if filter.has_embedding {
            Some(
                UnifiedEmbedding::from_weights(&weights, "model.embed_tokens", group_size, bits)
                    .map_err(anyhow::Error::msg)?,
            )
        } else {
            None
        };

        let mut layers = Vec::with_capacity(filter.num_layers());
        for layer_idx in filter.layer_range.clone() {
            let layer =
                models::deepseek_v3::DecoderLayer::from_weights(&weights, &config, layer_idx)
                    .map_err(anyhow::Error::msg)?;
            layers.push(DeepSeekV3StageLayer(layer));
        }

        let (norm, lm_head) = if filter.has_lm_head {
            let norm_weight = get_weight_copy(&weights, "model.norm.weight")?;
            let norm = RMSNorm::new(norm_weight, config.rms_norm_eps);
            let lm_head = UnifiedLinear::from_weights(&weights, "lm_head", group_size, bits)
                .map_err(anyhow::Error::msg)?;
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

impl FamilyStageExecutor for DeepSeekV3StageExecutor {
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

struct DeepSeekV3StageLayer(models::deepseek_v3::DecoderLayer);

impl TransformerStageLayer for DeepSeekV3StageLayer {
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
