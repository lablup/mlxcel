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

//! Llama 4 Scout stage-local executor for pipeline-parallel inference.
//!
//! This executor covers only the **text-only language tower** of Llama 4
//! Scout. Vision encoder stage splitting is explicitly out of scope of
//! and of the parent VLM inputs that require the
//! vision tower should not be routed through this stage executor.
//!
//! Structural notes:
//!
//! - Weights live under the `language_model.*` prefix (`language_model.model
//!   .embed_tokens`, `language_model.model.layers.{i}.*`,
//!   `language_model.model.norm.weight`, `language_model.lm_head.*`). These
//!   prefixes are already understood by the partial-loading filter.
//! - The decoder block mixes dense and MoE MLPs via
//!   `interleave_moe_layer_step`, plus iGQA with a chunked attention cache
//!   every fourth layer. The Llama-4–specific caches belong to the
//!   full-model `Llama4Wrapper`; in pipeline mode we use the legacy
//!   `TransformerBlock::forward(x, cache, mask)` path that takes the shared
//!   [`KVCache`](mlxcel_core::layers::KVCache). This intentionally trades
//!   the iGQA optimisation for a transport-compatible cache layout — the
//!   stage protocol stays unchanged, and Llama-4 parity against the
//!   non-pipeline path is still verified via the real-model integration
//!   test in `tests/pipeline_stage_executor_real_models.rs`.
//! - Stage boundaries are safe at any layer index between 0 and
//!   `num_hidden_layers`; no cross-layer state other than the KV cache is
//!   carried over.

use std::path::Path;

use anyhow::{Result, anyhow, ensure};
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, copy};

use crate::distributed::pipeline::partial_loading::filter_weight_map;
use crate::models;

use super::common::{TransformerStageLayer, TransformerStageModel};
use super::{FamilyStageExecutor, LayerFilter, StageExecutionInput, StageExecutionOutput};

pub struct Llama4StageExecutor {
    model: TransformerStageModel<Llama4StageLayer>,
}

impl Llama4StageExecutor {
    pub fn load(model_dir: &Path, filter: &LayerFilter, stage_index: usize) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|err| anyhow!("failed to read {}: {}", config_path.display(), err))?;
        let config: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|err| anyhow!("failed to parse {}: {}", config_path.display(), err))?;

        // Llama 4 configs nest the text-side args under `text_config`; fall back
        // to the root for minimal test fixtures.
        let text_config = config.get("text_config").unwrap_or(&config);
        let args: models::llama4::TextArgs = serde_json::from_value(text_config.clone())
            .map_err(|err| anyhow!("failed to parse Llama 4 text config: {}", err))?;

        let mut weights = models::load_text_weights(model_dir, None).map_err(anyhow::Error::msg)?;
        // Llama 4 Scout does not expose `tie_word_embeddings` — the
        // `language_model.lm_head.*` weights are always stored separately
        // from `language_model.model.embed_tokens.*`, so no tied-head
        // adjustment is required.
        filter_weight_map(&mut weights, filter);

        let group_size = args.group_size();
        let bits = args.bits();

        let embed_tokens = if filter.has_embedding {
            Some(
                UnifiedEmbedding::from_weights(
                    &weights,
                    "language_model.model.embed_tokens",
                    group_size,
                    bits,
                )
                .map_err(anyhow::Error::msg)?,
            )
        } else {
            None
        };

        let mut layers = Vec::with_capacity(filter.num_layers());
        for layer_idx in filter.layer_range.clone() {
            let layer = models::llama4::TransformerBlock::from_weights(&weights, &args, layer_idx)
                .map_err(anyhow::Error::msg)?;
            layers.push(Llama4StageLayer(layer));
        }

        let (norm, lm_head) = if filter.has_lm_head {
            let norm_weight = get_weight_copy(&weights, "language_model.model.norm.weight")?;
            let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);
            let lm_head =
                UnifiedLinear::from_weights(&weights, "language_model.lm_head", group_size, bits)
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

impl FamilyStageExecutor for Llama4StageExecutor {
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

struct Llama4StageLayer(models::llama4::TransformerBlock);

impl TransformerStageLayer for Llama4StageLayer {
    fn forward(
        &self,
        hidden: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Use the legacy forward path (KVCache) — see the module-level docs
        // for why the iGQA ChunkedKVCache path is intentionally skipped in
        // pipeline mode.
        self.0.forward(hidden, cache, mask)
    }
}

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>> {
    weights
        .get(name)
        .map(|weight| copy(weight))
        .ok_or_else(|| anyhow!("weight not found: {}", name))
}
