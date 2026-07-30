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

use std::path::Path;

use anyhow::{Result, anyhow};
use mlxcel_core::MlxArray;
use mlxcel_core::layers::KVCache;

use crate::distributed::pipeline::partial_loading::filter_weight_map;
use crate::models::deepseek_v32::DeepSeekV32StageModel;
use crate::models::glm_moe_dsa::ModelArgs;
use crate::models::{self, DeepSeekV32Model};

use super::{FamilyStageExecutor, LayerFilter, StageExecutionInput, StageExecutionOutput};

pub struct GlmMoeDsaStageExecutor {
    model: DeepSeekV32StageModel,
}

impl GlmMoeDsaStageExecutor {
    pub fn load(model_dir: &Path, filter: &LayerFilter, stage_index: usize) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|err| anyhow!("failed to read {}: {}", config_path.display(), err))?;
        let args: ModelArgs = serde_json::from_str(&config_str)
            .map_err(|err| anyhow!("failed to parse {}: {}", config_path.display(), err))?;

        let dsv32_args = args.to_dsv32_args();
        let mut weights = models::load_text_weights(model_dir, None).map_err(anyhow::Error::msg)?;
        weights = DeepSeekV32Model::sanitize_weights_with_args(weights, &dsv32_args)
            .map_err(anyhow::Error::msg)?;

        let mut effective_filter = filter.clone();
        if dsv32_args.tie_word_embeddings && filter.has_lm_head {
            effective_filter.has_embedding = true;
        }
        filter_weight_map(&mut weights, &effective_filter);

        Ok(Self {
            model: DeepSeekV32StageModel::from_filtered_weights(
                &weights,
                &dsv32_args,
                filter,
                stage_index,
            )
            .map_err(anyhow::Error::msg)?,
        })
    }
}

impl FamilyStageExecutor for GlmMoeDsaStageExecutor {
    fn make_caches(&self) -> Vec<KVCache> {
        self.model.make_caches()
    }

    fn execute(
        &self,
        input: StageExecutionInput<'_>,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> Result<StageExecutionOutput> {
        match input {
            StageExecutionInput::TokenIds(input_ids) => self
                .model
                .execute_from_token_ids(input_ids, caches)
                .map_err(anyhow::Error::msg),
            StageExecutionInput::HiddenStates(hidden_states) => self
                .model
                .execute_from_hidden_states(mlxcel_core::copy(hidden_states), caches)
                .map_err(anyhow::Error::msg),
        }
    }
}
