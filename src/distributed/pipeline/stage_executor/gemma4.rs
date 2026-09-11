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

use std::path::Path;

use anyhow::Result;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, copy};

use crate::models::gemma4::{Cache as Gemma4Cache, Gemma4StageModel};

use super::common::PointerOwnedCacheStore;
use super::{FamilyStageExecutor, LayerFilter, StageExecutionInput, StageExecutionOutput};

pub struct Gemma4StageExecutor {
    model: Gemma4StageModel,
    cache_store: PointerOwnedCacheStore<Gemma4Cache>,
}

impl Gemma4StageExecutor {
    pub fn load(model_dir: &Path, filter: &LayerFilter, stage_index: usize) -> Result<Self> {
        Ok(Self {
            model: Gemma4StageModel::load(model_dir, filter, stage_index)
                .map_err(anyhow::Error::msg)?,
            cache_store: PointerOwnedCacheStore::default(),
        })
    }
}

impl FamilyStageExecutor for Gemma4StageExecutor {
    fn make_caches(&self) -> Vec<KVCache> {
        (0..self.model.num_layers())
            .map(|_| KVCache::new())
            .collect()
    }

    fn release_caches(&self, caches: &[KVCache]) {
        self.cache_store.release_caches(caches);
    }

    fn execute(
        &self,
        input: StageExecutionInput<'_>,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> Result<StageExecutionOutput> {
        let mut internal_caches = self.cache_store.caches_for_sequence(
            caches,
            || self.model.make_caches(),
            Gemma4Cache::offset,
            "gemma4 sequence cache entry must exist",
        );

        let output = match input {
            StageExecutionInput::TokenIds(input_ids) => self
                .model
                .execute_from_token_ids(input_ids, &mut internal_caches),
            StageExecutionInput::HiddenStates(hidden_states) => self
                .model
                .execute_from_hidden_states(copy(hidden_states), &mut internal_caches),
        }
        .map_err(anyhow::Error::msg)?;

        PointerOwnedCacheStore::sync_external_offsets(
            caches,
            &internal_caches,
            Gemma4Cache::offset,
        );
        Ok(output)
    }
}
