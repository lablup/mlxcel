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
    /// Per-layer-input checkpoints (E2B / E4B) pack the remaining layers'
    /// per-layer inputs into the stage-boundary activation, so the inbound
    /// width is wider than `hidden_size`.
    fn inbound_activation_width(&self) -> Option<u64> {
        Some(self.model.inbound_activation_width())
    }

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
        mask: Option<&MlxArray>,
    ) -> Result<StageExecutionOutput> {
        self.execute_last_row(input, caches, mask, None)
    }

    /// The final stage slices the hidden state at `last_pos` before the final
    /// norm and LM head, so the head runs on one row (#3081). A non-final
    /// stage ignores `last_pos`.
    fn execute_last_row(
        &self,
        input: StageExecutionInput<'_>,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
        last_pos: Option<usize>,
    ) -> Result<StageExecutionOutput> {
        let mut internal_caches = self.cache_store.caches_for_sequence(
            caches,
            || self.model.make_caches(),
            Gemma4Cache::offset,
            "gemma4 sequence cache entry must exist",
        );

        let output = match input {
            StageExecutionInput::TokenIds(input_ids) => {
                self.model
                    .execute_from_token_ids(input_ids, &mut internal_caches, last_pos)
            }
            StageExecutionInput::HiddenStates(hidden_states) => self
                .model
                .execute_from_hidden_states(copy(hidden_states), &mut internal_caches, last_pos),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::pipeline::stage_executor::last_row_test_support::assert_last_row_matches_full;
    use crate::models::gemma4::gemma4_stage_tests::{LAYERS, args, weights};

    /// The final stage slices the hidden row before the norm, head and
    /// softcap and returns the full forward's row (#3081).
    #[test]
    fn last_row_matches_the_full_forward_row() {
        let (args, weights) = (args(), weights());
        assert_last_row_matches_full(
            |filter| {
                Box::new(Gemma4StageExecutor {
                    model: Gemma4StageModel::from_filtered_weights(
                        &weights,
                        &args,
                        filter,
                        0,
                        Default::default(),
                    )
                    .expect("fixture stage must load"),
                    cache_store: PointerOwnedCacheStore::default(),
                })
            },
            LAYERS,
            64,
            2e-2,
        );
    }
}
