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

//! Pipeline-parallel stage executor for Qwen3-Next.
//!
//! Qwen3-Next is a hybrid model that interleaves linear (GatedDeltaNet,
//! `GatedDeltaCache`) and full-attention (`KVCache`) layers. This executor is
//! the sibling of the Qwen 3.5 stage executor (`qwen35.rs`): it wraps a
//! [`Qwen3NextStageModel`] holding only the stage's layer range and keeps the
//! stage-local heterogeneous cache vector in a
//! [`PointerOwnedCacheStore`], keyed off the identity of the external
//! `KVCache` handle slice the runtime hands in.
//!
//! The external handles carry only the sequence offset back to admission and
//! scheduler code; the real conv-state / SSM-state / KV tensors live in the
//! internal [`Qwen3NextCache`] vector, one entry per LOCAL layer, whose
//! `Linear`/`Attention` variant is chosen from the GLOBAL layer index so the
//! cache type and mask match the single-process path exactly.
//!
//! Like the Qwen 3.5 executor, the speculative-decoding hooks are not
//! propagated across the stage boundary: mlxcel's pipeline-parallel runner
//! does not support speculative decoding for any family today.

use std::path::Path;

use anyhow::Result;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, copy};

use crate::models::qwen3_next::{Qwen3NextCache, Qwen3NextStageModel};

use super::common::PointerOwnedCacheStore;
use super::{FamilyStageExecutor, LayerFilter, StageExecutionInput, StageExecutionOutput};

pub struct Qwen3NextStageExecutor {
    model: Qwen3NextStageModel,
    cache_store: PointerOwnedCacheStore<Qwen3NextCache>,
}

impl Qwen3NextStageExecutor {
    pub fn load(model_dir: &Path, filter: &LayerFilter, stage_index: usize) -> Result<Self> {
        Ok(Self {
            model: Qwen3NextStageModel::load(model_dir, filter, stage_index)
                .map_err(anyhow::Error::msg)?,
            cache_store: PointerOwnedCacheStore::default(),
        })
    }
}

impl FamilyStageExecutor for Qwen3NextStageExecutor {
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
            Qwen3NextCache::offset,
            "qwen3-next sequence cache entry must exist",
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
            Qwen3NextCache::offset,
        );
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::pipeline::stage_executor::last_row_test_support::assert_last_row_matches_full;
    use crate::models::qwen3_5_tests::two_layer_weights_and_config;

    /// The final stage slices the hidden row before the norm and head and
    /// returns the full forward's row (#3081). The Qwen3-Next stage loads the
    /// Qwen 3.5 fixture's tensors, with the gated-delta layer's four separate
    /// projections replaced by Qwen3-Next's fused `in_proj_qkvz` and `in_proj_ba`.
    #[test]
    fn last_row_matches_the_full_forward_row() {
        let (mut weights, config) = two_layer_weights_and_config(8);
        let config = config.to_qwen3next_config();
        let linear = "model.layers.0.linear_attn";
        for name in ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"] {
            weights.remove(&format!("{linear}.{name}.weight"));
        }
        for (name, rows) in [("in_proj_qkvz", 16), ("in_proj_ba", 4)] {
            let values: Vec<f32> = (0..rows * 8)
                .map(|i| 0.01 + (i % 17) as f32 * 0.003)
                .collect();
            weights.insert(
                format!("{linear}.{name}.weight"),
                mlxcel_core::from_slice_f32(&values, &[rows, 8]),
            );
        }
        assert_last_row_matches_full(
            |filter| {
                Box::new(Qwen3NextStageExecutor {
                    model: Qwen3NextStageModel::from_filtered_weights(&weights, &config, filter, 0)
                        .expect("fixture stage must load"),
                    cache_store: PointerOwnedCacheStore::default(),
                })
            },
            2,
            8,
            1e-4,
        );
    }
}
