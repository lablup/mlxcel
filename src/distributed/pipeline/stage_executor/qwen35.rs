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

//! Pipeline-parallel stage executor for Qwen 3.5.
//!
//! Speculative-decoding hooks (`forward_speculative`, `rollback_speculative_cache` —) are **NOT** propagated through
//! this stage executor's Phase 1. Rationale:
//!
//! - The DFlash drafter must run against a *single coherent* target. The
//!   speculative round loop needs the verify-pass logits, per-layer hidden
//!   captures, and per-GDN-layer rollback snapshots within the same forward
//!   pass — splitting that across pipeline stages would require cross-stage
//!   plumbing of hidden-state and GDN-state snapshots that does not exist
//!   today.
//! - mlxcel's pipeline-parallel runner currently does not support speculative
//!   decoding for *any* model family, so opening the door specifically for
//!   Qwen 3.5 would create an isolated, untested code path.
//!
//! Follow-up: when speculative + pipeline-parallel becomes a supported combo,
//! reopen this file and surface a `forward_speculative_stage` analog that
//! emits and consumes hidden / GDN snapshots over the stage boundary. Until
//! then, callers that need DFlash on Qwen 3.5 must use the non-pipeline
//! `Qwen35Model::forward_speculative` path.

use std::path::Path;

use anyhow::Result;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, copy};

use crate::models::qwen3_5::Qwen35StageModel;
use crate::models::qwen3_next::Qwen3NextCache;

use super::common::PointerOwnedCacheStore;
use super::{FamilyStageExecutor, LayerFilter, StageExecutionInput, StageExecutionOutput};

pub struct Qwen35StageExecutor {
    model: Qwen35StageModel,
    cache_store: PointerOwnedCacheStore<Qwen3NextCache>,
}

impl Qwen35StageExecutor {
    pub fn load(model_dir: &Path, filter: &LayerFilter, stage_index: usize) -> Result<Self> {
        Ok(Self {
            model: Qwen35StageModel::load(model_dir, filter, stage_index)
                .map_err(anyhow::Error::msg)?,
            cache_store: PointerOwnedCacheStore::default(),
        })
    }
}

impl FamilyStageExecutor for Qwen35StageExecutor {
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
            Qwen3NextCache::offset,
            "qwen3.5 sequence cache entry must exist",
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
            Qwen3NextCache::offset,
        );
        Ok(output)
    }
}
