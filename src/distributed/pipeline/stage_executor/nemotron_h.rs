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

//! Nemotron-H (hybrid Mamba2 + Transformer + MoE) stage-local executor.
//!
//! # SSM-split design note
//!
//! Nemotron-H mixes four block types inside one decoder stack: Mamba2,
//! Attention, MLP, and MoE. Mamba2 blocks carry persistent `conv_state` and
//! `ssm_state` through every token, so splitting the execution of a single
//! Mamba2 block across two pipeline stages is **not supported** — the
//! current wire protocol only transmits hidden states, not SSM state.
//!
//! Concretely:
//!
//! - Stage boundaries may land between any two blocks, regardless of their
//!   type. `LayerFilter::layer_range` is naturally block-granular.
//! - Stage boundaries may **not** land inside a Mamba2 block. Because the
//!   partitioner only ever picks indices into the block list, this
//!   invariant is enforced by construction.
//! - Stateless blocks (MLP, MoE) do not need a cache slot on the stage;
//!   only attention and Mamba2 blocks do. Nemotron-H reports this via
//!   [`NemotronHModel::num_cache_layers`](crate::models::nemotron_h) and
//!   [`NemotronHModel::layer_needs_cache`](crate::models::nemotron_h).
//!
//! # Known limitation (bring-up)
//!
//! This executor currently loads the **full** Nemotron-H model on every
//! stage and only executes the layer range assigned to the local stage.
//! The memory footprint therefore matches the non-PP path on each stage;
//! partial weight loading for Mamba-family hybrids is tracked as a
//! follow-up optimisation. The correctness invariants above (SSM blocks
//! whole within one stage) are unaffected by this bring-up strategy.

use std::path::Path;

use anyhow::{Result, anyhow, bail, ensure};
use mlxcel_core::MlxArray;
use mlxcel_core::layers::KVCache;

use crate::models::{self, NemotronHModel, nemotron_h::NemotronLayerCache};

use super::common::PointerOwnedCacheStore;
use super::{FamilyStageExecutor, LayerFilter, StageExecutionInput, StageExecutionOutput};

pub struct NemotronHStageExecutor {
    model: NemotronHModel,
    filter: LayerFilter,
    num_local_cache_layers: usize,
    cache_store: PointerOwnedCacheStore<NemotronLayerCache>,
}

impl NemotronHStageExecutor {
    pub fn load(model_dir: &Path, filter: &LayerFilter, stage_index: usize) -> Result<Self> {
        ensure!(
            filter.layer_range.end > filter.layer_range.start,
            "stage {} has an empty layer range",
            stage_index
        );
        let model_dir_str = model_dir
            .to_str()
            .ok_or_else(|| anyhow!("nemotron_h model directory path must be valid UTF-8"))?;
        let (model, _config) =
            models::NemotronHModel::load(model_dir_str).map_err(|err| anyhow!("{}", err))?;

        ensure!(
            filter.layer_range.end <= model.num_layers(),
            "stage {} layer range {}..{} exceeds Nemotron-H depth {}",
            stage_index,
            filter.layer_range.start,
            filter.layer_range.end,
            model.num_layers()
        );

        let num_local_cache_layers = filter
            .layer_range
            .clone()
            .filter(|&abs| model.layer_needs_cache(abs))
            .count();

        Ok(Self {
            model,
            filter: filter.clone(),
            num_local_cache_layers,
            cache_store: PointerOwnedCacheStore::default(),
        })
    }
}

impl FamilyStageExecutor for NemotronHStageExecutor {
    fn make_caches(&self) -> Vec<KVCache> {
        // One external KVCache slot per *stateful* layer inside the stage.
        // Stateless MLP / MoE blocks do not get a transport-side slot, so
        // the schedule layer's cache admission only accounts for real
        // per-sequence state.
        (0..self.num_local_cache_layers)
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
        ensure!(
            caches.len() == self.num_local_cache_layers,
            "nemotron_h stage cache count mismatch: expected {}, got {}",
            self.num_local_cache_layers,
            caches.len()
        );

        let layer_range = self.filter.layer_range.clone();
        let model = &self.model;
        let mut internal_caches = self.cache_store.caches_for_sequence(
            caches,
            || {
                layer_range
                    .clone()
                    .filter_map(|abs| model.make_layer_cache(abs))
                    .collect()
            },
            NemotronLayerCache::offset,
            "nemotron_h sequence cache entry must exist",
        );

        let raw_input: &MlxArray = match input {
            StageExecutionInput::TokenIds(tokens) => {
                if !self.filter.has_embedding {
                    bail!(
                        "nemotron_h stage received token IDs but does not host the embedding layer"
                    );
                }
                tokens
            }
            StageExecutionInput::HiddenStates(hidden) => {
                if self.filter.has_embedding {
                    bail!("nemotron_h entry stage expects token IDs, not hidden states");
                }
                hidden
            }
        };

        let out = self.model.forward_stage(
            raw_input,
            self.filter.layer_range.clone(),
            self.filter.has_embedding,
            self.filter.has_lm_head,
            &mut internal_caches,
        );

        PointerOwnedCacheStore::sync_external_offsets(
            caches,
            &internal_caches,
            NemotronLayerCache::offset,
        );

        if self.filter.has_lm_head {
            Ok(StageExecutionOutput::Logits(out))
        } else {
            Ok(StageExecutionOutput::HiddenStates(out))
        }
    }
}
