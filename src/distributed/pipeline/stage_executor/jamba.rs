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

//! Jamba (hybrid Transformer + Mamba) stage-local executor.
//!
//! # SSM-split design note
//!
//! Jamba interleaves dense attention blocks with Mamba state-space blocks.
//! Each Mamba block maintains a `JambaMambaCache` that carries persistent
//! conv-state and SSM-state across every token — the cache is updated
//! in-place on every forward pass and the update depends on the previous
//! state. Splitting the execution of one Mamba block across two pipeline
//! stages is therefore **not supported**; doing so would require
//! serialising SSM state across a wire boundary, and the current activation
//! wire protocol only transmits hidden states.
//!
//! The rule this executor enforces is simpler than it sounds:
//!
//! - Stage boundaries are allowed **between** any two decoder layers,
//!   including between a Mamba layer and the next attention layer.
//! - Stage boundaries are **not** allowed inside a single decoder layer.
//!
//! The current partition API (`LayerFilter::layer_range`) is naturally
//! layer-granular — it can only land on block boundaries — so no additional
//! validation is required beyond loading the full model on every stage.
//!
//! # Known limitation (bring-up)
//!
//! This executor currently loads the **full** Jamba model on every stage
//! and only executes the layer range assigned to the local stage. This
//! preserves the correctness of the SSM state update (the stage physically
//! holds every block it touches) and unlocks PP functionality for
//! zero-host-count clusters. The memory footprint is therefore equivalent
//! to running the non-PP path on each stage; partial weight loading for
//! Mamba-family hybrids is tracked as a follow-up optimisation so the
//! transport wire protocol remains the single source of truth for stage
//! boundary invariants.

use std::path::Path;

use anyhow::{Result, anyhow, bail, ensure};
use mlxcel_core::MlxArray;
use mlxcel_core::layers::KVCache;

use crate::models::{self, JambaModel, jamba::JambaLayerCache};

use super::common::PointerOwnedCacheStore;
use super::{FamilyStageExecutor, LayerFilter, StageExecutionInput, StageExecutionOutput};

pub struct JambaStageExecutor {
    model: JambaModel,
    filter: LayerFilter,
    cache_store: PointerOwnedCacheStore<JambaLayerCache>,
}

impl JambaStageExecutor {
    pub fn load(model_dir: &Path, filter: &LayerFilter, stage_index: usize) -> Result<Self> {
        ensure!(
            filter.layer_range.end > filter.layer_range.start,
            "stage {} has an empty layer range",
            stage_index
        );
        let model_dir_str = model_dir
            .to_str()
            .ok_or_else(|| anyhow!("jamba model directory path must be valid UTF-8"))?;
        let (model, _config) =
            models::JambaModel::load(model_dir_str).map_err(|err| anyhow!("{}", err))?;

        ensure!(
            filter.layer_range.end <= model.num_layers(),
            "stage {} layer range {}..{} exceeds Jamba depth {}",
            stage_index,
            filter.layer_range.start,
            filter.layer_range.end,
            model.num_layers()
        );

        Ok(Self {
            model,
            filter: filter.clone(),
            cache_store: PointerOwnedCacheStore::default(),
        })
    }
}

impl FamilyStageExecutor for JambaStageExecutor {
    fn make_caches(&self) -> Vec<KVCache> {
        // Each local layer in the stage corresponds to exactly one external
        // KVCache slot (used purely as a pointer-identity handle). The real
        // hybrid cache is stored internally keyed off of that identity.
        (0..self.filter.num_layers())
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
            caches.len() == self.filter.num_layers(),
            "jamba stage cache count mismatch: expected {}, got {}",
            self.filter.num_layers(),
            caches.len()
        );

        let block_types: Vec<String> = self.model.layer_block_types().to_vec();
        let mut internal_caches = self.cache_store.caches_for_sequence(
            caches,
            || {
                self.filter
                    .layer_range
                    .clone()
                    .map(|abs| {
                        if block_types.get(abs).map(|s| s.as_str()) == Some("attention") {
                            JambaLayerCache::Attention(KVCache::new())
                        } else {
                            JambaLayerCache::Mamba(models::jamba::JambaMambaCache::new())
                        }
                    })
                    .collect()
            },
            JambaLayerCache::offset,
            "jamba sequence cache entry must exist",
        );

        let raw_input: &MlxArray = match input {
            StageExecutionInput::TokenIds(tokens) => {
                if !self.filter.has_embedding {
                    bail!("jamba stage received token IDs but does not host the embedding layer");
                }
                tokens
            }
            StageExecutionInput::HiddenStates(hidden) => {
                if self.filter.has_embedding {
                    bail!("jamba entry stage expects token IDs, not hidden states");
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

        // Propagate the sequence offset from the first attention cache slot
        // back into the external handle so admission and scheduler code that
        // reads `KVCache::offset` sees the real value.
        PointerOwnedCacheStore::sync_external_offsets(
            caches,
            &internal_caches,
            JambaLayerCache::offset,
        );

        if self.filter.has_lm_head {
            Ok(StageExecutionOutput::Logits(out))
        } else {
            Ok(StageExecutionOutput::HiddenStates(out))
        }
    }
}
