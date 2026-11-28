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

//! Shared stage-local execution helpers for text-only pipeline backends.
//!
//! Used by: Llama stage executor, future GPT-OSS/Gemma/Qwen/GLM stage executors

use std::cell::{RefCell, RefMut};
use std::collections::HashMap;

use anyhow::{Result, anyhow, bail, ensure};
use mlxcel_core::layers::{KVCache, RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::pipeline_hint;
use mlxcel_core::{MlxArray, UniquePtr, copy};

use crate::distributed::pipeline::LayerFilter;

use super::{StageExecutionInput, StageExecutionOutput};

pub struct PointerOwnedCacheStore<C> {
    cache_sets: RefCell<HashMap<usize, Vec<C>>>,
}

impl<C> Default for PointerOwnedCacheStore<C> {
    fn default() -> Self {
        Self {
            cache_sets: RefCell::new(HashMap::new()),
        }
    }
}

impl<C> PointerOwnedCacheStore<C> {
    pub fn cache_key(caches: &[KVCache]) -> usize {
        caches.as_ptr() as usize
    }

    pub fn release_caches(&self, caches: &[KVCache]) {
        self.cache_sets
            .borrow_mut()
            .remove(&Self::cache_key(caches));
    }

    pub fn sync_external_offsets(
        external_caches: &mut [KVCache],
        internal_caches: &[C],
        cache_offset: impl Fn(&C) -> i32,
    ) {
        for (external, internal) in external_caches.iter_mut().zip(internal_caches.iter()) {
            external.offset = cache_offset(internal);
        }
    }

    pub fn caches_for_sequence<'a>(
        &'a self,
        external_caches: &[KVCache],
        make_caches: impl FnOnce() -> Vec<C>,
        cache_offset: impl Fn(&C) -> i32,
        missing_message: &'static str,
    ) -> RefMut<'a, Vec<C>> {
        let cache_key = Self::cache_key(external_caches);
        let needs_reset = {
            let cache_sets = self.cache_sets.borrow();
            cache_sets.get(&cache_key).is_some_and(|internal_caches| {
                external_caches.iter().all(|cache| cache.offset == 0)
                    && internal_caches.iter().any(|cache| cache_offset(cache) > 0)
            })
        };

        let mut cache_sets = self.cache_sets.borrow_mut();
        if needs_reset || !cache_sets.contains_key(&cache_key) {
            cache_sets.insert(cache_key, make_caches());
        }
        RefMut::map(cache_sets, |cache_sets| {
            cache_sets.get_mut(&cache_key).expect(missing_message)
        })
    }
}

pub trait TransformerStageLayer {
    fn forward(
        &self,
        hidden: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray>;
}

pub struct TransformerStageModel<L> {
    filter: LayerFilter,
    embed_tokens: Option<UnifiedEmbedding>,
    layers: Vec<L>,
    norm: Option<RMSNorm>,
    lm_head: Option<UnifiedLinear>,
}

impl<L> TransformerStageModel<L> {
    pub fn new(
        filter: LayerFilter,
        embed_tokens: Option<UnifiedEmbedding>,
        layers: Vec<L>,
        norm: Option<RMSNorm>,
        lm_head: Option<UnifiedLinear>,
    ) -> Result<Self> {
        ensure!(
            !layers.is_empty(),
            "stage-local transformer model requires at least one layer"
        );
        Ok(Self {
            filter,
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    pub fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }
}

impl<L: TransformerStageLayer> TransformerStageModel<L> {
    pub fn execute(
        &self,
        input: StageExecutionInput<'_>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> Result<StageExecutionOutput> {
        self.execute_last_row(input, caches, mask, None)
    }

    /// [`Self::execute`] that, on the stage hosting the LM head, projects only
    /// hidden row `last_pos` (`[batch, 1, vocab]` logits) instead of every
    /// position. A non-final stage ignores `last_pos` and returns its full
    /// hidden states (#3074).
    pub fn execute_last_row(
        &self,
        input: StageExecutionInput<'_>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: Option<usize>,
    ) -> Result<StageExecutionOutput> {
        let hidden = self.prepare_hidden(input)?;
        self.execute_hidden(hidden, caches, mask, last_pos)
    }

    fn prepare_hidden(&self, input: StageExecutionInput<'_>) -> Result<UniquePtr<MlxArray>> {
        Ok(match input {
            StageExecutionInput::TokenIds(input_ids) => self
                .embed_tokens
                .as_ref()
                .ok_or_else(|| {
                    anyhow!("stage does not host embeddings; hidden-state input required")
                })?
                .forward(input_ids),
            StageExecutionInput::HiddenStates(hidden_states) => {
                if self.filter.has_embedding {
                    bail!("entry stage expects token IDs, not hidden states");
                }
                copy(hidden_states)
            }
        })
    }

    fn execute_hidden(
        &self,
        mut hidden: UniquePtr<MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: Option<usize>,
    ) -> Result<StageExecutionOutput> {
        ensure!(
            caches.len() == self.layers.len(),
            "stage cache count mismatch: expected {}, got {}",
            self.layers.len(),
            caches.len()
        );

        let n = self.layers.len();
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &mut caches[i], mask);
            pipeline_hint(&hidden, i, n);
        }

        match (&self.norm, &self.lm_head) {
            (Some(norm), Some(lm_head)) => {
                // Norm and head act per position, so slicing first yields the
                // row the full projection would have (#3074).
                let hidden = match last_pos {
                    Some(pos) => mlxcel_core::generate::logits_at_position(&hidden, pos),
                    None => hidden,
                };
                let hidden = norm.forward(&hidden);
                Ok(StageExecutionOutput::Logits(lm_head.forward(&hidden)))
            }
            _ => Ok(StageExecutionOutput::HiddenStates(hidden)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoLayer;

    impl TransformerStageLayer for EchoLayer {
        fn forward(
            &self,
            hidden: &MlxArray,
            _cache: &mut KVCache,
            _mask: Option<&MlxArray>,
        ) -> UniquePtr<MlxArray> {
            copy(hidden)
        }
    }

    #[test]
    fn shared_transformer_stage_model_preserves_hidden_states() {
        let model = TransformerStageModel::new(
            LayerFilter {
                layer_range: 1..3,
                has_embedding: false,
                has_lm_head: false,
            },
            None,
            vec![EchoLayer, EchoLayer],
            None,
            None,
        )
        .unwrap();
        let input = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let mut caches = model.make_caches();

        let output = model
            .execute(
                StageExecutionInput::HiddenStates(input.as_ref().unwrap()),
                &mut caches,
                None,
            )
            .unwrap()
            .into_hidden_states()
            .unwrap();

        let close = mlxcel_core::allclose(input.as_ref().unwrap(), &output, 0.0, 0.0);
        assert!(mlxcel_core::item_bool(&close));
    }

    #[test]
    fn shared_transformer_stage_model_rejects_cache_count_mismatch() {
        let model = TransformerStageModel::new(
            LayerFilter {
                layer_range: 1..3,
                has_embedding: false,
                has_lm_head: false,
            },
            None,
            vec![EchoLayer, EchoLayer],
            None,
            None,
        )
        .unwrap();
        let input = mlxcel_core::from_slice_f32(&[1.0, 2.0], &[1, 1, 2]);
        let mut caches = vec![KVCache::new()];
        let err = match model.execute(
            StageExecutionInput::HiddenStates(input.as_ref().unwrap()),
            &mut caches,
            None,
        ) {
            Ok(_) => panic!("expected cache count mismatch"),
            Err(err) => err,
        };

        assert!(err.to_string().contains("stage cache count mismatch"));
    }

    /// A last-stage model with a real norm and head: [`execute_last_row`]
    /// returns the one requested row, equal to that row of the full forward
    /// (#3074).
    #[test]
    fn last_row_execution_matches_the_full_forward_row() {
        let norm_weight = mlxcel_core::from_slice_f32(&[1.0, 0.5, 2.0, 1.5], &[4]);
        let head_weight = mlxcel_core::from_slice_f32(
            &(0..20).map(|v| (v as f32) * 0.1 - 0.7).collect::<Vec<_>>(),
            &[5, 4],
        );
        let model = TransformerStageModel::new(
            LayerFilter {
                layer_range: 0..1,
                has_embedding: false,
                has_lm_head: true,
            },
            None,
            vec![EchoLayer],
            Some(RMSNorm::new(norm_weight, 1e-6)),
            Some(UnifiedLinear::Regular(mlxcel_core::layers::Linear::new(
                head_weight,
                None,
            ))),
        )
        .unwrap();
        let input = mlxcel_core::from_slice_f32(
            &(0..12).map(|v| (v as f32) * 0.3 - 1.0).collect::<Vec<_>>(),
            &[1, 3, 4],
        );

        let full = model
            .execute(
                StageExecutionInput::HiddenStates(input.as_ref().unwrap()),
                &mut model.make_caches(),
                None,
            )
            .unwrap()
            .into_logits()
            .unwrap();
        assert_eq!(mlxcel_core::array_shape(&full), vec![1, 3, 5]);

        for pos in [0usize, 1, 2] {
            let row = model
                .execute_last_row(
                    StageExecutionInput::HiddenStates(input.as_ref().unwrap()),
                    &mut model.make_caches(),
                    None,
                    Some(pos),
                )
                .unwrap()
                .into_logits()
                .unwrap();
            assert_eq!(mlxcel_core::array_shape(&row), vec![1, 1, 5]);
            let want = mlxcel_core::generate::logits_at_position(&full, pos);
            let close = mlxcel_core::allclose(&row, &want, 1e-5, 1e-5);
            assert!(mlxcel_core::item_bool(&close), "row {pos} differs");
        }

        // `None` keeps every position.
        let all = model
            .execute_last_row(
                StageExecutionInput::HiddenStates(input.as_ref().unwrap()),
                &mut model.make_caches(),
                None,
                None,
            )
            .unwrap()
            .into_logits()
            .unwrap();
        assert_eq!(mlxcel_core::array_shape(&all), vec![1, 3, 5]);
    }

    /// A stage that hosts no LM head ignores `last_pos` and still returns its
    /// whole hidden state for the next stage.
    #[test]
    fn non_final_stage_ignores_last_pos() {
        let model = TransformerStageModel::new(
            LayerFilter {
                layer_range: 0..1,
                has_embedding: false,
                has_lm_head: false,
            },
            None,
            vec![EchoLayer],
            None,
            None,
        )
        .unwrap();
        let input = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let hidden = model
            .execute_last_row(
                StageExecutionInput::HiddenStates(input.as_ref().unwrap()),
                &mut model.make_caches(),
                None,
                Some(1),
            )
            .unwrap()
            .into_hidden_states()
            .unwrap();
        assert_eq!(mlxcel_core::array_shape(&hidden), vec![1, 2, 2]);
    }
}
