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

//! `LanguageModel` for [`Raon1MiniModel`]: model-owned hybrid state keyed by
//! `SequenceId`, as Kimi K3 and Kimi Linear do.

use super::super::raon_rda::RaonDeltaCache;
use super::{Mixer, Raon1MiniLayerCache, Raon1MiniModel, StepMasks};
use mlxcel_core::cache::{CachePool, LayerStateKind, SequenceId, SequenceStateLayout};
use mlxcel_core::generate::{BatchedDecodeKind, LanguageModel, route_batched_decode_with_ids};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

impl Raon1MiniModel {
    /// Prefill or decode against the sequence's state (the fallback slot
    /// when `seq_id` is `None`), for every position or only `row`.
    ///
    /// A multi-token forward without a sequence id starts a new prompt and
    /// resets the fallback slot, which is why chunked prefill is declined.
    pub(crate) fn forward_rows_for_sequence(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        row: Option<usize>,
    ) -> UniquePtr<MlxArray> {
        let seq_len = mlxcel_core::array_shape(input_ids)[1];
        if seq_id.is_none() && seq_len > 1 {
            self.sequence_state
                .replace_internal(self.make_layer_caches());
        }
        let run = |caches: &mut [Raon1MiniLayerCache]| {
            let h = self.embed_tokens.forward(input_ids);
            self.forward_hidden(h, caches, &StepMasks::default(), row)
        };
        match seq_id {
            Some(id) => self
                .sequence_state
                .with_existing_sequence_state(id, run)
                .unwrap_or_else(|err| panic!("raon1_mini {err}")),
            None => self.sequence_state.with_sequence_state(None, run),
        }
    }

    /// One row at a time against each row's own state (the fallback when
    /// the rows cannot share one batched step).
    fn forward_rows_with_states(
        &self,
        input_ids: &MlxArray,
        rows: &mut [Vec<Raon1MiniLayerCache>],
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(input_ids);
        let outputs: Vec<_> = rows
            .iter_mut()
            .enumerate()
            .map(|(row, caches)| {
                let r = row as i32;
                let input = mlxcel_core::slice(input_ids, &[r, 0], &[r + 1, shape[1]]);
                let h = self.embed_tokens.forward(&input);
                self.forward_hidden(h, caches, &StepMasks::default(), None)
            })
            .collect();
        mlxcel_core::concatenate_many(&outputs.iter().map(|o| &**o).collect::<Vec<_>>(), 0)
    }

    fn validate_batch_cache_layout(&self, rows: &[Vec<Raon1MiniLayerCache>]) {
        for (row_idx, row) in rows.iter().enumerate() {
            assert_eq!(
                row.len(),
                self.layers.len(),
                "raon1_mini batched decode cache count mismatch for row {row_idx}"
            );
            for (layer_idx, (cache, layer)) in row.iter().zip(&self.layers).enumerate() {
                assert!(
                    matches!(
                        (cache, layer.is_rda),
                        (Raon1MiniLayerCache::Delta(_), true)
                            | (Raon1MiniLayerCache::Attn(_), false)
                    ),
                    "raon1_mini batched decode cache kind mismatch for row {row_idx}, layer {layer_idx}"
                );
            }
        }
    }

    fn delta_states(rows: &[Vec<Raon1MiniLayerCache>], layer_idx: usize) -> Vec<&RaonDeltaCache> {
        rows.iter()
            .map(|row| match &row[layer_idx] {
                Raon1MiniLayerCache::Delta(state) => state,
                Raon1MiniLayerCache::Attn(_) => unreachable!("preflighted cache kind"),
            })
            .collect()
    }

    fn delta_rows_uniform(rows: &[&RaonDeltaCache]) -> bool {
        use super::super::recurrent_batch::optional_rows_uniform;
        optional_rows_uniform(&rows.iter().map(|s| s.conv.as_deref()).collect::<Vec<_>>())
            && optional_rows_uniform(&rows.iter().map(|s| s.ssm.as_deref()).collect::<Vec<_>>())
    }

    /// Every RDA layer's rows are all fresh or all initialized with one
    /// shape, so they stack into a single `[B, ...]` state.
    fn batch_states_uniform(&self, rows: &[Vec<Raon1MiniLayerCache>]) -> bool {
        self.layers.iter().enumerate().all(|(layer_idx, layer)| {
            !layer.is_rda || Self::delta_rows_uniform(&Self::delta_states(rows, layer_idx))
        })
    }

    fn stack_delta(rows: &[&RaonDeltaCache]) -> RaonDeltaCache {
        use super::super::recurrent_batch::stack_optional_rows;
        RaonDeltaCache {
            conv: stack_optional_rows(&rows.iter().map(|s| s.conv.as_deref()).collect::<Vec<_>>())
                .into_option(),
            ssm: stack_optional_rows(&rows.iter().map(|s| s.ssm.as_deref()).collect::<Vec<_>>())
                .into_option(),
            offset: 0,
        }
    }

    fn split_delta(state: &RaonDeltaCache, offsets: &[i32]) -> Vec<RaonDeltaCache> {
        use super::super::recurrent_batch::split_rows;
        let batch = offsets.len();
        let split = |field: &Option<UniquePtr<MlxArray>>| match field {
            Some(field) => split_rows(field, batch).into_iter().map(Some).collect(),
            None => (0..batch).map(|_| None).collect::<Vec<_>>(),
        };
        split(&state.conv)
            .into_iter()
            .zip(split(&state.ssm))
            .zip(offsets)
            .map(|((conv, ssm), offset)| RaonDeltaCache {
                conv,
                ssm,
                offset: offset + 1,
            })
            .collect()
    }

    /// One decode token for every row in a single forward: the RDA states
    /// are stacked into one `[B, ...]` state for the step and split back,
    /// and the MLA layers read each row's own latent cache, so the weights
    /// (128 experts, the 260K-row head) are read once per step instead of
    /// once per row. The caller has checked [`Self::batch_states_uniform`].
    fn forward_batched_decode_with_caches(
        &self,
        input_ids: &MlxArray,
        rows: &mut [Vec<Raon1MiniLayerCache>],
    ) -> UniquePtr<MlxArray> {
        let h = self.embed_tokens.forward(input_ids);
        let mut streams = super::raon_mhc::expand_streams(&h, self.streams);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            streams = layer.forward_with_mixer(&streams, |mixer, x| match mixer {
                Mixer::Rda(attn) => {
                    let states = Self::delta_states(rows, layer_idx);
                    let offsets: Vec<i32> = states.iter().map(|s| s.offset).collect();
                    let mut stacked = Self::stack_delta(&states);
                    let out = attn.forward(x, &mut stacked, None);
                    for (row, state) in rows.iter_mut().zip(Self::split_delta(&stacked, &offsets)) {
                        row[layer_idx] = Raon1MiniLayerCache::Delta(state);
                    }
                    out
                }
                Mixer::Mla(attn) => {
                    let mut caches: Vec<&mut KVCache> = rows
                        .iter_mut()
                        .map(|row| match &mut row[layer_idx] {
                            Raon1MiniLayerCache::Attn(cache) => cache,
                            Raon1MiniLayerCache::Delta(_) => {
                                unreachable!("preflighted cache kind")
                            }
                        })
                        .collect();
                    attn.forward_batched_decode(x, &mut caches)
                }
            });
        }
        self.head(&streams, None)
    }
}

impl LanguageModel for Raon1MiniModel {
    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// MLA layers hold their (latent-pair) KV mode, RDA layers no KV.
    fn kv_cache_storage_modes(&self) -> Option<Vec<Option<mlxcel_core::cache::KVCacheMode>>> {
        Some(
            self.layers
                .iter()
                .enumerate()
                .map(|(idx, layer)| {
                    let kind = if layer.is_rda {
                        LayerStateKind::None
                    } else {
                        LayerStateKind::Full
                    };
                    kind.stored_mode(super::super::mla_batched::latent_pair_cache_mode(
                        self.kv_cache_layer_modes.mode_for_layer(idx),
                    ))
                })
                .collect(),
        )
    }

    fn set_kv_cache_layer_modes(&self, modes: Vec<mlxcel_core::cache::KVCacheMode>) {
        self.kv_cache_layer_modes.set(modes);
        self.sequence_state
            .replace_internal(self.make_layer_caches());
    }

    fn kv_cache_layer_modes(&self) -> Option<Vec<mlxcel_core::cache::KVCacheMode>> {
        self.kv_cache_layer_modes.clone_modes()
    }

    /// Off, as for every recurrent family: the scheduler's padded prefill
    /// carries no per-row pad counts into the RDA state.
    fn supports_padded_prefill(&self) -> bool {
        false
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn batched_decode_kind(&self) -> BatchedDecodeKind {
        BatchedDecodeKind::Native
    }

    /// A multi-token forward without a sequence id resets the fallback
    /// state (see [`Raon1MiniModel::forward_rows_for_sequence`]), so a
    /// chunked prefill would drop every chunk but the last; the prompt runs
    /// in one pass, as for Kimi K3.
    fn supports_chunked_prefill(&self) -> bool {
        false
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.sequence_state
            .replace_internal(self.make_layer_caches());
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }

    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_rows_for_sequence(input_ids, None, None)
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        SequenceStateLayout::model_owned(self.layers.len())
    }

    fn sync_sequence_storage(
        &self,
        seq_id: SequenceId,
        cache_pool: &mut CachePool,
    ) -> Result<(), String> {
        self.sequence_state
            .with_existing_sequence_state(seq_id, |caches| {
                let lengths: Vec<_> = caches
                    .iter()
                    .map(|cache| match cache {
                        Raon1MiniLayerCache::Attn(kv) => kv.live_len().max(0) as usize,
                        Raon1MiniLayerCache::Delta(_) => 0,
                    })
                    .collect();
                cache_pool.sync_paged_state_with_lengths(seq_id, &lengths)
            })?
    }

    fn reset_runtime_state(&self) {
        self.sequence_state
            .replace_internal(self.make_layer_caches());
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.sequence_state
            .prepare_sequence_state(seq_id, self.make_layer_caches());
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.sequence_state.release_sequence_state(seq_id);
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_rows_for_sequence(input_ids, seq_id, None)
    }

    fn last_row_logits(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> Option<UniquePtr<MlxArray>> {
        // No embeddings forward: decline so the default keeps its behavior.
        if input_embeddings.is_some() {
            return None;
        }
        Some(self.forward_rows_for_sequence(input_ids, seq_id, Some(last_pos)))
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_batched_with_ids(input_ids, None, batch_caches, mask)
    }

    fn forward_batched_with_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(input_ids);
        assert_eq!(shape.len(), 2, "raon1_mini batched input must be rank 2");
        assert_eq!(
            shape[0] as usize,
            batch_caches.len(),
            "raon1_mini input/cache row mismatch"
        );
        route_batched_decode_with_ids(self, input_ids, seq_ids, batch_caches, |_| {
            let ids = seq_ids.expect("raon1_mini model-owned batched decode requires sequence ids");
            assert_eq!(
                ids.len(),
                shape[0] as usize,
                "raon1_mini: one sequence id per row"
            );
            self.sequence_state
                .validate_sequence_states(ids)
                .unwrap_or_else(|err| panic!("raon1_mini batched decode: {err}"));
            self.sequence_state
                .with_batched_sequence_states(ids, |rows| {
                    self.validate_batch_cache_layout(rows);
                    if shape[1] != 1 || mask.is_some() || !self.batch_states_uniform(rows) {
                        self.forward_rows_with_states(input_ids, rows)
                    } else {
                        self.forward_batched_decode_with_caches(input_ids, rows)
                    }
                })
                .unwrap_or_else(|err| panic!("raon1_mini batched decode: {err}"))
        })
    }
}
