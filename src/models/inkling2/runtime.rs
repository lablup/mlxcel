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

//! `LanguageModel` for Inkling 2: model-owned per-visit state keyed by
//! sequence id, as Inkling 1.

use mlxcel_core::cache::{CachePool, SequenceId};
use mlxcel_core::generate::{BatchedDecodeKind, LanguageModel, ModelStateSnapshot};
use mlxcel_core::inkling_layer::InklingLayerCache;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use super::Inkling2Model;
use super::conv::{can_rewind, rewind_history};

const SNAPSHOT_FAMILY: &str = "inkling2";

impl Inkling2Model {
    /// Remove the newest `n` tokens from every visit, or change nothing and
    /// fail when a sliding window or a convolution history cannot rewind that
    /// far (see the KV layout notes in `attention.rs` and `conv.rs`).
    pub(crate) fn trim_caches(
        &self,
        caches: &mut [InklingLayerCache],
        n: i32,
    ) -> Result<(), String> {
        for (visit, cache) in caches.iter().enumerate() {
            let layer = &self.layers[self.layer_indices[visit]];
            let kernel = layer.conv_kernel();
            let attention = layer.attention();
            let (live, offset) = (cache.kv.live_len(), cache.kv.offset);
            if n > offset {
                return Err(format!(
                    "Inkling 2 visit {visit}: cannot trim {n} tokens from a {offset}-token cache"
                ));
            }
            if attention.is_sliding() && live - n < (offset - n).min(attention.window() - 1) {
                return Err(format!(
                    "Inkling 2 visit {visit}: the sliding window keeps {live} keys, too few to trim {n}"
                ));
            }
            if !cache
                .conv
                .iter()
                .all(|state| can_rewind(state.as_deref(), n, kernel))
            {
                return Err(format!(
                    "Inkling 2 visit {visit}: convolution history is too short to trim {n} tokens"
                ));
            }
        }
        for (visit, cache) in caches.iter_mut().enumerate() {
            let kernel = self.layers[self.layer_indices[visit]].conv_kernel();
            cache.kv.trim(n);
            for state in &mut cache.conv {
                if let Some(history) = state.as_ref().and_then(|value| value.as_ref()) {
                    *state = Some(rewind_history(history, n, kernel)?);
                }
            }
        }
        Ok(())
    }
}

impl LanguageModel for Inkling2Model {
    fn forward(
        &self,
        input: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state
            .with_sequence_state(None, |state| self.token_logits(input, state, None))
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
        Some(self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |state| match input_embeddings {
                Some(embeddings) => {
                    let normalized = self.embed_norm.forward(embeddings);
                    self.logits(&normalized, state, Some(last_pos))
                }
                None => self.token_logits(input_ids, state, Some(last_pos)),
            },
        ))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.sequence_state
            .replace_internal(self.make_internal_caches());
        (0..self.layer_indices.len())
            .map(|_| KVCache::new())
            .collect()
    }

    fn num_layers(&self) -> usize {
        self.layer_indices.len()
    }

    /// Every visit keeps an FP16 `KVCache` beside its convolution states
    /// (issue #2790: report the real storage rather than a mode table).
    fn kv_cache_storage_modes(&self) -> Option<Vec<Option<mlxcel_core::cache::KVCacheMode>>> {
        crate::models::model_owned::table_less_storage_modes(self.layer_indices.len(), |_| {
            mlxcel_core::cache::LayerStateKind::Full
        })
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }

    fn supports_batching(&self) -> bool {
        true
    }

    /// Each row runs its own single-sequence forward against its own state.
    fn batched_decode_kind(&self) -> BatchedDecodeKind {
        BatchedDecodeKind::PerRowFallback
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_batched_with_ids(input_ids, None, caches, mask)
    }

    fn forward_batched_with_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let ids = seq_ids.expect("Inkling 2 batched decode requires one sequence id per row");
        assert_eq!(
            ids.len(),
            batch_caches.len(),
            "Inkling 2 requires one sequence id per row"
        );
        let outputs: Vec<_> = batch_caches
            .iter_mut()
            .enumerate()
            .map(|(row, caches)| {
                let input =
                    mlxcel_core::utils::slice_axis(input_ids, 0, row as i32, row as i32 + 1);
                self.forward_with_sequence_id(&input, Some(ids[row]), caches, mask)
            })
            .collect();
        let refs: Vec<_> = outputs.iter().map(|output| &**output).collect();
        mlxcel_core::concatenate_many(&refs, 0)
    }

    fn sequence_state_layout(&self) -> mlxcel_core::cache::SequenceStateLayout {
        mlxcel_core::cache::SequenceStateLayout::model_owned(self.layer_indices.len())
    }

    fn supports_padded_prefill(&self) -> bool {
        false
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.sequence_state
            .prepare_sequence_state(seq_id, self.make_internal_caches());
    }

    fn sync_sequence_storage(
        &self,
        seq_id: SequenceId,
        cache_pool: &mut CachePool,
    ) -> Result<(), String> {
        self.sequence_state
            .with_sequence_state_ref(seq_id, |caches| {
                let lens: Vec<_> = caches
                    .iter()
                    .map(|cache| cache.kv.seq_len().max(0) as usize)
                    .collect();
                cache_pool.sync_paged_state_with_lengths(seq_id, &lens)
            })
            .unwrap_or_else(|| {
                Err(format!(
                    "missing Inkling 2 sequence state for sequence {seq_id}"
                ))
            })
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.sequence_state.release_sequence_state(seq_id);
    }

    fn forward_with_sequence_id(
        &self,
        input: &MlxArray,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |state| self.token_logits(input, state, None),
        )
    }

    fn forward_with_embeddings(
        &self,
        input: &MlxArray,
        embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_embeddings_and_sequence_id(input, embeddings, None, caches, mask)
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input: &MlxArray,
        embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |state| match embeddings {
                Some(embeddings) => self.logits(&self.embed_norm.forward(embeddings), state, None),
                None => self.token_logits(input, state, None),
            },
        )
    }

    fn embed_tokens(&self, input: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.embed_tokens.forward(input))
    }

    fn embed_tokens_module(&self) -> Option<mlxcel_core::layers::UnifiedEmbedding> {
        Some(self.embed_tokens.clone_shared())
    }

    fn inkling2_dspark_binding(
        &self,
    ) -> Option<mlxcel_core::drafter::inkling2_dspark::Inkling2DsparkBinding> {
        Some(self.dspark_binding())
    }

    fn supports_snapshot_reuse(&self) -> bool {
        true
    }

    fn snapshot_sequence_state(
        &self,
        seq_id: SequenceId,
        token_len: usize,
    ) -> Option<ModelStateSnapshot> {
        self.sequence_state
            .with_sequence_state_ref(seq_id, |state| {
                let mut snapshot = ModelStateSnapshot::new(SNAPSHOT_FAMILY, token_len);
                for (visit, cache) in state.iter().enumerate() {
                    cache.snapshot_into(&mut snapshot, &format!("visit{visit}"));
                }
                snapshot
            })
    }

    fn restore_sequence_state(
        &self,
        seq_id: SequenceId,
        snapshot: &ModelStateSnapshot,
    ) -> Result<(), String> {
        if snapshot.family() != SNAPSHOT_FAMILY {
            return Err(format!(
                "cannot restore {} snapshot into Inkling 2",
                snapshot.family()
            ));
        }
        let mut state = self.make_internal_caches();
        for (visit, cache) in state.iter_mut().enumerate() {
            cache.restore_from(snapshot, &format!("visit{visit}"))?;
        }
        self.sequence_state.replace_sequence_state(seq_id, state);
        Ok(())
    }

    /// A verify block keeps at least as many older convolution rows and
    /// sliding keys as it appends (`conv::history_margin`, the sliding
    /// margin in `attention.rs`), so [`Self::rewind_decode_appends`] can stop
    /// anywhere inside it. This is what lets prompt-lookup verify rounds, and
    /// with them the grouped expert kernel (#3359), run on Inkling 2.
    fn supports_speculative_verify_rewind(&self) -> bool {
        true
    }

    /// Unwind the newest `n` positions of `seq_id` through [`Self::trim_state`],
    /// which refuses without mutation when a margin is too short.
    fn rewind_decode_appends(&self, seq_id: SequenceId, n: i32) -> Result<(), String> {
        self.trim_state(Some(seq_id), n)
    }

    fn trim_state(&self, seq: Option<SequenceId>, excess: i32) -> Result<(), String> {
        if excess <= 0 {
            return Ok(());
        }
        match seq {
            None => self
                .sequence_state
                .with_sequence_state(None, |state| self.trim_caches(state, excess)),
            Some(seq_id) => self
                .sequence_state
                .with_existing_sequence_state(seq_id, |state| self.trim_caches(state, excess))?,
        }
    }
}
