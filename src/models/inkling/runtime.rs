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

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use super::InklingModel;
use crate::models::recurrent_snapshot::{push_i32, push_optional, restore_i32, restore_optional};

pub(crate) struct InklingLayerCache {
    pub(crate) kv: KVCache,
    pub(crate) conv: [Option<UniquePtr<MlxArray>>; 4],
}

impl InklingLayerCache {
    pub(crate) fn new() -> Self {
        Self {
            kv: KVCache::new(),
            conv: std::array::from_fn(|_| None),
        }
    }

    fn snapshot_into(
        &self,
        snapshot: &mut mlxcel_core::generate::ModelStateSnapshot,
        prefix: &str,
    ) {
        let (keys, values) = self
            .kv
            .visible_state()
            .map_or((None, None), |(keys, values)| (Some(keys), Some(values)));
        push_optional(snapshot, format!("{prefix}.keys"), &keys);
        push_optional(snapshot, format!("{prefix}.values"), &values);
        push_i32(snapshot, format!("{prefix}.offset"), self.kv.offset);
        for (idx, state) in self.conv.iter().enumerate() {
            push_optional(snapshot, format!("{prefix}.conv{idx}"), state);
        }
    }

    fn restore_from(
        &mut self,
        snapshot: &mlxcel_core::generate::ModelStateSnapshot,
        prefix: &str,
    ) -> Result<(), String> {
        let keys = restore_optional(snapshot, format!("{prefix}.keys"));
        let values = restore_optional(snapshot, format!("{prefix}.values"));
        let offset = restore_i32(snapshot, format!("{prefix}.offset"))
            .unwrap_or(snapshot.token_len() as i32);
        self.kv.restore_fp16_live_window(keys, values, offset)?;
        for (idx, state) in self.conv.iter_mut().enumerate() {
            *state = restore_optional(snapshot, format!("{prefix}.conv{idx}"));
        }
        Ok(())
    }
}

impl LanguageModel for InklingModel {
    fn forward(
        &self,
        input: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state
            .with_sequence_state(None, |state| self.forward_with_caches(input, state))
    }

    fn forward_last_logits(
        &self,
        input: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_sequence_state(None, |state| {
            let embeddings = self.embed_tokens.forward(input);
            self.forward_last_embeddings_with_caches(&embeddings, state, last_pos)
        })
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.sequence_state
            .replace_internal(self.make_internal_caches());
        (0..self.layers.len()).map(|_| KVCache::new()).collect()
    }
    fn num_layers(&self) -> usize {
        self.layers.len()
    }
    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
    fn supports_batching(&self) -> bool {
        false
    }
    fn supports_padded_prefill(&self) -> bool {
        false
    }
    fn prepare_sequence_state(&self, seq_id: mlxcel_core::cache::SequenceId) {
        self.sequence_state
            .prepare_sequence_state(seq_id, self.make_internal_caches());
    }
    fn release_sequence_state_by_id(&self, seq_id: mlxcel_core::cache::SequenceId) {
        self.sequence_state.release_sequence_state(seq_id);
    }
    fn forward_with_sequence_id(
        &self,
        input: &MlxArray,
        seq_id: Option<mlxcel_core::cache::SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |state| self.forward_with_caches(input, state),
        )
    }
    fn forward_last_logits_with_sequence_id(
        &self,
        input: &MlxArray,
        seq_id: Option<mlxcel_core::cache::SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |state| {
                let embeddings = self.embed_tokens.forward(input);
                self.forward_last_embeddings_with_caches(&embeddings, state, last_pos)
            },
        )
    }
    fn forward_with_embeddings(
        &self,
        input: &MlxArray,
        embeddings: Option<&MlxArray>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state
            .with_sequence_state(None, |state| match embeddings {
                Some(embeddings) => self.forward_embeddings_with_caches(embeddings, state),
                None => self.forward_with_caches(input, state),
            })
    }
    fn forward_with_embeddings_and_sequence_id(
        &self,
        input: &MlxArray,
        embeddings: Option<&MlxArray>,
        seq_id: Option<mlxcel_core::cache::SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |state| match embeddings {
                Some(embeddings) => self.forward_embeddings_with_caches(embeddings, state),
                None => self.forward_with_caches(input, state),
            },
        )
    }
    fn forward_last_logits_with_embeddings_and_sequence_id(
        &self,
        input: &MlxArray,
        embeddings: Option<&MlxArray>,
        seq_id: Option<mlxcel_core::cache::SequenceId>,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |state| match embeddings {
                Some(embeddings) => {
                    self.forward_last_embeddings_with_caches(embeddings, state, last_pos)
                }
                None => {
                    let embeddings = self.embed_tokens.forward(input);
                    self.forward_last_embeddings_with_caches(&embeddings, state, last_pos)
                }
            },
        )
    }
    fn embed_tokens(&self, input: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.embed_tokens.forward(input))
    }
    fn supports_snapshot_reuse(&self) -> bool {
        true
    }
    fn snapshot_sequence_state(
        &self,
        seq_id: mlxcel_core::cache::SequenceId,
        token_len: usize,
    ) -> Option<mlxcel_core::generate::ModelStateSnapshot> {
        self.sequence_state
            .with_sequence_state_ref(seq_id, |state| {
                let mut snapshot =
                    mlxcel_core::generate::ModelStateSnapshot::new("inkling", token_len);
                for (i, cache) in state.iter().enumerate() {
                    cache.snapshot_into(&mut snapshot, &format!("layer{i}"));
                }
                snapshot
            })
    }
    fn restore_sequence_state(
        &self,
        seq_id: mlxcel_core::cache::SequenceId,
        snapshot: &mlxcel_core::generate::ModelStateSnapshot,
    ) -> Result<(), String> {
        if snapshot.family() != "inkling" {
            return Err(format!(
                "cannot restore {} snapshot into Inkling",
                snapshot.family()
            ));
        }
        let mut state = self.make_internal_caches();
        for (i, cache) in state.iter_mut().enumerate() {
            cache.restore_from(snapshot, &format!("layer{i}"))?;
        }
        self.sequence_state.replace_sequence_state(seq_id, state);
        Ok(())
    }
    fn trim_internal_caches(&self, excess: i32) {
        if excess <= 0 {
            return;
        }
        self.sequence_state.with_sequence_state(None, |state| {
            for cache in state {
                // This hook removes speculative or padded TAIL tokens. The
                // recurrent conv state cannot be positionally rewound, so
                // clear it exactly as the other recurrent model families do.
                cache.kv.trim(excess);
                for state in &mut cache.conv {
                    *state = None;
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlxcel_core::dtype;
    use mlxcel_core::generate::ModelStateSnapshot;

    #[test]
    fn trimmed_sliding_cache_snapshot_restores_absolute_window() {
        let mut cache = InklingLayerCache::new();
        let keys = mlxcel_core::zeros(&[1, 1, 10, 2], dtype::FLOAT32);
        let values = mlxcel_core::ones(&[1, 1, 10, 2], dtype::FLOAT32);
        let _ = cache.kv.update_and_fetch(keys, values);
        assert_eq!(cache.kv.trim_front(6), 6);
        let next_k = mlxcel_core::zeros(&[1, 1, 1, 2], dtype::FLOAT32);
        let next_v = mlxcel_core::ones(&[1, 1, 1, 2], dtype::FLOAT32);
        let _ = cache.kv.update_and_fetch(next_k, next_v);
        cache.conv[0] = Some(mlxcel_core::ones(&[1, 3, 2], dtype::FLOAT32));

        let mut snapshot = ModelStateSnapshot::new("inkling", 11);
        cache.snapshot_into(&mut snapshot, "layer0");
        let mut restored = InklingLayerCache::new();
        restored.restore_from(&snapshot, "layer0").unwrap();
        assert_eq!(restored.kv.offset, 11);
        assert_eq!(
            mlxcel_core::array_shape(restored.kv.keys.as_deref().unwrap()),
            [1, 1, 5, 2]
        );
        assert_eq!(
            mlxcel_core::array_shape(restored.conv[0].as_deref().unwrap()),
            [1, 3, 2]
        );

        let next_k = mlxcel_core::zeros(&[1, 1, 1, 2], dtype::FLOAT32);
        let next_v = mlxcel_core::ones(&[1, 1, 1, 2], dtype::FLOAT32);
        let (visible_k, _) = restored.kv.update_and_fetch(next_k, next_v);
        assert_eq!(restored.kv.offset, 12);
        assert_eq!(mlxcel_core::array_shape(&visible_k), [1, 1, 6, 2]);
    }
}
