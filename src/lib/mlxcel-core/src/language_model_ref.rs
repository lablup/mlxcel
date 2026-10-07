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

//! [`LanguageModel`] for a shared reference to a model.
//!
//! [`crate::engine::Engine`] owns its model by value, which is the shape the
//! server's `BatchScheduler` wants. A raw-completion client that borrows a
//! model its caller still needs afterwards (`mlxcel generate` loads the model,
//! generates, then runs the Qwen3-Omni speech pass over the same instance)
//! opens an `Engine<&M>` instead (#2176). Every method forwards, provided ones
//! included, so a family's override is never shadowed by a trait default.

use crate::cache::{CachePool, KVCacheMode, SequenceId, SequenceStateLayout};
use crate::generate::{LanguageModel, ModelStateSnapshot};
use crate::layers::KVCache;
use crate::{MlxArray, UniquePtr};

impl<M: LanguageModel + ?Sized> LanguageModel for &M {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        (**self).forward(input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        (**self).make_caches()
    }

    fn set_kv_cache_layer_modes(&self, modes: Vec<KVCacheMode>) {
        (**self).set_kv_cache_layer_modes(modes)
    }

    fn kv_cache_layer_modes(&self) -> Option<Vec<KVCacheMode>> {
        (**self).kv_cache_layer_modes()
    }

    fn num_layers(&self) -> usize {
        (**self).num_layers()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        (**self).eos_token_ids()
    }

    fn output_suppressed_token_ids(&self) -> Vec<i32> {
        (**self).output_suppressed_token_ids()
    }

    fn supports_chunked_prefill(&self) -> bool {
        (**self).supports_chunked_prefill()
    }

    fn forward_last_logits(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        (**self).forward_last_logits(input_ids, caches, mask, last_pos)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        (**self).forward_with_embeddings(input_ids, input_embeddings, caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        (**self).embed_tokens(input_ids)
    }

    fn embed_tokens_module(&self) -> Option<crate::layers::UnifiedEmbedding> {
        (**self).embed_tokens_module()
    }

    fn lm_head_module(&self) -> Option<crate::layers::UnifiedLinear> {
        (**self).lm_head_module()
    }

    fn final_norm_module(&self) -> Option<crate::layers::RMSNorm> {
        (**self).final_norm_module()
    }

    fn after_prefill(&self) {
        (**self).after_prefill()
    }

    fn trim_state(&self, seq: Option<SequenceId>, excess: i32) -> Result<(), String> {
        (**self).trim_state(seq, excess)
    }

    fn supports_decode_lookahead_rewind(&self) -> bool {
        (**self).supports_decode_lookahead_rewind()
    }

    fn rewind_decode_appends(&self, seq_id: SequenceId, n: i32) -> Result<(), String> {
        (**self).rewind_decode_appends(seq_id, n)
    }

    fn reset_runtime_state(&self) {
        (**self).reset_runtime_state()
    }

    fn release_sequence_state(&self, caches: &mut [KVCache]) {
        (**self).release_sequence_state(caches)
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        (**self).prepare_sequence_state(seq_id)
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        (**self).release_sequence_state_by_id(seq_id)
    }

    fn rope_table_regime(&self, total_prompt_len: usize) -> Option<u8> {
        (**self).rope_table_regime(total_prompt_len)
    }

    fn supports_snapshot_reuse(&self) -> bool {
        (**self).supports_snapshot_reuse()
    }

    fn snapshot_sequence_state(
        &self,
        seq_id: SequenceId,
        token_len: usize,
    ) -> Option<ModelStateSnapshot> {
        (**self).snapshot_sequence_state(seq_id, token_len)
    }

    fn restore_sequence_state(
        &self,
        seq_id: SequenceId,
        snapshot: &ModelStateSnapshot,
    ) -> Result<(), String> {
        (**self).restore_sequence_state(seq_id, snapshot)
    }

    fn snapshot_truncatable_to(&self, snapshot: &ModelStateSnapshot, target_len: usize) -> bool {
        (**self).snapshot_truncatable_to(snapshot, target_len)
    }

    fn restore_sequence_state_truncated(
        &self,
        seq_id: SequenceId,
        snapshot: &ModelStateSnapshot,
        target_len: usize,
    ) -> Result<(), String> {
        (**self).restore_sequence_state_truncated(seq_id, snapshot, target_len)
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        (**self).sequence_state_layout()
    }

    fn supports_padded_prefill(&self) -> bool {
        (**self).supports_padded_prefill()
    }

    fn supports_maskless_padded_prefill(&self) -> bool {
        (**self).supports_maskless_padded_prefill()
    }

    fn supports_batching(&self) -> bool {
        (**self).supports_batching()
    }

    fn supports_paged_decode_backend(&self) -> bool {
        (**self).supports_paged_decode_backend()
    }

    fn supports_batched_prefill(&self) -> bool {
        (**self).supports_batched_prefill()
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        (**self).forward_with_sequence_id(input_ids, seq_id, caches, mask)
    }

    fn forward_last_logits_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        (**self).forward_last_logits_with_sequence_id(input_ids, seq_id, caches, mask, last_pos)
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        (**self).forward_with_embeddings_and_sequence_id(
            input_ids,
            input_embeddings,
            seq_id,
            caches,
            mask,
        )
    }

    fn forward_last_logits_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        (**self).forward_last_logits_with_embeddings_and_sequence_id(
            input_ids,
            input_embeddings,
            seq_id,
            caches,
            mask,
            last_pos,
        )
    }

    fn sync_sequence_storage(
        &self,
        seq_id: SequenceId,
        cache_pool: &mut CachePool,
    ) -> Result<(), String> {
        (**self).sync_sequence_storage(seq_id, cache_pool)
    }

    fn forward_batched_with_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        (**self).forward_batched_with_ids(input_ids, seq_ids, batch_caches, mask)
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        (**self).forward_batched(input_ids, batch_caches, mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model whose `make_caches` override differs from any default, so a
    /// forwarding that fell back to a trait default would be visible.
    struct Overrides;

    impl LanguageModel for Overrides {
        fn forward(
            &self,
            _input_ids: &MlxArray,
            _caches: &mut [KVCache],
            _mask: Option<&MlxArray>,
        ) -> UniquePtr<MlxArray> {
            crate::ffi::zeros(&[1, 1, 1], crate::dtype::FLOAT32)
        }

        fn make_caches(&self) -> Vec<KVCache> {
            Vec::new()
        }

        fn num_layers(&self) -> usize {
            7
        }

        fn eos_token_ids(&self) -> Vec<i32> {
            vec![2]
        }

        fn supports_batching(&self) -> bool {
            false
        }

        fn supports_paged_decode_backend(&self) -> bool {
            true
        }

        fn output_suppressed_token_ids(&self) -> Vec<i32> {
            vec![9]
        }
    }

    fn through_ref<M: LanguageModel>(model: M) -> (usize, bool, bool, Vec<i32>) {
        (
            model.num_layers(),
            model.supports_batching(),
            model.supports_paged_decode_backend(),
            model.output_suppressed_token_ids(),
        )
    }

    #[test]
    fn reference_forwards_overridden_provided_methods() {
        let model = Overrides;
        assert_eq!(through_ref(&model), (7, false, true, vec![9]));
        assert_eq!(
            (&model).sequence_state_layout(),
            model.sequence_state_layout()
        );
    }
}
