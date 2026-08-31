use mlxcel_core::cache::SequenceId;
use mlxcel_core::{MlxArray, UniquePtr};

use super::InklingModel;

impl InklingModel {
    pub(crate) fn forward_prefill_with_hidden_for_sequence(
        &self,
        input: &MlxArray,
        seq_id: Option<SequenceId>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |caches| {
                let embeddings = self.embed_tokens.forward(input);
                let hidden_pre = self.pre_norm_hidden_embeddings_with_caches(&embeddings, caches);
                let hidden_post = self.norm.forward(&hidden_pre);
                (self.project_hidden(&hidden_post), hidden_pre)
            },
        )
    }

    /// Forward a B=1 verify block after capturing every layer's exact KV and
    /// four-convolution pre-forward state.
    pub(crate) fn forward_speculative_for_sequence(
        &self,
        input: &MlxArray,
        seq_id: Option<SequenceId>,
    ) -> (
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        Vec<UniquePtr<MlxArray>>,
        Vec<i32>,
    ) {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |caches| {
                let mut snapshot_tensors = Vec::new();
                let mut snapshot_scalars = Vec::with_capacity(caches.len() * 2);
                for cache in caches.iter() {
                    cache.capture_flat(&mut snapshot_tensors, &mut snapshot_scalars);
                }
                let embeddings = self.embed_tokens.forward(input);
                let hidden_pre = self.pre_norm_hidden_embeddings_with_caches(&embeddings, caches);
                let hidden_post = self.norm.forward(&hidden_pre);
                let logits = self.project_hidden(&hidden_post);
                (logits, hidden_pre, snapshot_tensors, snapshot_scalars)
            },
        )
    }

    /// Restore the pre-verify state and replay the accepted prefix including
    /// its bonus token; tail trimming is insufficient for recurrent conv state.
    pub(crate) fn restore_and_replay_speculative_for_sequence(
        &self,
        replay_input: &MlxArray,
        seq_id: Option<SequenceId>,
        snapshot_tensors: Vec<UniquePtr<MlxArray>>,
        snapshot_scalars: &[i32],
    ) -> Result<(), String> {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |caches| {
                if snapshot_scalars.len() != caches.len() * 2 {
                    return Err(format!(
                        "Inkling snapshot has {} scalars for {} layers",
                        snapshot_scalars.len(),
                        caches.len()
                    ));
                }
                let mut tensors = snapshot_tensors.into_iter();
                let mut scalar_index = 0;
                for cache in caches.iter_mut() {
                    cache.restore_flat(&mut tensors, snapshot_scalars, &mut scalar_index)?;
                }
                if tensors.next().is_some() {
                    return Err("Inkling snapshot contains trailing tensors".into());
                }
                let embeddings = self.embed_tokens.forward(replay_input);
                let _ = self.pre_norm_hidden_embeddings_with_caches(&embeddings, caches);
                Ok(())
            },
        )
    }

    pub(crate) fn speculative_cache_offset_for_sequence(&self, seq_id: Option<SequenceId>) -> i32 {
        self.sequence_state.with_or_create_sequence_state(
            seq_id,
            || self.make_internal_caches(),
            |caches| caches.first().map_or(0, |cache| cache.kv.offset),
        )
    }
}
