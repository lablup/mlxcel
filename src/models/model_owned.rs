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

use std::cell::RefCell;
use std::collections::HashMap;

use mlxcel_core::cache::{PagedDecodeMetadata, SequenceId};
use mlxcel_core::generate::DecodeBatchContext;
use mlxcel_core::{MlxArray, UniquePtr};

pub(crate) struct ModelOwnedSequenceState<T> {
    internal: RefCell<Vec<T>>,
    sequences: RefCell<HashMap<SequenceId, Vec<T>>>,
}

impl<T> ModelOwnedSequenceState<T> {
    pub(crate) fn new(internal: Vec<T>) -> Self {
        Self {
            internal: RefCell::new(internal),
            sequences: RefCell::new(HashMap::new()),
        }
    }

    pub(crate) fn replace_internal(&self, internal: Vec<T>) {
        *self.internal.borrow_mut() = internal;
    }

    pub(crate) fn prepare_sequence_state(&self, seq_id: SequenceId, state: Vec<T>) {
        self.sequences.borrow_mut().insert(seq_id, state);
    }

    pub(crate) fn replace_sequence_state(&self, seq_id: SequenceId, state: Vec<T>) {
        self.sequences.borrow_mut().insert(seq_id, state);
    }

    pub(crate) fn with_sequence_state_ref<R>(
        &self,
        seq_id: SequenceId,
        f: impl FnOnce(&[T]) -> R,
    ) -> Option<R> {
        let sequences = self.sequences.borrow();
        sequences.get(&seq_id).map(|state| f(state.as_slice()))
    }

    pub(crate) fn with_sequence_state<R>(
        &self,
        seq_id: Option<SequenceId>,
        f: impl FnOnce(&mut [T]) -> R,
    ) -> R {
        let sequence_state = seq_id.and_then(|id| self.sequences.borrow_mut().remove(&id));
        if let Some(mut sequence_state) = sequence_state {
            let result = f(&mut sequence_state);
            self.sequences
                .borrow_mut()
                .insert(seq_id.expect("sequence id must exist"), sequence_state);
            return result;
        }

        let mut internal = self.internal.borrow_mut();
        f(&mut internal)
    }

    pub(crate) fn with_or_create_sequence_state<R>(
        &self,
        seq_id: Option<SequenceId>,
        init: impl FnOnce() -> Vec<T>,
        f: impl FnOnce(&mut [T]) -> R,
    ) -> R {
        if let Some(seq_id) = seq_id {
            let mut sequence_state = self
                .sequences
                .borrow_mut()
                .remove(&seq_id)
                .unwrap_or_else(init);
            let result = f(&mut sequence_state);
            self.sequences.borrow_mut().insert(seq_id, sequence_state);
            return result;
        }

        let mut internal = self.internal.borrow_mut();
        f(&mut internal)
    }

    pub(crate) fn with_batched_sequence_states<R>(
        &self,
        seq_ids: &[SequenceId],
        f: impl FnOnce(&mut [Vec<T>]) -> R,
    ) -> Result<R, String> {
        let mut extracted = Vec::with_capacity(seq_ids.len());
        {
            let mut sequences = self.sequences.borrow_mut();
            for &seq_id in seq_ids {
                let state = sequences.remove(&seq_id).ok_or_else(|| {
                    format!("missing model-owned sequence state for sequence {seq_id}")
                })?;
                extracted.push(state);
            }
        }

        let result = f(&mut extracted);

        let mut sequences = self.sequences.borrow_mut();
        for (seq_id, state) in seq_ids.iter().copied().zip(extracted) {
            sequences.insert(seq_id, state);
        }
        Ok(result)
    }

    pub(crate) fn release_sequence_state(&self, seq_id: SequenceId) {
        self.sequences.borrow_mut().remove(&seq_id);
    }
}

// Used by: Gemma3 paged decode, model-owned cache families with materialized visible views
pub(crate) fn dispatch_paged_decode_from_visible_caches<C, F>(
    q_batched: &MlxArray,
    k_batched: &MlxArray,
    v_batched: &MlxArray,
    caches: &mut [&mut C],
    scale: f32,
    context: &DecodeBatchContext,
    mut update_and_fetch_visible: F,
) -> Result<Option<UniquePtr<MlxArray>>, String>
where
    F: FnMut(
        &mut C,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>), String>,
{
    if !context.is_paged_decode() {
        return Ok(None);
    }

    let q_shape = mlxcel_core::array_shape(q_batched);
    if q_shape.len() != 4 || q_shape[2] != 1 {
        return Ok(None);
    }

    let mut visible_keys = Vec::with_capacity(caches.len());
    let mut visible_values = Vec::with_capacity(caches.len());
    let mut cache_keys = Vec::with_capacity(caches.len());
    let mut cache_values = Vec::with_capacity(caches.len());
    let mut kv_lens = Vec::with_capacity(caches.len());

    for (batch_idx, cache) in caches.iter_mut().enumerate() {
        let k_i = mlxcel_core::slice(
            k_batched,
            &[batch_idx as i32, 0, 0, 0],
            &[batch_idx as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
        );
        let v_i = mlxcel_core::slice(
            v_batched,
            &[batch_idx as i32, 0, 0, 0],
            &[batch_idx as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
        );

        let (visible_k, visible_v) = update_and_fetch_visible(cache, k_i, v_i)?;
        let visible_len = mlxcel_core::array_shape(&visible_k)
            .get(2)
            .copied()
            .unwrap_or_default();

        kv_lens.push(visible_len);
        cache_keys.push(
            visible_k
                .as_ref()
                .ok_or_else(|| "visible key cache missing backing array".to_string())?
                as *const MlxArray,
        );
        cache_values.push(
            visible_v
                .as_ref()
                .ok_or_else(|| "visible value cache missing backing array".to_string())?
                as *const MlxArray,
        );
        visible_keys.push(visible_k);
        visible_values.push(visible_v);
    }

    let metadata = PagedDecodeMetadata::from_visible_lengths(&kv_lens, context.paged_block_size)?;
    let attn = if context.use_native_paged_kernel {
        mlxcel_core::layers::paged_decode_attention_dense_compat(
            q_batched,
            &cache_keys,
            &cache_values,
            &metadata,
            scale,
        )
    } else {
        mlxcel_core::layers::paged_decode_attention_dense_fallback(
            q_batched,
            &cache_keys,
            &cache_values,
            &metadata,
            scale,
        )
    }?;

    Ok(Some(attn))
}

// Used by: Llama4 paged decode, future chunked/sliding model-owned cache families
pub(crate) fn dispatch_paged_decode_from_backing_caches<C, F, G, H, I>(
    q_batched: &MlxArray,
    k_batched: &MlxArray,
    v_batched: &MlxArray,
    caches: &mut [&mut C],
    scale: f32,
    context: &DecodeBatchContext,
    mut update_cache: F,
    mut keys_ptr: G,
    mut values_ptr: H,
    mut visible_len: I,
) -> Result<Option<UniquePtr<MlxArray>>, String>
where
    F: FnMut(&mut C, UniquePtr<MlxArray>, UniquePtr<MlxArray>) -> Result<(), String>,
    G: FnMut(&C) -> Result<*const MlxArray, String>,
    H: FnMut(&C) -> Result<*const MlxArray, String>,
    I: FnMut(&C) -> Result<i32, String>,
{
    if !context.is_paged_decode() {
        return Ok(None);
    }

    let q_shape = mlxcel_core::array_shape(q_batched);
    if q_shape.len() != 4 || q_shape[2] != 1 {
        return Ok(None);
    }

    let mut cache_keys = Vec::with_capacity(caches.len());
    let mut cache_values = Vec::with_capacity(caches.len());
    let mut kv_lens = Vec::with_capacity(caches.len());

    for (batch_idx, cache) in caches.iter_mut().enumerate() {
        let k_i = mlxcel_core::slice(
            k_batched,
            &[batch_idx as i32, 0, 0, 0],
            &[batch_idx as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
        );
        let v_i = mlxcel_core::slice(
            v_batched,
            &[batch_idx as i32, 0, 0, 0],
            &[batch_idx as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
        );

        update_cache(cache, k_i, v_i)?;
        kv_lens.push(visible_len(cache)?);
        cache_keys.push(keys_ptr(cache)?);
        cache_values.push(values_ptr(cache)?);
    }

    let metadata = PagedDecodeMetadata::from_visible_lengths(&kv_lens, context.paged_block_size)?;
    let attn = if context.use_native_paged_kernel {
        mlxcel_core::layers::paged_decode_attention_dense_compat(
            q_batched,
            &cache_keys,
            &cache_values,
            &metadata,
            scale,
        )
    } else {
        mlxcel_core::layers::paged_decode_attention_dense_fallback(
            q_batched,
            &cache_keys,
            &cache_values,
            &metadata,
            scale,
        )
    }?;

    Ok(Some(attn))
}
