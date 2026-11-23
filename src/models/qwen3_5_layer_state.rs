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

//! Qwen 3.5 over scheduler-owned layer state (issue #3254, ADR 0010).
//!
//! The pool holds one [`LayerState`] per layer: a `Kv` for a full-attention
//! layer and a `Recurrent` with the GatedDeltaNet slots for a linear one. The
//! decoder layers, the batched decode, the speculative verify and its
//! rollback are written over [`Qwen3NextCache`], which Qwen3-Next and
//! Qwen4-Exp share and still own. A forward therefore borrows the sequence's
//! state as that typed view for its duration: [`LentCaches`] moves each
//! layer's cache out of the pool's vector and moves it back when dropped. No
//! tensor is copied, and between forwards the state lives only in the pool.

use std::ops::{Deref, DerefMut};

use mlxcel_core::cache::{LayerState, RecurrentState};

use super::gated_delta::GatedDeltaCache;
use super::qwen3_next::Qwen3NextCache;

/// The GatedDeltaNet tensors of a linear layer, in snapshot order.
pub(crate) const RECURRENT_SLOTS: [&str; 2] = ["conv_state", "state_cache"];

/// The pool form of one layer's cache.
pub(crate) fn into_layer_state(cache: Qwen3NextCache) -> LayerState {
    match cache {
        Qwen3NextCache::Attention(kv) => LayerState::Kv(kv),
        Qwen3NextCache::Linear(delta) => {
            let mut state = RecurrentState::with_slots(&RECURRENT_SLOTS);
            state.put(RECURRENT_SLOTS[0], delta.conv_state);
            state.put(RECURRENT_SLOTS[1], delta.state_cache);
            state.offset = delta.offset;
            LayerState::Recurrent(state)
        }
    }
}

/// One sequence's layer state, borrowed as the typed caches the Qwen 3.5
/// forward runs over. Dropping it returns every cache to the pool's vector.
pub(crate) struct LentCaches<'a> {
    state: &'a mut [LayerState],
    caches: Vec<Qwen3NextCache>,
}

impl<'a> LentCaches<'a> {
    /// Borrow `state`. Every layer is checked before any is moved, so a
    /// refusal leaves the state as it was.
    ///
    /// # Errors
    ///
    /// A layer that is neither attention KV nor recurrent state.
    pub(crate) fn take(state: &'a mut [LayerState]) -> Result<Self, String> {
        if let Some((layer_idx, layer)) = state
            .iter()
            .enumerate()
            .find(|(_, layer)| !matches!(layer, LayerState::Kv(_) | LayerState::Recurrent(_)))
        {
            return Err(format!(
                "Qwen 3.5 forward over a {} layer {layer_idx}: the layer state does not match \
                 layer_state_kinds()",
                layer.storage_name()
            ));
        }
        // A recurrent layer keeps its (now empty) slot table in place while
        // the tensors are lent, so returning them allocates nothing per
        // layer and step.
        let caches = state
            .iter_mut()
            .map(|layer| match layer {
                LayerState::Recurrent(recurrent) => Qwen3NextCache::Linear(GatedDeltaCache {
                    conv_state: recurrent.take(RECURRENT_SLOTS[0]),
                    state_cache: recurrent.take(RECURRENT_SLOTS[1]),
                    offset: recurrent.offset,
                }),
                _ => match std::mem::replace(layer, LayerState::Empty) {
                    LayerState::Kv(kv) => Qwen3NextCache::Attention(kv),
                    _ => unreachable!("every layer was checked above"),
                },
            })
            .collect();
        Ok(Self { state, caches })
    }

    /// The caches as an owned vector, for the batched decode that takes one
    /// per row. [`Self::replace`] hands them back.
    pub(crate) fn take_vec(&mut self) -> Vec<Qwen3NextCache> {
        std::mem::take(&mut self.caches)
    }

    /// Hand back the caches [`Self::take_vec`] moved out, or install the
    /// caches a batched prefill split off for this sequence.
    pub(crate) fn replace(&mut self, caches: Vec<Qwen3NextCache>) {
        debug_assert_eq!(caches.len(), self.state.len());
        self.caches = caches;
    }
}

impl Deref for LentCaches<'_> {
    type Target = [Qwen3NextCache];

    fn deref(&self) -> &[Qwen3NextCache] {
        &self.caches
    }
}

impl DerefMut for LentCaches<'_> {
    fn deref_mut(&mut self) -> &mut [Qwen3NextCache] {
        &mut self.caches
    }
}

impl Drop for LentCaches<'_> {
    fn drop(&mut self) {
        for (layer, cache) in self.state.iter_mut().zip(self.caches.drain(..)) {
            match (layer, cache) {
                // Back into the slot table `take` left in place.
                (LayerState::Recurrent(recurrent), Qwen3NextCache::Linear(delta)) => {
                    recurrent.put(RECURRENT_SLOTS[0], delta.conv_state);
                    recurrent.put(RECURRENT_SLOTS[1], delta.state_cache);
                    recurrent.offset = delta.offset;
                }
                (layer, cache) => *layer = into_layer_state(cache),
            }
        }
    }
}
