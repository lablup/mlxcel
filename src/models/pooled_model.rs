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

//! Driver for a model outside a scheduler (issue #3254).
//!
//! A scheduler-owned family keeps no per-sequence state in its wrapper, so a
//! caller that holds only the model can no longer prepare a sequence on it and
//! forward it by id. [`Pooled`] owns the model and a real [`CachePool`], and
//! exposes the by-id `LanguageModel` calls such callers were written against:
//! each one resolves the caller's sequence id to a pool sequence and goes
//! through the entry points the engine and the prompt cache use
//! (`forward_layer_state`, the per-layer rewinds,
//! `server::prompt_cache::state_snapshot`). A family that still owns its state
//! is passed through unchanged, so one driver serves any model. It derefs to
//! the model for the wrapper's own methods.
//!
//! Used by: model-level tests, the logit-trace example and the parity tests
//! under `tests/`. The server and the CLI run on `mlxcel_core::engine::Engine`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Deref;

use mlxcel_core::cache::{
    CachePool, KVCacheMode, LayerState, LayerStateForward, PagedKvLayout, SequenceId,
    SequenceStateLayout,
};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct Pooled<M: LanguageModel> {
    model: M,
    /// Whether the model's state is the pool's. `false` passes every call
    /// through to the model.
    scheduler_owned: bool,
    pool: RefCell<CachePool>,
    /// The test's sequence ids, mapped to the ids the pool assigned.
    ids: RefCell<HashMap<SequenceId, SequenceId>>,
    /// The layout every sequence is opened with, `None` for the model's.
    layout: Option<SequenceStateLayout>,
}

impl<M: LanguageModel> Deref for Pooled<M> {
    type Target = M;

    fn deref(&self) -> &M {
        &self.model
    }
}

impl<M: LanguageModel> std::ops::DerefMut for Pooled<M> {
    fn deref_mut(&mut self) -> &mut M {
        &mut self.model
    }
}

impl<M: LanguageModel> Pooled<M> {
    pub fn new(model: M) -> Self {
        Self {
            scheduler_owned: model.layer_state_kinds().is_some(),
            model,
            pool: RefCell::new(CachePool::new(64)),
            ids: RefCell::new(HashMap::new()),
            layout: None,
        }
    }

    /// Open every sequence on `block_size`-token paged blocks, the layout a
    /// paged sliding family's sequences take (issue #3256).
    pub fn with_paged_blocks(model: M, block_size: usize) -> Self {
        let layout = PagedKvLayout::uniform(model.num_layers(), block_size, block_size)
            .expect("paged test layout");
        Self {
            layout: Some(SequenceStateLayout::paged_kv_cache(layout)),
            ..Self::new(model)
        }
    }

    /// Run `f` over the pool, for block accounting and detach tests.
    pub fn with_pool<R>(&self, f: impl FnOnce(&mut CachePool) -> R) -> R {
        f(&mut self.pool.borrow_mut())
    }

    /// The pool id behind the test's `seq`, opening it on first use.
    pub fn pool_sequence(&self, seq: SequenceId) -> SequenceId {
        self.pool_id(seq)
    }

    /// The pool sequence behind `seq`, opened on first use.
    fn pool_id(&self, seq: SequenceId) -> SequenceId {
        if let Some(id) = self.ids.borrow().get(&seq) {
            return *id;
        }
        let id = self
            .pool
            .borrow_mut()
            .allocate_with_layout(&self.model, self.layout.clone())
            .expect("pool sequence");
        self.ids.borrow_mut().insert(seq, id);
        id
    }

    /// Whether `seq` has a pool sequence.
    pub fn has_sequence_state(&self, seq: SequenceId) -> bool {
        self.ids.borrow().contains_key(&seq)
    }

    /// Read `seq`'s layer state.
    pub fn with_layer_state<R>(&self, seq: SequenceId, f: impl FnOnce(&[LayerState]) -> R) -> R {
        let id = self.pool_id(seq);
        let pool = self.pool.borrow();
        f(pool.layer_state(id).expect("scheduler-owned sequence"))
    }

    /// Mutate `seq`'s layer state.
    pub fn with_layer_state_mut<R>(
        &self,
        seq: SequenceId,
        f: impl FnOnce(&mut [LayerState]) -> R,
    ) -> R {
        let id = self.pool_id(seq);
        let mut pool = self.pool.borrow_mut();
        f(pool.layer_state_mut(id).expect("scheduler-owned sequence"))
    }

    /// The KV cache mode each of `seq`'s attention layers was built with.
    pub fn sequence_kv_modes(&self, seq: SequenceId) -> Vec<KVCacheMode> {
        self.with_layer_state(seq, |state| {
            state
                .iter()
                .filter_map(|layer| match layer {
                    LayerState::Kv(cache) => Some(cache.mode),
                    LayerState::Rotating(cache) => Some(cache.mode),
                    _ => None,
                })
                .collect()
        })
    }

    fn forward_row(&self, seq: SequenceId, request: LayerStateForward<'_>) -> UniquePtr<MlxArray> {
        let ids = [seq];
        let request = request.with_seq_ids(Some(&ids));
        self.with_layer_state_mut(seq, |state| {
            self.model
                .forward_layer_state(&request, &mut [state])
                .expect("scheduler-owned forward")
        })
    }

    /// The batched forward over the rows' pool state. `Err` is the model's
    /// refusal or the pool's (a duplicate or unknown row).
    pub fn try_forward_batched_with_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        _batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let seqs = seq_ids.ok_or("a scheduler-owned batch needs sequence ids")?;
        if let [seq] = seqs {
            // One row keeps the single-sequence contract, lazy open included.
            return Ok(self.forward_row(*seq, LayerStateForward::tokens(input_ids).with_mask(mask)));
        }
        let ids = self.ids.borrow();
        let pool_ids: Vec<SequenceId> = seqs
            .iter()
            .map(|seq| {
                ids.get(seq).copied().ok_or_else(|| {
                    format!("missing scheduler-owned sequence state for sequence {seq}")
                })
            })
            .collect::<Result<_, _>>()?;
        let mut pool = self.pool.borrow_mut();
        let mut rows = pool.batch_layer_state_mut(&pool_ids)?;
        let request = LayerStateForward::tokens(input_ids)
            .with_mask(mask)
            .with_seq_ids(Some(seqs));
        self.model.forward_layer_state(&request, &mut rows)
    }
}

#[path = "pooled_model_language_model.rs"]
mod language_model;
