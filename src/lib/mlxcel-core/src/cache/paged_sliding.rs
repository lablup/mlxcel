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

//! A sliding-window layer stored as window-bounded blocks of the shared
//! paged pool (issue #3256).
//!
//! The storage is a pool-backed [`KVCache`] whose paged layer carries the
//! window ([`super::PagedLayerState::window`]): every append first moves the
//! layer's `logical_start` to `len - (window - 1)` and retires the head blocks
//! it passed, so the gathered keys are the `window - 1` earlier positions plus
//! the step's own. Attention mirrors [`super::RotatingKVCache::attend`] over
//! that same key set: an unmasked multi-token step runs the windowed causal
//! kernel once the keys outgrow the window, a masked one passes the window as
//! the hint, and a decode step sees exactly `window` keys. Unlike the ring,
//! logical token `t` stays at block `t / block_size`, so the state truncates
//! to any block boundary.

use std::cell::RefCell;
use std::rc::Rc;

use cxx::UniquePtr;

use super::{KVCache, PagedBlockPool, PagedSequenceState};
use crate::ffi::{self, MlxArray};

/// One window-bounded sliding layer of a pool-backed sequence.
pub struct PagedSlidingKVCache {
    kv: KVCache,
    window: i32,
}

impl PagedSlidingKVCache {
    /// A sliding layer over `layer_idx` of `state`, whose paged layer must
    /// already carry `window` (the pool settles it).
    pub fn new(
        pool: Rc<RefCell<PagedBlockPool>>,
        state: Rc<RefCell<PagedSequenceState>>,
        layer_idx: usize,
        window: i32,
    ) -> Self {
        Self {
            kv: KVCache::new_paged(pool, state, layer_idx),
            window,
        }
    }

    /// The attention window.
    pub fn window(&self) -> i32 {
        self.window
    }

    /// Tokens consumed: the RoPE position of the next token.
    pub fn offset(&self) -> i32 {
        self.kv.offset
    }

    /// The pool-backed cache that reads and writes the blocks.
    pub fn kv(&self) -> &KVCache {
        &self.kv
    }

    /// Mutable access for the whole-batch paged decode launch.
    pub fn kv_mut(&mut self) -> &mut KVCache {
        &mut self.kv
    }

    fn with_layer<T>(&self, f: impl FnOnce(&super::PagedLayerState) -> T) -> Option<T> {
        let backing = self.kv.paged_backing.as_ref()?;
        let state = backing.state.borrow();
        state.layer(backing.layer_idx).map(f)
    }

    /// Earlier keys the next append returns with its own: the visible window
    /// once the pool settles it, `min(visible, window - 1)`. Sizes prefill
    /// masks the way `RotatingKVCache::visible_len` does for the ring.
    pub fn live_len(&self) -> i32 {
        let window = self.window.max(1) as usize;
        self.with_layer(|layer| {
            let start = layer
                .logical_start
                .max(layer.len.saturating_sub(window - 1));
            layer.len.saturating_sub(start) as i32
        })
        .unwrap_or(0)
    }

    /// Keys the last step attended (the paged visible window).
    pub fn visible_len(&self) -> usize {
        self.with_layer(super::PagedLayerState::visible_len)
            .unwrap_or(0)
    }

    /// Append `keys` / `values` and return the attended key window.
    pub fn update_and_fetch(
        &mut self,
        keys: UniquePtr<MlxArray>,
        values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        self.kv.update_and_fetch(keys, values)
    }

    /// Append this step's K/V and attend over the window, the
    /// [`super::RotatingKVCache::attend`] contract on paged storage.
    pub fn attend(
        &mut self,
        q: &MlxArray,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let q_len = ffi::array_shape(q)[2];
        let (cache_k, cache_v) = self.kv.update_and_fetch(new_keys, new_values);
        let k_len = ffi::array_shape(&cache_k)[2];
        if mask.is_none() && q_len > 1 {
            let window = if k_len <= self.window { 0 } else { self.window };
            return crate::causal_attention(q, &cache_k, &cache_v, scale, 0.0, window);
        }
        crate::layers::attention(q, &cache_k, &cache_v, scale, mask, 0.0, self.window)
    }

    /// Drop the last `n` positions, bringing back retired head blocks the
    /// shorter window needs. Returns how many were dropped.
    ///
    /// # Errors
    ///
    /// A head block the shorter window needs was already released.
    pub fn rewind(&mut self, n: i32) -> Result<i32, String> {
        let n = n.min(self.kv.offset);
        if n <= 0 {
            return Ok(0);
        }
        let backing = self
            .kv
            .paged_backing
            .clone()
            .ok_or_else(|| "paged sliding layer without pool backing".to_string())?;
        let trimmed = backing.pool.borrow_mut().rewind_tokens(
            &mut backing.state.borrow_mut(),
            backing.layer_idx,
            n as usize,
        )? as i32;
        self.kv.offset -= trimmed;
        Ok(trimmed)
    }

    /// Bytes of the live blocks this layer reads (the pool holds them; the
    /// layer owns no dense buffer).
    pub fn nbytes(&self) -> usize {
        0
    }
}
