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

//! Growing a dense FP16 [`KVCache`] buffer, and releasing the buffer it
//! replaces before the step's graph runs (issue #3084).
//!
//! A multi-token append that outgrows the buffer concatenates the live prefix
//! with a zero block. Inside a lazy chunked-prefill graph every layer does
//! this, and each superseded buffer is freed only when its concatenate runs.
//! The freed buffer goes back to MLX's free-buffer cache, where nothing can
//! reuse it: every later request in the same piece asks for the larger size,
//! and MLX only hands out a cached buffer within two pages of the request. So
//! the whole previous cache sits in the allocator cache until the piece ends.
//! On DeepSeek-V2-Lite, whose default path caches decompressed per-head K/V
//! (10 KiB per token per layer), that was 7.4 GB of the process footprint
//! at a 28.7K-token prompt in 2048-token pieces.
//!
//! [`KVCache::reserve_rows`] performs the same growth layer by layer before
//! the step's graph is built, queues the copy, and empties the allocator
//! cache at the next layer, so the superseded buffers are returned instead of
//! parked. It cannot batch the layers: growing every layer at once holds every
//! old and every new buffer together, which is the peak it set out to remove.
//! Each grown buffer is the one [`KVCache::update_fp16`] would have built, so
//! the appends that follow find room and write exactly the bytes they always
//! wrote.

use super::{KVCache, KVCacheMode};
use crate::concatenate;
use crate::ffi;
use crate::ffi::MlxArray;

/// Shape and dtypes of a K/V append, the inputs of one FP16 growth.
struct AppendGeometry {
    batch: i32,
    heads: i32,
    rows: i32,
    k_head_dim: i32,
    v_head_dim: i32,
    k_dtype: i32,
    v_dtype: i32,
}

impl KVCache {
    /// Whether an FP16 append of `new_seq_len` rows must (re)allocate.
    pub(super) fn fp16_append_needs_growth(&self, new_seq_len: i32) -> bool {
        self.keys.is_none() || (self.buffer_idx() + new_seq_len) > self.buffer_seq_len()
    }

    /// Grow the FP16 buffers to fit an append shaped like `new_keys` /
    /// `new_values`. The lazy growth inside [`Self::update_fp16`].
    pub(super) fn grow_fp16_buffers(&mut self, new_keys: &MlxArray, new_values: &MlxArray) {
        let key_shape = ffi::array_shape(new_keys);
        self.grow_fp16_to(&AppendGeometry {
            batch: key_shape[0],
            heads: key_shape[1],
            rows: key_shape[2],
            k_head_dim: key_shape[3],
            v_head_dim: ffi::array_shape(new_values)[3],
            k_dtype: ffi::array_dtype(new_keys),
            v_dtype: ffi::array_dtype(new_values),
        });
    }

    /// The single growth rule for the dense FP16 path, keeping the live prefix
    /// and dropping dead slots. Both the lazy growth and the eager reservation
    /// go through it, which is what keeps the two byte-identical.
    fn grow_fp16_to(&mut self, g: &AppendGeometry) {
        let prev = self.buffer_idx();
        let n_steps = (self.step + g.rows - 1) / self.step;
        let buf_size = n_steps * self.step;

        let new_k = ffi::zeros(&[g.batch, g.heads, buf_size, g.k_head_dim], g.k_dtype);
        let new_v = ffi::zeros(&[g.batch, g.heads, buf_size, g.v_head_dim], g.v_dtype);

        let (Some(keys), Some(values)) = (self.keys.take(), self.values.take()) else {
            self.keys = Some(new_k);
            self.values = Some(new_v);
            return;
        };
        // Normalize the buffer to the live prefix before extending. The
        // buffer can be longer than `prev`: a logical `trim` keeps the old
        // capacity (issue #1209), and a partial growth chunk leaves a
        // non-step-aligned length. Either way the slots past `prev` are dead
        // and must not survive the concatenate.
        let (keys, values) = if prev != ffi::array_shape(&keys)[2] {
            (
                ffi::slice(
                    &keys,
                    &[0, 0, 0, 0],
                    &[g.batch, g.heads, prev, g.k_head_dim],
                ),
                ffi::slice(
                    &values,
                    &[0, 0, 0, 0],
                    &[g.batch, g.heads, prev, g.v_head_dim],
                ),
            )
        } else {
            (keys, values)
        };
        self.keys = Some(concatenate(&keys, &new_k, 2));
        self.values = Some(concatenate(&values, &new_v, 2));
    }

    /// Grow this cache, lazily, for a coming append of `new_rows` rows shaped
    /// like the rows it already holds. Returns `true` when it grew.
    ///
    /// Declines (returns `false`) unless the cache is plain
    /// `KVCacheMode::Fp16`, not pool-backed, already holds a buffer, and that
    /// buffer is too small for a multi-token append: the quantized and paged
    /// layouts grow through their own paths, a first fill has nothing to
    /// release, a decode row writes in place or inside the step slack, and an
    /// append that fits allocates nothing.
    ///
    /// The append that follows must carry the cached dtypes (it always does:
    /// the buffer was created from the same projection's first append).
    fn grow_for_rows(&mut self, new_rows: i32) -> bool {
        if self.mode != KVCacheMode::Fp16
            || self.paged_backing.is_some()
            || new_rows <= 1
            || !self.fp16_append_needs_growth(new_rows)
        {
            return false;
        }
        let (Some(keys), Some(values)) = (self.keys.as_ref(), self.values.as_ref()) else {
            return false;
        };
        let ks = ffi::array_shape(keys);
        let geometry = AppendGeometry {
            batch: ks[0],
            heads: ks[1],
            rows: new_rows,
            k_head_dim: ks[3],
            v_head_dim: ffi::array_shape(values)[3],
            k_dtype: ffi::array_dtype(keys),
            v_dtype: ffi::array_dtype(values),
        };
        self.grow_fp16_to(&geometry);
        true
    }

    /// Grow the buffer for a coming `new_rows`-row append now, before the
    /// step's graph is built, and hand back the buffers earlier reservations
    /// replaced instead of leaving them in MLX's allocator cache. Returns
    /// `true` when it grew.
    ///
    /// Call it once per layer, before that layer's [`Self::update_and_fetch`].
    /// The grown buffer (a copy of the already-evaluated cache into a larger
    /// one) is queued with `async_eval`, so the CPU keeps building the graph;
    /// once that copy runs, the superseded buffer is freed into the allocator
    /// cache, and the next layer's reservation empties the cache before it
    /// grows. At most the last few layers' old buffers are still parked when
    /// the step's graph runs, instead of every layer's.
    ///
    /// Declines, at no cost, for a decode row, a first fill, an append that
    /// fits, and any layout other than dense FP16 (see `grow_for_rows`). The
    /// grown buffer is the one [`Self::update_fp16`] would have built, so the
    /// append writes the bytes it always wrote.
    ///
    /// Used by: DeepSeek V2 MLA attention, decompressed and latent caches
    /// (issue #3084).
    pub fn reserve_rows(&mut self, new_rows: i32) -> bool {
        if new_rows <= 1 || self.mode != KVCacheMode::Fp16 || self.paged_backing.is_some() {
            return false;
        }
        if !self.fp16_append_needs_growth(new_rows) || self.keys.is_none() {
            return false;
        }
        // Return what the previous layers' grown copies released.
        crate::memory::clear_cache();
        if !self.grow_for_rows(new_rows) {
            return false;
        }
        let keys = self.keys.as_ref().expect("grow_for_rows sets keys");
        let values = self.values.as_ref().expect("grow_for_rows sets values");
        ffi::async_eval_pair(keys, values);
        true
    }
}
