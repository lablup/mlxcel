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
//! replaces before the step's graph runs (issues #3084, #3092).
//!
//! A multi-token append that outgrows the buffer concatenates the live prefix
//! with a zero block. Inside a lazy chunked-prefill graph every layer does
//! this, and each superseded buffer is freed only when its concatenate runs.
//! The freed buffer goes back to MLX's free-buffer cache, where nothing can
//! reuse it: every later request in the same piece asks for the larger size,
//! and MLX only hands out a cached buffer within two pages of the request. So
//! the whole previous cache would sit in the allocator cache until the piece
//! ends and the engine flushes it (#672). On DeepSeek-V2-Lite, whose default
//! path caches decompressed per-head K/V (10 KiB per token per layer), that
//! was 7.4 GB of the process footprint at a 28.7K-token prompt in 2048-token
//! pieces.
//!
//! So a multi-token growth of an existing buffer, the one a chunked-prefill
//! piece or a follow-up turn makes, runs eagerly inside `update_fp16`, for
//! every model on this cache: it empties the allocator cache (returning what
//! the previous layers' growths released), builds the grown buffer, and
//! queues its copy with `async_eval`, so the CPU keeps building the graph.
//! Once that copy runs, the superseded buffer is freed, and the next layer's
//! growth returns it. At most the last few layers' old buffers are still
//! parked when the step's graph runs, instead of every layer's.
//!
//! The growth cannot be batched across layers at the start of the forward:
//! growing every layer at once holds every old and every new buffer together,
//! which is the peak it sets out to remove (measured in #3091). A first fill
//! has nothing to release, a decode row writes in place or inside the step
//! slack, and an append that fits allocates nothing, so none of them changes.
//! The quantized, rotating and pool-backed layouts never reach this path. The
//! grown buffer is exactly the one the lazy growth would build, so the append
//! writes the bytes it always wrote.

use super::KVCache;
use crate::concatenate;
use crate::ffi;
use crate::ffi::MlxArray;

impl KVCache {
    /// Whether an FP16 append of `new_seq_len` rows must (re)allocate.
    pub(super) fn fp16_append_needs_growth(&self, new_seq_len: i32) -> bool {
        self.keys.is_none() || (self.buffer_idx() + new_seq_len) > self.buffer_seq_len()
    }

    /// Whether growing for an append of `new_rows` rows releases the
    /// superseded buffer eagerly: a multi-token append to a buffer that
    /// already exists.
    pub(super) fn fp16_growth_releases(&self, new_rows: i32) -> bool {
        new_rows > 1 && self.keys.is_some() && self.values.is_some()
    }

    /// Grow the FP16 buffers to fit an append shaped like `new_keys` /
    /// `new_values`, keeping the live prefix and dropping dead slots.
    ///
    /// A multi-token growth of an existing buffer also hands back the buffers
    /// earlier layers' growths replaced and queues its own copy (see the
    /// module docs); every other growth stays lazy. Returns whether it
    /// released.
    ///
    /// Used by: every model on the dense FP16 `KVCache` (through
    /// `KVCache::update_fp16`), including DeepSeek V2's decompressed and
    /// latent MLA caches (issues #3084, #3092).
    pub(super) fn grow_fp16_buffers(&mut self, new_keys: &MlxArray, new_values: &MlxArray) -> bool {
        let key_shape = ffi::array_shape(new_keys);
        let (batch, heads, rows, k_head_dim) =
            (key_shape[0], key_shape[1], key_shape[2], key_shape[3]);
        let v_head_dim = ffi::array_shape(new_values)[3];
        let release = self.fp16_growth_releases(rows);
        if release {
            // Return what the previous layers' grown copies released.
            crate::memory::clear_cache();
        }

        let prev = self.buffer_idx();
        let n_steps = (self.step + rows - 1) / self.step;
        let buf_size = n_steps * self.step;
        let new_k = ffi::zeros(
            &[batch, heads, buf_size, k_head_dim],
            ffi::array_dtype(new_keys),
        );
        let new_v = ffi::zeros(
            &[batch, heads, buf_size, v_head_dim],
            ffi::array_dtype(new_values),
        );

        let (Some(keys), Some(values)) = (self.keys.take(), self.values.take()) else {
            self.keys = Some(new_k);
            self.values = Some(new_v);
            return false;
        };
        // Normalize the buffer to the live prefix before extending. The
        // buffer can be longer than `prev`: a logical `trim` keeps the old
        // capacity (issue #1209), and a partial growth chunk leaves a
        // non-step-aligned length. Either way the slots past `prev` are dead
        // and must not survive the concatenate.
        let (keys, values) = if prev != ffi::array_shape(&keys)[2] {
            (
                ffi::slice(&keys, &[0, 0, 0, 0], &[batch, heads, prev, k_head_dim]),
                ffi::slice(&values, &[0, 0, 0, 0], &[batch, heads, prev, v_head_dim]),
            )
        } else {
            (keys, values)
        };
        let grown_k = concatenate(&keys, &new_k, 2);
        let grown_v = concatenate(&values, &new_v, 2);
        if release {
            ffi::async_eval_pair(&grown_k, &grown_v);
        }
        self.keys = Some(grown_k);
        self.values = Some(grown_v);
        release
    }
}
