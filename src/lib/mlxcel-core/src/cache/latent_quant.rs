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

//! Affine-quantized MLA latent rows stored in a [`KVCache`] (issue #3155).
//!
//! An absorbed MLA cache holds the shared latent `ckv` and the rope stream
//! `k_pe`. Under [`KVCacheMode::Int8`] the latent is stored as the MLX affine
//! triple `quantize_weights` produces, so the decode step can hand it to
//! `quantized_matmul` without dequantizing the window, and `k_pe` stays FP16.
//! The layout is bit-width generic (`bits` below); 8 is the only width a mode
//! selects, because 4-bit failed the #3155 quality gate
//! (`docs/benchmark_results/mla-quantized-latent-m3ultra-2026-11-19.md`). The four per-token buffers reuse the slots a per-head
//! INT8 cache already has, which is what lets `trim`, `trim_front`,
//! `gather_positions`, detach/adopt and the prompt-cache store move them
//! together with no latent-specific code:
//!
//! | slot         | holds                 | shape                              | dtype  |
//! |--------------|-----------------------|------------------------------------|--------|
//! | `keys`       | packed `ckv`          | `[B, 1, L, rank * bits / 32]`      | uint32 |
//! | `values`     | `k_pe`                | `[B, 1, L, qk_rope_head_dim]`      | fp16   |
//! | `key_scales` | affine group scales   | `[B, 1, L, rank / group_size]`     | fp16   |
//! | `val_scales` | affine group biases   | `[B, 1, L, rank / group_size]`     | fp16   |
//!
//! The mode is fixed for a cache's lifetime and the family decides once per
//! cache whether it stores a latent, so a cache never mixes these rows with
//! per-head INT8 rows.
//!
//! Used by: `mla::cache::MlaLatentCache`.

use cxx::UniquePtr;

use super::{KVCache, KVCacheMode, direct_prefill_cache_store_enabled};
use crate::concatenate;
use crate::dtype;
use crate::ffi::{self, MlxArray};

/// Affine group size of a quantized MLA latent.
///
/// Fixed: `kv_lora_rank` is 512 in every shipping checkpoint, 64 divides it,
/// and it is the group size the MLX quantized kernels are tuned for.
pub const LATENT_QUANT_GROUP_SIZE: i32 = 64;

impl KVCacheMode {
    /// Bit width this mode stores an MLA latent at, or `None` when the mode is
    /// not an affine latent mode.
    ///
    /// Used by: `mla::cache` (storage and window), `mla::latent_layout_supports_mode`.
    #[must_use]
    pub const fn latent_affine_bits(self) -> Option<i32> {
        match self {
            Self::Int8 => Some(8),
            _ => None,
        }
    }

    /// Whether this mode keeps per-token sidecars in `key_scales` /
    /// `val_scales` that every trim and slice must move with the rows.
    #[must_use]
    pub const fn uses_scale_sidecars(self) -> bool {
        matches!(self, Self::Int8)
    }
}

/// One step's (or one window's) quantized latent rows.
///
/// All four arrays share axes 0 to 2; see the module table for axis 3.
pub struct QuantizedLatentRows {
    /// Packed latent, uint32.
    pub packed: UniquePtr<MlxArray>,
    /// Affine scale per group.
    pub scales: UniquePtr<MlxArray>,
    /// Affine bias per group.
    pub biases: UniquePtr<MlxArray>,
    /// Rope key stream, FP16.
    pub kpe: UniquePtr<MlxArray>,
}

/// Slice `[.., 0..len, ..]` of a rank-4 buffer along the sequence axis.
fn seq_prefix(arr: &MlxArray, len: i32) -> UniquePtr<MlxArray> {
    let s = ffi::array_shape(arr);
    ffi::slice(arr, &[0, 0, 0, 0], &[s[0], s[1], len, s[3]])
}

/// Extend `slot` so it has room for `extra` more rows shaped like `like`,
/// keeping the first `prev` rows.
fn grow_slot(slot: &mut Option<UniquePtr<MlxArray>>, like: &MlxArray, prev: i32, extra: i32) {
    let s = ffi::array_shape(like);
    let pad = ffi::zeros(&[s[0], s[1], extra, s[3]], ffi::array_dtype(like));
    *slot = Some(match slot.as_ref() {
        Some(existing) if prev > 0 => {
            // Normalize to the live prefix first: the buffer can be longer
            // than `prev` after a logical `trim` (issue #1209).
            if ffi::array_shape(existing)[2] == prev {
                concatenate(existing, &pad, 2)
            } else {
                concatenate(&seq_prefix(existing, prev), &pad, 2)
            }
        }
        _ => pad,
    });
}

/// Write `rows` into `slot` at sequence positions `[prev, live_len)`.
fn write_slot(slot: &mut Option<UniquePtr<MlxArray>>, rows: &MlxArray, prev: i32, live_len: i32) {
    let buf = slot.as_ref().expect("grown before the write");
    let s = ffi::array_shape(buf);
    *slot = Some(ffi::slice_update(
        buf,
        rows,
        &[0, 0, prev, 0],
        &[s[0], s[1], live_len, s[3]],
    ));
}

impl KVCache {
    /// Whether this cache currently holds quantized latent rows.
    ///
    /// True only after [`Self::update_latent_quantized`] has written to it:
    /// the packed latent is the one uint32 `keys` buffer any mode stores.
    #[must_use]
    pub fn holds_quantized_latent(&self) -> bool {
        self.mode.latent_affine_bits().is_some()
            && self
                .keys
                .as_ref()
                .is_some_and(|k| ffi::array_dtype(k) == dtype::UINT32)
    }

    /// Append one step's quantized latent rows.
    ///
    /// Same bookkeeping as the per-head update paths: buffer-slot coordinates
    /// (`offset - live_start`), `step`-sized growth, and a monotonic `offset`,
    /// so `trim` and `trim_front` behave exactly as they do for per-head INT8.
    pub(crate) fn update_latent_quantized(&mut self, rows: QuantizedLatentRows) {
        debug_assert!(
            self.mode.latent_affine_bits().is_some(),
            "quantized latent rows need an affine latent mode, got {}",
            self.mode
        );
        debug_assert!(
            self.keys.is_none() || self.holds_quantized_latent(),
            "a cache never mixes per-head and latent rows"
        );
        let new_len = ffi::array_shape(&rows.packed)[2];
        let prev = self.buffer_idx();

        if prev == 0 && self.keys.is_none() && direct_prefill_cache_store_enabled() {
            self.keys = Some(ffi::contiguous(&rows.packed, false));
            self.values = Some(ffi::contiguous(&rows.kpe, false));
            self.key_scales = Some(ffi::contiguous(&rows.scales, false));
            self.val_scales = Some(ffi::contiguous(&rows.biases, false));
            self.offset += new_len;
            return;
        }

        if self.keys.is_none() || prev + new_len > self.buffer_seq_len() {
            let extra = (self.step + new_len - 1) / self.step * self.step;
            grow_slot(&mut self.keys, &rows.packed, prev, extra);
            grow_slot(&mut self.values, &rows.kpe, prev, extra);
            grow_slot(&mut self.key_scales, &rows.scales, prev, extra);
            grow_slot(&mut self.val_scales, &rows.biases, prev, extra);
        }

        self.offset += new_len;
        let live_len = self.buffer_idx();
        write_slot(&mut self.keys, &rows.packed, prev, live_len);
        write_slot(&mut self.values, &rows.kpe, prev, live_len);
        write_slot(&mut self.key_scales, &rows.scales, prev, live_len);
        write_slot(&mut self.val_scales, &rows.biases, prev, live_len);
    }

    /// The live window of a quantized latent cache, or `None` when the cache
    /// holds none.
    pub(crate) fn latent_quantized_window(&self) -> Option<QuantizedLatentRows> {
        if !self.holds_quantized_latent() {
            return None;
        }
        let live_len = self.buffer_idx();
        Some(QuantizedLatentRows {
            packed: seq_prefix(self.keys.as_ref()?, live_len),
            scales: seq_prefix(self.key_scales.as_ref()?, live_len),
            biases: seq_prefix(self.val_scales.as_ref()?, live_len),
            kpe: seq_prefix(self.values.as_ref()?, live_len),
        })
    }
}
