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

//! One attention entry per KV cache (issue #2171, epic #2166 Phase 4a).
//!
//! Which kernel answers a step is a property of the storage behind the cache,
//! not of the model forward that calls it. Before this module every transformer
//! family that could run on the server's paged pool carried the same branch in
//! its own attention block (`l == 1 && mask.is_none() && cache.is_paged_backed()`,
//! plus a batched twin that read a scheduler-supplied `DecodeBatchContext`), so
//! two families could disagree on when the paged kernel applies and a storage
//! policy change meant editing model files.
//!
//! [`KVCache::attend`] and [`attend_batched`] own that decision now. The rule,
//! recorded in ADR 0008:
//!
//! | storage behind the cache | single-token, unmasked step | any other step |
//! |---|---|---|
//! | pool-backed paged (`new_paged`) | [`paged_batch_decode_attention`]: fused v2 or the ADR 0001 gather fallback, chosen inside | pool append, gathered window, fused SDPA |
//! | dense Turbo4Asym / Turbo4 / Turbo4Delegated | the dequant-first SDPA variants the cache mode and its env gates already select | dense update, full dequant, SDPA |
//! | dense FP16 / Int8 | dense update, fused SDPA | dense update, causal or masked SDPA |
//!
//! A masked step (prefill, speculative or MTP verify) never takes the paged
//! single-token kernel; that is the `mask.is_none()` guard the models used to
//! carry. Dispatch is a match on the cache's own fields rather than a trait
//! object: ADR 0004 rejected per-op dynamic dispatch on the decode hot loop,
//! and the cache already resolves its quantization modes the same way.
//!
//! Storage is chosen once, where the cache is built: `CachePool` wires
//! pool-backed caches for a sequence on the paged backend and dense caches
//! otherwise, so the per-sequence storage policy the scheduler used to carry
//! in `DecodeBatchContext` is the cache itself for these families.
//!
//! The model-owned families (issue #2172, epic #2166 Phase 4b) hold their
//! per-sequence state as their own cache enums rather than a pool `KVCache`,
//! and that state is never pool-backed, so for them the rule reduces to the
//! dense rows of the table. [`RotatingKVCache::attend`] (Gemma 3's sliding
//! layers) and [`ChunkedKVCache::attend`] (Llama 4's chunked layers) are the
//! dense entries for those storages, [`KvAttention`] is the trait the model
//! enums implement by matching to the inner cache, and [`attend_batched_rows`]
//! is the per-row batched loop every non-pooled batch takes. The
//! `DecodeBatchContext` those families used to read is gone: the
//! dense-pointer "paged compat" kernels it selected
//! (`paged_decode_attention_dense_compat`, `_rotating_compat`) were per-row
//! loops of block slices, a concat and one SDPA call per row, so a batch now
//! runs the same attention per row through its own cache without the concat
//! copy, the change ADR 0008 already made for Qwen3 and Llama 3.

use cxx::UniquePtr;

use super::{ChunkedKVCache, KVCache, RotatingKVCache, paged_batch_decode_attention, turbo};
use crate::ffi::{self, MlxArray};

/// One attention entry per KV storage: append the step's K/V and run
/// attention through the kernel that suits the storage behind the cache.
///
/// Implemented by [`KVCache`], [`RotatingKVCache`] and [`ChunkedKVCache`] in
/// the core, and by the model-owned cache enums (`gemma3::Cache`,
/// `llama4::Llama4Cache`) by matching to the inner cache. Static dispatch: a
/// batched caller is monomorphized over the enum, so there is no virtual call
/// per layer per token (ADR 0004).
///
/// `q` is `[B, Hq, L, D]`; `new_keys` and `new_values` are `[B, Hkv, L, D]`
/// with RoPE and any Q/K norm already applied; `mask` is the optional
/// additive mask the model built. Returns `[B, Hq, L, D]`.
pub trait KvAttention {
    fn attend(
        &mut self,
        q: &MlxArray,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray>;
}

impl KvAttention for KVCache {
    #[inline]
    fn attend(
        &mut self,
        q: &MlxArray,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        KVCache::attend(self, q, new_keys, new_values, scale, mask)
    }
}

impl KvAttention for RotatingKVCache {
    #[inline]
    fn attend(
        &mut self,
        q: &MlxArray,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        RotatingKVCache::attend(self, q, new_keys, new_values, scale, mask)
    }
}

impl KvAttention for ChunkedKVCache {
    #[inline]
    fn attend(
        &mut self,
        q: &MlxArray,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        ChunkedKVCache::attend(self, q, new_keys, new_values, scale, mask)
    }
}

impl RotatingKVCache {
    /// Append this step's K/V and run attention over the window the ring
    /// returns.
    ///
    /// A rotating cache is dense storage that is never pool-backed, so this
    /// is the dense row of the ADR 0008 table. The write goes through
    /// [`Self::update_and_fetch`], which stays the one write path: it is
    /// where the decode undo log (#2182) records the rows a single-token write
    /// inside a `DecodeLookaheadAppendScope` overwrites, and where every
    /// multi-token or Turbo append clears that log, so a speculative write
    /// made through this entry is as rewindable as one made through
    /// `update_and_fetch` directly. No host sync and no allocation beyond the
    /// shape reads.
    ///
    /// A multi-token unmasked step whose returned window fits the ring is a
    /// prefill whose mask would be plain causal and takes the causal helper;
    /// anything else runs the masked or unmasked SDPA with the ring's
    /// `max_size` as the window hint (what Gemma 3 passed as `window_size`,
    /// read only by the Metal 4 kernel).
    ///
    /// Used by: `models::gemma3::Cache::attend` (sliding layers),
    /// [`attend_batched_rows`].
    pub fn attend(
        &mut self,
        q: &MlxArray,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let q_len = query_len(q);
        let (cache_k, cache_v) = self.update_and_fetch(new_keys, new_values);
        let k_len = ffi::array_shape(&cache_k)[2];
        if mask.is_none() && q_len > 1 && k_len <= self.max_size {
            return crate::causal_attention(q, &cache_k, &cache_v, scale, 0.0, 0);
        }
        crate::layers::attention(q, &cache_k, &cache_v, scale, mask, 0.0, self.max_size)
    }
}

impl ChunkedKVCache {
    /// Append this step's K/V and run attention over the visible chunk.
    ///
    /// Dense storage, never pool-backed: the write goes through
    /// [`Self::update_and_fetch`] and the attention is the causal helper for
    /// an unmasked multi-token step, else the masked or unmasked SDPA, the
    /// pair Llama 4's single-row forward spelled out. The caller still runs
    /// [`Self::maybe_trim_front`] before the step, as before.
    ///
    /// Used by: `models::llama4::Llama4Cache::attend` (chunked layers),
    /// [`attend_batched_rows`].
    pub fn attend(
        &mut self,
        q: &MlxArray,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let q_len = query_len(q);
        let (cache_k, cache_v) = self.update_and_fetch(new_keys, new_values);
        if q_len > 1 && mask.is_none() {
            return crate::causal_attention(q, &cache_k, &cache_v, scale, 0.0, 0);
        }
        crate::layers::attention(q, &cache_k, &cache_v, scale, mask, 0.0, 0)
    }
}

impl KVCache {
    /// Append this step's K/V and run attention through the kernel that suits
    /// the storage behind this cache.
    ///
    /// `q` is `[B, Hq, L, D]`; `new_keys` and `new_values` are `[B, Hkv, L, D]`
    /// with RoPE and any Q/K norm already applied; `mask` is the optional
    /// additive mask the model built (`None` for a plain decode step and for
    /// causal prefill, which runs through the causal helper). Returns
    /// `[B, Hq, L, D]`.
    ///
    /// Cache mutation is exactly what the pre-existing paths performed: the
    /// pooled launch appends through `write_paged`, every dense route goes
    /// through [`Self::update`] or [`Self::update_and_fetch`], and nothing here
    /// touches `offset` on its own. No host sync and no allocation beyond the
    /// shape read.
    ///
    /// Used by: `models::qwen3::Attention::forward`,
    /// `models::llama3::Attention::forward` (and the families that reuse them:
    /// Qwen2, Qwen2.5, Helium, the VLM text backbones), [`attend_batched`].
    pub fn attend(
        &mut self,
        q: &MlxArray,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let q_len = query_len(q);
        // Pool-backed single-token decode (#899). The batch-1 launch is the
        // `[1, H, 1, D]` shape the whole-batch entry point serves, and the
        // server reaches it through the single-sequence forward whenever one
        // request is decoding. The call declines (`None`) before touching the
        // pool for anything it cannot serve, so the dense route below still
        // runs its pool intercept (append plus gather) unchanged.
        if q_len == 1
            && mask.is_none()
            && self.paged_backing.is_some()
            && let Some(out) = paged_batch_decode_attention(
                q,
                &new_keys,
                &new_values,
                &mut [&mut *self],
                scale,
                0.0,
            )
        {
            return out;
        }
        self.attend_dense(q, q_len, new_keys, new_values, scale, mask)
    }

    /// Dense-storage attention: the Turbo decode variants at a single token,
    /// otherwise update, fetch the live window and run SDPA.
    fn attend_dense(
        &mut self,
        q: &MlxArray,
        q_len: i32,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        if q_len == 1 {
            // Turbo4Asym (FP16 K, 4-bit V) decodes dequant-first by default:
            // exact and 3-6x faster than the lossy sparse-V weighted sum, which
            // stays reachable behind `MLXCEL_TURBO4_ASYM_DEQUANT_SDPA=0`.
            // Symmetric Turbo4 and Turbo4Delegated mirror mlx-swift-lm's
            // dequant-first policy. Each gate is one cached `OnceLock` read.
            if turbo::sparse_v::turbo4_asym_dequant_sdpa_enabled()
                && self.turbo4_asym_dequant_sdpa_available()
            {
                return self.update_and_turbo4_asym_dequant_sdpa_attention(
                    q, new_keys, new_values, scale, mask,
                );
            }
            if self.sparse_v_available() {
                return self
                    .update_and_sparse_v_attention(q, new_keys, new_values, scale, mask)
                    .expect(
                        "update_and_sparse_v_attention returned None despite sparse_v_available",
                    );
            }
            if turbo::sparse_v::turbo4_dequant_sdpa_enabled()
                && self.turbo4_dequant_sdpa_available()
            {
                return self.update_and_turbo4_dequant_sdpa_attention(
                    q, new_keys, new_values, scale, mask,
                );
            }
            if turbo::sparse_v::turbo4_delegated_compressed_attention_enabled()
                && self.turbo4_delegated_available()
            {
                return self
                    .update_and_turbo4_delegated_attention(q, new_keys, new_values, scale, mask);
            }
        }
        let (cache_k, cache_v) = self.update_and_fetch(new_keys, new_values);
        match mask {
            // Causal prefill without an explicit mask uses the shared causal
            // helper; decode and masked or padded prefill use the unified
            // dispatcher, which also accepts a null mask.
            None if q_len > 1 => crate::causal_attention(q, &cache_k, &cache_v, scale, 0.0, 0),
            _ => {
                let mask_ptr = mask.map_or(std::ptr::null(), |m| m as *const MlxArray);
                // SAFETY: `mask_ptr` is null or points at `mask`, a borrow that
                // outlives this call; `attention_from_ptr` only reads it.
                unsafe {
                    crate::layers::attention_from_ptr(
                        q, &cache_k, &cache_v, scale, mask_ptr, 0.0, 0,
                    )
                }
            }
        }
    }
}

/// Batched counterpart of [`KVCache::attend`]: one layer's caches for every
/// sequence of a decode or batched-prefill step.
///
/// `q_batched` is `[B, Hq, T, D]`, `k_batched` and `v_batched` are
/// `[B, Hkv, T, D]`, `caches[b]` is sequence `b`'s cache for this layer, and
/// `mask` is the optional `[B, T, S]` additive mask whose row `b` goes to
/// sequence `b`. Returns `[B, Hq, T, D]`; the batch-axis concat commutes with
/// the transpose and reshape the caller applies afterward.
///
/// A single-token unmasked step over pool-backed caches is one whole-batch
/// launch ([`paged_batch_decode_attention`], which declines before writing
/// when it cannot serve the batch). Every other shape, and a declined batch,
/// goes through [`KVCache::attend`] per sequence, so a row decodes the same way
/// whether it was scheduled alone or in a batch.
///
/// Used by: `models::qwen3::Attention::forward_split_attention`,
/// `models::llama3::Attention::forward_split_attention`.
pub fn attend_batched(
    q_batched: &MlxArray,
    k_batched: &MlxArray,
    v_batched: &MlxArray,
    caches: &mut [&mut KVCache],
    scale: f32,
    mask: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    let batch = caches.len();
    assert!(batch > 0, "attend_batched: empty batch");
    let seq_len = query_len(q_batched);
    if seq_len == 1
        && mask.is_none()
        && caches[0].is_paged_backed()
        && let Some(out) =
            paged_batch_decode_attention(q_batched, k_batched, v_batched, caches, scale, 0.0)
    {
        return out;
    }
    attend_batched_rows(q_batched, k_batched, v_batched, caches, scale, mask)
}

/// Per-row batched attention: row `b` of the batch goes through
/// `caches[b].attend` on its own storage and the outputs are concatenated on
/// the batch axis.
///
/// This is the batched route for every storage that has no whole-batch
/// launch: the model-owned enums of Gemma 3 and Llama 4, and the dense or
/// declined batches [`attend_batched`] hands over. Same shapes and mask
/// convention as [`attend_batched`]. One slice per row and one concat per
/// extra row, which is what the per-row loops it replaced did.
///
/// Used by: [`attend_batched`], `models::gemma3::Attention::forward_batched_decode`,
/// `models::llama4::CxxAttention::forward_batched_decode_rows`.
pub fn attend_batched_rows<C: KvAttention>(
    q_batched: &MlxArray,
    k_batched: &MlxArray,
    v_batched: &MlxArray,
    caches: &mut [&mut C],
    scale: f32,
    mask: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    let batch = caches.len();
    assert!(batch > 0, "attend_batched_rows: empty batch");
    let seq_len = query_len(q_batched);
    let mut outputs: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(batch);
    for (b, cache) in caches.iter_mut().enumerate() {
        let q_b = slice_row(q_batched, b);
        let k_b = slice_row(k_batched, b);
        let v_b = slice_row(v_batched, b);
        let mask_b = mask.map(|m| {
            let row = ffi::slice(m, &[b as i32, 0, 0], &[b as i32 + 1, seq_len, i32::MAX]);
            ffi::squeeze_axis(&row, 0)
        });
        outputs.push(cache.attend(&q_b, k_b, v_b, scale, mask_b.as_deref()));
    }
    let mut outputs = outputs.into_iter();
    let mut result = outputs.next().expect("a non-empty batch has a first row");
    for out in outputs {
        result = crate::concatenate(&result, &out, 0);
    }
    result
}

/// Query length of a `[B, H, L, D]` tensor (a shape read, no sync).
#[inline]
fn query_len(q: &MlxArray) -> i32 {
    ffi::array_shape(q)[2]
}

/// Slice row `b` out of a `[B, H, T, D]` tensor, keeping the batch axis.
#[inline]
fn slice_row(arr: &MlxArray, b: usize) -> UniquePtr<MlxArray> {
    ffi::slice(
        arr,
        &[b as i32, 0, 0, 0],
        &[b as i32 + 1, i32::MAX, i32::MAX, i32::MAX],
    )
}

#[cfg(test)]
#[path = "attend_tests.rs"]
mod attend_tests;
