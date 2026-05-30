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

//! Cross-sequence KV cache reuse: trim / detach / adopt primitives.
//!
//! This module extends [`super::KVCache`] and [`super::CachePool`] with the
//! primitives required by the cross-request prompt prefix cache:
//!
//! * [`KVCache::trim_to`] — shrink the logical cache length to an exact value
//!   while keeping the pre-allocated backing buffer around.
//! * [`KVCache::clone_handle`] — move the underlying `MlxArray` ownership out
//!   into an inert [`DetachedKVCache`] that can outlive the original sequence.
//! * [`CachePool::detach`] — lift a whole [`super::SequenceCacheSet`] off the
//!   active HashMap and return it as an owned [`DetachedCacheSet`] without
//!   freeing the MLX buffers.
//! * [`CachePool::adopt`] — install a previously-detached cache set under a
//!   fresh [`super::SequenceId`] and prime the model's sidecar state for it.
//!
//! Only the **dense** KV cache backend (`SequenceStateBackend::DenseKvCache`)
//! is handled directly by this file. Paged sequences are handled by the
//! parallel API surface in [`super::paged_detach`]; this
//! module delegates through the shared `DetachedHandle` namespace so parking
//! remains a single pool-level abstraction.
//!
//! ## Memory accounting
//!
//! While a detached cache set is in-flight — e.g. inside a scheduler that has
//! taken it out of `CachePool` but is about to re-adopt it — callers can
//! [`park`] the set so that the pool's
//! [`CachePool::memory_usage_bytes`] keeps including the bytes. Parking is
//! optional; `detach` + `adopt` work end-to-end without it.
//!
//! ## INT8 preservation
//!
//! Both the INT8 key/value tensors and the per-token FP16 scale tensors are
//! moved through detach/adopt unchanged, so `KVCache::mode == Int8` sequences
//! round-trip losslessly.
//!
//! ## Aliasing with `MLXCEL_ENABLE_DIRECT_PREFILL_CACHE_STORE`
//!
//! The direct-prefill-store fast path in [`super::KVCache::update`] installs
//! the incoming FP16 tensor directly as the cache buffer (with a
//! `contiguous` call) when the cache is empty and the env var is set. Detach
//! simply moves that buffer out via [`UniquePtr::take`]; no aliasing survives
//! because `MlxArray` buffers are functional — every operation produces a
//! fresh array, and the move semantics of `UniquePtr` prevent concurrent
//! access. Adopting that same buffer into a new sequence is therefore safe.
//!
//! [`park`]: CachePool::park_detached

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Instant;

use cxx::UniquePtr;

use crate::ffi;
use crate::ffi::MlxArray;

use super::{
    CachePool, KVCache, KVCacheMode, RotatingKVCache, SequenceCacheSet, SequenceId,
    SequenceStateBackend,
};

// ---------------------------------------------------------------------------
// DetachedKVCache
// ---------------------------------------------------------------------------

/// Inert, model-agnostic snapshot of a single [`KVCache`] that can outlive the
/// sequence which produced it.
///
/// `DetachedKVCache` owns the underlying MLX `UniquePtr<MlxArray>` buffers
/// directly, so detach/adopt never allocates a new tensor or copies data.
/// INT8-mode caches carry their per-token scale tensors alongside the INT8
/// key/value buffers so dequantization behavior is bit-identical after adopt.
/// Turbo4Asym-mode caches additionally carry the packed-V buffer, the
/// per-token V norms, and the deterministic Turbo4 seed so the adopted cache
/// can rebuild [`crate::cache::turbo::TurboQuantParams`] without referring
/// back to the originating `KVCache`.
/// Turbo4 (symmetric) caches also carry packed-K + per-token K norms.
/// Turbo4Delegated-mode caches (K unified) carry
/// the unified FP16 K buffer in `keys` (same shape contract as `Fp16` mode),
/// the V-side cold/hot split state (`cold_offset`), and the configured
/// hot-V fold threshold so the adopted cache resumes decoding from the same
/// V cold/hot boundary it left at detach time. There is no separate
/// `cold_keys` field — K is unified. When the opt-in FP16 fast path is active,
/// the detached `values` tensor is the unified FP16 V buffer rather than a hot
/// ring; `delegated_fp16_fast_path` preserves that interpretation on adopt and
/// `delegated_fp16_sidecar_policy` preserves the foreground compaction policy.
/// Turbo3Asym-mode caches use the same `(keys, v_packed, v_norms)` triple
/// as `Turbo4Asym` but the V buffer carries the 24-bit-grouped 3-bit indices
/// The `mode` field on the handle preserves the bit-width
/// distinction so adopt rebuilds the right `TurboQuantParams3` instance.
pub struct DetachedKVCache {
    pub(super) keys: Option<UniquePtr<MlxArray>>,
    pub(super) values: Option<UniquePtr<MlxArray>>,
    pub(super) offset: i32,
    pub(super) step: i32,
    pub(super) mode: KVCacheMode,
    pub(super) key_scales: Option<UniquePtr<MlxArray>>,
    pub(super) val_scales: Option<UniquePtr<MlxArray>>,
    pub(super) v_packed: Option<UniquePtr<MlxArray>>,
    pub(super) v_norms: Option<UniquePtr<MlxArray>>,
    /// Turbo4-V Sparse-V kernel rescale sidecar. Lockstep with
    /// `v_norms`; round-trips through detach/adopt so paged + prefix-cache
    /// reuse paths preserve the precomputed rescale.
    pub(super) v_rescale: Option<UniquePtr<MlxArray>>,
    pub(super) k_packed: Option<UniquePtr<MlxArray>>,
    pub(super) k_norms: Option<UniquePtr<MlxArray>>,
    pub(super) turbo_seed: u32,
    pub(super) cold_offset: i32,
    pub(super) hot_threshold: i32,
    pub(super) delegated_fp16_fast_path: bool,
    pub(super) delegated_fp16_sidecar_policy: super::turbo::DelegatedFp16SidecarPolicy,
}

impl DetachedKVCache {
    /// Logical length of the stored cache (matches the live
    /// [`KVCache::seq_len`] at detach time).
    pub fn seq_len(&self) -> i32 {
        self.offset
    }

    /// Quantization mode of the detached cache.
    pub fn mode(&self) -> KVCacheMode {
        self.mode
    }

    /// Total byte footprint of the detached tensors (keys + values + INT8
    /// scales + Turbo4Asym v_packed/v_norms + Turbo4 symmetric
    /// k_packed/k_norms). Turbo4Delegated caches no longer carry
    /// a separate `cold_keys` tensor — the unified K buffer is already
    /// counted under `keys`.
    pub fn nbytes(&self) -> usize {
        let k = self.keys.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let v = self.values.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let ks = self.key_scales.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let vs = self.val_scales.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let vp = self.v_packed.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let vn = self.v_norms.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let vr = self.v_rescale.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let kp = self.k_packed.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let kn = self.k_norms.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        k + v + ks + vs + vp + vn + vr + kp + kn
    }

    /// Whether the detached handle carries no data (all tensors were `None`).
    pub fn is_empty(&self) -> bool {
        self.keys.is_none() && self.k_packed.is_none()
    }

    /// Read-only access to the detached keys tensor.
    pub fn keys(&self) -> Option<&MlxArray> {
        self.keys.as_deref()
    }

    /// Read-only access to the detached values tensor.
    pub fn values(&self) -> Option<&MlxArray> {
        self.values.as_deref()
    }

    /// Shrink the inert detached cache to exactly `new_len` tokens along the
    /// sequence-length axis (axis 2 of the per-layer 4-D tensors).
    ///
    /// Mirrors [`KVCache::trim`] semantics on the post-detach handle:
    ///
    /// * `new_len == 0` drops every backing buffer (keys, values, INT8 scale
    ///   sidecars, Turbo4 packed sidecars / norms / V-rescale, K-side packed
    ///   sidecars and norms when in symmetric Turbo4) and resets `offset`.
    ///   The mode tag and `step` are preserved so a subsequent
    ///   [`KVCache::install_detached`] round-trip stays consistent.
    /// * `0 < new_len < self.offset` re-slices each tensor's seq-len axis to
    ///   `new_len` via the same `ffi::slice` primitive `KVCache::trim` uses.
    ///   For `Turbo4Delegated` the K side is unified so the shape contract is
    ///   identical to `Fp16`; the V-side cold/hot split is rebuilt from
    ///   `cold_offset.min(new_len)` and `delegated_fp16_fast_path` decides
    ///   whether `values` is a unified FP16 buffer or a hot ring.
    /// * `new_len == self.offset` is a no-op (zero allocations).
    /// * `new_len < 0` or `new_len > self.offset` returns `Err` and leaves
    ///   the handle untouched.
    ///
    /// This is the inert-side counterpart of [`KVCache::trim_to`] and is the
    /// load-bearing primitive that lets the scheduler adopt only the first
    /// `matched_len` tokens' worth of KV state when an Automatic Prefix
    /// Caching (APC) lookup returns a block-aligned matched
    /// length shorter than the full cached entry. See server-side
    /// `try_adopt_cached_prefix` in `src/server/batch/scheduler.rs`.
    ///
    /// INT8 scale tensors and the Turbo4* per-token sidecars (`v_packed`,
    /// `v_norms`, `v_rescale`, `k_packed`, `k_norms`) are sliced in lockstep
    /// so a subsequent install + dequantize stays bit-identical to the
    /// already-trimmed live cache.
    ///
    /// Used by: [`DetachedCacheSet::truncate_to`] (— APC block-level partial cache adoption in the scheduler).
    pub fn trim_to(&mut self, new_len: i32) -> Result<(), String> {
        if new_len < 0 {
            return Err(format!(
                "DetachedKVCache::trim_to: new_len must be non-negative, got {new_len}"
            ));
        }
        if new_len > self.offset {
            return Err(format!(
                "DetachedKVCache::trim_to: new_len ({new_len}) exceeds current offset ({})",
                self.offset
            ));
        }
        if new_len == self.offset {
            return Ok(());
        }

        let prev_offset = self.offset;

        if new_len == 0 {
            // Drop every backing buffer and reset offsets / cold-V state.
            self.keys = None;
            self.values = None;
            self.key_scales = None;
            self.val_scales = None;
            self.v_packed = None;
            self.v_norms = None;
            self.v_rescale = None;
            self.k_packed = None;
            self.k_norms = None;
            self.cold_offset = 0;
            self.offset = 0;
            return Ok(());
        }

        // Slice axis 2 (seq-len) of each tensor to the new length. Width axes
        // (B, H, head_dim / packed_dim) come from the existing shape.
        let trim_axis_seq =
            |a: &Option<UniquePtr<MlxArray>>, tail_axis: i32| -> Option<UniquePtr<MlxArray>> {
                a.as_ref().map(|arr| {
                    let shape = ffi::array_shape(arr);
                    let last = if tail_axis == 0 { shape[3] } else { tail_axis };
                    ffi::slice(arr, &[0, 0, 0, 0], &[shape[0], shape[1], new_len, last])
                })
            };

        if self.mode == KVCacheMode::Turbo4Delegated {
            // K is unified — same shape contract as Fp16. Slice to `new_len`.
            self.keys = trim_axis_seq(&self.keys, 0);

            let new_cold = self.cold_offset.min(new_len);
            let new_hot_len = new_len - new_cold;

            // V buffer interpretation depends on the FP16 fast path. When the
            // fast path is on, V is a unified FP16 working set sliced to the
            // total length; when off, V is the hot ring sliced to the new hot
            // tail length.
            if let Some(ref v) = self.values {
                let v_shape = ffi::array_shape(v);
                let visible_v_len = if self.delegated_fp16_fast_path {
                    new_len
                } else {
                    new_hot_len
                };
                // NOTE: when `visible_v_len == 0` we explicitly drop the V
                // buffer (set to `None`). This diverges intentionally from
                // `KVCache::trim`, which leaves the backing buffer in place
                // in the same scenario (its `if visible_v_len > 0` guard
                // simply skips the slice call). In the live cache the
                // retained stale buffer is harmless because `update_and_fetch`
                // checks `offset == 0` before reading V; in the detached
                // handle the stale buffer would be re-installed via
                // `install_detached` and could be decoded against, so we zero
                // it out defensively here.
                self.values = if visible_v_len > 0 {
                    Some(ffi::slice(
                        v,
                        &[0, 0, 0, 0],
                        &[v_shape[0], v_shape[1], visible_v_len, v_shape[3]],
                    ))
                } else {
                    None
                };
            }

            // Cold-V sidecars only need re-slicing when cold actually shrank.
            // Re-slice unconditionally to `new_cold` so we never carry cold
            // state past the new logical boundary.
            if new_cold > 0 {
                if let Some(ref vp) = self.v_packed {
                    let vp_shape = ffi::array_shape(vp);
                    self.v_packed = Some(ffi::slice(
                        vp,
                        &[0, 0, 0, 0],
                        &[vp_shape[0], vp_shape[1], new_cold, vp_shape[3]],
                    ));
                }
                if let Some(ref vn) = self.v_norms {
                    let vn_shape = ffi::array_shape(vn);
                    self.v_norms = Some(ffi::slice(
                        vn,
                        &[0, 0, 0, 0],
                        &[vn_shape[0], vn_shape[1], new_cold, 1],
                    ));
                }
                if let Some(ref vr) = self.v_rescale {
                    let vr_shape = ffi::array_shape(vr);
                    self.v_rescale = Some(ffi::slice(
                        vr,
                        &[0, 0, 0, 0],
                        &[vr_shape[0], vr_shape[1], new_cold, 1],
                    ));
                }
            } else {
                // Cold portion fully erased.
                self.v_packed = None;
                self.v_norms = None;
                self.v_rescale = None;
            }

            self.cold_offset = new_cold;
            self.offset = new_len;
        } else {
            // Non-delegated modes: simple buffer-prefix trim along axis 2.
            self.keys = trim_axis_seq(&self.keys, 0);
            self.values = trim_axis_seq(&self.values, 0);

            // INT8 scale sidecars carry a width-1 head-axis (axis 3).
            if self.mode == KVCacheMode::Int8 {
                self.key_scales = trim_axis_seq(&self.key_scales, 1);
                self.val_scales = trim_axis_seq(&self.val_scales, 1);
            }

            // Turbo4* V sidecars (per-token packed + norms + rescale).
            if matches!(
                self.mode,
                KVCacheMode::Turbo4Asym | KVCacheMode::Turbo4 | KVCacheMode::Turbo3Asym
            ) {
                self.v_packed = trim_axis_seq(&self.v_packed, 0);
                self.v_norms = trim_axis_seq(&self.v_norms, 1);
                self.v_rescale = trim_axis_seq(&self.v_rescale, 1);
            }

            // K-side sidecars exist only in symmetric Turbo4.
            if self.mode == KVCacheMode::Turbo4 {
                self.k_packed = trim_axis_seq(&self.k_packed, 0);
                self.k_norms = trim_axis_seq(&self.k_norms, 1);
            }

            self.offset = new_len;
        }

        debug_assert!(
            self.offset == new_len,
            "DetachedKVCache::trim_to: post-condition failed: offset {} != new_len {} (was {prev_offset})",
            self.offset,
            new_len
        );
        Ok(())
    }
}

impl std::fmt::Debug for DetachedKVCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DetachedKVCache")
            .field("offset", &self.offset)
            .field("step", &self.step)
            .field("mode", &self.mode)
            .field("has_keys", &self.keys.is_some())
            .field("has_values", &self.values.is_some())
            .field("has_key_scales", &self.key_scales.is_some())
            .field("has_val_scales", &self.val_scales.is_some())
            .field("has_v_packed", &self.v_packed.is_some())
            .field("has_v_norms", &self.v_norms.is_some())
            .field("has_v_rescale", &self.v_rescale.is_some())
            .field("has_k_packed", &self.k_packed.is_some())
            .field("has_k_norms", &self.k_norms.is_some())
            .field("turbo_seed", &self.turbo_seed)
            .field("cold_offset", &self.cold_offset)
            .field("hot_threshold", &self.hot_threshold)
            .field("delegated_fp16_fast_path", &self.delegated_fp16_fast_path)
            .field(
                "delegated_fp16_sidecar_policy",
                &self.delegated_fp16_sidecar_policy,
            )
            .finish()
    }
}

// ---------------------------------------------------------------------------
// KVCache extensions
// ---------------------------------------------------------------------------

impl KVCache {
    /// Shrink the logical cache length to exactly `new_len`.
    ///
    /// Semantics:
    /// * `new_len == 0` fully rewinds the cache (equivalent to
    ///   `trim(self.offset)`) and drops all backing buffers.
    /// * `0 < new_len < self.offset` keeps the pre-allocated buffer but
    ///   re-slices its visible region to `new_len`. This mirrors
    ///   [`KVCache::trim`] but takes an absolute target instead of a delta.
    /// * `new_len == self.offset` is a no-op.
    /// * `new_len < 0` or `new_len > self.offset` returns `Err`.
    ///
    /// INT8 mode: the per-token scale buffers are trimmed in lock-step so
    /// subsequent `update_and_fetch` dequantization stays consistent.
    ///
    /// Used by: prompt prefix cache reuse, speculative decode rewinds,
    /// server scheduler trim-to-exact-prefix paths.
    pub fn trim_to(&mut self, new_len: i32) -> Result<(), String> {
        if new_len < 0 {
            return Err(format!(
                "KVCache::trim_to: new_len must be non-negative, got {new_len}"
            ));
        }
        if new_len > self.offset {
            return Err(format!(
                "KVCache::trim_to: new_len ({new_len}) exceeds current offset ({})",
                self.offset
            ));
        }

        let delta = self.offset - new_len;
        if delta == 0 {
            return Ok(());
        }

        let trimmed = self.trim(delta);
        debug_assert_eq!(
            trimmed, delta,
            "KVCache::trim returned {trimmed} but trim_to computed delta {delta}"
        );
        Ok(())
    }

    /// Move the underlying MLX buffers out of this cache into a
    /// [`DetachedKVCache`] handle.
    ///
    /// After this call the source `KVCache` is empty (`is_empty() == true`,
    /// `offset == 0`) but retains its quantization mode and step size so it
    /// can be reused for a new sequence. The returned `DetachedKVCache`
    /// carries the original tensors unchanged — including INT8 scale buffers
    /// when `mode == Int8` — so adopt is a zero-copy operation. For the
    /// delegated FP16 fast path this first compacts any missing packed V
    /// sidecars so lazy generation can skip foreground folds without donating
    /// a sidecar-incomplete prompt-cache entry.
    ///
    /// Used by: prompt prefix cache detach/adopt, cross-request reuse
    /// handoff inside `CachePool::detach`.
    pub fn clone_handle(&mut self) -> DetachedKVCache {
        self.compact_turbo4_delegated_fp16_sidecars();

        let handle = DetachedKVCache {
            keys: self.keys.take(),
            values: self.values.take(),
            offset: std::mem::replace(&mut self.offset, 0),
            step: self.step,
            mode: self.mode,
            key_scales: self.key_scales.take(),
            val_scales: self.val_scales.take(),
            v_packed: self.v_packed.take(),
            v_norms: self.v_norms.take(),
            v_rescale: self.v_rescale.take(),
            k_packed: self.k_packed.take(),
            k_norms: self.k_norms.take(),
            turbo_seed: self.turbo_seed,
            cold_offset: std::mem::replace(&mut self.cold_offset, 0),
            hot_threshold: self.hot_threshold,
            delegated_fp16_fast_path: self.delegated_fp16_fast_path,
            delegated_fp16_sidecar_policy: self.delegated_fp16_sidecar_policy,
        };
        // Clear turbo_params on the source so the next quantize call rebuilds
        // it from scratch (required if the slot is reused with a different
        // head_dim after detach). LOW-1 fix. The 3-bit
        // `turbo3_params` follows the same contract.
        self.turbo_params = None;
        self.turbo3_params = None;
        // retired the cold-V dequant memo — nothing to drop on
        // the source.
        handle
    }

    /// Re-install a previously detached cache into this `KVCache` slot.
    ///
    /// This is the inverse of [`KVCache::clone_handle`]. The receiver must be
    /// empty (`is_empty() == true`) to guarantee no live buffer is silently
    /// dropped; callers that need to overwrite a populated cache should
    /// `trim_to(0)` first.
    ///
    /// Used by: `CachePool::adopt` when re-hydrating per-layer caches for a
    /// freshly allocated sequence id.
    pub fn install_detached(&mut self, detached: DetachedKVCache) -> Result<(), String> {
        if !self.is_empty() {
            return Err(
                "KVCache::install_detached: target cache is not empty; trim_to(0) first".into(),
            );
        }
        self.keys = detached.keys;
        self.values = detached.values;
        self.offset = detached.offset;
        self.step = detached.step;
        self.mode = detached.mode;
        self.key_scales = detached.key_scales;
        self.val_scales = detached.val_scales;
        self.v_packed = detached.v_packed;
        self.v_norms = detached.v_norms;
        self.v_rescale = detached.v_rescale;
        self.k_packed = detached.k_packed;
        self.k_norms = detached.k_norms;
        self.turbo_seed = detached.turbo_seed;
        self.cold_offset = detached.cold_offset;
        self.hot_threshold = detached.hot_threshold;
        self.delegated_fp16_fast_path = detached.delegated_fp16_fast_path;
        self.delegated_fp16_sidecar_policy = detached.delegated_fp16_sidecar_policy;
        // turbo_params is rebuilt lazily on the next quantize call, but if we
        // can already see the V head_dim from v_packed we may as well prebuild
        // so dequantize-only consumers (which don't go through update_*) still
        // work. Detect it from v_packed (or k_packed for symmetric Turbo4)
        // shape: [B, H, T, head_dim/2].
        if matches!(
            self.mode,
            KVCacheMode::Turbo4Asym | KVCacheMode::Turbo4 | KVCacheMode::Turbo4Delegated
        ) {
            let probe = self.v_packed.as_ref().or(self.k_packed.as_ref());
            if let Some(p) = probe {
                let shape = ffi::array_shape(p);
                if shape.len() == 4 && shape[3] > 0 {
                    let head_dim = (shape[3] as u32) * 2;
                    self.turbo_params = Some(super::turbo::TurboQuantParams::new(
                        head_dim,
                        self.turbo_seed,
                    ));
                }
            }
        }
        // Turbo3Asym: rebuild the 3-bit params from v_packed
        // shape. Inverse of `head_dim * 3 / 8`: head_dim = packed_dim * 8 / 3.
        // Mirrors the Turbo4 prebuild above so dequantize-only consumers see
        // a ready-to-go cache after install.
        if self.mode == KVCacheMode::Turbo3Asym {
            if let Some(p) = self.v_packed.as_ref() {
                let shape = ffi::array_shape(p);
                if shape.len() == 4 && shape[3] > 0 {
                    let head_dim = (shape[3] as u32) * 8 / 3;
                    self.turbo3_params = Some(super::turbo::quant3::TurboQuantParams3::new(
                        head_dim,
                        self.turbo_seed,
                    ));
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DetachedRotatingKVCache (B9)
// ---------------------------------------------------------------------------

/// Inert snapshot of a single [`RotatingKVCache`] (sliding-window cache) that
/// can outlive the sequence which produced it.
///
/// Mirrors [`DetachedKVCache`] for the rotating cache backend, including the
/// `Turbo4Asym` packed sidecar buffers (`v_packed`, `v_norms`) and the
/// deterministic seed so the adopted cache can rebuild
/// [`crate::cache::turbo::TurboQuantParams`] without consulting the originating
/// cache. Adds the rotating-specific `max_size` and `idx` fields so the ring
/// position is preserved across the round-trip — without `idx`, a wrap-around
/// state would silently fall back to "no wraparound yet" semantics.
///
/// Used by: prompt prefix cache reuse for sliding-window models (Gemma 3/4,
/// Ministral 3, GPT-OSS, RecurrentGemma, Exaone) under the same
/// architecture as the dense `DetachedKVCache`.
pub struct DetachedRotatingKVCache {
    pub(super) keys: Option<UniquePtr<MlxArray>>,
    pub(super) values: Option<UniquePtr<MlxArray>>,
    pub(super) max_size: i32,
    pub(super) offset: i32,
    pub(super) idx: i32,
    pub(super) step: i32,
    pub(super) mode: KVCacheMode,
    pub(super) key_scales: Option<UniquePtr<MlxArray>>,
    pub(super) val_scales: Option<UniquePtr<MlxArray>>,
    pub(super) v_packed: Option<UniquePtr<MlxArray>>,
    pub(super) v_norms: Option<UniquePtr<MlxArray>>,
    /// Sparse-V rescale sidecar for rotating Turbo4Asym caches.
    pub(super) v_rescale: Option<UniquePtr<MlxArray>>,
    pub(super) turbo_seed: u32,
}

impl DetachedRotatingKVCache {
    /// Logical sequence length at detach time (matches `RotatingKVCache::offset`).
    pub fn seq_len(&self) -> i32 {
        self.offset
    }

    /// Quantization mode of the detached cache.
    pub fn mode(&self) -> KVCacheMode {
        self.mode
    }

    /// Sliding window upper bound preserved across the round-trip.
    pub fn max_size(&self) -> i32 {
        self.max_size
    }

    /// Total byte footprint of the detached tensors.
    pub fn nbytes(&self) -> usize {
        let k = self.keys.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let v = self.values.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let ks = self.key_scales.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let vs = self.val_scales.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let vp = self.v_packed.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let vn = self.v_norms.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        let vr = self.v_rescale.as_ref().map_or(0, |a| ffi::array_nbytes(a));
        k + v + ks + vs + vp + vn + vr
    }

    /// Whether the detached handle carries no data (all tensors were `None`).
    pub fn is_empty(&self) -> bool {
        self.keys.is_none()
    }
}

impl std::fmt::Debug for DetachedRotatingKVCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DetachedRotatingKVCache")
            .field("max_size", &self.max_size)
            .field("offset", &self.offset)
            .field("idx", &self.idx)
            .field("step", &self.step)
            .field("mode", &self.mode)
            .field("has_keys", &self.keys.is_some())
            .field("has_values", &self.values.is_some())
            .field("has_v_packed", &self.v_packed.is_some())
            .field("has_v_norms", &self.v_norms.is_some())
            .field("has_v_rescale", &self.v_rescale.is_some())
            .field("turbo_seed", &self.turbo_seed)
            .finish()
    }
}

impl RotatingKVCache {
    /// Move the underlying MLX buffers out of this rotating cache into a
    /// [`DetachedRotatingKVCache`] handle.
    ///
    /// After this call the source `RotatingKVCache` is empty
    /// (`is_empty() == true`, `offset == 0`, `idx == 0`) but retains its
    /// `max_size`, quantization mode, step, and `turbo_seed` so it can be
    /// reused for a new sequence. The returned handle carries the original
    /// tensors unchanged — including `v_packed` / `v_norms` for `Turbo4Asym`
    /// — so adopt is a zero-copy operation.
    ///
    /// Used by: sliding-window prompt prefix cache detach/adopt (B9; dense counterpart is [`KVCache::clone_handle`]).
    pub fn clone_handle(&mut self) -> DetachedRotatingKVCache {
        let handle = DetachedRotatingKVCache {
            keys: self.keys.take(),
            values: self.values.take(),
            max_size: self.max_size,
            offset: std::mem::replace(&mut self.offset, 0),
            idx: std::mem::replace(&mut self.idx, 0),
            step: self.step,
            mode: self.mode,
            key_scales: self.key_scales.take(),
            val_scales: self.val_scales.take(),
            v_packed: self.v_packed.take(),
            v_norms: self.v_norms.take(),
            v_rescale: self.v_rescale.take(),
            turbo_seed: self.turbo_seed,
        };
        // Mirror `KVCache::clone_handle` (LOW-1): clear cached
        // turbo_params on the source so the next quantize call rebuilds them
        // from scratch (slot may be reused with a different head_dim).
        self.turbo_params = None;
        handle
    }

    /// Re-install a previously detached rotating cache into this slot.
    ///
    /// Inverse of [`RotatingKVCache::clone_handle`]. The receiver must be
    /// empty (`is_empty() == true`) so no live buffer is silently dropped;
    /// callers that need to overwrite a populated cache should construct a
    /// fresh `RotatingKVCache::new_with_mode_and_seed` and install into that.
    ///
    /// Block alignment: the adopted state is bit-identical to the source's,
    /// including `idx`. Because per-token Turbo4 quantization is independent
    /// across slots, no alignment-recovery work is needed at install time.
    pub fn install_detached(&mut self, detached: DetachedRotatingKVCache) -> Result<(), String> {
        if !self.is_empty() {
            return Err("RotatingKVCache::install_detached: target cache is not empty".into());
        }
        self.keys = detached.keys;
        self.values = detached.values;
        self.max_size = detached.max_size;
        self.offset = detached.offset;
        self.idx = detached.idx;
        self.step = detached.step;
        self.mode = detached.mode;
        self.key_scales = detached.key_scales;
        self.val_scales = detached.val_scales;
        self.v_packed = detached.v_packed;
        self.v_norms = detached.v_norms;
        self.v_rescale = detached.v_rescale;
        self.turbo_seed = detached.turbo_seed;
        // Pre-build turbo_params from v_packed shape if available so the
        // first dequantize-only consumer doesn't need to wait for an update
        // call (mirrors `KVCache::install_detached`).
        if self.mode == KVCacheMode::Turbo4Asym {
            if let Some(ref vp) = self.v_packed {
                let shape = ffi::array_shape(vp);
                if shape.len() == 4 && shape[3] > 0 {
                    let head_dim = (shape[3] as u32) * 2;
                    self.turbo_params = Some(super::turbo::TurboQuantParams::new(
                        head_dim,
                        self.turbo_seed,
                    ));
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DetachedCacheSet
// ---------------------------------------------------------------------------

/// Inert snapshot of a whole sequence's per-layer KV caches.
///
/// Produced by [`CachePool::detach`] and consumed by [`CachePool::adopt`].
/// Only the dense backend is supported; paged sequences produce `None` on
/// detach.
pub struct DetachedCacheSet {
    /// Per-layer detached caches, one per model layer.
    pub caches: Vec<DetachedKVCache>,
    /// Logical backend tag (always `DenseKvCache` in this API surface).
    pub backend: SequenceStateBackend,
    /// Prompt length as recorded on the originating [`SequenceCacheSet`].
    pub prompt_len: usize,
    /// Last known decode offset at detach time.
    pub current_offset: i32,
    /// Timestamp of the originating allocation (preserved across handoffs).
    pub created_at: Instant,
    /// Timestamp of the most recent detach.
    pub detached_at: Instant,
    /// Original sequence id this cache set was last installed under (for
    /// logging / observability; the adopt path always assigns a fresh id).
    pub origin_seq_id: SequenceId,
}

impl DetachedCacheSet {
    /// Summed tensor bytes across all layer caches.
    pub fn nbytes(&self) -> usize {
        self.caches.iter().map(|c| c.nbytes()).sum()
    }

    /// Number of layer caches carried by this set.
    pub fn num_layers(&self) -> usize {
        self.caches.len()
    }

    /// Logical token length of the first non-empty layer (or 0).
    ///
    /// All transformer layers share a common prefix length by construction,
    /// so the first layer's `offset` is a faithful summary of the set.
    pub fn seq_len(&self) -> i32 {
        self.caches.first().map(|c| c.offset).unwrap_or(0)
    }

    /// Shrink every per-layer detached cache to exactly `new_len` tokens.
    ///
    /// Walks each [`DetachedKVCache`] and calls [`DetachedKVCache::trim_to`]
    /// in lockstep, then updates the set-wide `prompt_len` and
    /// `current_offset` so accounting downstream of `CachePool::adopt`
    /// matches the new logical length. An empty cache set or `new_len`
    /// equal to the existing length is a no-op.
    ///
    /// The intended caller is the scheduler's `try_adopt_cached_prefix` when
    /// an APC lookup returns a block-aligned `matched_len`
    /// shorter than the candidate entry's full token length — i.e. the
    /// request and the cached entry agree on the first N blocks but diverge
    /// at block N+1. Truncating the detached set to `matched_len` before
    /// adoption gives the model worker a KV cache whose logical length
    /// exactly matches the prefix the prefill loop will skip, so the next
    /// `update_and_fetch` writes at the correct seq-len offset.
    ///
    /// Returns `Err(_)` if any layer's `trim_to` rejects the request (e.g.
    /// `new_len > seq_len`). On error, layers already truncated stay at
    /// the new length — the caller should drop the set rather than retry.
    ///
    /// Used by: [`crate::cache::CachePool::adopt`] callers that need
    /// per-block partial adoption.
    #[must_use = "truncate_to returns Err on partial failure; on error some layers are already trimmed and the set must be dropped, not retried"]
    pub fn truncate_to(&mut self, new_len: i32) -> Result<(), String> {
        if new_len < 0 {
            return Err(format!(
                "DetachedCacheSet::truncate_to: new_len must be non-negative, got {new_len}"
            ));
        }
        if self.caches.is_empty() {
            // Empty set is a degenerate value; truncation is vacuously OK.
            self.current_offset = new_len;
            self.prompt_len = (new_len as usize).min(self.prompt_len);
            return Ok(());
        }
        // Sanity: every layer must agree on the pre-truncate seq length so
        // we never silently divergent-trim a set produced by a different
        // sequence layout.
        let head = self.caches[0].offset;
        debug_assert!(
            self.caches.iter().all(|c| c.offset == head),
            "DetachedCacheSet::truncate_to: layers disagree on seq_len: {:?}",
            self.caches.iter().map(|c| c.offset).collect::<Vec<_>>()
        );
        if new_len == head {
            return Ok(());
        }
        if new_len > head {
            return Err(format!(
                "DetachedCacheSet::truncate_to: new_len ({new_len}) exceeds current seq_len ({head})"
            ));
        }

        for (i, cache) in self.caches.iter_mut().enumerate() {
            cache.trim_to(new_len).map_err(|e| {
                format!("DetachedCacheSet::truncate_to: layer {i} trim_to failed: {e}")
            })?;
        }

        self.current_offset = new_len;
        // `prompt_len` is the originating prompt size at detach time. After a
        // partial adoption the request shares only `new_len` of those tokens,
        // so clamp to that — never grow beyond.
        let new_prompt = new_len.max(0) as usize;
        if new_prompt < self.prompt_len {
            self.prompt_len = new_prompt;
        }
        Ok(())
    }
}

impl std::fmt::Debug for DetachedCacheSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DetachedCacheSet")
            .field("backend", &self.backend)
            .field("num_layers", &self.num_layers())
            .field("seq_len", &self.seq_len())
            .field("prompt_len", &self.prompt_len)
            .field("current_offset", &self.current_offset)
            .field("origin_seq_id", &self.origin_seq_id)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Parking (in-flight holding)
// ---------------------------------------------------------------------------

/// Opaque handle returned by [`CachePool::park_detached`].
///
/// Parking is an optional escape hatch: a scheduler can hand a
/// [`DetachedCacheSet`] back to the pool for the duration of a cross-request
/// lookup so that [`CachePool::memory_usage_bytes`] keeps accounting for the
/// tensors that the pool logically still holds in-flight. The normal
/// `detach` → store in external cache → `adopt` flow does not require
/// parking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DetachedHandle(u64);

impl DetachedHandle {
    /// Raw numeric representation of this handle, useful for logging and
    /// metric labels.
    pub fn as_u64(self) -> u64 {
        self.0
    }

    /// Construct a handle from a raw id. Provided for cross-module builders
    /// (e.g. the paged detach surface in [`super::paged_detach`]) that mint
    /// handles out of the same `CachePool::next_id` space.
    pub(super) fn from_raw(id: u64) -> Self {
        Self(id)
    }
}

impl std::fmt::Display for DetachedHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "detached-{}", self.0)
    }
}

/// Internal map of parked detached cache sets, keyed by handle. This map is
/// attached to [`CachePool`] as `detached: DetachedMap` via the
/// `detached` field declared in the parent module. The map stores a
/// [`super::paged_detach::ParkedCache`] enum so dense and paged variants
/// share the same handle namespace.
pub(super) type DetachedMap = HashMap<DetachedHandle, super::paged_detach::ParkedCache>;

// ---------------------------------------------------------------------------
// CachePool extensions
// ---------------------------------------------------------------------------

impl CachePool {
    /// Remove `seq_id` from the active set and return its per-layer caches as
    /// a `DetachedCacheSet` without freeing the MLX buffers.
    ///
    /// Returns `None` if:
    /// * `seq_id` is not currently active, or
    /// * the sequence uses the paged backend (paged detach is's responsibility — this method deliberately rejects it).
    ///
    /// The caller is responsible for re-homing the detached set, either by
    /// passing it to [`CachePool::adopt`] or by parking it via
    /// [`CachePool::park_detached`]. Dropping the returned set releases the
    /// underlying MLX memory normally.
    ///
    /// Used by: prompt prefix cache store, scheduler request-boundary
    /// handoff.
    pub fn detach(&mut self, seq_id: SequenceId) -> Option<DetachedCacheSet> {
        // Peek first so we can refuse non-dense backends without destructive
        // side effects.
        {
            let sequence = self.active.get(&seq_id)?;
            if sequence.backend != SequenceStateBackend::DenseKvCache {
                return None;
            }
        }

        let mut sequence = self.active.remove(&seq_id)?;
        let detached_caches: Vec<DetachedKVCache> = sequence
            .caches
            .iter_mut()
            .map(|cache| cache.clone_handle())
            .collect();

        Some(DetachedCacheSet {
            caches: detached_caches,
            backend: sequence.backend,
            prompt_len: sequence.prompt_len,
            current_offset: sequence.current_offset,
            created_at: sequence.created_at,
            detached_at: Instant::now(),
            origin_seq_id: sequence.seq_id,
        })
    }

    /// Install a previously-detached cache set under a fresh `SequenceId`.
    ///
    /// Capacity is checked against `max_sequences` before allocation. On
    /// success the model's
    /// [`prepare_sequence_state`](crate::generate::LanguageModel::prepare_sequence_state)
    /// hook is invoked with the new id so any per-model sidecar maps
    /// (mixed-cache models, quantized sidecars, etc.) are initialized
    /// consistently with a freshly allocated sequence.
    ///
    /// Only `DenseKvCache` sets are supported; attempting to adopt a paged
    /// set returns an error and the original set is dropped (its tensors
    /// freed) to avoid leaks. Use [`CachePool::adopt_preserving`] when the
    /// caller wants the set back on failure.
    ///
    /// Used by: prompt prefix cache re-entry, scheduler fast-path
    /// when a new request reuses an existing prefix.
    pub fn adopt(
        &mut self,
        model: &dyn crate::generate::LanguageModel,
        detached: DetachedCacheSet,
    ) -> Result<SequenceId, String> {
        self.adopt_preserving(model, detached)
            .map_err(|(err, _)| err)
    }

    /// Like [`CachePool::adopt`] but returns the original [`DetachedCacheSet`]
    /// back to the caller on failure so it can be retried or routed
    /// elsewhere.
    pub fn adopt_preserving(
        &mut self,
        model: &dyn crate::generate::LanguageModel,
        detached: DetachedCacheSet,
    ) -> Result<SequenceId, (String, DetachedCacheSet)> {
        if detached.backend != SequenceStateBackend::DenseKvCache {
            return Err((
                format!(
                    "CachePool::adopt: backend {:?} is not supported (paged adopt is tracked)",
                    detached.backend
                ),
                detached,
            ));
        }
        if self.active.len() >= self.max_sequences {
            return Err((
                format!(
                    "CachePool::adopt: max capacity ({}) reached, cannot adopt new sequence",
                    self.max_sequences
                ),
                detached,
            ));
        }

        let id = SequenceId::from_raw(self.next_id.fetch_add(1, Ordering::Relaxed));

        // Reconstruct the live per-layer caches from the detached handles.
        // `KVCache::install_detached` demands an empty target, which
        // `KVCache::new()` trivially satisfies.
        let DetachedCacheSet {
            caches,
            backend,
            prompt_len,
            current_offset,
            created_at,
            detached_at: _,
            origin_seq_id: _,
        } = detached;

        let mut live: Vec<KVCache> = Vec::with_capacity(caches.len());
        for detached_cache in caches {
            let mut cache = KVCache::new_with_mode(detached_cache.mode);
            cache
                .install_detached(detached_cache)
                .expect("freshly constructed KVCache is empty");
            live.push(cache);
        }

        let mut entry = SequenceCacheSet::dense_external(id, live);
        // Preserve the originating metadata across the handoff so scheduler
        // stats and reuse bookkeeping stay coherent.
        entry.backend = backend;
        entry.prompt_len = prompt_len;
        entry.current_offset = current_offset;
        entry.created_at = created_at;
        self.active.insert(id, entry);

        // Hook in the model-side sidecar state for the new id, matching the
        // normal `allocate` -> `prepare_sequence_state` sequencing that the
        // batch scheduler uses today.
        model.prepare_sequence_state(id);

        Ok(id)
    }

    /// Park a detached set inside the pool so its bytes remain visible to
    /// [`CachePool::memory_usage_bytes`].
    ///
    /// Returns an opaque [`DetachedHandle`] that can later be consumed by
    /// [`CachePool::take_parked`] or [`CachePool::adopt_parked`]. Parked
    /// caches do **not** count toward `active_count()` and do not consume
    /// an `allocate()` slot — they only contribute to memory accounting.
    pub fn park_detached(&mut self, detached: DetachedCacheSet) -> DetachedHandle {
        let handle = DetachedHandle(self.next_id.fetch_add(1, Ordering::Relaxed));
        self.detached
            .insert(handle, super::paged_detach::ParkedCache::Dense(detached));
        handle
    }

    /// Retrieve a previously parked dense set, leaving the pool.
    ///
    /// Returns `None` if the handle was never parked, already taken, or
    /// points to a paged set (use [`CachePool::take_parked_paged`] for
    /// paged sets).
    pub fn take_parked(&mut self, handle: DetachedHandle) -> Option<DetachedCacheSet> {
        match self.detached.remove(&handle) {
            Some(super::paged_detach::ParkedCache::Dense(set)) => Some(set),
            Some(other) => {
                // Wrong variant — put it back so the caller can dispatch to
                // the paged-side take_parked.
                self.detached.insert(handle, other);
                None
            }
            None => None,
        }
    }

    /// Read-only peek at a parked dense set (for inspection / metrics).
    ///
    /// Returns `None` if the handle points to a paged set.
    pub fn peek_parked(&self, handle: DetachedHandle) -> Option<&DetachedCacheSet> {
        match self.detached.get(&handle) {
            Some(super::paged_detach::ParkedCache::Dense(set)) => Some(set),
            _ => None,
        }
    }

    /// Convenience: consume a parked handle and re-adopt it in one call.
    pub fn adopt_parked(
        &mut self,
        model: &dyn crate::generate::LanguageModel,
        handle: DetachedHandle,
    ) -> Result<SequenceId, String> {
        let detached = self
            .take_parked(handle)
            .ok_or_else(|| format!("CachePool::adopt_parked: unknown handle {handle}"))?;
        self.adopt(model, detached)
    }

    /// Number of currently parked detached sets (dense and paged combined).
    pub fn parked_count(&self) -> usize {
        self.detached.len()
    }

    /// Summed bytes across all parked detached sets (dense and paged).
    pub fn parked_bytes(&self) -> usize {
        self.detached.values().map(|d| d.nbytes()).sum()
    }
}

// Tests live in the companion `detach_tests.rs` so this file stays at a
// comfortable implementation-only size (see `docs/code-guidelines.md`).
#[cfg(test)]
#[path = "detach_tests.rs"]
mod tests;
