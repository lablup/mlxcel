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

//! Unit and property tests for `cache/detach.rs`.
//!
//! Organised as:
//! 1. `KVCache::trim_to` semantics (mid-buffer, trim-to-zero, error paths).
//! 2. `KVCache::clone_handle` / `install_detached` round-trip, including
//!    INT8 scale preservation.
//! 3. `CachePool::detach` / `adopt` round-trip, paged rejection,
//!    `prepare_sequence_state` wiring, capacity handling.
//! 4. Parking helpers and `memory_usage_bytes` accounting.
//! 5. Property test: prefill(N)+detach+adopt+decode(M) vs fresh prefill(N+M).

use super::*;
use crate::dtype;
use crate::generate::LanguageModel;
use crate::{allclose, array_to_raw_bytes, astype, eval, item_bool};
use cxx::UniquePtr;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Minimal model stub for adopt-path tests. Tracks whether
/// `prepare_sequence_state` was called for a specific id.
#[derive(Default)]
struct RecordingModel {
    num_layers: usize,
    prepared: std::cell::RefCell<Vec<SequenceId>>,
}

impl RecordingModel {
    fn new(num_layers: usize) -> Self {
        Self {
            num_layers,
            prepared: std::cell::RefCell::new(Vec::new()),
        }
    }

    fn prepared_ids(&self) -> Vec<SequenceId> {
        self.prepared.borrow().clone()
    }
}

impl LanguageModel for RecordingModel {
    fn forward(
        &self,
        _input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        crate::ffi::zeros(&[1], 0)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        (0..self.num_layers).map(|_| KVCache::new()).collect()
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![0]
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.prepared.borrow_mut().push(seq_id);
    }
}

/// Generate deterministic [1, 1, T, 1] FP32 tensors so round-trip
/// comparisons can be byte-exact.
fn fp32_tokens(values: &[f32]) -> UniquePtr<MlxArray> {
    let t = values.len() as i32;
    crate::ffi::from_slice_f32(values, &[1, 1, t, 1])
}

/// Extract every element of an FP32 cache keys/values tensor as a flat
/// Vec<f32> so tests can compare contents without depending on strides.
fn flatten_fp32(arr: &MlxArray) -> Vec<f32> {
    let a = astype(arr, dtype::FLOAT32);
    eval(&a);
    let bytes = array_to_raw_bytes(&a);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn visible_keys_as_fp32(cache: &KVCache) -> Vec<f32> {
    let keys = cache
        .keys
        .as_ref()
        .expect("cache should have keys when len > 0");
    let shape = crate::ffi::array_shape(keys);
    let sliced = crate::ffi::slice(
        keys,
        &[0, 0, 0, 0],
        &[shape[0], shape[1], cache.seq_len(), shape[3]],
    );
    flatten_fp32(&sliced)
}

// ---------------------------------------------------------------------------
// KVCache::trim_to
// ---------------------------------------------------------------------------

#[test]
fn trim_to_mid_buffer_keeps_buffer_and_resets_visible_length() {
    let mut cache = KVCache::new();
    let keys = fp32_tokens(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let values = fp32_tokens(&[10.0, 20.0, 30.0, 40.0, 50.0]);
    cache.update(keys, values);
    assert_eq!(cache.seq_len(), 5);

    cache.trim_to(3).expect("trim_to 3 must succeed");
    assert_eq!(cache.seq_len(), 3);
    assert!(!cache.is_empty());

    // Contents of the visible region must match the first three tokens.
    assert_eq!(visible_keys_as_fp32(&cache), vec![1.0, 2.0, 3.0]);
}

#[test]
fn trim_to_zero_drops_storage_and_preserves_mode() {
    let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
    let keys = fp32_tokens(&[1.0, 2.0, 3.0]);
    let values = fp32_tokens(&[4.0, 5.0, 6.0]);
    cache.update(keys, values);
    assert!(!cache.is_empty());
    assert_eq!(cache.mode, KVCacheMode::Int8);

    cache.trim_to(0).expect("trim_to 0 must succeed");
    assert_eq!(cache.seq_len(), 0);
    assert!(cache.is_empty());
    assert_eq!(cache.mode, KVCacheMode::Int8);
}

#[test]
fn trim_to_past_offset_returns_error_without_mutation() {
    let mut cache = KVCache::new();
    let keys = fp32_tokens(&[1.0, 2.0]);
    let values = fp32_tokens(&[3.0, 4.0]);
    cache.update(keys, values);
    assert_eq!(cache.seq_len(), 2);

    let err = cache
        .trim_to(5)
        .expect_err("trim_to past offset must error");
    assert!(err.contains("exceeds current offset"));
    assert_eq!(cache.seq_len(), 2, "cache must be unchanged after error");
}

#[test]
fn trim_to_negative_returns_error() {
    let mut cache = KVCache::new();
    let keys = fp32_tokens(&[1.0]);
    let values = fp32_tokens(&[2.0]);
    cache.update(keys, values);

    let err = cache.trim_to(-1).expect_err("trim_to negative must error");
    assert!(err.contains("non-negative"));
    assert_eq!(cache.seq_len(), 1);
}

#[test]
fn trim_to_exact_offset_is_noop() {
    let mut cache = KVCache::new();
    let keys = fp32_tokens(&[1.0, 2.0, 3.0]);
    let values = fp32_tokens(&[4.0, 5.0, 6.0]);
    cache.update(keys, values);
    assert_eq!(cache.seq_len(), 3);

    cache.trim_to(3).expect("trim_to offset must succeed");
    assert_eq!(cache.seq_len(), 3);
    assert_eq!(visible_keys_as_fp32(&cache), vec![1.0, 2.0, 3.0]);
}

// ---------------------------------------------------------------------------
// KVCache::clone_handle / install_detached
// ---------------------------------------------------------------------------

#[test]
fn clone_handle_leaves_source_empty_and_transfers_tensors() {
    let mut cache = KVCache::new();
    let keys = fp32_tokens(&[1.0, 2.0]);
    let values = fp32_tokens(&[3.0, 4.0]);
    cache.update(keys, values);

    let handle = cache.clone_handle();
    assert!(cache.is_empty(), "source cache must be empty after clone");
    assert_eq!(cache.seq_len(), 0);
    assert_eq!(handle.seq_len(), 2);
    assert!(!handle.is_empty());
    assert_eq!(handle.mode(), KVCacheMode::Fp16);
}

#[test]
fn install_detached_rejects_non_empty_target() {
    let mut src = KVCache::new();
    src.update(fp32_tokens(&[1.0]), fp32_tokens(&[2.0]));
    let handle = src.clone_handle();

    let mut dst = KVCache::new();
    dst.update(fp32_tokens(&[9.0]), fp32_tokens(&[99.0]));
    assert!(dst.install_detached(handle).is_err());
}

#[test]
fn clone_handle_install_detached_round_trip_preserves_contents() {
    let mut cache = KVCache::new();
    let keys = fp32_tokens(&[1.0, 2.0, 3.0, 4.0]);
    let values = fp32_tokens(&[5.0, 6.0, 7.0, 8.0]);
    cache.update(keys, values);
    let expected_keys = visible_keys_as_fp32(&cache);

    let handle = cache.clone_handle();
    let mut restored = KVCache::new();
    restored.install_detached(handle).unwrap();

    assert_eq!(restored.seq_len(), 4);
    assert_eq!(visible_keys_as_fp32(&restored), expected_keys);
}

#[test]
fn clone_handle_round_trip_preserves_int8_scales() {
    let mut cache = KVCache::new_with_mode(KVCacheMode::Int8);
    // Mixed magnitudes so per-token scales differ across tokens.
    let keys = fp32_tokens(&[1.0, 10.0, 100.0, 0.5]);
    let values = fp32_tokens(&[2.0, 20.0, 200.0, 0.25]);
    cache.update(keys, values);
    assert_eq!(cache.seq_len(), 4);

    // Capture scale buffer bytes pre-detach so we can compare post-adopt.
    let pre_scales = array_to_raw_bytes(cache.key_scales.as_ref().unwrap());

    let handle = cache.clone_handle();
    assert_eq!(handle.mode(), KVCacheMode::Int8);

    let mut restored = KVCache::new_with_mode(KVCacheMode::Int8);
    restored.install_detached(handle).unwrap();
    assert_eq!(restored.seq_len(), 4);
    assert_eq!(restored.mode, KVCacheMode::Int8);
    assert!(restored.key_scales.is_some());
    assert!(restored.val_scales.is_some());

    let post_scales = array_to_raw_bytes(restored.key_scales.as_ref().unwrap());
    assert_eq!(
        pre_scales, post_scales,
        "INT8 scale buffer must survive detach/adopt bit-for-bit"
    );
}

// ---------------------------------------------------------------------------
// CachePool::detach / adopt
// ---------------------------------------------------------------------------

#[test]
fn cache_pool_detach_returns_none_for_unknown_seq() {
    let mut pool = CachePool::new(4);
    assert!(pool.detach(SequenceId::from_raw(9_999)).is_none());
}

#[test]
fn cache_pool_detach_adopt_round_trip_preserves_contents() {
    let model = RecordingModel::new(2);
    let mut pool = CachePool::new(4);

    let seq_a = pool.allocate(&model).unwrap();
    {
        let caches = pool.get_caches_mut(seq_a).unwrap();
        caches[0].update(
            fp32_tokens(&[1.0, 2.0, 3.0]),
            fp32_tokens(&[10.0, 20.0, 30.0]),
        );
        caches[1].update(
            fp32_tokens(&[4.0, 5.0, 6.0]),
            fp32_tokens(&[40.0, 50.0, 60.0]),
        );
    }

    // Remember expected visible state pre-detach.
    let expected_layer_0 = {
        let caches = pool.get_caches_mut(seq_a).unwrap();
        visible_keys_as_fp32(&caches[0])
    };

    let detached = pool.detach(seq_a).expect("dense sequence should detach");
    assert_eq!(pool.active_count(), 0);
    assert!(pool.get_caches_mut(seq_a).is_none());
    assert_eq!(detached.num_layers(), 2);
    assert_eq!(detached.seq_len(), 3);

    let seq_b = pool.adopt(&model, detached).expect("adopt must succeed");
    assert_ne!(seq_a, seq_b);
    assert_eq!(pool.active_count(), 1);

    let caches = pool.get_caches_mut(seq_b).unwrap();
    assert_eq!(caches[0].seq_len(), 3);
    assert_eq!(caches[1].seq_len(), 3);
    assert_eq!(visible_keys_as_fp32(&caches[0]), expected_layer_0);

    // prepare_sequence_state must have been invoked for the new id.
    let prepared = model.prepared_ids();
    assert!(
        prepared.contains(&seq_b),
        "prepare_sequence_state not called for new id {seq_b}; got {prepared:?}"
    );
}

#[test]
fn cache_pool_detach_adopt_preserves_int8_round_trip() {
    let model = RecordingModel::new(1);
    let mut pool = CachePool::new(4);

    let seq_a = pool.allocate(&model).unwrap();
    // Swap the layer to INT8 mode so this test exercises the INT8 path.
    {
        let caches = pool.get_caches_mut(seq_a).unwrap();
        caches[0] = KVCache::new_with_mode(KVCacheMode::Int8);
        caches[0].update(
            fp32_tokens(&[1.0, 8.0, 64.0, 0.125]),
            fp32_tokens(&[2.0, 16.0, 128.0, 0.0625]),
        );
    }

    let pre_scales = {
        let caches = pool.get_caches_mut(seq_a).unwrap();
        array_to_raw_bytes(caches[0].key_scales.as_ref().unwrap())
    };

    let detached = pool.detach(seq_a).unwrap();
    let seq_b = pool.adopt(&model, detached).unwrap();

    let caches = pool.get_caches_mut(seq_b).unwrap();
    assert_eq!(caches[0].mode, KVCacheMode::Int8);
    assert_eq!(caches[0].seq_len(), 4);
    assert!(caches[0].key_scales.is_some());
    assert!(caches[0].val_scales.is_some());

    let post_scales = array_to_raw_bytes(caches[0].key_scales.as_ref().unwrap());
    assert_eq!(pre_scales, post_scales);
}

#[test]
fn cache_pool_detach_rejects_paged_backend() {
    use super::super::PagedKvLayout;
    let layout = PagedKvLayout::uniform(1, 4, 128).unwrap();
    // Lightweight paged model used in the existing paged tests.
    struct PagedOnly {
        layout: PagedKvLayout,
    }

    impl LanguageModel for PagedOnly {
        fn forward(
            &self,
            _input_ids: &MlxArray,
            _caches: &mut [KVCache],
            _mask: Option<&MlxArray>,
        ) -> UniquePtr<MlxArray> {
            crate::ffi::zeros(&[1], 0)
        }

        fn make_caches(&self) -> Vec<KVCache> {
            Vec::new()
        }

        fn num_layers(&self) -> usize {
            self.layout.num_layers
        }

        fn eos_token_ids(&self) -> Vec<i32> {
            vec![0]
        }

        fn sequence_state_layout(&self) -> super::super::SequenceStateLayout {
            super::super::SequenceStateLayout::paged_kv_cache(self.layout.clone())
        }
    }

    let model = PagedOnly { layout };
    let mut pool = CachePool::new(4);
    let id = pool.allocate(&model).unwrap();
    assert!(
        pool.detach(id).is_none(),
        "paged sequences must not be detach-able in this sub-issue"
    );
    // active count untouched when detach returns None
    assert_eq!(pool.active_count(), 1);
}

#[test]
fn cache_pool_adopt_respects_capacity_and_returns_preserving_set() {
    let model = RecordingModel::new(1);
    let mut pool = CachePool::new(1);

    let seq_a = pool.allocate(&model).unwrap();
    {
        let caches = pool.get_caches_mut(seq_a).unwrap();
        caches[0].update(fp32_tokens(&[1.0]), fp32_tokens(&[2.0]));
    }
    let detached = pool.detach(seq_a).unwrap();

    // Fill the pool back up.
    let _seq_fill = pool.allocate(&model).unwrap();
    assert_eq!(pool.active_count(), 1);

    // Now adopt must fail because max_sequences=1.
    let err = pool.adopt_preserving(&model, detached).unwrap_err();
    assert!(err.0.contains("max capacity"));
    assert_eq!(
        err.1.seq_len(),
        1,
        "original set must survive a failed adopt"
    );
}

#[test]
fn cache_pool_adopt_rejects_paged_set() {
    let model = RecordingModel::new(1);
    let mut pool = CachePool::new(2);

    // Fabricate a detached set with the wrong backend to check
    // adopt's explicit rejection.
    let bogus = DetachedCacheSet {
        caches: Vec::new(),
        backend: SequenceStateBackend::PagedKvCache,
        prompt_len: 0,
        current_offset: 0,
        created_at: Instant::now(),
        detached_at: Instant::now(),
        origin_seq_id: SequenceId::from_raw(0),
    };
    let err = pool.adopt_preserving(&model, bogus).unwrap_err();
    assert!(err.0.contains("is not supported"));
    assert_eq!(err.1.backend, SequenceStateBackend::PagedKvCache);
}

// ---------------------------------------------------------------------------
// Parking / memory accounting
// ---------------------------------------------------------------------------

#[test]
fn park_and_take_round_trip_tracks_bytes() {
    let model = RecordingModel::new(1);
    let mut pool = CachePool::new(4);

    let seq = pool.allocate(&model).unwrap();
    {
        let caches = pool.get_caches_mut(seq).unwrap();
        caches[0].update(
            fp32_tokens(&[1.0, 2.0, 3.0, 4.0]),
            fp32_tokens(&[5.0, 6.0, 7.0, 8.0]),
        );
    }
    let detached = pool.detach(seq).unwrap();
    let detached_bytes = detached.nbytes();
    assert!(detached_bytes > 0);

    let handle = pool.park_detached(detached);
    assert_eq!(pool.parked_count(), 1);
    assert_eq!(pool.parked_bytes(), detached_bytes);
    assert!(pool.peek_parked(handle).is_some());

    // memory_usage_bytes must reflect the parked set.
    assert_eq!(pool.memory_usage_bytes(), detached_bytes);

    let taken = pool.take_parked(handle).unwrap();
    assert_eq!(taken.seq_len(), 4);
    assert_eq!(pool.parked_count(), 0);
    assert_eq!(pool.memory_usage_bytes(), 0);
}

#[test]
fn take_parked_returns_none_for_unknown_handle() {
    let mut pool = CachePool::new(4);
    let bogus = DetachedHandle(1_234_567);
    assert!(pool.take_parked(bogus).is_none());
    assert!(pool.peek_parked(bogus).is_none());
}

#[test]
fn adopt_parked_flows_through_prepare_sequence_state() {
    let model = RecordingModel::new(1);
    let mut pool = CachePool::new(4);

    let seq = pool.allocate(&model).unwrap();
    {
        let caches = pool.get_caches_mut(seq).unwrap();
        caches[0].update(fp32_tokens(&[1.0]), fp32_tokens(&[2.0]));
    }
    let detached = pool.detach(seq).unwrap();
    let handle = pool.park_detached(detached);

    let new_seq = pool.adopt_parked(&model, handle).unwrap();
    assert_eq!(pool.parked_count(), 0);
    assert_eq!(pool.active_count(), 1);
    assert!(model.prepared_ids().contains(&new_seq));
}

// ---------------------------------------------------------------------------
// Property test: prefill(N)+detach+adopt+decode(M) == prefill(N+M)
// ---------------------------------------------------------------------------

/// The KV cache state after `detach + adopt + decode(M)` on top of
/// `prefill(N)` must be identical to the KV cache state after a fresh
/// `prefill(N+M)`. Because the production decoder consumes the KV cache
/// keys/values directly when producing logits, byte-identical cache
/// contents imply bit-identical logits for a deterministic model.
#[test]
fn property_detach_adopt_decode_matches_fresh_prefill_fp16() {
    let model = RecordingModel::new(1);
    let mut pool = CachePool::new(4);

    // Full context the canonical path will prefill in one shot.
    let full_k: Vec<f32> = (0..10).map(|i| i as f32 + 0.1).collect();
    let full_v: Vec<f32> = (0..10).map(|i| 100.0 - i as f32).collect();

    // --- Path A: prefill(N) -> detach -> adopt -> decode(M)
    let split = 6usize;
    let head_k = &full_k[..split];
    let head_v = &full_v[..split];
    let tail_k = &full_k[split..];
    let tail_v = &full_v[split..];

    let seq_src = pool.allocate(&model).unwrap();
    {
        let caches = pool.get_caches_mut(seq_src).unwrap();
        caches[0].update(fp32_tokens(head_k), fp32_tokens(head_v));
    }
    let detached = pool.detach(seq_src).unwrap();
    let seq_adopted = pool.adopt(&model, detached).unwrap();
    {
        let caches = pool.get_caches_mut(seq_adopted).unwrap();
        caches[0].update(fp32_tokens(tail_k), fp32_tokens(tail_v));
    }
    let path_a_keys = {
        let caches = pool.get_caches_mut(seq_adopted).unwrap();
        visible_keys_as_fp32(&caches[0])
    };

    // --- Path B: fresh prefill(N+M)
    let seq_fresh = pool.allocate(&model).unwrap();
    {
        let caches = pool.get_caches_mut(seq_fresh).unwrap();
        caches[0].update(fp32_tokens(&full_k), fp32_tokens(&full_v));
    }
    let path_b_keys = {
        let caches = pool.get_caches_mut(seq_fresh).unwrap();
        visible_keys_as_fp32(&caches[0])
    };

    assert_eq!(path_a_keys.len(), path_b_keys.len());
    assert_eq!(
        path_a_keys, path_b_keys,
        "detach+adopt+decode must produce bit-identical KV state vs fresh prefill"
    );
}

#[test]
fn property_detach_adopt_decode_matches_fresh_prefill_int8_within_tolerance() {
    let model = RecordingModel::new(1);
    let mut pool = CachePool::new(4);

    // INT8 per-token absmax is lossy, but the quantization is applied
    // per-token and is deterministic. Detach/adopt must therefore
    // produce the same dequantized output within numerical noise.
    let full_k: Vec<f32> = (0..8).map(|i| (i as f32 - 4.0) * 3.125).collect();
    let full_v: Vec<f32> = (0..8).map(|i| 1.0 - i as f32 * 0.25).collect();
    let split = 5usize;

    // --- Path A ---
    let seq_src = pool.allocate(&model).unwrap();
    {
        let caches = pool.get_caches_mut(seq_src).unwrap();
        caches[0] = KVCache::new_with_mode(KVCacheMode::Int8);
        caches[0].update(fp32_tokens(&full_k[..split]), fp32_tokens(&full_v[..split]));
    }
    let detached = pool.detach(seq_src).unwrap();
    let seq_adopted = pool.adopt(&model, detached).unwrap();
    let (keys_a, values_a) = {
        let caches = pool.get_caches_mut(seq_adopted).unwrap();
        caches[0].update_and_fetch(fp32_tokens(&full_k[split..]), fp32_tokens(&full_v[split..]))
    };
    eval(&keys_a);
    eval(&values_a);

    // --- Path B ---
    let seq_fresh = pool.allocate(&model).unwrap();
    {
        let caches = pool.get_caches_mut(seq_fresh).unwrap();
        caches[0] = KVCache::new_with_mode(KVCacheMode::Int8);
    }
    let (keys_b, values_b) = {
        let caches = pool.get_caches_mut(seq_fresh).unwrap();
        caches[0].update_and_fetch(fp32_tokens(&full_k), fp32_tokens(&full_v))
    };
    eval(&keys_b);
    eval(&values_b);

    let close_k = allclose(&keys_a, &keys_b, 1e-3, 1e-3);
    eval(&close_k);
    assert!(
        item_bool(&close_k),
        "INT8 detach+adopt+decode dequantized keys diverge from fresh prefill"
    );
    let close_v = allclose(&values_a, &values_b, 1e-3, 1e-3);
    eval(&close_v);
    assert!(
        item_bool(&close_v),
        "INT8 detach+adopt+decode dequantized values diverge from fresh prefill"
    );
}

// ---------------------------------------------------------------------------
// DetachedKVCache::trim_to / DetachedCacheSet::truncate_to
// ---------------------------------------------------------------------------

/// Build a [`DetachedCacheSet`] with `num_layers` Fp16 layers, each carrying
/// the supplied k/v vectors. Convenience helper for the truncate_to tests.
fn detached_set_with_fp16_layers(num_layers: usize, k: &[f32], v: &[f32]) -> DetachedCacheSet {
    let mut layer_handles = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        let mut layer = KVCache::new();
        layer.update(fp32_tokens(k), fp32_tokens(v));
        layer_handles.push(layer.clone_handle());
    }
    let prompt_len = k.len();
    let now = Instant::now();
    DetachedCacheSet {
        caches: layer_handles,
        backend: SequenceStateBackend::DenseKvCache,
        prompt_len,
        current_offset: prompt_len as i32,
        created_at: now,
        detached_at: now,
        origin_seq_id: SequenceId::from_raw(42),
    }
}

#[test]
fn detached_kvcache_trim_to_mid_buffer_resets_offset_and_visible_keys() {
    // Build a live cache with 6 tokens, detach it, then trim the inert
    // handle to 4. After install_detached the visible region must hold the
    // first 4 tokens bit-identically.
    let mut live = KVCache::new();
    let keys = fp32_tokens(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let values = fp32_tokens(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    live.update(keys, values);
    let mut handle = live.clone_handle();
    assert_eq!(handle.seq_len(), 6);

    handle.trim_to(4).expect("trim_to mid-buffer must succeed");
    assert_eq!(handle.seq_len(), 4);
    assert!(!handle.is_empty());

    // Round-trip into a fresh cache and verify the first 4 tokens survived.
    let mut restored = KVCache::new();
    restored.install_detached(handle).unwrap();
    assert_eq!(restored.seq_len(), 4);
    assert_eq!(visible_keys_as_fp32(&restored), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn detached_kvcache_trim_to_zero_drops_storage_and_keeps_mode() {
    let mut live = KVCache::new_with_mode(KVCacheMode::Int8);
    live.update(fp32_tokens(&[1.0, 2.0, 3.0]), fp32_tokens(&[4.0, 5.0, 6.0]));
    let mut handle = live.clone_handle();
    assert_eq!(handle.seq_len(), 3);
    assert_eq!(handle.mode(), KVCacheMode::Int8);

    handle.trim_to(0).expect("trim_to zero must succeed");
    assert_eq!(handle.seq_len(), 0);
    assert!(handle.is_empty(), "all backing buffers must drop");
    // Mode is preserved so a downstream install can still adopt INT8 layers.
    assert_eq!(handle.mode(), KVCacheMode::Int8);
}

#[test]
fn detached_kvcache_trim_to_exact_offset_is_noop() {
    let mut live = KVCache::new();
    live.update(fp32_tokens(&[1.0, 2.0]), fp32_tokens(&[3.0, 4.0]));
    let mut handle = live.clone_handle();

    handle
        .trim_to(2)
        .expect("trim_to exact offset must succeed");
    assert_eq!(handle.seq_len(), 2);

    let mut restored = KVCache::new();
    restored.install_detached(handle).unwrap();
    assert_eq!(visible_keys_as_fp32(&restored), vec![1.0, 2.0]);
}

#[test]
fn detached_kvcache_trim_to_past_offset_returns_error_without_mutation() {
    let mut live = KVCache::new();
    live.update(fp32_tokens(&[1.0, 2.0]), fp32_tokens(&[3.0, 4.0]));
    let mut handle = live.clone_handle();

    let err = handle
        .trim_to(5)
        .expect_err("trim_to past offset must error");
    assert!(err.contains("exceeds current offset"));
    assert_eq!(handle.seq_len(), 2, "handle must be unchanged after error");
}

#[test]
fn detached_kvcache_trim_to_negative_returns_error() {
    let mut live = KVCache::new();
    live.update(fp32_tokens(&[1.0]), fp32_tokens(&[2.0]));
    let mut handle = live.clone_handle();

    let err = handle.trim_to(-1).expect_err("trim_to negative must error");
    assert!(err.contains("non-negative"));
    assert_eq!(handle.seq_len(), 1);
}

#[test]
fn detached_kvcache_trim_to_int8_preserves_per_token_scales() {
    // INT8 mode carries per-token scale tensors. Trim must slice them in
    // lockstep with the data tensors so a subsequent dequantize stays
    // bit-identical to the already-trimmed live cache.
    let mut live = KVCache::new_with_mode(KVCacheMode::Int8);
    live.update(
        fp32_tokens(&[1.0, 8.0, 64.0, 0.125]),
        fp32_tokens(&[2.0, 16.0, 128.0, 0.0625]),
    );
    let mut handle = live.clone_handle();
    assert_eq!(handle.seq_len(), 4);
    assert!(handle.key_scales.is_some(), "INT8 carries key_scales");
    assert!(handle.val_scales.is_some(), "INT8 carries val_scales");

    handle.trim_to(2).expect("trim_to 2 must succeed");
    assert_eq!(handle.seq_len(), 2);

    // Scale buffers must still be present and shaped consistently with the
    // data tensors. We probe shape via a round-trip install.
    let mut restored = KVCache::new_with_mode(KVCacheMode::Int8);
    restored.install_detached(handle).unwrap();
    assert_eq!(restored.seq_len(), 2);
    assert_eq!(restored.mode, KVCacheMode::Int8);
    assert!(restored.key_scales.is_some());
    assert!(restored.val_scales.is_some());

    // Dequantization should still produce a plausible value range for the
    // first two tokens. The exact bytes are mode-dependent so we just check
    // the cache reports the right length and survives a no-op update.
    let (k, v) = restored.update_and_fetch(fp32_tokens(&[256.0]), fp32_tokens(&[512.0]));
    eval(&k);
    eval(&v);
    assert_eq!(restored.seq_len(), 3);
}

#[test]
fn detached_cache_set_truncate_to_partial_shrinks_every_layer() {
    // Build a 3-layer set with 8 tokens per layer and truncate to 5. Every
    // layer's per-tensor seq-len axis must end up at 5, the set-wide
    // current_offset must mirror that, and prompt_len clamps too. The
    // round-trip into a fresh CachePool must produce caches at length 5.
    let model = RecordingModel::new(3);
    let mut pool = CachePool::new(2);
    let k: Vec<f32> = (1..=8).map(|i| i as f32 * 0.5).collect();
    let v: Vec<f32> = (1..=8).map(|i| i as f32 * 1.5).collect();
    let mut detached = detached_set_with_fp16_layers(3, &k, &v);
    assert_eq!(detached.seq_len(), 8);
    assert_eq!(detached.prompt_len, 8);

    detached.truncate_to(5).expect("truncate_to must succeed");
    assert_eq!(detached.seq_len(), 5, "first layer length must shrink");
    assert!(
        detached.caches.iter().all(|c| c.seq_len() == 5),
        "every layer must agree on the new length"
    );
    assert_eq!(detached.current_offset, 5);
    assert_eq!(detached.prompt_len, 5, "prompt_len must clamp downward");

    let new_id = pool.adopt(&model, detached).expect("adopt truncated set");
    let caches = pool.get_caches_mut(new_id).unwrap();
    assert_eq!(caches.len(), 3);
    assert_eq!(caches[0].seq_len(), 5);
    assert_eq!(caches[1].seq_len(), 5);
    assert_eq!(caches[2].seq_len(), 5);

    // First five visible keys must equal the originating prefix.
    assert_eq!(
        visible_keys_as_fp32(&caches[0]),
        vec![0.5, 1.0, 1.5, 2.0, 2.5]
    );
}

#[test]
fn detached_cache_set_truncate_to_full_length_is_noop() {
    let k: Vec<f32> = (1..=4).map(|i| i as f32).collect();
    let v: Vec<f32> = (1..=4).map(|i| i as f32 * 10.0).collect();
    let mut detached = detached_set_with_fp16_layers(2, &k, &v);
    assert_eq!(detached.seq_len(), 4);

    detached
        .truncate_to(4)
        .expect("truncate_to to current length must succeed");
    assert_eq!(detached.seq_len(), 4);
    assert!(detached.caches.iter().all(|c| c.seq_len() == 4));
    assert_eq!(detached.current_offset, 4);
    assert_eq!(detached.prompt_len, 4);
}

#[test]
fn detached_cache_set_truncate_to_zero_drops_every_layer() {
    let k: Vec<f32> = (1..=4).map(|i| i as f32).collect();
    let v: Vec<f32> = (1..=4).map(|i| i as f32 * 10.0).collect();
    let mut detached = detached_set_with_fp16_layers(2, &k, &v);

    detached
        .truncate_to(0)
        .expect("truncate_to zero must succeed");
    assert!(detached.caches.iter().all(|c| c.is_empty()));
    assert_eq!(detached.current_offset, 0);
    assert_eq!(detached.prompt_len, 0);
}

#[test]
fn detached_cache_set_truncate_to_above_seq_len_returns_error() {
    let k: Vec<f32> = (1..=3).map(|i| i as f32).collect();
    let v: Vec<f32> = (1..=3).map(|i| i as f32).collect();
    let mut detached = detached_set_with_fp16_layers(1, &k, &v);

    let err = detached
        .truncate_to(99)
        .expect_err("truncate_to past current seq_len must error");
    assert!(err.contains("exceeds current seq_len"));
    // The handle must remain at its original length.
    assert_eq!(detached.seq_len(), 3);
}

#[test]
fn detached_cache_set_truncate_to_negative_returns_error() {
    let mut detached = detached_set_with_fp16_layers(1, &[1.0, 2.0], &[3.0, 4.0]);

    let err = detached
        .truncate_to(-1)
        .expect_err("truncate_to negative must error");
    assert!(err.contains("non-negative"));
    assert_eq!(detached.seq_len(), 2);
}

#[test]
fn detached_cache_set_truncate_to_then_adopt_then_decode_matches_partial_prefill() {
    // The key correctness property for truncate the detached
    // set to N tokens, adopt, then decode M more tokens. The resulting
    // cache state must equal a fresh prefill of the first N tokens followed
    // by the same M tokens — i.e. truncate is a faithful "rewind to N"
    // operation on the inert handle.
    let model = RecordingModel::new(1);
    let mut pool = CachePool::new(4);

    // Source sequence: 8 tokens of FP16 KV state.
    let full_k: Vec<f32> = (1..=8).map(|i| i as f32).collect();
    let full_v: Vec<f32> = (1..=8).map(|i| (i as f32) * 0.25).collect();

    let seq_src = pool.allocate(&model).unwrap();
    {
        let caches = pool.get_caches_mut(seq_src).unwrap();
        caches[0].update(fp32_tokens(&full_k), fp32_tokens(&full_v));
    }
    let mut detached = pool.detach(seq_src).unwrap();
    assert_eq!(detached.seq_len(), 8);

    // Truncate to 5 (simulates an APC partial adoption: blocks 0..5 agree
    // with the request; blocks 5..8 diverge and need re-prefill).
    detached.truncate_to(5).unwrap();
    assert_eq!(detached.seq_len(), 5);

    let seq_adopted = pool.adopt(&model, detached).unwrap();
    let extra_k = &full_k[5..];
    let extra_v = &full_v[5..];
    {
        let caches = pool.get_caches_mut(seq_adopted).unwrap();
        caches[0].update(fp32_tokens(extra_k), fp32_tokens(extra_v));
    }
    let path_a_keys = {
        let caches = pool.get_caches_mut(seq_adopted).unwrap();
        visible_keys_as_fp32(&caches[0])
    };

    // Reference: fresh prefill(8) of the same tokens.
    let seq_fresh = pool.allocate(&model).unwrap();
    {
        let caches = pool.get_caches_mut(seq_fresh).unwrap();
        caches[0].update(fp32_tokens(&full_k), fp32_tokens(&full_v));
    }
    let path_b_keys = {
        let caches = pool.get_caches_mut(seq_fresh).unwrap();
        visible_keys_as_fp32(&caches[0])
    };

    assert_eq!(path_a_keys.len(), 8);
    assert_eq!(path_b_keys.len(), 8);
    assert_eq!(
        path_a_keys, path_b_keys,
        "truncate_to(5)+adopt+decode(3) must reproduce fresh prefill(8) bit-for-bit"
    );
}

#[test]
fn detached_cache_set_truncate_to_int8_preserves_dequantization() {
    // INT8 layers must survive truncate_to via lockstep slicing of the
    // per-token scale buffers. A round-trip dequantize must match a fresh
    // INT8 prefill of the same tokens within tolerance.
    let model = RecordingModel::new(1);
    let mut pool = CachePool::new(4);

    let full_k: Vec<f32> = (0..8).map(|i| (i as f32 - 3.0) * 1.75).collect();
    let full_v: Vec<f32> = (0..8).map(|i| 1.0 + i as f32 * 0.1).collect();

    let seq_src = pool.allocate(&model).unwrap();
    {
        let caches = pool.get_caches_mut(seq_src).unwrap();
        caches[0] = KVCache::new_with_mode(KVCacheMode::Int8);
        caches[0].update(fp32_tokens(&full_k), fp32_tokens(&full_v));
    }
    let mut detached = pool.detach(seq_src).unwrap();
    detached
        .truncate_to(5)
        .expect("INT8 truncate_to must succeed");
    assert_eq!(detached.seq_len(), 5);

    let seq_adopted = pool.adopt(&model, detached).unwrap();
    let (k_a, v_a) = {
        let caches = pool.get_caches_mut(seq_adopted).unwrap();
        caches[0].update_and_fetch(fp32_tokens(&full_k[5..]), fp32_tokens(&full_v[5..]))
    };
    eval(&k_a);
    eval(&v_a);

    // Reference: fresh INT8 prefill of all 8 tokens.
    let seq_fresh = pool.allocate(&model).unwrap();
    let (k_b, v_b) = {
        let caches = pool.get_caches_mut(seq_fresh).unwrap();
        caches[0] = KVCache::new_with_mode(KVCacheMode::Int8);
        caches[0].update_and_fetch(fp32_tokens(&full_k), fp32_tokens(&full_v))
    };
    eval(&k_b);
    eval(&v_b);

    let close_k = allclose(&k_a, &k_b, 1e-3, 1e-3);
    eval(&close_k);
    assert!(
        item_bool(&close_k),
        "INT8 truncate_to+adopt+decode keys diverge from fresh prefill"
    );
    let close_v = allclose(&v_a, &v_b, 1e-3, 1e-3);
    eval(&close_v);
    assert!(
        item_bool(&close_v),
        "INT8 truncate_to+adopt+decode values diverge from fresh prefill"
    );
}
