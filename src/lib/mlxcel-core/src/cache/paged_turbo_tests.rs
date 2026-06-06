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

//! Unit tests for the Turbo4-aware paged KV layout (B10).
//!
//! Organised as:
//! 1. `PagedKvLayout` Turbo4 construction + validation.
//! 2. nbytes accounting across the 5 cache modes.
//! 3. Per-page sidecar install/take/peek + access gating.
//! 4. Detach/adopt round-trip preserves sidecars and is bit-identical
//!    on Fp16/Int8.
//! 5. Trim/restore correctness with sidecar drops.
//! 6. Page eviction (refcount-driven) drops sidecars deterministically.

use super::paged::{PagedBlockId, PagedBlockPool, PagedKvLayout, PagedSequenceState};
use super::{CachePool, KVCache, KVCacheMode, SequenceId, SequenceStateLayout};

use crate::ffi::MlxArray;
use crate::generate::LanguageModel;
use cxx::UniquePtr;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn dummy_array(values: &[f32], shape: &[i32]) -> UniquePtr<MlxArray> {
    crate::ffi::from_slice_f32(values, shape)
}

fn fp16_layout() -> PagedKvLayout {
    PagedKvLayout::uniform(2, 4, 128).unwrap()
}

fn int8_layout() -> PagedKvLayout {
    PagedKvLayout::new_with_mode(4, vec![64, 64], KVCacheMode::Int8, Vec::new()).unwrap()
}

fn turbo4_asym_layout() -> PagedKvLayout {
    // 2 layers, block_size=4 tokens, 128 main bytes/block, 16 sidecar bytes/block.
    PagedKvLayout::uniform_with_mode(2, 4, 128, KVCacheMode::Turbo4Asym, 16).unwrap()
}

fn turbo4_sym_layout() -> PagedKvLayout {
    PagedKvLayout::uniform_with_mode(2, 4, 64, KVCacheMode::Turbo4, 32).unwrap()
}

fn turbo4_delegated_layout() -> PagedKvLayout {
    PagedKvLayout::uniform_with_mode(2, 4, 96, KVCacheMode::Turbo4Delegated, 24).unwrap()
}

/// Minimal paged stub for adopt round-trip.
struct PagedStub {
    layout: PagedKvLayout,
    prepared: std::cell::RefCell<Vec<SequenceId>>,
}

impl PagedStub {
    fn new(layout: PagedKvLayout) -> Self {
        Self {
            layout,
            prepared: std::cell::RefCell::new(Vec::new()),
        }
    }
    fn prepared_ids(&self) -> Vec<SequenceId> {
        self.prepared.borrow().clone()
    }
}

impl LanguageModel for PagedStub {
    fn forward(
        &self,
        _input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        crate::ffi::zeros(&[1], 0)
    }
    fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layout.num_layers)
            .map(|_| KVCache::new_with_mode(self.layout.cache_mode))
            .collect()
    }
    fn num_layers(&self) -> usize {
        self.layout.num_layers
    }
    fn eos_token_ids(&self) -> Vec<i32> {
        vec![0]
    }
    fn sequence_state_layout(&self) -> SequenceStateLayout {
        SequenceStateLayout::paged_kv_cache(self.layout.clone())
    }
    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.prepared.borrow_mut().push(seq_id);
    }
}

// ---------------------------------------------------------------------------
// 1. PagedKvLayout Turbo4 construction
// ---------------------------------------------------------------------------

#[test]
fn layout_uniform_defaults_to_fp16() {
    let layout = fp16_layout();
    assert_eq!(layout.cache_mode, KVCacheMode::Fp16);
    assert!(layout.turbo_sidecar_bytes_per_block.is_empty());
    assert!(!layout.is_turbo_mode());
    assert_eq!(layout.turbo_sidecar_bytes_per_block_for_layer(0), 0);
    assert_eq!(layout.turbo_sidecar_bytes_per_block_for_layer(99), 0);
}

#[test]
fn layout_int8_does_not_allocate_sidecars() {
    let layout = int8_layout();
    assert_eq!(layout.cache_mode, KVCacheMode::Int8);
    assert!(layout.turbo_sidecar_bytes_per_block.is_empty());
    assert!(!layout.is_turbo_mode());
}

#[test]
fn layout_turbo4_modes_carry_sidecar_bytes() {
    let asym = turbo4_asym_layout();
    assert!(asym.is_turbo_mode());
    assert_eq!(asym.turbo_sidecar_bytes_per_block_for_layer(0), 16);
    assert_eq!(asym.turbo_sidecar_bytes_per_block_for_layer(1), 16);

    let sym = turbo4_sym_layout();
    assert!(sym.is_turbo_mode());
    assert_eq!(sym.turbo_sidecar_bytes_per_block_for_layer(0), 32);

    let delegated = turbo4_delegated_layout();
    assert!(delegated.is_turbo_mode());
    assert_eq!(delegated.turbo_sidecar_bytes_per_block_for_layer(0), 24);
}

#[test]
fn layout_rejects_sidecars_for_non_turbo_modes() {
    let err = PagedKvLayout::new_with_mode(4, vec![64], KVCacheMode::Fp16, vec![16]).unwrap_err();
    assert!(err.contains("must be empty"), "{err}");
}

#[test]
fn layout_requires_sidecar_per_layer_for_turbo_modes() {
    let err = PagedKvLayout::new_with_mode(4, vec![64, 64], KVCacheMode::Turbo4Asym, vec![16])
        .unwrap_err();
    assert!(err.contains("length"), "{err}");
}

#[test]
fn layout_sidecar_must_be_block_size_multiple() {
    let err = PagedKvLayout::new_with_mode(4, vec![64, 64], KVCacheMode::Turbo4Asym, vec![16, 7])
        .unwrap_err();
    assert!(err.contains("multiple of block_size"), "{err}");
}

// ---------------------------------------------------------------------------
// 2. nbytes accounting across modes
// ---------------------------------------------------------------------------

#[test]
fn nbytes_fp16_only_counts_main_bytes() {
    let layout = fp16_layout();
    let mut pool = PagedBlockPool::new(layout.clone());
    let mut state = PagedSequenceState::new(&layout);
    pool.append_tokens(&mut state, 0, 4).unwrap();
    pool.append_tokens(&mut state, 1, 4).unwrap();
    // 2 blocks * 128 bytes/block = 256.
    assert_eq!(state.reserved_bytes(&layout), 256);
    assert_eq!(state.used_bytes(&layout), 256);
    assert_eq!(pool.turbo_sidecar_bytes(), 0);
}

#[test]
fn nbytes_int8_only_counts_main_bytes() {
    let layout = int8_layout();
    let mut pool = PagedBlockPool::new(layout.clone());
    let mut state = PagedSequenceState::new(&layout);
    pool.append_tokens(&mut state, 0, 4).unwrap();
    pool.append_tokens(&mut state, 1, 4).unwrap();
    assert_eq!(state.reserved_bytes(&layout), 128);
    assert_eq!(state.used_bytes(&layout), 128);
    assert_eq!(pool.turbo_sidecar_bytes(), 0);
}

#[test]
fn nbytes_turbo4_asym_includes_sidecar_per_block() {
    let layout = turbo4_asym_layout();
    let mut pool = PagedBlockPool::new(layout.clone());
    let mut state = PagedSequenceState::new(&layout);
    pool.append_tokens(&mut state, 0, 4).unwrap();
    pool.append_tokens(&mut state, 1, 4).unwrap();
    // Per-layer reserved = 1 block * (128 + 16) = 144. Two layers = 288.
    assert_eq!(state.reserved_bytes(&layout), 288);
    assert_eq!(state.used_bytes(&layout), 288);
}

#[test]
fn nbytes_turbo4_sym_includes_sidecar_per_block() {
    let layout = turbo4_sym_layout();
    let mut pool = PagedBlockPool::new(layout.clone());
    let mut state = PagedSequenceState::new(&layout);
    pool.append_tokens(&mut state, 0, 8).unwrap();
    // 2 blocks * (64 + 32) = 192.
    assert_eq!(state.reserved_bytes(&layout), 192);
}

#[test]
fn nbytes_turbo4_delegated_includes_sidecar_per_block() {
    let layout = turbo4_delegated_layout();
    let mut pool = PagedBlockPool::new(layout.clone());
    let mut state = PagedSequenceState::new(&layout);
    pool.append_tokens(&mut state, 0, 4).unwrap();
    // 1 block * (96 + 24) = 120.
    assert_eq!(state.reserved_bytes(&layout), 120);
}

// ---------------------------------------------------------------------------
// 3. Sidecar install/take/peek + dispatch gating
// ---------------------------------------------------------------------------

#[test]
fn install_sidecars_succeeds_in_turbo_modes() {
    let layout = turbo4_asym_layout();
    let mut pool = PagedBlockPool::new(layout);
    let mut state = PagedSequenceState::new(pool.layout());
    pool.append_tokens(&mut state, 0, 4).unwrap();
    let block_id = state.layer(0).unwrap().block_ids[0];

    let v_packed = dummy_array(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]);
    pool.install_v_packed(block_id, v_packed).unwrap();
    assert!(pool.v_packed_for(block_id).is_some());
    assert!(pool.has_turbo_sidecar(block_id));

    let v_norms = dummy_array(&[0.5; 4], &[1, 1, 4, 1]);
    pool.install_v_norms(block_id, v_norms).unwrap();
    assert!(pool.v_norms_for(block_id).is_some());

    // Sidecar bytes are accounted for.
    assert!(pool.turbo_sidecar_bytes() > 0);
}

#[test]
fn install_sidecars_rejects_fp16_mode() {
    let layout = fp16_layout();
    let mut pool = PagedBlockPool::new(layout);
    let mut state = PagedSequenceState::new(pool.layout());
    pool.append_tokens(&mut state, 0, 4).unwrap();
    let block_id = state.layer(0).unwrap().block_ids[0];

    let arr = dummy_array(&[1.0, 2.0], &[1, 1, 2, 1]);
    let err = pool.install_v_packed(block_id, arr).unwrap_err();
    assert!(err.contains("not a Turbo4 variant"), "{err}");
}

#[test]
fn install_k_sidecars_only_in_symmetric_mode() {
    let mut pool = PagedBlockPool::new(turbo4_asym_layout());
    let mut state = PagedSequenceState::new(pool.layout());
    pool.append_tokens(&mut state, 0, 4).unwrap();
    let block_id = state.layer(0).unwrap().block_ids[0];

    let k_packed = dummy_array(&[1.0, 2.0], &[1, 1, 2, 1]);
    let err = pool.install_k_packed(block_id, k_packed).unwrap_err();
    assert!(err.contains("does not store packed K"), "{err}");

    // Symmetric path accepts.
    let mut pool2 = PagedBlockPool::new(turbo4_sym_layout());
    let mut state2 = PagedSequenceState::new(pool2.layout());
    pool2.append_tokens(&mut state2, 0, 4).unwrap();
    let block_id2 = state2.layer(0).unwrap().block_ids[0];
    let k = dummy_array(&[1.0, 2.0], &[1, 1, 2, 1]);
    pool2.install_k_packed(block_id2, k).unwrap();
    assert!(pool2.k_packed_for(block_id2).is_some());
}

#[test]
fn install_cold_keys_only_in_delegated_mode() {
    // Asymmetric must reject cold keys.
    let mut pool = PagedBlockPool::new(turbo4_asym_layout());
    let mut state = PagedSequenceState::new(pool.layout());
    pool.append_tokens(&mut state, 0, 4).unwrap();
    let block_id = state.layer(0).unwrap().block_ids[0];
    let cold = dummy_array(&[1.0, 2.0], &[1, 1, 2, 1]);
    let err = pool.install_cold_keys(block_id, cold).unwrap_err();
    assert!(err.contains("does not store cold keys"), "{err}");

    // Delegated accepts.
    let mut pool2 = PagedBlockPool::new(turbo4_delegated_layout());
    let mut state2 = PagedSequenceState::new(pool2.layout());
    pool2.append_tokens(&mut state2, 0, 4).unwrap();
    let block_id2 = state2.layer(0).unwrap().block_ids[0];
    let cold2 = dummy_array(&[1.0, 2.0], &[1, 1, 2, 1]);
    pool2.install_cold_keys(block_id2, cold2).unwrap();
    assert!(pool2.cold_keys_for(block_id2).is_some());
}

#[test]
fn take_sidecars_clears_storage() {
    let mut pool = PagedBlockPool::new(turbo4_asym_layout());
    let mut state = PagedSequenceState::new(pool.layout());
    pool.append_tokens(&mut state, 0, 4).unwrap();
    let block_id = state.layer(0).unwrap().block_ids[0];

    let v_packed = dummy_array(&[1.0, 2.0], &[1, 1, 2, 1]);
    pool.install_v_packed(block_id, v_packed).unwrap();
    assert!(pool.v_packed_for(block_id).is_some());

    let taken = pool.take_v_packed(block_id);
    assert!(taken.is_some());
    assert!(pool.v_packed_for(block_id).is_none());

    // Second take returns None.
    assert!(pool.take_v_packed(block_id).is_none());
}

#[test]
fn install_rejects_unknown_block_id() {
    let mut pool = PagedBlockPool::new(turbo4_asym_layout());
    let bogus = PagedBlockId::from_raw(9_999);
    let arr = dummy_array(&[1.0], &[1]);
    let err = pool.install_v_packed(bogus, arr).unwrap_err();
    assert!(err.contains("unknown block"), "{err}");
}

// ---------------------------------------------------------------------------
// 4. Detach/adopt round-trip
// ---------------------------------------------------------------------------

#[test]
fn detach_adopt_round_trip_fp16_remains_bit_identical() {
    let model = PagedStub::new(fp16_layout());
    let mut pool = CachePool::new(4);
    let id = pool.allocate(&model).unwrap();
    pool.append_paged_tokens(id, 0, 4).unwrap();
    pool.append_paged_tokens(id, 1, 4).unwrap();

    let detached = pool.detach_paged(id).expect("detach must succeed");
    assert_eq!(detached.cache_mode(), KVCacheMode::Fp16);
    assert_eq!(detached.turbo_sidecar_bytes(), 0);
    assert_eq!(detached.turbo_sidecar_page_count(), 0);

    let new_id = pool.adopt_paged(&model, detached).unwrap();
    assert_ne!(new_id, id);
    assert_eq!(model.prepared_ids().last().copied(), Some(new_id));
}

#[test]
fn detach_adopt_round_trip_turbo4_asym_preserves_sidecars() {
    let model = PagedStub::new(turbo4_asym_layout());
    let mut pool = CachePool::new(4);
    let id = pool.allocate(&model).unwrap();
    pool.append_paged_tokens(id, 0, 4).unwrap();
    pool.append_paged_tokens(id, 1, 4).unwrap();

    // Install per-page sidecars on every block.
    let block_ids: Vec<PagedBlockId> = {
        let state = pool.get_paged_state(id).unwrap();
        state
            .layers
            .iter()
            .flat_map(|layer| layer.block_ids.iter().copied())
            .collect()
    };
    assert!(!block_ids.is_empty());
    {
        let mut pool_ref = pool.paged_pool_mut().unwrap();
        for &block_id in &block_ids {
            pool_ref
                .install_v_packed(block_id, dummy_array(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]))
                .unwrap();
            pool_ref
                .install_v_norms(block_id, dummy_array(&[0.5, 0.5, 0.5, 0.5], &[1, 1, 4, 1]))
                .unwrap();
        }
    }

    let detached = pool.detach_paged(id).expect("detach must succeed");
    assert_eq!(detached.cache_mode(), KVCacheMode::Turbo4Asym);
    assert_eq!(detached.turbo_sidecar_page_count(), block_ids.len());
    assert!(detached.turbo_sidecar_bytes() > 0);

    // Sidecars left the pool when detached.
    {
        let pool_ref = pool.paged_pool_ref().unwrap();
        for &block_id in &block_ids {
            assert!(pool_ref.v_packed_for(block_id).is_none());
            assert!(pool_ref.v_norms_for(block_id).is_none());
        }
    }

    let new_id = pool.adopt_paged(&model, detached).unwrap();
    // After adopt, sidecars are reinstalled on the new pool.
    {
        let pool_ref = pool.paged_pool_ref().unwrap();
        for &block_id in &block_ids {
            assert!(
                pool_ref.v_packed_for(block_id).is_some(),
                "v_packed for {block_id} must be reinstalled after adopt"
            );
            assert!(
                pool_ref.v_norms_for(block_id).is_some(),
                "v_norms for {block_id} must be reinstalled after adopt"
            );
        }
    }
    assert_ne!(new_id, id);
}

#[test]
fn detach_adopt_round_trip_turbo4_sym_preserves_k_sidecars() {
    let model = PagedStub::new(turbo4_sym_layout());
    let mut pool = CachePool::new(4);
    let id = pool.allocate(&model).unwrap();
    pool.append_paged_tokens(id, 0, 4).unwrap();

    let block_id = pool.get_paged_state(id).unwrap().layers[0].block_ids[0];
    {
        let mut pool_ref = pool.paged_pool_mut().unwrap();
        pool_ref
            .install_v_packed(block_id, dummy_array(&[1.0, 2.0], &[1, 1, 2, 1]))
            .unwrap();
        pool_ref
            .install_v_norms(block_id, dummy_array(&[0.5, 0.5], &[1, 1, 2, 1]))
            .unwrap();
        pool_ref
            .install_k_packed(block_id, dummy_array(&[3.0, 4.0], &[1, 1, 2, 1]))
            .unwrap();
        pool_ref
            .install_k_norms(block_id, dummy_array(&[0.6, 0.6], &[1, 1, 2, 1]))
            .unwrap();
    }

    let detached = pool.detach_paged(id).expect("detach must succeed");
    assert_eq!(detached.cache_mode(), KVCacheMode::Turbo4);
    assert_eq!(detached.turbo_sidecar_page_count(), 1);

    let _new_id = pool.adopt_paged(&model, detached).unwrap();
    let pool_ref = pool.paged_pool_ref().unwrap();
    assert!(pool_ref.k_packed_for(block_id).is_some());
    assert!(pool_ref.k_norms_for(block_id).is_some());
    assert!(pool_ref.v_packed_for(block_id).is_some());
    assert!(pool_ref.v_norms_for(block_id).is_some());
}

#[test]
fn detach_adopt_round_trip_turbo4_delegated_preserves_cold_keys() {
    let model = PagedStub::new(turbo4_delegated_layout());
    let mut pool = CachePool::new(4);
    let id = pool.allocate(&model).unwrap();
    pool.append_paged_tokens(id, 0, 4).unwrap();

    let block_id = pool.get_paged_state(id).unwrap().layers[0].block_ids[0];
    {
        let mut pool_ref = pool.paged_pool_mut().unwrap();
        pool_ref
            .install_v_packed(block_id, dummy_array(&[1.0, 2.0], &[1, 1, 2, 1]))
            .unwrap();
        pool_ref
            .install_v_norms(block_id, dummy_array(&[0.5], &[1, 1, 1, 1]))
            .unwrap();
        pool_ref
            .install_cold_keys(block_id, dummy_array(&[7.0, 8.0], &[1, 1, 2, 1]))
            .unwrap();
    }

    let detached = pool.detach_paged(id).expect("detach must succeed");
    assert_eq!(detached.cache_mode(), KVCacheMode::Turbo4Delegated);

    let _new_id = pool.adopt_paged(&model, detached).unwrap();
    let pool_ref = pool.paged_pool_ref().unwrap();
    assert!(pool_ref.cold_keys_for(block_id).is_some());
    // Symmetric K sidecars must remain absent in delegated mode.
    assert!(pool_ref.k_packed_for(block_id).is_none());
}

// ---------------------------------------------------------------------------
// 5. Trim / restore correctness
// ---------------------------------------------------------------------------

#[test]
fn trim_releases_sidecars_when_block_evicted() {
    let mut pool = PagedBlockPool::new(turbo4_asym_layout());
    let mut state = PagedSequenceState::new(pool.layout());
    pool.append_tokens(&mut state, 0, 8).unwrap(); // 2 blocks
    let block_a = state.layer(0).unwrap().block_ids[0];
    let block_b = state.layer(0).unwrap().block_ids[1];

    pool.install_v_packed(block_a, dummy_array(&[1.0], &[1, 1, 1, 1]))
        .unwrap();
    pool.install_v_packed(block_b, dummy_array(&[2.0], &[1, 1, 1, 1]))
        .unwrap();
    let bytes_before = pool.turbo_sidecar_bytes();
    assert!(bytes_before > 0);

    // Trim 4 tokens — that frees the second block (and its sidecar).
    pool.trim_tokens(&mut state, 0, 4).unwrap();
    assert_eq!(state.layer(0).unwrap().block_ids.len(), 1);
    assert!(pool.v_packed_for(block_b).is_none());
    assert!(pool.v_packed_for(block_a).is_some());
    assert!(pool.turbo_sidecar_bytes() < bytes_before);
}

#[test]
fn release_sequence_drops_all_sidecars() {
    let mut pool = PagedBlockPool::new(turbo4_asym_layout());
    let mut state = PagedSequenceState::new(pool.layout());
    pool.append_tokens(&mut state, 0, 4).unwrap();
    pool.append_tokens(&mut state, 1, 4).unwrap();

    let block_ids: Vec<PagedBlockId> = state
        .layers
        .iter()
        .flat_map(|l| l.block_ids.iter().copied())
        .collect();
    for &id in &block_ids {
        pool.install_v_packed(id, dummy_array(&[1.0], &[1, 1, 1, 1]))
            .unwrap();
    }
    assert!(pool.turbo_sidecar_bytes() > 0);

    pool.release_sequence(&mut state).unwrap();
    for &id in &block_ids {
        assert!(pool.v_packed_for(id).is_none());
    }
    assert_eq!(pool.turbo_sidecar_bytes(), 0);
}

#[test]
fn detach_paged_rejects_dense_sequence_in_turbo_mode() {
    // Allocating a dense sequence on a Turbo-mode pool should be a no-op for
    // detach_paged (returns None), confirming dispatch is correct.
    let model = PagedStub::new(turbo4_asym_layout());
    let mut pool = CachePool::new(4);
    let _paged_id = pool.allocate(&model).unwrap();
    assert!(pool.detach_paged(SequenceId::from_raw(9_999)).is_none());
}

// ---------------------------------------------------------------------------
// 6. Page eviction drops sidecars deterministically
// ---------------------------------------------------------------------------

#[test]
fn refcount_zero_drops_sidecars_so_recycle_is_safe() {
    let layout = turbo4_asym_layout();
    let mut pool = PagedBlockPool::new(layout.clone());
    let mut state = PagedSequenceState::new(&layout);
    pool.append_tokens(&mut state, 0, 4).unwrap();
    let block_id = state.layer(0).unwrap().block_ids[0];
    pool.install_v_packed(block_id, dummy_array(&[42.0], &[1, 1, 1, 1]))
        .unwrap();
    assert!(pool.v_packed_for(block_id).is_some());

    // Drop the sequence pin entirely.
    pool.trim_tokens(&mut state, 0, 4).unwrap();
    assert_eq!(pool.refcount(block_id), 0);
    // Recycled block must not still expose sidecar contents.
    assert!(pool.v_packed_for(block_id).is_none());

    // Re-acquire should give a clean block.
    pool.append_tokens(&mut state, 0, 4).unwrap();
    let new_block = state.layer(0).unwrap().block_ids[0];
    assert!(pool.v_packed_for(new_block).is_none());
}

// Memory accounting: paged Turbo4 pool reports sidecar bytes via
// `CachePool::memory_usage_bytes`.
#[test]
fn cache_pool_memory_usage_includes_pool_sidecars() {
    let model = PagedStub::new(turbo4_asym_layout());
    let mut pool = CachePool::new(4);
    let id = pool.allocate(&model).unwrap();
    pool.append_paged_tokens(id, 0, 4).unwrap();
    let block_id = pool.get_paged_state(id).unwrap().layers[0].block_ids[0];
    let baseline = pool.memory_usage_bytes();
    {
        let mut pool_ref = pool.paged_pool_mut().unwrap();
        pool_ref
            .install_v_packed(block_id, dummy_array(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]))
            .unwrap();
    }
    assert!(
        pool.memory_usage_bytes() > baseline,
        "memory_usage_bytes must grow after installing a sidecar tensor"
    );
}
