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

use cxx::UniquePtr;

use super::*;
use crate::cache::{
    CachePool, PagedKvLayout, SequenceStateBackend, SequenceStateLayout, turbo::BLOCK_SIZE,
};
use crate::generate::LanguageModel;

const WINDOW: i32 = 8;

fn kinds() -> Vec<LayerStateKind> {
    vec![
        LayerStateKind::sliding(WINDOW),
        LayerStateKind::Full,
        LayerStateKind::Recurrent,
        LayerStateKind::None,
        LayerStateKind::Chunked { chunk: 4 },
    ]
}

fn kv(rows: i32, base: f32) -> UniquePtr<MlxArray> {
    let data: Vec<f32> = (0..rows * 4).map(|i| base + i as f32).collect();
    crate::from_slice_f32(&data, &[1, 1, rows, 4])
}

#[test]
fn stored_mode_follows_the_layer_kind() {
    let mode = KVCacheMode::Int8;
    assert_eq!(LayerStateKind::Full.stored_mode(mode), Some(mode));
    assert_eq!(
        LayerStateKind::sliding(WINDOW).stored_mode(mode),
        Some(RotatingKVCache::stored_mode(WINDOW, mode))
    );
    assert_eq!(
        LayerStateKind::Chunked { chunk: 4 }.stored_mode(mode),
        Some(KVCacheMode::Fp16)
    );
    assert_eq!(LayerStateKind::Recurrent.stored_mode(mode), None);
    assert_eq!(LayerStateKind::None.stored_mode(mode), None);
}

#[test]
fn a_misaligned_turbo4asym_window_builds_fp16() {
    let misaligned = LayerStateKind::sliding(BLOCK_SIZE as i32 + 1);
    assert_eq!(
        misaligned.build_mode(KVCacheMode::Turbo4Asym),
        KVCacheMode::Fp16
    );
    let aligned = LayerStateKind::sliding(BLOCK_SIZE as i32 * 2);
    assert_eq!(
        aligned.build_mode(KVCacheMode::Turbo4Asym),
        KVCacheMode::Turbo4Asym
    );
    // Every other mode passes through, as the rotating cache stores FP16 for
    // it anyway.
    assert_eq!(misaligned.build_mode(KVCacheMode::Int8), KVCacheMode::Int8);
    assert_eq!(
        LayerStateKind::Full.build_mode(KVCacheMode::Turbo4Asym),
        KVCacheMode::Turbo4Asym
    );
}

#[test]
fn for_kinds_builds_the_storage_of_each_kind() {
    let modes = [KVCacheMode::Fp16, KVCacheMode::Int8];
    let state = LayerState::for_kinds(&kinds(), Some(&modes), 0);
    let names: Vec<&str> = state.iter().map(LayerState::storage_name).collect();
    assert_eq!(
        names,
        vec!["sliding", "full", "recurrent", "empty", "chunked"]
    );
    for (layer, kind) in state.iter().zip(kinds()) {
        assert!(layer.matches_kind(kind), "{kind:?}");
    }
    assert_eq!(state[1].as_kv().map(|c| c.mode), Some(KVCacheMode::Int8));
    assert_eq!(state[0].as_rotating().map(|c| c.max_size), Some(WINDOW));
    // A kind from another table does not match the storage.
    assert!(!state[0].matches_kind(LayerStateKind::sliding(WINDOW + 1)));
    assert!(!state[1].matches_kind(LayerStateKind::Recurrent));
}

#[test]
fn recurrent_state_keeps_slot_order_and_moves_tensors() {
    let mut state = RecurrentState::with_slots(&["conv_state", "state_cache"]);
    assert_eq!(
        state.names().collect::<Vec<_>>(),
        vec!["conv_state", "state_cache"]
    );
    assert!(state.get("conv_state").is_none());
    state.put("state_cache", Some(kv(2, 0.0)));
    state.put("conv_state", Some(kv(1, 9.0)));
    state.put("extra", Some(kv(1, 3.0)));
    assert_eq!(
        state.names().collect::<Vec<_>>(),
        vec!["conv_state", "state_cache", "extra"],
        "an existing slot keeps its position and a new one is appended"
    );
    assert_eq!(state.slots().filter(|(_, t)| t.is_some()).count(), 3);
    assert!(state.nbytes() > 0);
    let taken = state.take("state_cache").expect("stored tensor");
    assert_eq!(crate::array_shape(&taken), vec![1, 1, 2, 4]);
    assert!(state.get("state_cache").is_none());
    state.offset = 7;
    state.clear();
    assert_eq!(state.offset, 0);
    assert_eq!(state.nbytes(), 0);
    assert_eq!(state.names().count(), 3, "clear keeps the declared slots");
}

#[test]
fn rewinds_cover_attention_layers_and_refuse_the_rest() {
    let mut full = LayerState::new(LayerStateKind::Full, KVCacheMode::Fp16);
    let _ = full
        .as_kv_mut()
        .expect("full layer")
        .update_and_fetch(kv(6, 0.0), kv(6, 100.0));
    assert_eq!(full.offset(), 6);
    assert_eq!(full.visible_len(), 6);
    full.rewind_padded_prefill(2).expect("pad trim");
    assert_eq!(full.offset(), 4);
    full.rewind_decode_appends(1).expect("decode rewind");
    assert_eq!(full.offset(), 3);
    assert!(full.rewind_decode_appends(9).is_err());

    let mut empty = LayerState::Empty;
    assert_eq!(empty.rewind_padded_prefill(3), Ok(()));
    assert_eq!(empty.rewind_decode_appends(3), Ok(()));
    assert_eq!(empty.visible_len(), 0);

    let mut recurrent = LayerState::new(LayerStateKind::Recurrent, KVCacheMode::Fp16);
    let err = recurrent.rewind_padded_prefill(1).expect_err("recurrent");
    assert!(err.contains("recurrent layer cannot drop"), "{err}");
    let err = recurrent.rewind_decode_appends(1).expect_err("recurrent");
    assert!(err.contains("recurrent layer cannot rewind"), "{err}");
    assert_eq!(recurrent.visible_len(), 0);
}

#[test]
#[should_panic(expected = "attention over a recurrent layer")]
fn attention_over_a_layer_without_kv_is_a_table_mismatch() {
    let mut recurrent = LayerState::new(LayerStateKind::Recurrent, KVCacheMode::Fp16);
    let q = kv(1, 0.0);
    let _ = KvAttention::attend(&mut recurrent, &q, kv(1, 0.0), kv(1, 0.0), 1.0, None);
}

/// A scheduler-owned stub: its forward appends to every attention layer.
struct SchedulerOwnedStub {
    snapshot: bool,
}

impl LanguageModel for SchedulerOwnedStub {
    fn forward(
        &self,
        _input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        crate::zeros(&[1, 1, 4], crate::dtype::FLOAT32)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        Vec::new()
    }

    fn num_layers(&self) -> usize {
        2
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![0]
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn layer_state_kinds(&self) -> Option<Vec<LayerStateKind>> {
        Some(vec![LayerStateKind::sliding(WINDOW), LayerStateKind::Full])
    }

    fn layer_state_snapshot_spec(&self) -> Option<LayerStateSnapshotSpec> {
        self.snapshot.then_some(LayerStateSnapshotSpec {
            family: "stub",
            label: "Stub",
            encoding: LayerStateEncoding::Windowed {
                standard: "standard",
                rotating: "rotating",
            },
        })
    }

    fn supports_decode_lookahead_rewind(&self) -> bool {
        true
    }
}

#[test]
fn a_family_with_a_kind_table_is_scheduler_owned_and_gets_pool_state() {
    let model = SchedulerOwnedStub { snapshot: true };
    assert_eq!(
        model.sequence_state_layout(),
        SequenceStateLayout::scheduler_owned(2)
    );
    assert!(
        SequenceStateBackend::SchedulerOwned.keeps_state_outside_kv_slice()
            && SequenceStateBackend::ModelOwned.keeps_state_outside_kv_slice()
            && !SequenceStateBackend::DenseKvCache.keeps_state_outside_kv_slice()
            && !SequenceStateBackend::PagedKvCache.keeps_state_outside_kv_slice()
    );

    let mut pool = CachePool::new(4);
    let a = pool.allocate(&model).expect("allocate");
    let b = pool.allocate(&model).expect("allocate");
    let set = pool.get(a).expect("active");
    assert_eq!(set.backend, SequenceStateBackend::SchedulerOwned);
    assert!(set.caches.is_empty(), "the KVCache slice is a placeholder");
    let layers = set.layer_state().expect("layer state");
    assert_eq!(layers.len(), 2);
    assert_eq!(layers[0].storage_name(), "sliding");
    assert_eq!(layers[1].storage_name(), "full");

    // Two sequences borrow disjoint state in one batch; an alias is refused.
    {
        let mut rows = pool.batch_layer_state_mut(&[a, b]).expect("batch");
        assert_eq!(rows.len(), 2);
        let _ = rows[0][1]
            .as_kv_mut()
            .expect("full layer")
            .update_and_fetch(kv(3, 0.0), kv(3, 1.0));
    }
    assert_eq!(pool.layer_state(a).expect("a")[1].offset(), 3);
    assert_eq!(pool.layer_state(b).expect("b")[1].offset(), 0);
    let err = pool
        .batch_layer_state_mut(&[a, a])
        .err()
        .expect("duplicate ids");
    assert!(err.contains("duplicate sequence"), "{err}");

    // A restore installs a whole vector, and a wrong layer count is refused
    // without touching the state.
    let set = pool.get_mut(a).expect("active");
    let err = set
        .install_layer_state(Vec::new())
        .expect_err("layer count");
    assert!(err.contains("cannot replace"), "{err}");
    assert_eq!(set.layer_state().expect("kept")[1].offset(), 3);
    set.install_layer_state(model.make_layer_state().expect("fresh"))
        .expect("install");
    assert_eq!(set.layer_state().expect("fresh")[1].offset(), 0);

    pool.release(a);
    assert!(pool.layer_state(a).is_none());
}

#[test]
fn the_paged_override_keeps_layer_state_beside_the_shadow_table() {
    // Under the paged decode override every family is allocated as
    // `PagedKvCache` for block accounting. A scheduler-owned family's real
    // state is still its layer state, mirrored into the shadow by length.
    let model = SchedulerOwnedStub { snapshot: false };
    let mut pool = CachePool::new(4);
    let layout = PagedKvLayout::uniform(2, 4, 4).expect("layout");
    let id = pool
        .allocate_with_layout(&model, Some(SequenceStateLayout::paged_kv_cache(layout)))
        .expect("allocate");
    let set = pool.get(id).expect("active");
    assert_eq!(set.backend, SequenceStateBackend::PagedKvCache);
    assert!(set.layer_state().is_some());

    let layers = pool.layer_state_mut(id).expect("layer state");
    let _ = layers[1]
        .as_kv_mut()
        .expect("full layer")
        .update_and_fetch(kv(5, 0.0), kv(5, 1.0));
    pool.sync_paged_state_with_layer_state(id).expect("sync");
    let paged = pool.get_paged_state(id).expect("paged shadow");
    assert_eq!(paged.layers[0].len, 0);
    assert_eq!(paged.layers[1].len, 5);
}

#[test]
fn a_scheduler_owned_layout_needs_a_model_with_a_kind_table() {
    struct Dense;
    impl LanguageModel for Dense {
        fn forward(
            &self,
            _input_ids: &MlxArray,
            _caches: &mut [KVCache],
            _mask: Option<&MlxArray>,
        ) -> UniquePtr<MlxArray> {
            crate::zeros(&[1, 1, 4], crate::dtype::FLOAT32)
        }
        fn make_caches(&self) -> Vec<KVCache> {
            vec![KVCache::new()]
        }
        fn num_layers(&self) -> usize {
            1
        }
        fn eos_token_ids(&self) -> Vec<i32> {
            vec![0]
        }
        fn supports_batching(&self) -> bool {
            true
        }
    }
    let mut pool = CachePool::new(2);
    let err = pool
        .allocate_with_layout(&Dense, Some(SequenceStateLayout::scheduler_owned(1)))
        .expect_err("no kind table");
    assert!(err.contains("layer_state_kinds"), "{err}");
    // The default layout of the same model is untouched.
    let id = pool.allocate(&Dense).expect("dense");
    assert!(pool.layer_state(id).is_none());
    assert_eq!(
        pool.get(id).expect("active").backend,
        SequenceStateBackend::DenseKvCache
    );
}
