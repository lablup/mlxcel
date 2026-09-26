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

use super::*;

fn append(cache: &mut RotatingKVCache, count: i32) {
    let values: Vec<f32> = (0..count).map(|i| (cache.offset + i) as f32).collect();
    let _ = cache.update_and_fetch(
        ffi::from_slice_f32(&values, &[1, 1, count, 1]),
        ffi::from_slice_f32(&values, &[1, 1, count, 1]),
    );
}

#[test]
fn reference_ring_survives_prefill_chunking_rollback_compaction_and_snapshots() {
    // Same logical offset, three different physical layouts. The origin must
    // come from the cache cursor, not offset % window.
    for chunks in [vec![3], vec![10], vec![5, 5], vec![1; 10]] {
        let mut cache = RotatingKVCache::new(8);
        for chunk in chunks {
            append(&mut cache, chunk);
        }
        let initial = cache.offset;
        let expected = if ffi::array_shape(cache.keys.as_ref().unwrap())[2] > 8 {
            0
        } else {
            cache.buffer_write_idx() % 8
        };
        cache.enable_speculative_buffer(4).unwrap();
        assert_eq!(cache.speculative_ring_cursor(), Some(expected));
        for _ in 0..10 {
            append(&mut cache, 4);
            cache.trim(2);
            assert_eq!(
                cache.speculative_ring_cursor(),
                Some((expected + cache.offset - initial) % 8)
            );
        }
        let state = cache.snapshot_state();
        let mut restored = RotatingKVCache::new(8);
        restored
            .restore_fp16_snapshot_state(
                state,
                cache.keys.as_ref().map(|k| ffi::array_handle_clone(k)),
                cache.values.as_ref().map(|v| ffi::array_handle_clone(v)),
            )
            .unwrap();
        assert_eq!(
            restored.speculative_ring_cursor(),
            cache.speculative_ring_cursor()
        );
        restored.enable_speculative_buffer(8).unwrap();
        assert_eq!(
            restored.speculative_ring_cursor(),
            cache.speculative_ring_cursor()
        );
        let detached = restored.clone_handle();
        assert_eq!(restored.speculative_ring_cursor(), None);
        assert_eq!(restored.buffer_size, 0);
        restored.install_detached(detached).unwrap();
        assert_eq!(
            restored.speculative_ring_cursor(),
            cache.speculative_ring_cursor()
        );
        append(&mut restored, 1);
        assert_eq!(
            restored.speculative_ring_cursor(),
            Some((expected + restored.offset - initial) % 8)
        );
    }
}

#[test]
fn malformed_ring_snapshots_are_rejected_without_mutating_cache() {
    let mut source = RotatingKVCache::new(8);
    append(&mut source, 3);
    source.enable_speculative_buffer(4).unwrap();
    let good = source.snapshot_state();
    for corruption in 0..6 {
        let mut state = good;
        let mut k_shape = vec![1, 1, 8, 1];
        let mut v_shape = k_shape.clone();
        let mut v_dtype = dtype::FLOAT32;
        match corruption {
            0 => state.speculative_ring_origin = Some((state.offset + 1, 0)),
            1 => state.start_position = i32::MAX,
            2 => state.speculative_ring_origin = None,
            3 => {
                k_shape.pop();
            }
            4 => v_shape[1] = 2,
            _ => v_dtype = dtype::FLOAT16,
        }
        let mut target = RotatingKVCache::new(8);
        assert!(
            target
                .restore_fp16_snapshot_state(
                    state,
                    Some(ffi::zeros(&k_shape, dtype::FLOAT32)),
                    Some(ffi::zeros(&v_shape, v_dtype))
                )
                .is_err()
        );
        assert_eq!(target.offset, 0);
        assert!(target.keys.is_none());
    }
    source.trim(2);
    let state = source.snapshot_state();
    assert_eq!(state.speculative_ring_origin, Some((1, 1)));
    let mut restored = RotatingKVCache::new(8);
    restored
        .restore_fp16_snapshot_state(state, source.keys.take(), source.values.take())
        .unwrap();
    assert_eq!(restored.speculative_ring_cursor(), Some(1));
}
