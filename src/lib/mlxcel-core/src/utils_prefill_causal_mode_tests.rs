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

//! Tests for the prefill causal-mode choice (#3051): each branch of
//! `prefill_uses_causal_mode`, and that MLX's causal mode masks exactly the
//! keys `create_causal_mask` allows, at offset 0 and behind a cache.

use super::{
    create_causal_mask, parse_prefill_causal_mode, prefill_causal_mask, prefill_uses_causal_mode,
};
use crate::ffi;
use cxx::UniquePtr;

fn draw(shape: &[i32], seed: u32) -> UniquePtr<ffi::MlxArray> {
    let len: i32 = shape.iter().product();
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(12_345);
    let values: Vec<f32> = (0..len)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        })
        .collect();
    ffi::from_slice_f32(&values, shape)
}

/// Causal mode against the array mask for one `[1, heads, q_len, 64]` query
/// over `offset + q_len` keys, the shape a prefill piece behind a cache sees.
fn assert_causal_mode_matches_array_mask(q_len: i32, offset: i32) {
    let (heads, kv_heads, dim) = (4, 2, 64);
    let k_len = offset + q_len;
    let q = draw(&[1, heads, q_len, dim], 1 + offset as u32);
    let k = draw(&[1, kv_heads, k_len, dim], 2 + offset as u32);
    let v = draw(&[1, kv_heads, k_len, dim], 3 + offset as u32);
    let scale = 1.0 / (dim as f32).sqrt();
    let mask = create_causal_mask(q_len, offset);
    let reference = crate::layers::attention(&q, &k, &v, scale, Some(&mask), 0.0, 0);
    let causal = crate::causal_attention(&q, &k, &v, scale, 0.0, 0);
    assert_eq!(ffi::array_shape(&causal), ffi::array_shape(&reference));
    let close = ffi::allclose(&causal, &reference, 1e-4, 1e-5);
    ffi::eval(&close);
    assert!(
        ffi::item_bool(&close),
        "causal mode differs from the array mask at q_len={q_len}, offset={offset}"
    );
}

#[test]
fn a_cold_single_row_prefill_uses_causal_mode_and_matches_the_array_mask() {
    assert!(prefill_uses_causal_mode(37, 1, false));
    assert!(prefill_causal_mask(37, 0, 1).is_none());
    assert_causal_mode_matches_array_mask(37, 0);
}

#[test]
fn a_prefill_piece_behind_a_cache_uses_causal_mode_and_matches_the_array_mask() {
    // The offset is not an input to the choice: MLX aligns causal mode
    // bottom-right, which is `create_causal_mask`'s offset when the layer
    // attends `offset + q_len` keys.
    assert!(prefill_uses_causal_mode(16, 1, false));
    assert!(prefill_causal_mask(16, 300, 1).is_none());
    assert_causal_mode_matches_array_mask(16, 5);
    assert_causal_mode_matches_array_mask(16, 300);
}

#[test]
fn a_padded_prefill_keeps_its_caller_mask() {
    assert!(!prefill_uses_causal_mode(16, 1, true));
}

#[test]
fn a_batched_prefill_keeps_the_array_mask() {
    assert!(!prefill_uses_causal_mode(16, 2, false));
    let mask = prefill_causal_mask(16, 4, 2).expect("a batched prefill keeps the array mask");
    assert_eq!(ffi::array_shape(&mask), vec![16, 20]);
}

#[test]
fn a_decode_row_needs_no_mask() {
    assert!(!prefill_uses_causal_mode(1, 1, false));
    assert!(prefill_causal_mask(1, 300, 1).is_none());
    assert!(prefill_causal_mask(1, 300, 4).is_none());
}

#[test]
fn the_kill_switch_parses_off_values_only() {
    for off in ["0", "false", "OFF", " off "] {
        assert!(!parse_prefill_causal_mode(Some(off)), "{off:?}");
    }
    for on in [None, Some("1"), Some("on"), Some("")] {
        assert!(parse_prefill_causal_mode(on), "{on:?}");
    }
}
