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

//! Tests for the #3072 prefill mask helpers: each branch of
//! `prefill_sliding_window_mask` and `softcap_prefill_causal_mask`, the value
//! head dim `PrefillAttention` carries into MLX's verdict, and that
//! `maskless_prefill_attention` masks exactly the keys the arrays it replaces
//! allow, inside and past the window.

use super::{
    PrefillAttention, SlidingPrefillKeys, create_causal_mask, create_sliding_window_prefill_mask,
    create_sliding_window_prefill_mask_dense, prefill_causal_mask, prefill_causal_mode_enabled,
    prefill_sliding_window_mask, softcap_prefill_causal_mask,
};
use crate::cache::DecodeLookaheadAppendScope;
use crate::ffi;
use crate::window_sdpa::{
    WindowSdpaExtras, WindowSdpaShape, plan, rotating_prefill_key_len, sdpa_mode_is_fused,
};
use cxx::UniquePtr;

const ATTN: PrefillAttention = PrefillAttention {
    q_heads: 4,
    kv_heads: 2,
    head_dim: 64,
    value_dim: 64,
    sinks: false,
};
const F32: i32 = crate::dtype::FLOAT32;

fn shape(q_len: i32, k_len: i32) -> WindowSdpaShape {
    WindowSdpaShape::prefill(1, 4, 2, q_len, k_len, 64, F32)
}

fn sliding(seq_len: i32, prior: i32, window: i32, batch: i32, keys: SlidingPrefillKeys) -> bool {
    prefill_sliding_window_mask(seq_len, prior, window, batch, ATTN, F32, keys).is_none()
}

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

/// `[1, 4, q_len, 64]` queries over `[1, 2, k_len, 64]` keys and values.
fn qkv(q_len: i32, k_len: i32) -> [UniquePtr<ffi::MlxArray>; 3] {
    let seed = (q_len * 31 + k_len) as u32;
    [
        draw(&[1, 4, q_len, 64], seed),
        draw(&[1, 2, k_len, 64], seed + 1),
        draw(&[1, 2, k_len, 64], seed + 2),
    ]
}

fn assert_close(a: &ffi::MlxArray, b: &ffi::MlxArray, what: &str) {
    assert_eq!(ffi::array_shape(a), ffi::array_shape(b), "{what}: shape");
    let close = ffi::allclose(a, b, 1e-4, 1e-5);
    ffi::eval(&close);
    assert!(ffi::item_bool(&close), "{what}: values differ");
}

#[test]
fn a_decode_row_needs_no_sliding_mask() {
    assert!(sliding(1, 300, 64, 1, SlidingPrefillKeys::Dense));
    assert!(sliding(1, 300, 64, 4, SlidingPrefillKeys::Rotating));
}

#[test]
fn a_prefill_inside_the_window_takes_causal_mode_where_mlx_fuses_it() {
    let fused = sdpa_mode_is_fused(&shape(16, 40), false, true);
    let expected = fused && prefill_causal_mode_enabled();
    assert_eq!(sliding(16, 24, 64, 1, SlidingPrefillKeys::Dense), expected);
    assert_eq!(
        sliding(16, 24, 64, 1, SlidingPrefillKeys::Rotating),
        expected
    );
    // A non-positive window is no window: plain causal at any length.
    let fused_long = sdpa_mode_is_fused(&shape(16, 316), false, true);
    assert_eq!(
        sliding(16, 300, 0, 1, SlidingPrefillKeys::Dense),
        fused_long && prefill_causal_mode_enabled()
    );
}

#[test]
fn a_dense_prefill_past_the_window_follows_the_native_window_plan() {
    let native = plan(&shape(48, 88), 32, WindowSdpaExtras::default()).is_native();
    let mask = prefill_sliding_window_mask(48, 40, 32, 1, ATTN, F32, SlidingPrefillKeys::Dense);
    assert_eq!(mask.is_none(), native && prefill_causal_mode_enabled());
    if let Some(mask) = mask {
        let dense = create_sliding_window_prefill_mask_dense(48, 40, 32);
        assert_eq!(ffi::array_shape(&mask), vec![48, 88]);
        let same = ffi::array_equal(&mask, &dense, false);
        ffi::eval(&same);
        assert!(ffi::item_bool(&same), "the kept mask is the dense band");
    }
}

#[test]
fn a_rotating_prefill_past_the_window_sizes_its_plan_to_the_ring() {
    // 300 prior tokens behind a 64-wide ring: the cache returns 63 of them.
    let k_len = rotating_prefill_key_len(16, 300, 64);
    assert_eq!(k_len, 79);
    let native = plan(&shape(16, k_len), 64, WindowSdpaExtras::default()).is_native();
    let mask = prefill_sliding_window_mask(16, 300, 64, 1, ATTN, F32, SlidingPrefillKeys::Rotating);
    assert_eq!(mask.is_none(), native && prefill_causal_mode_enabled());
    if let Some(mask) = mask {
        let ring = create_sliding_window_prefill_mask(16, 300, 64);
        assert_eq!(ffi::array_shape(&mask), ffi::array_shape(&ring));
    }
}

#[test]
fn batched_and_speculative_sliding_prefills_keep_the_array() {
    let mask = prefill_sliding_window_mask(48, 40, 32, 2, ATTN, F32, SlidingPrefillKeys::Dense)
        .expect("a batched prefill keeps the band");
    assert_eq!(ffi::array_shape(&mask), vec![48, 88]);
    let _scope = DecodeLookaheadAppendScope::enter();
    let mask = prefill_sliding_window_mask(4, 100, 32, 1, ATTN, F32, SlidingPrefillKeys::Dense)
        .expect("a verify block keeps the band");
    assert_eq!(ffi::array_shape(&mask), vec![4, 104]);
}

#[test]
fn maskless_attention_inside_the_window_matches_the_causal_array() {
    let [q, k, v] = qkv(16, 40);
    let mask = create_causal_mask(16, 24);
    let reference = crate::layers::attention(&q, &k, &v, 0.125, Some(&mask), 0.0, 0);
    let maskless = crate::maskless_prefill_attention(&q, &k, &v, 0.125, 64);
    assert_close(&maskless, &reference, "inside the window");
}

#[test]
fn maskless_attention_past_the_window_matches_the_dense_band() {
    let [q, k, v] = qkv(48, 88);
    let band = create_sliding_window_prefill_mask_dense(48, 40, 32);
    let reference = crate::layers::attention(&q, &k, &v, 0.125, Some(&band), 0.0, 32);
    let maskless = crate::maskless_prefill_attention(&q, &k, &v, 0.125, 32);
    assert_close(&maskless, &reference, "past the window");
}

#[test]
fn the_value_head_dim_reaches_mlx_verdict() {
    // An MLA-style layer: 192-wide queries and keys over 128-wide values.
    let attention = PrefillAttention::new(4, 4, 192).with_value_dim(128);
    assert_eq!((attention.head_dim, attention.value_dim), (192, 128));
    let wide = WindowSdpaShape {
        value_dim: 128,
        ..WindowSdpaShape::prefill(1, 4, 4, 64, 64, 192, F32)
    };
    let fused = sdpa_mode_is_fused(&wide, false, true);
    let mask = prefill_causal_mask(64, 0, 1, attention, F32);
    assert_eq!(mask.is_none(), fused && prefill_causal_mode_enabled());
}

#[test]
fn a_softcapped_prefill_shares_the_mask_causal_attention_would_build() {
    assert!(softcap_prefill_causal_mask(1, 10, 50.0).is_none());
    assert!(softcap_prefill_causal_mask(16, 10, 0.0).is_none());
    let mask = softcap_prefill_causal_mask(16, 24, 50.0).expect("softcapped prefill mask");
    assert_eq!(ffi::array_shape(&mask), vec![16, 40]);
    // The shared mask runs the same masked softcap SDPA as the per-layer
    // one `causal_attention` builds, so the two agree bit for bit.
    let [q, k, v] = qkv(16, 40);
    let shared = crate::layers::attention(&q, &k, &v, 0.125, Some(&mask), 50.0, 0);
    let per_layer = crate::causal_attention(&q, &k, &v, 0.125, 50.0, 0);
    let same = ffi::array_equal(&shared, &per_layer, false);
    ffi::eval(&same);
    assert!(ffi::item_bool(&same), "shared and per-layer masks diverged");
}
