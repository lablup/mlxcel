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

//! [`materialized_state_tail`] returns the slice it names. The retention
//! guards, which read process-wide MLX memory, are in
//! `tests/state_tail_retention.rs` so concurrent lib tests cannot skew them.

use super::{STATE_TAIL_EAGER_MIN_ROWS, materialized_state_tail};
use crate::ffi;

#[test]
fn state_tail_matches_the_slice() {
    for rows in [4, STATE_TAIL_EAGER_MIN_ROWS + 5] {
        let src = ffi::reshape(&ffi::arange_f32(0.0, (rows * 6) as f32, 1.0), &[1, rows, 6]);
        let tail = materialized_state_tail(&src, 3);
        let expected = ffi::slice(&src, &[0, rows - 3, 0], &[1, rows, 6]);
        let same = ffi::array_equal(&tail, &expected, false);
        crate::eval(&same);
        assert!(
            ffi::item_bool(&same),
            "tail differs from the slice at {rows} rows"
        );
    }
}

/// Runs a single-pass first fill of a `[1, 1, ROWS, WIDTH]` prompt through a
/// rotating cache with a 64-token window, evaluates only the attention-side
/// consumer of the returned K/V (as the layer does), drops every handle but
/// the cache, and returns the bytes still active (issue #3062).
fn rotating_first_fill_retained(full_scope: bool) -> u64 {
    use crate::cache::{FullFirstFillScope, RotatingKVCache};
    ffi::clear_memory_cache();
    ffi::synchronize_default();
    let before = crate::memory::active_memory();
    let mut cache = RotatingKVCache::new(64);
    let ones = ffi::ones(&[1, 1, ROWS, WIDTH], crate::dtype::FLOAT16);
    let keys = ffi::multiply(&ones, &ffi::full_f32(&[1], 2.0, crate::dtype::FLOAT16));
    let values = ffi::multiply(&ones, &ffi::full_f32(&[1], 3.0, crate::dtype::FLOAT16));
    drop(ones);
    let (k, v) = {
        let _scope = full_scope.then(FullFirstFillScope::enter);
        cache.update_and_fetch(keys, values)
    };
    let consumer = ffi::add(&ffi::sum_all(&k), &ffi::sum_all(&v));
    drop(k);
    drop(v);
    crate::eval(&consumer);
    drop(consumer);
    ffi::synchronize_default();
    ffi::clear_memory_cache();
    let retained = crate::memory::active_memory().saturating_sub(before);
    crate::eval(cache.keys.as_ref().expect("cache holds keys"));
    retained
}

#[test]
fn rotating_first_fill_releases_the_full_prompt_after_attention() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let retained = rotating_first_fill_retained(false);
    assert!(
        retained < SOURCE_BYTES / 8,
        "the first fill kept {retained} bytes alive; K and V are {} together",
        2 * SOURCE_BYTES
    );
}

/// The control: inside a `FullFirstFillScope` the cache stores the whole
/// prompt, so the guard above cannot pass vacuously.
#[test]
fn scoped_rotating_first_fill_keeps_the_full_prompt() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let retained = rotating_first_fill_retained(true);
    assert!(
        retained >= SOURCE_BYTES,
        "the scoped first fill kept only {retained} bytes; K and V are {} together",
        2 * SOURCE_BYTES
    );
}

#[test]
fn axis_tail_matches_the_slice_on_the_time_axis() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for rows in [5, STATE_TAIL_EAGER_MIN_ROWS + 5] {
        let src = ffi::reshape(
            &ffi::arange_f32(0.0, (2 * rows * 3) as f32, 1.0),
            &[1, 2, rows, 3],
        );
        let tail = super::materialized_axis_tail(&src, 2, 4);
        let expected = ffi::slice(&src, &[0, 0, rows - 4, 0], &[1, 2, rows, 3]);
        let same = ffi::array_equal(&tail, &expected, false);
        crate::eval(&same);
        assert!(
            ffi::item_bool(&same),
            "tail differs from the slice at {rows} rows"
        );
    }
}
