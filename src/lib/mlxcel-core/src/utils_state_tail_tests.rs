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

#[test]
fn axis_tail_matches_the_slice_on_the_time_axis() {
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
