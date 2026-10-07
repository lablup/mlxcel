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

//! Verify-block RoPE for the Qwen 3.5 full-attention layers (issue #2191),
//! and the test-only capture the attribution diagnostics read.

use mlxcel_core::{MlxArray, UniquePtr};

/// Rotate a `[1, H, L, D]` block one row at a time, each row at
/// `offset + row`, so every row goes through the `L = 1` RoPE call classic
/// decode makes for that token.
pub(crate) fn fast_rope_rows_like_decode(
    x: &MlxArray,
    dims: i32,
    base: f32,
    offset: i32,
) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let (b, h, l, d) = (shape[0], shape[1], shape[2], shape[3]);
    let rows: Vec<UniquePtr<MlxArray>> = (0..l)
        .map(|row| {
            let one = mlxcel_core::slice(x, &[0, 0, row, 0], &[b, h, row + 1, d]);
            mlxcel_core::fast_rope(&one, dims, false, base, 1.0, offset + row)
        })
        .collect();
    let rows: Vec<&MlxArray> = rows
        .iter()
        .map(|row| row.as_ref().expect("fast_rope result is non-null"))
        .collect();
    mlxcel_core::concatenate_many(&rows, 2)
}

#[cfg(test)]
pub(crate) mod capture {
    use mlxcel_core::{MlxArray, UniquePtr};
    use std::cell::RefCell;

    thread_local! {
        static SLOTS: RefCell<Option<Vec<(&'static str, UniquePtr<MlxArray>)>>> =
            const { RefCell::new(None) };
    }

    /// Start recording sub-op outputs of every full-attention forward on
    /// this thread.
    pub(crate) fn begin() {
        SLOTS.with(|s| *s.borrow_mut() = Some(Vec::new()));
    }

    /// Stop recording and return what was captured, in call order.
    pub(crate) fn take() -> Vec<(&'static str, UniquePtr<MlxArray>)> {
        SLOTS.with(|s| s.borrow_mut().take().unwrap_or_default())
    }

    pub(crate) fn record(tag: &'static str, x: &MlxArray) {
        SLOTS.with(|s| {
            if let Some(v) = s.borrow_mut().as_mut() {
                let c = mlxcel_core::copy(x);
                mlxcel_core::eval(&c);
                v.push((tag, c));
            }
        });
    }
}
