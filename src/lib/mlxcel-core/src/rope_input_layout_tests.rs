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

//! `fast_rope` / `fast_rope_with_freqs` over inputs that are not dense
//! buffers.
//!
//! MLX's CUDA `RoPE::eval_gpu` rotated a donatable input in place and
//! otherwise gave the output the input's strides and data size. A broadcast
//! input (`tile`, `broadcast_to`) has a zero stride on the broadcast axis, so
//! its output kept one element for every index along that axis, where each
//! index needs its own rotation. Eight rows broadcast along the sequence axis
//! and rotated at positions 16..23 all came back as the rotation at position
//! 16 on a GB10. The DeepSeek V3.2 YaRN cache test builds its reference rows
//! that way (one cached row, tiled), which is how it was found: its prefill
//! assertion failed on CUDA with a cache that was correct. The overlay
//! `overlays/cuda/mlx/backend/cuda/rope.cu` copies a broadcast input into a
//! dense output before rotating it.
//!
//! Every layout here must rotate to what a dense buffer of the same values
//! rotates to, with the input held by the caller and with no other owner (the
//! case the kernel may rotate in place). The dense rotation is itself checked
//! against one computed on the host in f64, so the two cannot be wrong
//! together. The strided view is the layout attention hands the kernel; it
//! was right before the overlay and has to stay on the paths that do not copy.
//!
//!   cargo test -p mlxcel-core --lib rope_input_layout_tests

use super::*;
use crate::utils::{array_to_vec_f32, slice_axis};

/// Five heads: the kernels rotate four heads per thread, so this leaves a
/// partial group.
const HEADS: i32 = 5;
const ROWS: i32 = 8;
const WIDTH: i32 = 8;
/// The first position. Not zero, where every rotation is the identity.
const OFFSET: i32 = 16;
const BASE: f32 = 10000.0;
/// Periods for `fast_rope_with_freqs`, nothing like `BASE^(2i/dims)`, so a
/// path that fell back to the base could not match by accident.
const PERIODS: [f32; 4] = [1.0, 7.5, 300.0, 4000.0];

/// `count` distinct values in `[-1.9, 1.9]`.
fn values(count: i32) -> Vec<f32> {
    (0..count)
        .map(|i| ((i * 7 + 3) % 23) as f32 * 0.17 - 1.87)
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct Rotation {
    /// Rotated features: all of `WIDTH`, or the first half of it.
    dims: i32,
    traditional: bool,
    /// A frequency table rather than the base.
    table: bool,
}

impl Rotation {
    fn all() -> Vec<Self> {
        let mut all = Vec::new();
        for dims in [WIDTH, WIDTH / 2] {
            for traditional in [true, false] {
                for table in [false, true] {
                    all.push(Self {
                        dims,
                        traditional,
                        table,
                    });
                }
            }
        }
        all
    }

    fn apply(self, x: &MlxArray) -> UniquePtr<MlxArray> {
        if self.table {
            let pairs = self.dims / 2;
            let periods = from_slice_f32(&PERIODS[..pairs as usize], &[pairs]);
            fast_rope_with_freqs(x, self.dims, self.traditional, 1.0, OFFSET, &periods)
        } else {
            fast_rope(x, self.dims, self.traditional, BASE, 1.0, OFFSET)
        }
    }

    /// The rotation of `input` (`[.., ROWS, WIDTH]`, row-major) in f64: row
    /// `t` of each sequence at position `OFFSET + t`.
    fn host(self, input: &[f32]) -> Vec<f32> {
        let pairs = (self.dims / 2) as usize;
        let mut out = input.to_vec();
        for (row, features) in out.chunks_exact_mut(WIDTH as usize).enumerate() {
            let position = f64::from(OFFSET + row as i32 % ROWS);
            for (pair, period) in PERIODS.iter().enumerate().take(pairs) {
                let inverse = if self.table {
                    1.0 / f64::from(*period)
                } else {
                    f64::from(BASE).powf(-(pair as f64) / pairs as f64)
                };
                let (sin, cos) = (position * inverse).sin_cos();
                let (a, b) = if self.traditional {
                    (2 * pair, 2 * pair + 1)
                } else {
                    (pair, pair + pairs)
                };
                let (x1, x2) = (f64::from(features[a]), f64::from(features[b]));
                features[a] = (x1 * cos - x2 * sin) as f32;
                features[b] = (x1 * sin + x2 * cos) as f32;
            }
        }
        out
    }
}

/// The row that differs most between `got` and `want`, and by how much. A NaN
/// counts as an infinite difference.
fn worst_row(got: &[f32], want: &[f32]) -> (usize, f32) {
    assert_eq!(got.len(), want.len(), "element count");
    let width = WIDTH as usize;
    got.chunks_exact(width)
        .zip(want.chunks_exact(width))
        .map(|(got, want)| {
            got.iter()
                .zip(want)
                .map(|(g, w)| {
                    let diff = (g - w).abs();
                    if diff.is_nan() { f32::INFINITY } else { diff }
                })
                .fold(0.0, f32::max)
        })
        .enumerate()
        .fold(
            (0, 0.0),
            |worst, row| if row.1 > worst.1 { row } else { worst },
        )
}

/// Rotate what `make` builds, under every [`Rotation`], and require the
/// rotation of a dense buffer holding the same values.
fn assert_rotates_like_dense(layout: &str, make: impl Fn() -> UniquePtr<MlxArray>) {
    let input = make();
    let shape = array_shape(&input);
    assert_eq!(
        shape[shape.len() - 2..],
        [ROWS, WIDTH],
        "{layout}: unexpected shape {shape:?}"
    );
    // The values by index, read through a contiguous copy rather than through
    // the rotation under test.
    let logical = array_to_vec_f32(&input);
    let dense = from_slice_f32(&logical, &shape);

    for rotation in Rotation::all() {
        let want = array_to_vec_f32(&rotation.apply(&dense));
        let (row, diff) = worst_row(&want, &rotation.host(&logical));
        assert!(
            diff < 1e-3,
            "dense input, {rotation:?}: row {row} is {diff} from the host rotation"
        );

        let held = array_to_vec_f32(&rotation.apply(&input));
        let (row, diff) = worst_row(&held, &want);
        assert!(
            diff < 1e-5,
            "{layout}, {rotation:?}: row {row} is {diff} from the dense rotation"
        );

        let unowned = {
            let input = make();
            rotation.apply(&input)
        };
        let (row, diff) = worst_row(&array_to_vec_f32(&unowned), &want);
        assert!(
            diff < 1e-5,
            "{layout} with no other owner, {rotation:?}: row {row} is {diff} from the dense rotation"
        );
    }
}

/// One row per head, repeated along the sequence axis: what a reference built
/// from a single cached row looks like.
#[test]
fn rows_broadcast_along_the_sequence_axis_rotate_at_their_own_positions() {
    let rows = values(HEADS * WIDTH);
    assert_rotates_like_dense("tile along the sequence axis", || {
        let rows = from_slice_f32(&rows, &[1, HEADS, 1, WIDTH]);
        tile(&rows, &[1, 1, ROWS, 1])
    });
    assert_rotates_like_dense("broadcast_to along the sequence axis", || {
        let rows = from_slice_f32(&rows, &[1, HEADS, 1, WIDTH]);
        broadcast_to(&rows, &[1, HEADS, ROWS, WIDTH])
    });
    // No head axis: `[1, 1, WIDTH]` to `[1, ROWS, WIDTH]`.
    assert_rotates_like_dense("3-D broadcast_to along the sequence axis", || {
        let row = from_slice_f32(&rows[..WIDTH as usize], &[1, 1, WIDTH]);
        broadcast_to(&row, &[1, ROWS, WIDTH])
    });
}

/// One sequence shared by every head, and one batch row shared by every
/// batch row.
#[test]
fn inputs_broadcast_along_the_head_and_batch_axes_rotate_like_dense() {
    let sequence = values(ROWS * WIDTH);
    assert_rotates_like_dense("broadcast_to along the head axis", || {
        let sequence = from_slice_f32(&sequence, &[1, 1, ROWS, WIDTH]);
        broadcast_to(&sequence, &[1, HEADS, ROWS, WIDTH])
    });
    let heads = values(HEADS * ROWS * WIDTH);
    assert_rotates_like_dense("broadcast_to along the batch axis", || {
        let heads = from_slice_f32(&heads, &[1, HEADS, ROWS, WIDTH]);
        broadcast_to(&heads, &[3, HEADS, ROWS, WIDTH])
    });
}

/// `[1, ROWS, HEADS, WIDTH + 4]` sliced to its last `WIDTH` features and
/// transposed to `[1, HEADS, ROWS, WIDTH]`: strided on every axis and
/// broadcast on none.
#[test]
fn strided_views_rotate_like_dense() {
    let wide = values(ROWS * HEADS * (WIDTH + 4));
    assert_rotates_like_dense("transposed slice", || {
        let wide = from_slice_f32(&wide, &[1, ROWS, HEADS, WIDTH + 4]);
        let view = slice_axis(&wide, -1, 4, WIDTH + 4);
        transpose_axes(&view, &[0, 2, 1, 3])
    });
}
