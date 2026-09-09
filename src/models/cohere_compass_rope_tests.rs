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

//! Gates for the Compass MRoPE table against upstream's
//! `CohereCompassRotaryEmbedding`, recomputed here from its definition rather
//! than from this implementation.

use super::CompassMRoPE;
use crate::models::embedding_test_support::{max_abs_diff, mlx_test_guard, to_vec};

const HEAD_DIM: usize = 128;
const BASE: f32 = 50000.0;
const SECTION: [i32; 3] = [24, 20, 20];

/// Upstream's `inv_freq_3d` and per-channel axis, written out independently:
///
/// ```text
/// inv_freq[j]      = base ** (-2j / head_dim)
/// hw               = section[0] + section[1]
/// inv_freq_3d[:hw] = cat(inv_freq[:hw][0::2], inv_freq[:hw][1::2])
/// inv_freq_3d[hw:] = inv_freq[hw:]
/// axis(j)          = H for j < s_h, W for s_h <= j < hw, T otherwise
/// ```
fn reference_table(head_dim: usize, base: f32, section: [i32; 3]) -> (Vec<f32>, Vec<i32>) {
    let half = head_dim / 2;
    let natural: Vec<f32> = (0..half)
        .map(|j| base.powf(-2.0 * j as f32 / head_dim as f32))
        .collect();
    let (s_h, s_w) = (section[0] as usize, section[1] as usize);
    let hw = s_h + s_w;

    let mut inv_freq: Vec<f32> = Vec::with_capacity(half);
    let mut j = 0;
    while j < hw {
        inv_freq.push(natural[j]);
        j += 2;
    }
    let mut j = 1;
    while j < hw {
        inv_freq.push(natural[j]);
        j += 2;
    }
    inv_freq.extend_from_slice(&natural[hw..]);

    let axis = (0..half)
        .map(|j| {
            if j < s_h {
                1
            } else if j < hw {
                2
            } else {
                0
            }
        })
        .collect();
    (inv_freq, axis)
}

#[test]
fn frequencies_are_pre_permuted_over_the_hw_block() {
    let table = CompassMRoPE::new(HEAD_DIM, BASE, SECTION).expect("published section is valid");
    let (want_inv, _) = reference_table(HEAD_DIM, BASE, SECTION);
    assert_eq!(table.inv_freq().len(), 64);
    assert!(
        max_abs_diff(table.inv_freq(), &want_inv) < 1e-9,
        "inv_freq is not upstream's pre-rotated inv_freq_3d"
    );

    // The pre-rotation is a real permutation, not the identity: channel 1 must
    // carry the natural frequency at index 2, not index 1.
    let natural_1 = BASE.powf(-2.0 / HEAD_DIM as f32);
    assert!(
        (table.inv_freq()[1] - natural_1).abs() > 1e-9,
        "the table is in natural order, so the pre-rotation was skipped"
    );
}

/// Sections are contiguous and ordered `[H, W, T]`, so with `[24, 20, 20]`
/// channels 0..23 are H, 24..43 are W and 44..63 are T. This is what separates
/// Compass from Qwen3-VL's step-3 interleave, which over the same list would
/// give T channels `{0, 3, ..., 57, 60..63}`.
#[test]
fn sections_are_contiguous_h_then_w_then_t() {
    let table = CompassMRoPE::new(HEAD_DIM, BASE, SECTION).expect("published section is valid");
    let (_, want_axis) = reference_table(HEAD_DIM, BASE, SECTION);
    assert_eq!(table.axis_of_channel(), want_axis.as_slice());

    assert_eq!(table.axis_of_channel()[0], 1);
    assert_eq!(table.axis_of_channel()[23], 1);
    assert_eq!(table.axis_of_channel()[24], 2);
    assert_eq!(table.axis_of_channel()[43], 2);
    assert_eq!(table.axis_of_channel()[44], 0);
    assert_eq!(table.axis_of_channel()[63], 0);

    // The Qwen3-VL interleave would put H at 1, 4, 7, ...; here channel 1 is
    // still H only because the whole first section is H, and channel 3 is too.
    assert_eq!(table.axis_of_channel()[3], 1, "not a step-3 interleave");
}

/// End to end: cos/sin for distinct per-axis positions must equal the table
/// recomputed from upstream's definition, including the duplicate-halves
/// (split) layout.
#[test]
fn cos_sin_match_the_reference_table() {
    let _guard = mlx_test_guard();
    let table = CompassMRoPE::new(HEAD_DIM, BASE, SECTION).expect("published section is valid");
    let (inv_freq, axis) = reference_table(HEAD_DIM, BASE, SECTION);

    let pos = [7i32, 11, 13];
    let position_ids = mlxcel_core::from_slice_i32(&pos, &[3, 1, 1]);
    let (cos, sin) = table.forward(&position_ids);

    let mut want_cos: Vec<f32> = (0..64)
        .map(|j| (pos[axis[j] as usize] as f32 * inv_freq[j]).cos())
        .collect();
    let mut want_sin: Vec<f32> = (0..64)
        .map(|j| (pos[axis[j] as usize] as f32 * inv_freq[j]).sin())
        .collect();
    want_cos.extend_from_within(..);
    want_sin.extend_from_within(..);

    assert!(max_abs_diff(&to_vec(&cos), &want_cos) < 1e-4);
    assert!(max_abs_diff(&to_vec(&sin), &want_sin) < 1e-4);
}

/// A section that does not sum to `head_dim / 2` is refused instead of
/// producing a silently truncated table.
#[test]
fn a_section_that_does_not_cover_head_dim_is_rejected() {
    let err = match CompassMRoPE::new(HEAD_DIM, BASE, [24, 20, 8]) {
        Ok(_) => panic!("a short section must be refused"),
        Err(e) => e,
    };
    assert!(err.contains("mrope_section"), "{err}");
}
