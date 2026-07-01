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

use super::helpers::{build_projection_layout, split_conv_output_ranges};

#[test]
fn projection_layout_computes_expected_ranges_and_shapes() {
    let layout = build_projection_layout(&[2, 5], 2, 4, 4, 8);

    assert_eq!(layout.mixed_qkvz_shape, vec![2, 5, 2, -1]);
    assert_eq!(layout.mixed_ba_shape, vec![2, 5, 2, -1]);
    assert_eq!(layout.q_range, (0, 4));
    assert_eq!(layout.k_range, (4, 8));
    assert_eq!(layout.v_range, (8, 24));
    // z spans the same width as v; explicit stop (not -1) so mlxcel_core::slice
    // does not truncate the last column (stop = -1 means dim_size - 1).
    assert_eq!(layout.z_range, (24, 40));
    assert_eq!(layout.b_range, (0, 2));
    assert_eq!(layout.a_range, (2, 4));
    assert_eq!(layout.v_shape, vec![2, 5, -1, 8]);
    assert_eq!(layout.ba_final_shape, vec![2, 5, 4]);
}

#[test]
fn split_conv_output_ranges_partitions_q_k_v_segments() {
    let ranges = split_conv_output_ranges(128, 640);

    assert_eq!(ranges[0], (0, 128));
    assert_eq!(ranges[1], (128, 256));
    assert_eq!(ranges[2], (256, 640));
}
