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

//! A quantized plane whose `scales` cover fewer output rows than its `weight`
//! is refused at load (issue #3457). MLX validates only the batch axes and
//! the last axis, so before the check such a plane loaded and its qmv kernel
//! read past the scales buffer.

use super::{UnifiedLinear, check_scales_leading_axes};
use crate::{dtype, ffi, weights::WeightMap};

/// A 4-bit affine plane: `[64, 8]` packed words describe 64 inputs, one group
/// of 64 per row, with `scale_rows` rows of scales and biases.
fn affine_plane(scale_rows: i32) -> WeightMap {
    let mut weights = WeightMap::new();
    weights.insert("p.weight".into(), ffi::zeros(&[64, 8], dtype::UINT32));
    weights.insert(
        "p.scales".into(),
        ffi::zeros(&[scale_rows, 1], dtype::FLOAT16),
    );
    weights.insert(
        "p.biases".into(),
        ffi::zeros(&[scale_rows, 1], dtype::FLOAT16),
    );
    weights
}

#[test]
fn unified_linear_refuses_scales_with_fewer_rows_than_the_weight() {
    let weights = affine_plane(32);
    let err = UnifiedLinear::from_weights_with_mode(&weights, "p", 64, 4, "affine")
        .err()
        .expect("short scales must be refused");
    assert!(err.contains("p.scales"), "{err}");
    assert!(err.contains("[32, 1]") && err.contains("[64, 8]"), "{err}");

    UnifiedLinear::from_weights_with_mode(&affine_plane(64), "p", 64, 4, "affine")
        .expect("matching scales still load");
}

#[test]
fn leading_axes_check_names_each_mismatch() {
    assert!(check_scales_leading_axes("p", &[64, 8], &[64, 1], Some(&[64, 1])).is_ok());
    let rank = check_scales_leading_axes("p", &[64, 8], &[64], None).unwrap_err();
    assert!(
        rank.contains("p.scales") && rank.contains("same rank"),
        "{rank}"
    );
    let experts = check_scales_leading_axes("p", &[4, 64, 8], &[3, 64, 1], None).unwrap_err();
    assert!(
        experts.contains("p.scales") && experts.contains("[3, 64, 1]"),
        "{experts}"
    );
    let biases = check_scales_leading_axes("p", &[64, 8], &[64, 1], Some(&[32, 1])).unwrap_err();
    assert!(biases.contains("p.biases"), "{biases}");
}
