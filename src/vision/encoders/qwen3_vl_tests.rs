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

use super::{ensure_fused_sdpa, fused_sdpa_target_dim, sdpa_pad_width};
use mlxcel_core::dtype;

#[test]
fn fused_sdpa_target_dim_uses_next_supported_kernel_width() {
    assert_eq!(fused_sdpa_target_dim(64), 64);
    assert_eq!(fused_sdpa_target_dim(72), 80);
    assert_eq!(fused_sdpa_target_dim(96), 128);
    assert_eq!(fused_sdpa_target_dim(160), 160);
}

#[test]
fn sdpa_pad_width_only_pads_the_last_dimension() {
    assert_eq!(
        sdpa_pad_width(4, 96, 128),
        Some(vec![0, 0, 0, 0, 0, 0, 0, 32])
    );
    assert_eq!(sdpa_pad_width(4, 128, 128), None);
}

#[test]
fn ensure_fused_sdpa_restores_original_output_shape() {
    let q = mlxcel_core::ones(&[1, 2, 4, 96], dtype::FLOAT32);
    let k = mlxcel_core::ones(&[1, 2, 4, 96], dtype::FLOAT32);
    let v = mlxcel_core::ones(&[1, 2, 4, 96], dtype::FLOAT32);

    let output = ensure_fused_sdpa(&q, &k, &v, 1.0, None);
    mlxcel_core::eval(&output);

    assert_eq!(mlxcel_core::array_shape(&output), vec![1, 2, 4, 96]);
    assert_eq!(mlxcel_core::array_dtype(&output), dtype::FLOAT32);
}
