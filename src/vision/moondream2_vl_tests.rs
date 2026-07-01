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

use super::moondream2_vl::build_moondream2_attention_mask;

fn mask_value(mask: &mlxcel_core::MlxArray, row: i32, col: i32) -> f32 {
    let value = mlxcel_core::slice(mask, &[0, 0, row, col], &[1, 1, row + 1, col + 1]);
    // Cast to f32 before reading — MLX item<float>() on bfloat16 scalars
    // returns wrong values (confirmed in both v0.31.0 and v0.31.1).
    let value = mlxcel_core::astype(&value, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&value);
    mlxcel_core::item_f32(&value)
}

#[test]
fn moondream2_attention_mask_keeps_prefix_fully_connected() {
    let mask = build_moondream2_attention_mask(3, 2);
    assert_eq!(mlxcel_core::array_shape(&mask), vec![1, 1, 5, 5]);
    // Prefix rows attend to the whole prefix bidirectionally.
    assert_eq!(mask_value(&mask, 0, 0), 0.0);
    assert_eq!(mask_value(&mask, 0, 2), 0.0);
    // Prefix rows do not attend to prompt positions.
    assert!(mask_value(&mask, 0, 3).is_infinite());
}

#[test]
fn moondream2_attention_mask_uses_causal_prompt_rows_after_prefix() {
    let mask = build_moondream2_attention_mask(2, 3);
    // First prompt row (index 2) sees the prefix and itself, but not the future.
    assert_eq!(mask_value(&mask, 2, 0), 0.0);
    assert_eq!(mask_value(&mask, 2, 2), 0.0);
    assert!(mask_value(&mask, 2, 3).is_infinite());
    // Last prompt row sees itself.
    assert_eq!(mask_value(&mask, 4, 4), 0.0);
}
