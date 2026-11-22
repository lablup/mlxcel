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

//! Shared checks for the tiny-model tests of `--mla-latent-quant`
//! (issue #3179): what a latent cache stored, against the fp16 latent of the
//! same weights, and a quantized decode step against the dense step that
//! reads the same rows dequantized.

use mlxcel_core::layers::KVCache;
use mlxcel_core::mla::LatentWindow;
use mlxcel_core::{MlxArray, UniquePtr};

/// Bound on a 4-bit affine row against the fp16 row it encodes, relative to
/// the largest element: one step of a group's range, `2 / 15` of the largest
/// element at most. MLX's affine codec adjusts the scale so zero stays
/// representable, so an element can land up to a step away, not half of one
/// (0.087 measured on the DeepSeek V2 fixture).
pub(crate) const INT4_ROW_TOL: f32 = 0.14;
/// The same bound at 8 bits (`2 / 255`).
pub(crate) const INT8_ROW_TOL: f32 = 8e-3;
/// A single-query decode on the packed rows against the dense step over the
/// same rows dequantized: float accuracy of two summation orders.
pub(crate) const SAME_ROWS_TOL: f32 = 5e-3;

pub(crate) fn f32_values(a: &MlxArray) -> Vec<f32> {
    let a = mlxcel_core::astype(a, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&a);
    mlxcel_core::utils::array_to_vec_f32(&a)
}

pub(crate) fn max_rel_error(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let scale = want
        .iter()
        .fold(0.0f32, |acc, v| acc.max(v.abs()))
        .max(1e-6);
    got.iter()
        .zip(want)
        .fold(0.0f32, |acc, (g, w)| acc.max((g - w).abs() / scale))
}

/// The last `width` values: the final position of a `[1, L, width]` output.
pub(crate) fn last_position(values: &[f32], width: usize) -> Vec<f32> {
    values[values.len() - width..].to_vec()
}

/// Columns `[from, to)` of the key slot `quant` stores quantized must read
/// back as the same columns of the fp16 latent `fp16` holds for the same
/// inputs, within `tol`, and must not be those fp16 values exactly.
pub(crate) fn assert_stored_columns(quant: &KVCache, fp16: &KVCache, from: i32, to: i32, tol: f32) {
    let window = LatentWindow::stored(quant).expect("the cache holds a quantized latent");
    let live = fp16.seq_len();
    assert_eq!(window.seq_len(), live, "both caches hold the same tokens");
    let got: UniquePtr<MlxArray> = window.dense_key_columns(from, to);
    let keys = fp16
        .keys
        .as_ref()
        .expect("the fp16 cache holds latent rows");
    let s = mlxcel_core::array_shape(keys);
    let want = mlxcel_core::slice(keys, &[0, 0, 0, from], &[s[0], s[1], live, to]);
    let err = max_rel_error(&f32_values(&got), &f32_values(&want));
    assert!(err > 0.0, "columns [{from}, {to}) were stored unquantized");
    assert!(err < tol, "columns [{from}, {to}) round trip off by {err}");
}
