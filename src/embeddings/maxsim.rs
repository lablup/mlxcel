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

//! MaxSim: the late-interaction (ColBERT) scoring function shared by the
//! multi-vector embedding families.
//!
//! `maxsim(q, d) = sum_i max_j dot(q_i, d_j)` over a query matrix
//! `q: [Lq, D]` and a document matrix `d: [Ld, D]`. Every row a
//! multi-vector family emits is already L2-normalized, so each dot product
//! is a cosine and the score is bounded by `Lq`. The function is
//! deliberately asymmetric: swapping the arguments changes which side owns
//! the outer sum, and the query is always the outer one.
//!
//! Two implementations live here on purpose. [`maxsim`] works on the plain
//! `Vec<f32>` rows the engine reads back, which is what `mlxcel embed` and
//! any caller outside the MLX thread has; [`maxsim_mlx`] runs the same
//! contraction on device arrays for a caller that already holds them.
//!
//! Used by: `mlxcel embed` (the multi-vector similarity matrix) and the
//! ColIdefics3 / ColQwen2.5 real-checkpoint gates.

use mlxcel_core::{MlxArray, dtype};

/// `sum_i max_j dot(q_i, d_j)`.
///
/// Rows of either side may be borrowed (`&[f32]`) or owned (`Vec<f32>`).
/// An empty query or document scores `0.0`, and rows shorter than the
/// widest one contribute only over their common prefix, so a ragged input
/// cannot panic.
#[must_use]
pub fn maxsim<Q, D>(query: &[Q], document: &[D]) -> f32
where
    Q: AsRef<[f32]>,
    D: AsRef<[f32]>,
{
    if query.is_empty() || document.is_empty() {
        return 0.0;
    }
    query
        .iter()
        .map(|q| {
            let q = q.as_ref();
            document
                .iter()
                .map(|d| dot(q, d.as_ref()))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

/// Dot product over the common prefix of two rows.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// MLX form of [`maxsim`] over `query: [Lq, D]` and `document: [Ld, D]`.
///
/// Both arrays are cast to f32 first, so a f16 or bf16 activation dtype
/// does not change the score. Returns `0.0` for a rank that is not 2 or for
/// an empty side rather than tripping an MLX shape assertion, because the
/// caller is a scoring helper and not a forward pass.
#[must_use]
pub fn maxsim_mlx(query: &MlxArray, document: &MlxArray) -> f32 {
    let q_shape = mlxcel_core::array_shape(query);
    let d_shape = mlxcel_core::array_shape(document);
    if q_shape.len() != 2 || d_shape.len() != 2 {
        return 0.0;
    }
    if q_shape[0] == 0 || d_shape[0] == 0 {
        return 0.0;
    }
    let q = mlxcel_core::astype(query, dtype::FLOAT32);
    let d = mlxcel_core::astype(document, dtype::FLOAT32);
    let scores = mlxcel_core::matmul(&q, &mlxcel_core::transpose_axes(&d, &[1, 0]));
    let best = mlxcel_core::max_axis(&scores, -1, false);
    let total = mlxcel_core::sum_all(&best);
    mlxcel_core::eval(&total);
    mlxcel_core::item_f32(&total)
}

#[cfg(test)]
#[path = "maxsim_tests.rs"]
mod maxsim_tests;
