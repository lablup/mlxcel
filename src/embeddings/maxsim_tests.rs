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

//! Tests for the MaxSim late-interaction score.

use super::{maxsim, maxsim_mlx};
use crate::models::embedding_test_support::mlx_test_guard;

/// Rows of an orthonormal basis, as the multi-vector families emit them.
fn basis(rows: usize, width: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|i| {
            let mut row = vec![0.0f32; width];
            row[i % width] = 1.0;
            row
        })
        .collect()
}

#[test]
fn maxsim_identity_equals_query_length() {
    // Every unit row matches itself exactly, so the outer sum is one per
    // query row. This is the upper bound of the score for normalized rows.
    for rows in [1usize, 3, 8] {
        let q = basis(rows, 8);
        let score = maxsim(&q, &q);
        assert!(
            (score - rows as f32).abs() < 1e-6,
            "maxsim of {rows} orthonormal rows against themselves = {score}, expected {rows}"
        );
    }
}

#[test]
fn maxsim_is_asymmetric() {
    // Orthogonal rows can score the same in both directions, so they do not
    // by themselves prove the function is asymmetric.
    let query = vec![vec![1.0f32, 0.0], vec![0.0, 1.0]];
    let document = vec![vec![1.0f32, 0.0]];
    assert!((maxsim(&query, &document) - 1.0).abs() < 1e-6);
    assert!((maxsim(&document, &query) - 1.0).abs() < 1e-6);

    // Non-orthogonal rows make the two directions differ numerically: the
    // outer sum always runs over the first argument's rows.
    let a = vec![vec![1.0f32, 0.0], vec![0.6, 0.8]];
    let b = vec![vec![0.8f32, 0.6]];
    let forward = maxsim(&a, &b);
    let backward = maxsim(&b, &a);
    assert!(
        (forward - backward).abs() > 0.1,
        "expected an asymmetric score, got {forward} and {backward}"
    );
}

#[test]
fn maxsim_of_an_empty_side_is_zero() {
    let empty: Vec<Vec<f32>> = Vec::new();
    let rows = basis(3, 4);
    assert_eq!(maxsim(&empty, &rows), 0.0);
    assert_eq!(maxsim(&rows, &empty), 0.0);
}

#[test]
fn maxsim_ranks_the_matching_document_first() {
    // The query row points at basis vector 0; the matching document holds
    // it, the unrelated one does not.
    let query = vec![vec![1.0f32, 0.0, 0.0]];
    let matching = vec![vec![0.0f32, 1.0, 0.0], vec![1.0, 0.0, 0.0]];
    let unrelated = vec![vec![0.0f32, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
    assert!(maxsim(&query, &matching) > maxsim(&query, &unrelated));
}

#[test]
fn mlx_maxsim_matches_cpu() {
    let _guard = mlx_test_guard();

    // Deterministic pseudo-random rows, then L2-normalized the way a
    // multi-vector family emits them.
    let (lq, ld, width) = (5usize, 7usize, 6usize);
    let make = |seed: u32, rows: usize| -> Vec<Vec<f32>> {
        let mut state = seed;
        (0..rows)
            .map(|_| {
                let mut row: Vec<f32> = (0..width)
                    .map(|_| {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        ((state >> 8) as f32 / (1u32 << 24) as f32) - 0.5
                    })
                    .collect();
                let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-9);
                for v in &mut row {
                    *v /= norm;
                }
                row
            })
            .collect()
    };
    let query = make(7, lq);
    let document = make(9_001, ld);

    let flat = |rows: &[Vec<f32>]| -> Vec<f32> { rows.iter().flatten().copied().collect() };
    let q = mlxcel_core::from_slice_f32(&flat(&query), &[lq as i32, width as i32]);
    let d = mlxcel_core::from_slice_f32(&flat(&document), &[ld as i32, width as i32]);

    let cpu = maxsim(&query, &document);
    let gpu = maxsim_mlx(&q, &d);
    assert!(
        (cpu - gpu).abs() < 1e-5,
        "MLX MaxSim {gpu} does not match the CPU MaxSim {cpu}"
    );

    // The asymmetry survives the device implementation too.
    let cpu_rev = maxsim(&document, &query);
    let gpu_rev = maxsim_mlx(&d, &q);
    assert!((cpu_rev - gpu_rev).abs() < 1e-5);
}

#[test]
fn mlx_maxsim_rejects_a_non_matrix_input() {
    let _guard = mlx_test_guard();
    let vector = mlxcel_core::from_slice_f32(&[1.0, 0.0], &[2]);
    let matrix = mlxcel_core::from_slice_f32(&[1.0, 0.0], &[1, 2]);
    assert_eq!(maxsim_mlx(&vector, &matrix), 0.0);
    assert_eq!(maxsim_mlx(&matrix, &vector), 0.0);
}
