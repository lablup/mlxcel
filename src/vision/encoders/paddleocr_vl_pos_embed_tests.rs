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

//! Quantized `position_embedding` (#2657): the 4-bit PaddleOCR-VL export
//! quantizes the learned position table, so the bilinear interpolation must
//! gather dequantized rows, and the plain-table lookup must stay bit-identical.

use super::PositionEmbedding;
use mlxcel_core::weights::WeightMap;

const GRID: usize = 4; // 4x4 learned grid
const HIDDEN: usize = 128;
const GROUP: i32 = 64;
const BITS: i32 = 4;

/// A deterministic `[GRID*GRID, HIDDEN]` table with distinct rows.
fn dense_table() -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let data: Vec<f32> = (0..GRID * GRID * HIDDEN)
        .map(|i| {
            let (row, col) = ((i / HIDDEN) as f32, (i % HIDDEN) as f32);
            (0.31 * row + 0.017 * col).sin() * 0.8 + 0.01 * row
        })
        .collect();
    mlxcel_core::from_slice_f32(&data, &[(GRID * GRID) as i32, HIDDEN as i32])
}

fn values(a: &mlxcel_core::MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(a)
}

/// Target grids that need real interpolation (up- and down-sampling, and a
/// non-square aspect), plus the identity grid.
const GRIDS: &[(i32, i32)] = &[(6, 10), (3, 5), (4, 4)];

#[test]
fn quantized_pos_embed_matches_the_dequantized_dense_table() {
    let q = mlxcel_core::quantize_weights(&dense_table(), GROUP, BITS);
    let packed = mlxcel_core::quantized_weights_w(&q);
    let scales = mlxcel_core::quantized_weights_scales(&q);
    let biases = mlxcel_core::quantized_weights_biases(&q);
    // The packed width is not the hidden size; reading it as one was the bug.
    assert_ne!(mlxcel_core::array_shape(&packed)[1], HIDDEN as i32);
    let dequantized = unsafe {
        mlxcel_core::dequantize(
            &packed,
            &scales,
            &*biases as *const _,
            GROUP,
            BITS,
            "affine",
        )
    };

    let mut quantized = WeightMap::new();
    quantized.insert("pos.weight".to_string(), packed);
    quantized.insert("pos.scales".to_string(), scales);
    quantized.insert("pos.biases".to_string(), biases);
    let mut dense = WeightMap::new();
    dense.insert("pos.weight".to_string(), dequantized);

    let from_quantized =
        PositionEmbedding::from_weights(&quantized, "pos", GROUP, BITS).expect("quantized");
    let from_dense = PositionEmbedding::from_weights(&dense, "pos", 0, 0).expect("dense");
    for &(h, w) in GRIDS {
        let got = from_quantized.interpolate(h, w);
        let want = from_dense.interpolate(h, w);
        assert_eq!(
            mlxcel_core::array_shape(&got),
            vec![h * w, HIDDEN as i32],
            "grid {h}x{w}"
        );
        assert_eq!(
            mlxcel_core::array_shape(&got),
            mlxcel_core::array_shape(&want)
        );
        let (got, want) = (values(&got), values(&want));
        for (idx, (g, w_)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w_).abs() <= 1e-3,
                "grid {h}x{w} entry {idx}: quantized {g}, dense {w_}"
            );
        }
    }
}

#[test]
fn plain_pos_embed_lookup_is_bit_identical_to_the_row_gather() {
    let table = mlxcel_core::astype(&dense_table(), mlxcel_core::dtype::BFLOAT16);
    let idx = mlxcel_core::from_slice_i32(&[0, 5, 15, 3, 3, 10], &[6]);
    let gathered = mlxcel_core::take(&table, &idx, 0);
    let mut wm = WeightMap::new();
    wm.insert("pos.weight".to_string(), table);
    let pos = PositionEmbedding::from_weights(&wm, "pos", 0, 0).expect("plain");
    assert!(!pos.table.is_quantized());
    assert_eq!(pos.num_positions, (GRID * GRID) as i32);
    let looked_up = pos.table.forward(&idx);
    assert_eq!(
        mlxcel_core::array_dtype(&looked_up),
        mlxcel_core::dtype::BFLOAT16
    );
    let bits =
        |a: &mlxcel_core::MlxArray| -> Vec<u32> { values(a).iter().map(|v| v.to_bits()).collect() };
    assert_eq!(bits(&looked_up), bits(&gathered));
}

#[test]
fn non_square_pos_table_is_rejected_at_load() {
    for rows in [15, 17] {
        let data = vec![0.5f32; rows * HIDDEN];
        let table = mlxcel_core::from_slice_f32(&data, &[rows as i32, HIDDEN as i32]);
        let mut wm = WeightMap::new();
        wm.insert("pos.weight".to_string(), table);
        let err = PositionEmbedding::from_weights(&wm, "pos", 0, 0)
            .err()
            .expect("a non-square table must not load");
        assert!(err.contains("not a square position grid"), "{err}");
    }
}
