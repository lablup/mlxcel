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

use super::*;

const GROUP_SIZE: i32 = 64;
const BITS: i32 = 4;
const INIT_H: i32 = 2;
const INIT_W: i32 = 3;
const DIM: i32 = 64;

fn source_table() -> UniquePtr<MlxArray> {
    let values = (0..(INIT_H * INIT_W * DIM))
        .map(|index| ((index as f32 * 0.037) + 0.4).cos())
        .collect::<Vec<_>>();
    mlxcel_core::from_slice_f32(&values, &[INIT_H * INIT_W, DIM])
}

fn quantized_table(prefix: &str, stored_shape: &[i32]) -> WeightMap {
    let source = source_table();
    let quantized = mlxcel_core::quantize_weights(&source, GROUP_SIZE, BITS);
    let packed = mlxcel_core::quantized_weights_w(&quantized);
    let scales = mlxcel_core::quantized_weights_scales(&quantized);
    let biases = mlxcel_core::quantized_weights_biases(&quantized);
    let mut packed_shape = stored_shape.to_vec();
    *packed_shape.last_mut().expect("non-empty stored shape") =
        mlxcel_core::array_shape(&packed)[1];
    let mut sidecar_shape = stored_shape.to_vec();
    *sidecar_shape.last_mut().expect("non-empty stored shape") =
        mlxcel_core::array_shape(&scales)[1];
    let mut weights = WeightMap::new();
    weights.insert(
        format!("{prefix}.weight"),
        mlxcel_core::reshape(&packed, &packed_shape),
    );
    weights.insert(
        format!("{prefix}.scales"),
        mlxcel_core::reshape(&scales, &sidecar_shape),
    );
    weights.insert(
        format!("{prefix}.biases"),
        mlxcel_core::reshape(&biases, &sidecar_shape),
    );
    weights
}

fn explicit_dequantized_grid(weights: &WeightMap, prefix: &str) -> UniquePtr<MlxArray> {
    let packed = &weights[&format!("{prefix}.weight")];
    let scales = &weights[&format!("{prefix}.scales")];
    let biases = &weights[&format!("{prefix}.biases")];
    let packed_width = *mlxcel_core::array_shape(packed)
        .last()
        .expect("packed rank");
    let groups = *mlxcel_core::array_shape(scales)
        .last()
        .expect("scales rank");
    let packed = mlxcel_core::reshape(packed, &[INIT_H * INIT_W, packed_width]);
    let scales = mlxcel_core::reshape(scales, &[INIT_H * INIT_W, groups]);
    let biases = mlxcel_core::reshape(biases, &[INIT_H * INIT_W, groups]);
    // SAFETY: the affine planes come directly from `quantize_weights` and
    // remain alive for this call.
    let dense = unsafe {
        mlxcel_core::dequantize(
            &packed,
            &scales,
            biases.as_ref().expect("non-null test biases") as *const MlxArray,
            GROUP_SIZE,
            BITS,
            "affine",
        )
    };
    mlxcel_core::reshape(&dense, &[INIT_H, INIT_W, DIM])
}

fn assert_close(actual: &MlxArray, expected: &MlxArray) {
    assert_eq!(
        mlxcel_core::array_shape(actual),
        mlxcel_core::array_shape(expected)
    );
    let actual = mlxcel_core::utils::array_to_vec_f32(actual);
    let expected = mlxcel_core::utils::array_to_vec_f32(expected);
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-5,
            "entry {index}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn lfm2_vl_flat_quantized_position_table_matches_explicit_dequant_resample() {
    let prefix = "lfm2.position_embedding";
    let weights = quantized_table(prefix, &[INIT_H * INIT_W, DIM]);
    let loaded = Learnable2DInterpPosEmb::from_weights(
        &weights, prefix, INIT_H, INIT_W, DIM, GROUP_SIZE, BITS,
    )
    .expect("LFM2-VL flat packed position table")
    .with_resample(PosEmbResample::BilinearAntialias);
    assert_eq!(
        mlxcel_core::array_shape(&loaded.weight),
        [INIT_H, INIT_W, DIM]
    );
    let reference = Learnable2DInterpPosEmb::from_array(
        explicit_dequantized_grid(&weights, prefix),
        INIT_H,
        INIT_W,
        DIM,
    )
    .with_resample(PosEmbResample::BilinearAntialias);
    assert_close(&loaded.pos_for(3, 2), &reference.pos_for(3, 2));
}

#[test]
fn kimi_vl_grid_quantized_position_table_matches_explicit_dequant_resample() {
    let prefix = "kimi.position_embedding";
    let weights = quantized_table(prefix, &[INIT_H, INIT_W, DIM]);
    let loaded = Learnable2DInterpPosEmb::from_weights(
        &weights, prefix, INIT_H, INIT_W, DIM, GROUP_SIZE, BITS,
    )
    .expect("Kimi-VL grid-shaped packed position table");
    assert_eq!(
        mlxcel_core::array_shape(&loaded.weight),
        [INIT_H, INIT_W, DIM]
    );
    let reference = Learnable2DInterpPosEmb::from_array(
        explicit_dequantized_grid(&weights, prefix),
        INIT_H,
        INIT_W,
        DIM,
    );
    assert_close(&loaded.pos_for(3, 2), &reference.pos_for(3, 2));
}
