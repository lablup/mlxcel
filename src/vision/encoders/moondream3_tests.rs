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

const ROWS: i32 = 4;
const DIM: i32 = 64;

fn config() -> Moondream3VisionConfig {
    serde_json::from_value(serde_json::json!({
        "enc_dim": DIM,
        "enc_patch_size": 1,
        "enc_n_layers": 0,
        "enc_ff_dim": DIM,
        "enc_n_heads": 1,
        "proj_out_dim": DIM,
        "crop_size": 2,
        "in_channels": 3,
        "max_crops": 1,
        "overlap_margin": 0,
        "proj_inner_dim": DIM
    }))
    .expect("test Moondream config")
}

fn dense() -> UniquePtr<MlxArray> {
    let values = (0..ROWS * DIM)
        .map(|index| (index as f32 * 0.031).sin())
        .collect::<Vec<_>>();
    mlxcel_core::from_slice_f32(&values, &[ROWS, DIM])
}

fn assert_close(actual: &MlxArray, expected: &MlxArray) {
    assert_eq!(mlxcel_core::array_shape(actual), [1, ROWS, DIM]);
    let difference = mlxcel_core::reshape(&mlxcel_core::subtract(actual, expected), &[-1]);
    let maximum = mlxcel_core::max_axis(&mlxcel_core::abs(&difference), 0, false);
    mlxcel_core::eval(&maximum);
    assert!(mlxcel_core::item_f32(&maximum) <= 1e-6);
}

#[test]
fn dense_position_table_accepts_2d_and_singleton_3d_storage() {
    let dense = dense();
    let expected = mlxcel_core::reshape(&dense, &[1, ROWS, DIM]);
    for stored in [
        mlxcel_core::copy(&dense),
        mlxcel_core::reshape(&dense, &[1, ROWS, DIM]),
    ] {
        let mut weights = WeightMap::new();
        weights.insert("vision.pos_emb".into(), stored);
        let loaded = load_position_embedding(&weights, &config(), 64, 4)
            .expect("dense Moondream position table");
        assert_eq!(
            mlxcel_core::array_to_raw_bytes(&loaded),
            mlxcel_core::array_to_raw_bytes(&expected)
        );
    }
}

#[test]
fn quantized_position_table_accepts_2d_and_singleton_3d_storage() {
    let dense = dense();
    let quantized = mlxcel_core::quantize_weights(&dense, 64, 4);
    let packed = mlxcel_core::quantized_weights_w(&quantized);
    let scales = mlxcel_core::quantized_weights_scales(&quantized);
    let biases = mlxcel_core::quantized_weights_biases(&quantized);
    let reference = unsafe {
        mlxcel_core::dequantize(
            &packed,
            &scales,
            biases.as_ref().expect("affine biases") as *const _,
            64,
            4,
            "affine",
        )
    };
    let reference = mlxcel_core::reshape(&reference, &[1, ROWS, DIM]);
    for leading in [false, true] {
        let mut weights = WeightMap::new();
        if leading {
            weights.insert(
                "vision.pos_emb".into(),
                mlxcel_core::reshape(&packed, &[1, ROWS, 8]),
            );
            weights.insert(
                "vision.pos_emb.scales".into(),
                mlxcel_core::reshape(&scales, &[1, ROWS, 1]),
            );
            weights.insert(
                "vision.pos_emb.biases".into(),
                mlxcel_core::reshape(&biases, &[1, ROWS, 1]),
            );
        } else {
            weights.insert("vision.pos_emb".into(), mlxcel_core::copy(&packed));
            weights.insert("vision.pos_emb.scales".into(), mlxcel_core::copy(&scales));
            weights.insert("vision.pos_emb.biases".into(), mlxcel_core::copy(&biases));
        }
        let loaded = load_position_embedding(&weights, &config(), 64, 4)
            .expect("quantized Moondream position table");
        assert_close(&loaded, &reference);
    }
}

#[test]
fn malformed_position_table_layouts_return_errors() {
    for shape in [&[2, ROWS, DIM][..], &[1, ROWS - 1, DIM], &[ROWS, DIM, 1, 1]] {
        let mut weights = WeightMap::new();
        weights.insert(
            "vision.pos_emb".into(),
            mlxcel_core::zeros(shape, mlxcel_core::dtype::FLOAT32),
        );
        assert!(load_position_embedding(&weights, &config(), 64, 4).is_err());
    }

    let mut wrong_width = WeightMap::new();
    wrong_width.insert(
        "vision.pos_emb".into(),
        mlxcel_core::zeros(&[1, ROWS, DIM - 1], mlxcel_core::dtype::FLOAT32),
    );
    let error = load_position_embedding(&wrong_width, &config(), 64, 4)
        .err()
        .expect("wrong dense width must fail");
    assert!(error.contains("shape mismatch"), "{error}");
}
