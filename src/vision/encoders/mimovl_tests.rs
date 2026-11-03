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

//! Fixture-parity tests for the MiMo-VL vision tower.
//!
//! The fixture is dumped from upstream mlx-vlm
//! (`mlx_vlm.models.mimo_v2.vision.VisionModel`) on a tiny random config with
//! one full-attention block and two window blocks, so dropping the learned
//! sinks or the banded window mask changes the expected tensor and fails here.

use super::{MimovlVisionConfig, MimovlVisionEncoder};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, dtype};

const FIXTURE: &str = include_str!("mimovl_parity_fixture.json");

fn fixture() -> serde_json::Value {
    serde_json::from_str(FIXTURE).expect("the committed fixture is valid JSON")
}

fn tensor(map: &mut WeightMap, prefix: &str, value: &serde_json::Value) {
    let shape: Vec<i32> = value["shape"]
        .as_array()
        .expect("shape")
        .iter()
        .map(|v| v.as_i64().expect("dim") as i32)
        .collect();
    let data: Vec<f32> = value["data"]
        .as_array()
        .expect("data")
        .iter()
        .map(|v| v.as_f64().expect("value") as f32)
        .collect();
    map.insert(
        format!("vision_tower.{prefix}"),
        mlxcel_core::from_slice_f32(&data, &shape),
    );
}

fn read_all(array: &MlxArray) -> Vec<f32> {
    let flat = mlxcel_core::astype(array, dtype::FLOAT32);
    mlxcel_core::eval(&flat);
    let bytes = mlxcel_core::array_to_raw_bytes(&flat);
    assert!(
        bytes.len().is_multiple_of(4),
        "unexpected f32 snapshot size {}",
        bytes.len()
    );
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn tiny_mimovl_matches_upstream_fixture() {
    let fixture = fixture();
    let config: MimovlVisionConfig =
        serde_json::from_value(fixture["config"].clone()).expect("fixture config parses");
    config.validate().expect("fixture config is valid");
    assert_eq!(config.fullatt_block_indexes, vec![0]);
    assert_eq!(config.vit_window_attn_types, vec![-1, 1, 0]);
    assert!(config.use_sink);

    let mut weights = WeightMap::new();
    for (key, value) in fixture["weights"].as_object().expect("weights object") {
        tensor(&mut weights, key, value);
    }
    let encoder = MimovlVisionEncoder::from_weights(&weights, &config, "vision_tower")
        .expect("the tiny tower loads");

    let pixels = &fixture["pixel_values_tc"];
    let shape: Vec<i32> = pixels["shape"]
        .as_array()
        .expect("pixel shape")
        .iter()
        .map(|v| v.as_i64().expect("dim") as i32)
        .collect();
    let data: Vec<f32> = pixels["data"]
        .as_array()
        .expect("pixel data")
        .iter()
        .map(|v| v.as_f64().expect("value") as f32)
        .collect();
    let pixel_values = mlxcel_core::from_slice_f32(&data, &shape);

    let grid: Vec<(i32, i32, i32)> = fixture["grid_thw"]
        .as_array()
        .expect("grid")
        .iter()
        .map(|row| {
            let row = row.as_array().expect("grid row");
            (
                row[0].as_i64().expect("t") as i32,
                row[1].as_i64().expect("h") as i32,
                row[2].as_i64().expect("w") as i32,
            )
        })
        .collect();

    let actual = encoder
        .forward_with_grid(&pixel_values, &grid)
        .expect("tiny forward");
    let actual_values = read_all(&actual.hidden_states);
    let expected = &fixture["expected"];
    let expected_shape: Vec<i32> = expected["shape"]
        .as_array()
        .expect("expected shape")
        .iter()
        .map(|v| v.as_i64().expect("dim") as i32)
        .collect();
    let expected_values: Vec<f32> = expected["data"]
        .as_array()
        .expect("expected data")
        .iter()
        .map(|v| v.as_f64().expect("value") as f32)
        .collect();

    assert_eq!(
        mlxcel_core::array_shape(&actual.hidden_states),
        expected_shape,
        "the fixture's merged token shape"
    );
    assert_eq!(actual_values.len(), expected_values.len());
    let mut largest = 0.0f32;
    for (index, (actual, expected)) in actual_values.iter().zip(expected_values.iter()).enumerate()
    {
        let tolerance = 2e-4 * expected.abs().max(1.0);
        let difference = (actual - expected).abs();
        largest = largest.max(difference);
        assert!(
            difference <= tolerance,
            "MiMo-VL fixture mismatch at flat index {index}: got {actual}, expected {expected} \
             (difference {difference}, largest so far {largest}). Dropping the attention sinks or \
             the banded window mask changes this value."
        );
    }
}
