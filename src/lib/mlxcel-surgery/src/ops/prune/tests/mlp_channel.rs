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

//! `granularity: mlp_channel` unit tests for `PruneOp`.

use super::super::{PruneOp, PruneSelector};
use super::{make_cfg, read_f32_2d};
use crate::{SurgeryOp, WeightMap};
use mlxcel_core as ffi;
use mlxcel_core::dtype;

#[test]
fn mlp_channel_prune_zeros_up_and_gate_proj_rows() {
    // intermediate_size=16, hidden=8 -> up/gate shape [16, 8].
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.mlp.up_proj.weight".to_string(),
        ffi::ones(&[16, 8], dtype::FLOAT32),
    );
    weights.insert(
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        ffi::ones(&[16, 8], dtype::FLOAT32),
    );

    let op = PruneOp::new(
        "model.layers.0.mlp.*",
        PruneSelector::MlpChannel {
            channel_ids: vec![3, 7],
        },
    )
    .expect("compile");
    let cfg = make_cfg(2, 2, 8, 16, 1);
    op.apply(&mut weights, &cfg).expect("apply");

    let (_, up) = read_f32_2d(&weights, "model.layers.0.mlp.up_proj.weight");
    let (_, gate) = read_f32_2d(&weights, "model.layers.0.mlp.gate_proj.weight");
    for r in 0..16 {
        let row_sum_up: f32 = up[r * 8..(r + 1) * 8].iter().sum();
        let row_sum_gate: f32 = gate[r * 8..(r + 1) * 8].iter().sum();
        if r == 3 || r == 7 {
            assert!(row_sum_up == 0.0, "up_proj row {r} should be zero");
            assert!(row_sum_gate == 0.0, "gate_proj row {r} should be zero");
        } else {
            assert!(row_sum_up > 0.0, "up_proj row {r} should remain nonzero");
            assert!(
                row_sum_gate > 0.0,
                "gate_proj row {r} should remain nonzero"
            );
        }
    }
}

#[test]
fn mlp_channel_prune_zeros_down_proj_column_in_float() {
    let mut weights = WeightMap::new();
    // down_proj [hidden=8, intermediate=16]
    weights.insert(
        "model.layers.0.mlp.down_proj.weight".to_string(),
        ffi::ones(&[8, 16], dtype::FLOAT32),
    );
    let op = PruneOp::new(
        "model.layers.0.mlp.*",
        PruneSelector::MlpChannel {
            channel_ids: vec![5],
        },
    )
    .expect("compile");
    let cfg = make_cfg(2, 2, 8, 16, 1);
    op.apply(&mut weights, &cfg).expect("apply");

    let (_, floats) = read_f32_2d(&weights, "model.layers.0.mlp.down_proj.weight");
    for row in 0..8 {
        for col in 0..16 {
            let v = floats[row * 16 + col];
            if col == 5 {
                assert!(v == 0.0, "col {col} of row {row} should be zero");
            } else {
                assert!(v == 1.0, "col {col} of row {row} should be one");
            }
        }
    }
}

#[test]
fn mlp_channel_prune_rejects_out_of_range_channel() {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.mlp.up_proj.weight".to_string(),
        ffi::ones(&[16, 8], dtype::FLOAT32),
    );
    let op = PruneOp::new(
        "model.layers.0.mlp.*",
        PruneSelector::MlpChannel {
            channel_ids: vec![100],
        },
    )
    .expect("compile");
    let cfg = make_cfg(2, 2, 8, 16, 1);
    let err = op.apply(&mut weights, &cfg).expect_err("oor channel");
    assert!(format!("{err}").contains("channel_id 100"));
}

#[test]
fn mlp_channel_prune_rejects_combined_gate_up() {
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.mlp.gate_up_proj.weight".to_string(),
        ffi::ones(&[32, 8], dtype::FLOAT32),
    );
    let op = PruneOp::new(
        "model.layers.0.mlp.*",
        PruneSelector::MlpChannel {
            channel_ids: vec![0],
        },
    )
    .expect("compile");
    let cfg = make_cfg(2, 2, 8, 16, 1);
    let err = op.apply(&mut weights, &cfg).expect_err("combined gate_up");
    assert!(format!("{err}").contains("gate_up_proj"));
}

#[test]
fn quantized_down_proj_single_channel_is_rejected() {
    // Quantized down_proj cannot be channel-pruned along the IN
    // axis because the prune width (1) is below the pack factor.
    let mut weights = WeightMap::new();
    weights.insert(
        "model.layers.0.mlp.down_proj.weight".to_string(),
        ffi::ones(&[8, 2], dtype::UINT32),
    );
    let op = PruneOp::new(
        "model.layers.0.mlp.*",
        PruneSelector::MlpChannel {
            channel_ids: vec![0],
        },
    )
    .expect("compile");
    let cfg = make_cfg(2, 2, 8, 16, 1);
    let err = op
        .apply(&mut weights, &cfg)
        .expect_err("quantized down_proj single-channel must fail");
    assert!(format!("{err}").contains("Quantized MLP-channel"));
}
