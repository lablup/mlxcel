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

use super::{
    VisionBlockDiagnosticStage, ensure_fused_sdpa, fused_sdpa_target_dim, gelu_pytorch_tanh,
    sdpa_pad_width,
};
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

#[test]
fn block_gelu_matches_qwen3_vl_pytorch_tanh_contract() {
    let input = mlxcel_core::from_slice_f32(&[-3.0, -1.0, 0.0, 1.0, 3.0], &[5]);
    let output = gelu_pytorch_tanh(&input);
    mlxcel_core::eval(&output);

    let expected = [-0.003_637_433, -0.158_808, 0.0, 0.841_192, 2.996_362_7];
    for (index, expected) in expected.into_iter().enumerate() {
        let value = mlxcel_core::slice(&output, &[index as i32], &[index as i32 + 1]);
        assert!(
            (mlxcel_core::item_f32(&value) - expected).abs() <= 2.0e-6,
            "Qwen3-VL tanh GELU mismatch at {index}"
        );
    }
}

#[test]
fn block_2_diagnostic_stage_order_is_stable() {
    let stages = [
        VisionBlockDiagnosticStage::Input,
        VisionBlockDiagnosticStage::Norm1,
        VisionBlockDiagnosticStage::Attention,
        VisionBlockDiagnosticStage::PostAttentionResidual,
        VisionBlockDiagnosticStage::Norm2,
        VisionBlockDiagnosticStage::Mlp,
        VisionBlockDiagnosticStage::Output,
    ];
    assert_eq!(stages.len(), VisionBlockDiagnosticStage::COUNT);
    for (index, stage) in stages.into_iter().enumerate() {
        assert_eq!(stage.index(), index);
    }
}
