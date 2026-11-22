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
use crate::weights::{f16_to_f32, f32_to_f16_bits};

fn exact_in_f16(value: f32) -> bool {
    f16_to_f32(&f32_to_f16_bits(value).to_ne_bytes())[0].to_bits() == value.to_bits()
}

fn exact_in_bf16(value: f32) -> bool {
    value.to_bits() & 0xffff == 0
}

fn backend() -> ProductionBackend {
    ProductionBackend::new("IREE compiler version 3.12.0", "runtime-sha256-0", "cuda")
}

#[test]
fn operator_names_round_trip_and_cover_every_issue_operator() {
    assert_eq!(ProductionProbeOperator::ALL.len(), 11);
    for operator in ProductionProbeOperator::ALL {
        assert_eq!(
            ProductionProbeOperator::from_name(operator.name()),
            Some(operator)
        );
    }
    assert_eq!(ProductionProbeOperator::from_name("tf32-matmul"), None);
    let q4 = ProductionProbeOperator::ALL
        .into_iter()
        .filter(|operator| operator.is_affine_q4())
        .collect::<Vec<_>>();
    assert_eq!(
        q4,
        [
            ProductionProbeOperator::AffineQ4Qmv,
            ProductionProbeOperator::AffineQ4Qmm
        ]
    );
}

#[test]
fn every_probe_builds_a_production_contract_and_a_bounded_artifact() {
    for operator in ProductionProbeOperator::ALL {
        let probe = production_probe(operator).unwrap();
        let contract = probe.contract(&backend()).unwrap();
        let identity = contract.canonical_identity();
        assert!(identity.contains("micro-oracle."), "{identity}");
        assert!(
            identity.contains("mlxcel-xla-stablehlo-emitter-v1"),
            "{operator:?} must carry the production emitter identity: {identity}"
        );
        assert!(!identity.contains("canonical-decomposition-v1"));
        assert_eq!(probe.spec.operands, probe.artifact.operands);
        assert!(!probe.artifact.weights.is_empty());
        assert!(
            probe
                .artifact
                .dynamic_operand_indices
                .iter()
                .all(|index| *index < probe.artifact.operands.len())
        );
    }
}

#[test]
fn q4_probes_select_mlx_qmv_and_qmm_row_counts_and_keep_f16_metadata() {
    let qmv = production_probe_spec(ProductionProbeOperator::AffineQ4Qmv).unwrap();
    let qmm = production_probe_spec(ProductionProbeOperator::AffineQ4Qmm).unwrap();
    assert_eq!(qmv.operands()[0].shape(), [1, Q4_INPUTS]);
    assert!(qmm.operands()[0].shape()[0] >= 8);
    // `supports_qmm_sm80`: N % 128 == 0 and K % max(64, group) == 0.
    assert_eq!(Q4_OUTPUTS % 128, 0);
    assert_eq!(Q4_INPUTS % Q4_GROUP_SIZE.max(64), 0);
    assert_eq!(qmv.group_size(), Some(Q4_GROUP_SIZE));
    for spec in [&qmv, &qmm] {
        for operand in [
            &spec.operands()[0],
            &spec.operands()[2],
            &spec.operands()[3],
        ] {
            assert!(operand.values().iter().all(|value| exact_in_f16(*value)));
        }
        assert!(pack_affine_q4_lanes(&spec.operands()[1]).is_ok());
    }
    let contract = production_probe(ProductionProbeOperator::AffineQ4Qmm)
        .unwrap()
        .contract(&backend())
        .unwrap()
        .canonical_identity();
    assert!(contract.contains("class=affine-q4-dequantize"));
    assert!(contract.contains("weights=packed-affine-in-graph"));
    assert!(contract.contains("input=f16"));
    assert!(contract.contains("result=f16"));
}

#[test]
fn sscp_probe_uses_the_gemma3n_production_entry_and_bf16_exact_operands() {
    let spec = production_probe_spec(ProductionProbeOperator::Gemma3nSscpConvolution).unwrap();
    assert_eq!(spec.operands()[0].shape(), [1, 32, 128, 1]);
    assert_eq!(spec.operands()[1].shape(), [128, 3, 3, 1]);
    assert!(
        spec.operands()
            .iter()
            .flat_map(|operand| operand.values())
            .all(|value| exact_in_bf16(*value))
    );
    assert_eq!(
        spec.convolution(),
        Some(SscpConvolutionGeometry {
            padding_low: [0, 1],
            padding_high: [2, 1],
            stride: [2, 2],
        })
    );
    let contract = production_probe(ProductionProbeOperator::Gemma3nSscpConvolution)
        .unwrap()
        .contract(&backend())
        .unwrap()
        .canonical_identity();
    assert!(contract.contains("id=micro-oracle.subsample.convolution"));
    assert!(contract.contains("class=convolution"));
    assert!(contract.contains("input=bf16"));
    assert!(contract.contains("result=bf16"));
}

#[test]
fn dense_probes_reuse_the_canonical_operands_and_comparisons() {
    let dense = production_probe(ProductionProbeOperator::DenseMatmul).unwrap();
    let canonical = dense_definition().unwrap();
    assert_eq!(dense.artifact.operands, canonical.operands);
    assert_eq!(dense.artifact.comparison, ComparisonMode::ExactBits);
    let rms = production_probe_spec(ProductionProbeOperator::RmsNorm).unwrap();
    assert_eq!(rms.epsilon(), Some(super::super::RMS_EPSILON));
    let prefix = production_probe(ProductionProbeOperator::PrefixSum).unwrap();
    assert_eq!(prefix.artifact.comparison, ComparisonMode::ExactBits);
}

#[test]
fn q4_packing_rejects_non_lane_values_and_ragged_widths() {
    let ragged = NumericTensor::new("lanes", vec![1, 4], vec![0.0; 4]).unwrap();
    assert!(pack_affine_q4_lanes(&ragged).is_err());
    let out_of_range = NumericTensor::new("lanes", vec![1, 8], vec![16.0; 8]).unwrap();
    assert!(pack_affine_q4_lanes(&out_of_range).is_err());
    let lanes = NumericTensor::new(
        "lanes",
        vec![1, 8],
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    )
    .unwrap();
    assert_eq!(pack_affine_q4_lanes(&lanes).unwrap(), [0x7654_3210]);
}

#[test]
fn probe_devices_accept_gpu_targets_and_reject_unknown_ones() {
    use super::super::validate_probe_device;
    for device in ["local-task", "local-sync", "cuda", "hip"] {
        assert!(validate_probe_device(device).is_ok(), "{device}");
    }
    for device in ["metal", "vulkan", "cuda://0", ""] {
        assert!(validate_probe_device(device).is_err(), "{device}");
    }
}
