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

//! In-process IREE execution for bounded operator-level numeric probes.

use std::mem::size_of;
use std::path::Path;
use std::process::Command;

use sha2::{Digest, Sha256};

use crate::aux::{
    AuxiliaryInput, AuxiliaryOutput, AuxiliaryTensorDType, AuxiliaryWeight, AuxiliaryWeightDType,
    IreeAuxiliaryModule,
};
use crate::aux_manifest::{AuxiliaryArtifactContract, ensure_qualified_auxiliary_artifact};
use crate::emitter::numeric_ops::{
    DENSE_MATMUL_ENTRY, DenseMatmulProbeSpec, GELU_PROJECTION_ENTRY, LAYER_NORM_ENTRY,
    RESIDUAL_ADD_ENTRY, RMS_NORM_ENTRY, RowWiseProbeSpec, SILU_PROJECTION_ENTRY, SOFTMAX_ENTRY,
    emit_dense_matmul_probe, emit_gelu_projection_probe, emit_layer_norm_probe,
    emit_residual_add_probe, emit_rms_norm_probe, emit_silu_projection_probe, emit_softmax_probe,
};
use crate::iree::{cached_vmfb_path, compile_one_to, iree_compile_bin, target_flags};
use crate::numeric_dtype_contract::{NumericDType, WeightExecution};
use crate::numeric_oracle::{
    AlgorithmIdentity, ComparisonMode, NumericBackendIdentity, NumericOracleCase,
    NumericOracleClaim, NumericOracleReport, NumericTensor, run_bounded_numeric_oracle,
};
use crate::operator_numeric_contract::{
    AssociationPolicy, BackendNumericIdentity, CheckpointDType, OperatorClass,
    OperatorNumericContract, OperatorNumericContractSet, RoundingBoundary,
};

const DENSE_ROWS: usize = 2;
const DENSE_OUTPUTS: usize = 2;
const DENSE_CONTRACTION: usize = 3;
const DENSE_OPERATION: &str = "micro-oracle.dense-matmul.f32";
const ROWS: usize = 2;
const FEATURES: usize = 4;
const RMS_EPSILON: f32 = 1e-5;
const LAYER_NORM_EPSILON: f32 = 1e-5;
const PROJECTION_OUTPUTS: usize = 3;
const RESIDUAL_OPERATION: &str = "micro-oracle.residual-add.f32";
const RMS_NORM_OPERATION: &str = "micro-oracle.rms-norm.f32";
const SOFTMAX_OPERATION: &str = "micro-oracle.attention-softmax.f32";
const LAYER_NORM_OPERATION: &str = "micro-oracle.layer-norm.f32";
const SILU_PROJECTION_OPERATION: &str = "micro-oracle.silu-projection.f32";
const GELU_PROJECTION_OPERATION: &str = "micro-oracle.gelu-projection.f32";

struct ProbeDefinition {
    operation: &'static str,
    entry: &'static str,
    cache_key: &'static str,
    config_identity: String,
    mlir: String,
    operator_class: OperatorClass,
    rounding_boundaries: Vec<RoundingBoundary>,
    association: AssociationPolicy,
    operands: Vec<NumericTensor>,
    weights: Vec<AuxiliaryWeight>,
    dynamic_operand_indices: Vec<usize>,
    output_shape: Vec<usize>,
    comparison: ComparisonMode,
    reference_algorithm: &'static str,
    candidate_algorithm: &'static str,
    reference: fn(&[NumericTensor]) -> Result<NumericTensor, String>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(size_of::<f32>()) {
        return Err(format!(
            "f32 output has {} bytes, which is not divisible by {}",
            bytes.len(),
            size_of::<f32>()
        ));
    }
    Ok(bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four-byte f32 chunk")))
        .collect())
}

fn compiler_version(compiler: &Path) -> Result<String, String> {
    let output = Command::new(compiler)
        .arg("--version")
        .output()
        .map_err(|error| format!("run {} --version: {error}", compiler.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} --version failed: {}",
            compiler.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let version = String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if version.is_empty() {
        return Err(format!(
            "{} --version returned no identity",
            compiler.display()
        ));
    }
    Ok(version)
}

fn runtime_archive_identity(compiler: &Path) -> Result<String, String> {
    let dist = option_env!("MLXCEL_XLA_IREE_DIST")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("IREE_DIST").map(std::path::PathBuf::from))
        .or_else(|| compiler.parent().and_then(Path::parent).map(Path::to_owned))
        .ok_or_else(|| format!("derive IREE dist root from {}", compiler.display()))?;
    let runtime = dist.join("lib/libiree_runtime_unified.a");
    let bytes =
        std::fs::read(&runtime).map_err(|error| format!("read {}: {error}", runtime.display()))?;
    Ok(format!("libiree-runtime-sha256-{}", sha256_hex(&bytes)))
}

fn operator_identity(
    compiler_version: &str,
    runtime_identity: &str,
    device: &str,
    definition: &ProbeDefinition,
) -> Result<OperatorNumericContractSet, String> {
    let backend = BackendNumericIdentity::new(
        "canonical-decomposition-v1",
        runtime_identity,
        compiler_version.replace(';', ","),
        device,
        definition.candidate_algorithm,
    )?;
    let operation = OperatorNumericContract::new(
        definition.operation,
        definition.operator_class,
        CheckpointDType::F32,
        NumericDType::F32,
        NumericDType::F32,
        NumericDType::F32,
        definition.rounding_boundaries.iter().copied(),
        definition.association,
        WeightExecution::Dense,
        None,
        backend,
    )?;
    OperatorNumericContractSet::new([operation])
}

fn dense_operands() -> Result<[NumericTensor; 2], String> {
    Ok([
        NumericTensor::new(
            "input",
            vec![DENSE_ROWS, DENSE_CONTRACTION],
            vec![1.0, 2.0, -1.0, -2.0, 0.5, 3.0],
        )?,
        NumericTensor::new(
            "weight",
            vec![DENSE_OUTPUTS, DENSE_CONTRACTION],
            vec![2.0, -1.0, 0.5, -3.0, 0.25, 4.0],
        )?,
    ])
}

fn resident_weight(name: &str, weight: &NumericTensor) -> AuxiliaryWeight {
    AuxiliaryWeight {
        name: name.to_string(),
        bytes: f32_bytes(weight.values()),
        dtype: AuxiliaryWeightDType::Float32,
        shape: weight.shape().to_vec(),
    }
}

fn dummy_weight(operation: &str) -> AuxiliaryWeight {
    AuxiliaryWeight {
        name: format!("{operation}.abi_dummy"),
        bytes: f32_bytes(&[0.0]),
        dtype: AuxiliaryWeightDType::Float32,
        shape: vec![1],
    }
}

fn reference_dense_matmul(operands: &[NumericTensor]) -> Result<NumericTensor, String> {
    let [input, weight] = operands else {
        return Err(format!(
            "dense matmul reference requires 2 operands, got {}",
            operands.len()
        ));
    };
    if input.shape() != [DENSE_ROWS, DENSE_CONTRACTION]
        || weight.shape() != [DENSE_OUTPUTS, DENSE_CONTRACTION]
    {
        return Err(format!(
            "dense matmul reference shape mismatch: input {:?}, weight {:?}",
            input.shape(),
            weight.shape()
        ));
    }
    let mut output = Vec::with_capacity(DENSE_ROWS * DENSE_OUTPUTS);
    for row in 0..DENSE_ROWS {
        for column in 0..DENSE_OUTPUTS {
            let mut accumulator = 0.0f32;
            for contraction in 0..DENSE_CONTRACTION {
                accumulator += input.values()[row * DENSE_CONTRACTION + contraction]
                    * weight.values()[column * DENSE_CONTRACTION + contraction];
            }
            output.push(accumulator);
        }
    }
    NumericTensor::new("output", vec![DENSE_ROWS, DENSE_OUTPUTS], output)
}

fn residual_operands() -> Result<Vec<NumericTensor>, String> {
    Ok(vec![
        NumericTensor::new(
            "hidden",
            vec![ROWS, FEATURES],
            vec![1.0, -2.0, 3.5, 0.25, -4.0, 8.0, 0.5, -0.125],
        )?,
        NumericTensor::new(
            "residual",
            vec![ROWS, FEATURES],
            vec![-0.5, 1.0, 2.25, -0.75, 3.0, -2.0, 0.125, 4.0],
        )?,
    ])
}

fn reference_residual_add(operands: &[NumericTensor]) -> Result<NumericTensor, String> {
    let [hidden, residual] = operands else {
        return Err(format!(
            "residual add reference requires 2 operands, got {}",
            operands.len()
        ));
    };
    let expected_shape = [ROWS, FEATURES];
    if hidden.shape() != expected_shape || residual.shape() != expected_shape {
        return Err(format!(
            "residual add reference shape mismatch: hidden {:?}, residual {:?}",
            hidden.shape(),
            residual.shape()
        ));
    }
    NumericTensor::new(
        "output",
        expected_shape.to_vec(),
        hidden
            .values()
            .iter()
            .zip(residual.values())
            .map(|(hidden, residual)| hidden + residual)
            .collect(),
    )
}

fn rms_norm_operands() -> Result<Vec<NumericTensor>, String> {
    Ok(vec![
        NumericTensor::new(
            "input",
            vec![ROWS, FEATURES],
            vec![1.0, -2.0, 3.0, -4.0, 0.25, 0.5, -0.75, 1.25],
        )?,
        NumericTensor::new("weight", vec![FEATURES], vec![0.5, 1.0, 1.5, -0.75])?,
    ])
}

fn reference_rms_norm(operands: &[NumericTensor]) -> Result<NumericTensor, String> {
    let [input, weight] = operands else {
        return Err(format!(
            "RMSNorm reference requires 2 operands, got {}",
            operands.len()
        ));
    };
    if input.shape() != [ROWS, FEATURES] || weight.shape() != [FEATURES] {
        return Err(format!(
            "RMSNorm reference shape mismatch: input {:?}, weight {:?}",
            input.shape(),
            weight.shape()
        ));
    }
    let mut output = Vec::with_capacity(ROWS * FEATURES);
    for row in input.values().chunks_exact(FEATURES) {
        let sum_of_squares = row.iter().fold(0.0f32, |sum, value| sum + value * value);
        let inverse_rms = (sum_of_squares / FEATURES as f32 + RMS_EPSILON)
            .sqrt()
            .recip();
        output.extend(
            row.iter()
                .zip(weight.values())
                .map(|(value, weight)| value * inverse_rms * weight),
        );
    }
    NumericTensor::new("output", vec![ROWS, FEATURES], output)
}

fn softmax_operands() -> Result<Vec<NumericTensor>, String> {
    Ok(vec![NumericTensor::new(
        "scores",
        vec![ROWS, FEATURES],
        vec![12.0, 11.0, -3.0, 0.5, -8.0, 0.0, 8.0, 7.5],
    )?])
}

fn reference_softmax(operands: &[NumericTensor]) -> Result<NumericTensor, String> {
    let [scores] = operands else {
        return Err(format!(
            "attention softmax reference requires 1 operand, got {}",
            operands.len()
        ));
    };
    if scores.shape() != [ROWS, FEATURES] {
        return Err(format!(
            "attention softmax reference shape mismatch: scores {:?}",
            scores.shape()
        ));
    }
    let mut output = Vec::with_capacity(ROWS * FEATURES);
    for row in scores.values().chunks_exact(FEATURES) {
        let maximum = row
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, |maximum, value| maximum.max(value));
        let exponentials = row
            .iter()
            .map(|value| (value - maximum).exp())
            .collect::<Vec<_>>();
        let denominator = exponentials.iter().copied().sum::<f32>();
        output.extend(
            exponentials
                .into_iter()
                .map(|exponential| exponential / denominator),
        );
    }
    NumericTensor::new("output", vec![ROWS, FEATURES], output)
}

fn layer_norm_operands() -> Result<Vec<NumericTensor>, String> {
    Ok(vec![
        NumericTensor::new(
            "input",
            vec![ROWS, FEATURES],
            vec![1.0, -2.0, 3.0, -4.0, 0.25, 0.5, -0.75, 1.25],
        )?,
        NumericTensor::new("weight", vec![FEATURES], vec![0.5, 1.0, 1.5, -0.75])?,
        NumericTensor::new("bias", vec![FEATURES], vec![0.25, -0.5, 0.75, 1.0])?,
    ])
}

fn reference_layer_norm(operands: &[NumericTensor]) -> Result<NumericTensor, String> {
    let [input, weight, bias] = operands else {
        return Err(format!(
            "LayerNorm reference requires 3 operands, got {}",
            operands.len()
        ));
    };
    if input.shape() != [ROWS, FEATURES]
        || weight.shape() != [FEATURES]
        || bias.shape() != [FEATURES]
    {
        return Err(format!(
            "LayerNorm reference shape mismatch: input {:?}, weight {:?}, bias {:?}",
            input.shape(),
            weight.shape(),
            bias.shape()
        ));
    }
    let mut output = Vec::with_capacity(ROWS * FEATURES);
    for row in input.values().chunks_exact(FEATURES) {
        let mean = row.iter().copied().sum::<f32>() / FEATURES as f32;
        let centered = row.iter().map(|value| value - mean).collect::<Vec<_>>();
        let variance = centered
            .iter()
            .fold(0.0f32, |sum, value| sum + value * value)
            / FEATURES as f32;
        let inverse_std = (variance + LAYER_NORM_EPSILON).sqrt().recip();
        output.extend(
            centered
                .into_iter()
                .zip(weight.values())
                .zip(bias.values())
                .map(|((value, weight), bias)| value * inverse_std * weight + bias),
        );
    }
    NumericTensor::new("output", vec![ROWS, FEATURES], output)
}

fn projection_operands() -> Result<Vec<NumericTensor>, String> {
    Ok(vec![
        NumericTensor::new(
            "input",
            vec![ROWS, FEATURES],
            vec![1.0, -2.0, 0.5, 3.0, -0.25, 0.75, -1.5, 2.0],
        )?,
        NumericTensor::new(
            "weight",
            vec![PROJECTION_OUTPUTS, FEATURES],
            vec![
                0.5, -1.0, 2.0, 0.25, -2.0, 0.5, 0.75, -0.125, 1.25, 0.0, -0.5, 2.0,
            ],
        )?,
    ])
}

fn reference_projection(operands: &[NumericTensor]) -> Result<Vec<f32>, String> {
    let [input, weight] = operands else {
        return Err(format!(
            "activation projection reference requires 2 operands, got {}",
            operands.len()
        ));
    };
    if input.shape() != [ROWS, FEATURES] || weight.shape() != [PROJECTION_OUTPUTS, FEATURES] {
        return Err(format!(
            "activation projection reference shape mismatch: input {:?}, weight {:?}",
            input.shape(),
            weight.shape()
        ));
    }
    let mut projected = Vec::with_capacity(ROWS * PROJECTION_OUTPUTS);
    for row in 0..ROWS {
        for output in 0..PROJECTION_OUTPUTS {
            let mut accumulator = 0.0f32;
            for feature in 0..FEATURES {
                accumulator += input.values()[row * FEATURES + feature]
                    * weight.values()[output * FEATURES + feature];
            }
            projected.push(accumulator);
        }
    }
    Ok(projected)
}

fn reference_silu_projection(operands: &[NumericTensor]) -> Result<NumericTensor, String> {
    let output = reference_projection(operands)?
        .into_iter()
        .map(|value| {
            let sigmoid = 1.0f32 / (1.0f32 + (-value).exp());
            value * sigmoid
        })
        .collect();
    NumericTensor::new("output", vec![ROWS, PROJECTION_OUTPUTS], output)
}

fn reference_gelu_projection(operands: &[NumericTensor]) -> Result<NumericTensor, String> {
    let output = reference_projection(operands)?
        .into_iter()
        .map(|value| {
            let squared = value * value;
            let cubed = squared * value;
            let nonlinear = 0.044_715f32 * cubed;
            let inner = value + nonlinear;
            let scaled = 0.797_884_6f32 * inner;
            let cdf = 1.0f32 + scaled.tanh();
            let half_value = 0.5f32 * value;
            half_value * cdf
        })
        .collect();
    NumericTensor::new("output", vec![ROWS, PROJECTION_OUTPUTS], output)
}

fn dense_definition() -> Result<ProbeDefinition, String> {
    let operands = dense_operands()?.to_vec();
    Ok(ProbeDefinition {
        operation: DENSE_OPERATION,
        entry: DENSE_MATMUL_ENTRY,
        cache_key: "dense-matmul-f32",
        config_identity: format!(
            "rows={DENSE_ROWS};outputs={DENSE_OUTPUTS};contraction={DENSE_CONTRACTION};dtype=f32"
        ),
        mlir: emit_dense_matmul_probe(DenseMatmulProbeSpec {
            rows: DENSE_ROWS,
            outputs: DENSE_OUTPUTS,
            contraction: DENSE_CONTRACTION,
        })?,
        operator_class: OperatorClass::DenseMatmul,
        rounding_boundaries: vec![
            RoundingBoundary::OperatorInput,
            RoundingBoundary::AccumulatorResult,
            RoundingBoundary::OperatorOutput,
        ],
        association: AssociationPolicy::BackendAlgorithm,
        weights: vec![resident_weight("dense.weight", &operands[1])],
        operands,
        dynamic_operand_indices: vec![0],
        output_shape: vec![DENSE_ROWS, DENSE_OUTPUTS],
        comparison: ComparisonMode::ExactBits,
        reference_algorithm: "row-major-k-sequential-v1",
        candidate_algorithm: "iree-selected-contraction-unobserved",
        reference: reference_dense_matmul,
    })
}

fn residual_definition() -> Result<ProbeDefinition, String> {
    Ok(ProbeDefinition {
        operation: RESIDUAL_OPERATION,
        entry: RESIDUAL_ADD_ENTRY,
        cache_key: "residual-add-f32",
        config_identity: format!("rows={ROWS};features={FEATURES};dtype=f32"),
        mlir: emit_residual_add_probe(RowWiseProbeSpec {
            rows: ROWS,
            features: FEATURES,
        })?,
        operator_class: OperatorClass::ResidualAdd,
        rounding_boundaries: vec![
            RoundingBoundary::OperatorInput,
            RoundingBoundary::ResidualResult,
            RoundingBoundary::OperatorOutput,
        ],
        association: AssociationPolicy::Sequential,
        operands: residual_operands()?,
        weights: vec![dummy_weight("residual_add")],
        dynamic_operand_indices: vec![0, 1],
        output_shape: vec![ROWS, FEATURES],
        comparison: ComparisonMode::ExactBits,
        reference_algorithm: "elementwise-binary-add-v1",
        candidate_algorithm: "iree-selected-residual-add-unobserved",
        reference: reference_residual_add,
    })
}

fn rms_norm_definition() -> Result<ProbeDefinition, String> {
    let operands = rms_norm_operands()?;
    Ok(ProbeDefinition {
        operation: RMS_NORM_OPERATION,
        entry: RMS_NORM_ENTRY,
        cache_key: "rms-norm-f32",
        config_identity: format!(
            "rows={ROWS};features={FEATURES};epsilon={RMS_EPSILON:e};dtype=f32"
        ),
        mlir: emit_rms_norm_probe(
            RowWiseProbeSpec {
                rows: ROWS,
                features: FEATURES,
            },
            RMS_EPSILON,
        )?,
        operator_class: OperatorClass::RmsNorm,
        rounding_boundaries: vec![
            RoundingBoundary::OperatorInput,
            RoundingBoundary::AccumulatorResult,
            RoundingBoundary::OperatorOutput,
        ],
        association: AssociationPolicy::BackendAlgorithm,
        weights: vec![resident_weight("rms_norm.weight", &operands[1])],
        operands,
        dynamic_operand_indices: vec![0],
        output_shape: vec![ROWS, FEATURES],
        comparison: ComparisonMode::AbsoluteRelative {
            absolute: 2e-6,
            relative: 2e-5,
        },
        reference_algorithm: "row-major-square-sum-rsqrt-v1",
        candidate_algorithm: "iree-selected-rms-norm-unobserved",
        reference: reference_rms_norm,
    })
}

fn softmax_definition() -> Result<ProbeDefinition, String> {
    Ok(ProbeDefinition {
        operation: SOFTMAX_OPERATION,
        entry: SOFTMAX_ENTRY,
        cache_key: "attention-softmax-f32",
        config_identity: format!("rows={ROWS};features={FEATURES};dtype=f32"),
        mlir: emit_softmax_probe(RowWiseProbeSpec {
            rows: ROWS,
            features: FEATURES,
        })?,
        operator_class: OperatorClass::AttentionSoftmax,
        rounding_boundaries: vec![
            RoundingBoundary::OperatorInput,
            RoundingBoundary::AccumulatorResult,
            RoundingBoundary::ActivationResult,
            RoundingBoundary::OperatorOutput,
        ],
        association: AssociationPolicy::BackendAlgorithm,
        operands: softmax_operands()?,
        weights: vec![dummy_weight("attention_softmax")],
        dynamic_operand_indices: vec![0],
        output_shape: vec![ROWS, FEATURES],
        comparison: ComparisonMode::AbsoluteRelative {
            absolute: 2e-6,
            relative: 2e-5,
        },
        reference_algorithm: "stable-row-major-softmax-v1",
        candidate_algorithm: "iree-selected-attention-softmax-unobserved",
        reference: reference_softmax,
    })
}

fn layer_norm_definition() -> Result<ProbeDefinition, String> {
    let operands = layer_norm_operands()?;
    Ok(ProbeDefinition {
        operation: LAYER_NORM_OPERATION,
        entry: LAYER_NORM_ENTRY,
        cache_key: "layer-norm-f32",
        config_identity: format!(
            "rows={ROWS};features={FEATURES};epsilon={LAYER_NORM_EPSILON:e};dtype=f32"
        ),
        mlir: emit_layer_norm_probe(
            RowWiseProbeSpec {
                rows: ROWS,
                features: FEATURES,
            },
            LAYER_NORM_EPSILON,
        )?,
        operator_class: OperatorClass::LayerNorm,
        rounding_boundaries: vec![
            RoundingBoundary::OperatorInput,
            RoundingBoundary::AccumulatorResult,
            RoundingBoundary::BiasAdd,
            RoundingBoundary::OperatorOutput,
        ],
        association: AssociationPolicy::BackendAlgorithm,
        weights: vec![
            resident_weight("layer_norm.weight", &operands[1]),
            resident_weight("layer_norm.bias", &operands[2]),
        ],
        operands,
        dynamic_operand_indices: vec![0],
        output_shape: vec![ROWS, FEATURES],
        comparison: ComparisonMode::AbsoluteRelative {
            absolute: 2e-6,
            relative: 2e-5,
        },
        reference_algorithm: "row-major-centered-square-sum-rsqrt-v1",
        candidate_algorithm: "iree-selected-layer-norm-unobserved",
        reference: reference_layer_norm,
    })
}

fn activation_projection_definition(
    operation: &'static str,
    entry: &'static str,
    cache_key: &'static str,
    operator_class: OperatorClass,
    candidate_algorithm: &'static str,
    emitter: fn(DenseMatmulProbeSpec) -> Result<String, String>,
    reference: fn(&[NumericTensor]) -> Result<NumericTensor, String>,
) -> Result<ProbeDefinition, String> {
    let operands = projection_operands()?;
    let spec = DenseMatmulProbeSpec {
        rows: ROWS,
        outputs: PROJECTION_OUTPUTS,
        contraction: FEATURES,
    };
    Ok(ProbeDefinition {
        operation,
        entry,
        cache_key,
        config_identity: format!(
            "rows={ROWS};outputs={PROJECTION_OUTPUTS};contraction={FEATURES};dtype=f32"
        ),
        mlir: emitter(spec)?,
        operator_class,
        rounding_boundaries: vec![
            RoundingBoundary::OperatorInput,
            RoundingBoundary::AccumulatorResult,
            RoundingBoundary::ActivationResult,
            RoundingBoundary::OperatorOutput,
        ],
        association: AssociationPolicy::BackendAlgorithm,
        weights: vec![resident_weight(
            &format!("{cache_key}.weight"),
            &operands[1],
        )],
        operands,
        dynamic_operand_indices: vec![0],
        output_shape: vec![ROWS, PROJECTION_OUTPUTS],
        comparison: ComparisonMode::AbsoluteRelative {
            absolute: 3e-6,
            relative: 3e-5,
        },
        reference_algorithm: "row-major-projection-then-activation-v1",
        candidate_algorithm,
        reference,
    })
}

fn silu_projection_definition() -> Result<ProbeDefinition, String> {
    activation_projection_definition(
        SILU_PROJECTION_OPERATION,
        SILU_PROJECTION_ENTRY,
        "silu-projection-f32",
        OperatorClass::SiluProjection,
        "iree-selected-silu-projection-unobserved",
        emit_silu_projection_probe,
        reference_silu_projection,
    )
}

fn gelu_projection_definition() -> Result<ProbeDefinition, String> {
    activation_projection_definition(
        GELU_PROJECTION_OPERATION,
        GELU_PROJECTION_ENTRY,
        "gelu-projection-f32",
        OperatorClass::GeluProjection,
        "iree-selected-gelu-projection-unobserved",
        emit_gelu_projection_probe,
        reference_gelu_projection,
    )
}

fn checked_output_bytes(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(size_of::<f32>(), |bytes, dimension| {
        if *dimension == 0 {
            return Err("numeric probe output dimensions must be non-zero".to_string());
        }
        bytes
            .checked_mul(*dimension)
            .ok_or_else(|| "numeric probe output byte count overflowed".to_string())
    })
}

fn execute_probe(device: &str, definition: ProbeDefinition) -> Result<NumericOracleReport, String> {
    if !device.starts_with("local") {
        return Err(format!(
            "canonical operator probes support only local IREE CPU targets, got {device:?}"
        ));
    }
    if definition.dynamic_operand_indices.is_empty() {
        return Err(format!(
            "{} requires at least one dynamic operand",
            definition.operation
        ));
    }
    for &index in &definition.dynamic_operand_indices {
        if index >= definition.operands.len() {
            return Err(format!(
                "{} dynamic operand index {index} exceeds {} operands",
                definition.operation,
                definition.operands.len()
            ));
        }
    }
    let output_bytes = checked_output_bytes(&definition.output_shape)?;
    let compiler = iree_compile_bin()?;
    let flags = target_flags(device)?;
    let compiler_version = compiler_version(&compiler)?;
    let runtime_identity = runtime_archive_identity(&compiler)?;
    let operator_contract =
        operator_identity(&compiler_version, &runtime_identity, device, &definition)?;
    let generation_identity = format!(
        "compiler={};version={};runtime={runtime_identity};flags={flags:?};mlir_sha256={}",
        compiler.display(),
        compiler_version,
        sha256_hex(definition.mlir.as_bytes())
    );
    let ProbeDefinition {
        operation,
        entry,
        cache_key,
        config_identity,
        mlir,
        operands,
        weights,
        dynamic_operand_indices,
        output_shape,
        comparison,
        reference_algorithm,
        reference,
        ..
    } = definition;
    let artifact_contract = AuxiliaryArtifactContract::new_with_operator_numeric_contract(
        entry,
        config_identity,
        generation_identity,
        operator_contract.clone(),
    )?;
    let cache = std::env::temp_dir().join("mlxcel-xla-numeric-probes");
    std::fs::create_dir_all(&cache)
        .map_err(|error| format!("mkdir {}: {error}", cache.display()))?;
    let vmfb = cached_vmfb_path(&compiler, &mlir, flags, &cache, cache_key, 0);
    ensure_qualified_auxiliary_artifact(&vmfb, &artifact_contract, &weights, |output| {
        compile_one_to(&compiler, &mlir, flags, &cache, cache_key, 0, output)
    })?;
    let mut module = IreeAuxiliaryModule::load(device, &vmfb, &artifact_contract, weights)?;
    let artifact_fingerprint = format!("{:016x}", module.fingerprint());
    let reference_backend = NumericBackendIdentity::new(
        "rust-canonical-decomposition",
        env!("CARGO_PKG_VERSION"),
        format!("{cache_key}-v1"),
        "rust-f32",
        vec![reference_algorithm.to_string()],
        std::env::consts::ARCH,
        "host-cpu",
        format!("rust-{cache_key}-v1"),
        AlgorithmIdentity::observed(reference_algorithm)?,
    )?;
    let candidate_backend = NumericBackendIdentity::new(
        "iree",
        compiler_version,
        runtime_identity,
        compiler.display().to_string(),
        flags.iter().map(|flag| (*flag).to_string()).collect(),
        device,
        device,
        artifact_fingerprint,
        AlgorithmIdentity::unobserved(
            "IREE does not expose the selected CPU operator kernel identity",
        )?,
    )?;
    let case = NumericOracleCase::from_operator_contract(
        operation,
        NumericOracleClaim::CanonicalDecomposition,
        &operator_contract,
        reference_backend,
        candidate_backend,
        comparison,
    )?;
    run_bounded_numeric_oracle(case, &operands, reference, move |operands| {
        let input_storage = dynamic_operand_indices
            .iter()
            .map(|&index| f32_bytes(operands[index].values()))
            .collect::<Vec<_>>();
        let inputs = dynamic_operand_indices
            .iter()
            .zip(&input_storage)
            .map(|(&index, bytes)| AuxiliaryInput {
                bytes,
                dtype: AuxiliaryTensorDType::Float32,
                shape: operands[index].shape(),
            })
            .collect::<Vec<_>>();
        let mut output_storage = vec![0u8; output_bytes];
        module.invoke(
            &inputs,
            &mut [AuxiliaryOutput {
                bytes: &mut output_storage,
                dtype: AuxiliaryTensorDType::Float32,
                shape: &output_shape,
            }],
        )?;
        NumericTensor::new("output", output_shape, decode_f32(&output_storage)?)
    })
}

/// Compile and execute one deterministic dense-matmul probe through the native
/// auxiliary IREE ABI.
///
/// The reference is an explicitly reported canonical Rust decomposition, not a
/// production MLX kernel. Consequently this probe can validate the shared
/// emitter/runtime path but can never qualify a model family for production.
pub fn run_dense_matmul_probe(device: &str) -> Result<NumericOracleReport, String> {
    execute_probe(device, dense_definition()?)
}

/// Run the bounded CPU operator suite through real in-process IREE
/// compile/load/invoke boundaries.
///
/// Every report uses a canonical Rust decomposition and therefore remains
/// deliberately ineligible for production-family qualification.
pub fn run_core_operator_probes(device: &str) -> Result<Vec<NumericOracleReport>, String> {
    [
        dense_definition,
        residual_definition,
        rms_norm_definition,
        softmax_definition,
        layer_norm_definition,
        silu_projection_definition,
        gelu_projection_definition,
    ]
    .into_iter()
    .map(|definition| execute_probe(device, definition()?))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_dense_matmul_has_stable_expected_values() {
        let operands = dense_operands().unwrap();
        let output = reference_dense_matmul(&operands).unwrap();
        assert_eq!(output.values(), [-0.5, -6.5, -3.0, 18.125]);
    }

    #[test]
    fn canonical_residual_add_has_stable_expected_values() {
        let operands = residual_operands().unwrap();
        let output = reference_residual_add(&operands).unwrap();
        assert_eq!(
            output.values(),
            [0.5, -1.0, 5.75, -0.5, -1.0, 6.0, 0.625, 3.875]
        );
    }

    #[test]
    fn canonical_rms_norm_outputs_are_finite() {
        let operands = rms_norm_operands().unwrap();
        let output = reference_rms_norm(&operands).unwrap();
        assert!(output.values().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn canonical_softmax_rows_sum_to_one() {
        let operands = softmax_operands().unwrap();
        let output = reference_softmax(&operands).unwrap();
        for row in output.values().chunks_exact(FEATURES) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() <= f32::EPSILON);
        }
    }

    #[test]
    fn malformed_f32_output_and_byte_counts_fail_closed() {
        assert!(decode_f32(&[0, 0, 0]).is_err());
        assert!(checked_output_bytes(&[1, 0]).is_err());
        assert!(checked_output_bytes(&[usize::MAX, 2]).is_err());
    }

    #[test]
    fn canonical_layer_norm_and_activation_projections_are_finite() {
        let layer_norm = layer_norm_operands().unwrap();
        assert!(
            reference_layer_norm(&layer_norm)
                .unwrap()
                .values()
                .iter()
                .all(|value| value.is_finite())
        );
        let projection = projection_operands().unwrap();
        for output in [
            reference_silu_projection(&projection).unwrap(),
            reference_gelu_projection(&projection).unwrap(),
        ] {
            assert!(output.values().iter().all(|value| value.is_finite()));
        }
    }
}
