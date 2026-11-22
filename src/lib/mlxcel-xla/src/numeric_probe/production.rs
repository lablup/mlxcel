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

//! Production-MLX micro-oracles: the emitted IREE operation against a
//! caller-supplied reference on the same operands (issue #3212).
//!
//! `mlxcel-xla` never calls MLX. The root crate passes the MLX operation as a
//! closure together with its backend identity, and every report claims
//! [`NumericOracleClaim::ProductionMlx`]. The artifact contract comes from the
//! production operator builder (`production_operator_contract`), so a probe
//! artifact carries the same operator identity scheme the production v3
//! manifests do. The IREE dispatch kernel stays unobserved, which keeps
//! `production_qualified` false; the claim records which reference was used,
//! not a family qualification.

use serde::Serialize;

use super::{
    ProbeArtifact, ProbeToolchain, dense_definition, execute_artifact, f16_bytes,
    gelu_projection_definition, layer_norm_definition, prefix_sum_definition, resident_weight,
    residual_definition, rms_norm_definition, silu_projection_definition, softmax_definition,
    u32_bytes,
};
use crate::auxiliary::{AuxiliaryWeight, AuxiliaryWeightDType, AuxiliaryWeightStorage};
use crate::emitter::Precision;
use crate::emitter::production_probe_ops::{
    AFFINE_Q4_MATMUL_ENTRY, AffineQ4MatmulProbeSpec, SSCP_CONVOLUTION_ENTRY,
    SscpConvolutionProbeSpec, emit_affine_q4_matmul_probe, emit_sscp_convolution_probe,
};
use crate::numeric_dtype_contract::NumericDType;
use crate::numeric_oracle::{
    ComparisonMode, NumericBackendIdentity, NumericOracleClaim, NumericOracleReport, NumericTensor,
};
use crate::operator_numeric_contract::{
    AffineDequantizationContract, AffineEvaluationOrder, AffineExecution, AssociationPolicy,
    CheckpointDType, CheckpointWeights, DeclaredOperator, InputPolicy, OperandOrigin,
    OperatorClass, OperatorNumericContractSet, PackedLaneOrder, ProductionBackend,
    gemma3n_audio_encode_operators, production_operator_contract,
};

#[path = "production_fixtures.rs"]
mod fixtures;

/// Operator family recorded in every production probe contract entry.
const PROBE_FAMILY: &str = "micro-oracle";

/// `--atol` / `--rtol` defaults of the Qwen2.5-VL window and Qwen3-VL
/// DeepStack same-input controls (`examples/xla_qwen25_vl_window_check.rs`,
/// `examples/xla_qwen3_vl_deepstack_check.rs`), the families that consume the
/// production Q4 projections. Not a new tolerance.
const QWEN_SAME_INPUT_TOLERANCE: ComparisonMode = ComparisonMode::AbsoluteRelative {
    absolute: 0.08,
    relative: 0.08,
};

/// The Gemma3n audio intermediate gate's per-stage `max_abs` bound
/// (`examples/xla_gemma3n_audio_reference_check.rs`). Its RMS bound has no
/// element-wise equivalent and is reported through the comparison summary.
const GEMMA3N_AUDIO_STAGE_TOLERANCE: ComparisonMode = ComparisonMode::AbsoluteRelative {
    absolute: 5e-3,
    relative: 0.0,
};

const Q4_GROUP_SIZE: usize = 64;
const Q4_OUTPUTS: usize = 128;
const Q4_INPUTS: usize = 256;
/// MLX CUDA keeps `M * B < 8` on qmv and takes `qmm_sm80` from eight rows.
const Q4_QMM_ROWS: usize = 16;

/// The released Gemma3n audio subsampler's first convolution:
/// 128 mel bins, 128 output channels, 3x3 kernel, 2x2 stride.
const SSCP_SPEC: SscpConvolutionProbeSpec = SscpConvolutionProbeSpec {
    time: 32,
    frequency: 128,
    channels: 128,
    kernel: [3, 3],
    stride: [2, 2],
};

/// One production MLX operation the micro-oracle compares IREE against.
///
/// Every variant documents the operand order its [`ProductionProbeSpec`]
/// carries and the MLX operation the reference must run on them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum ProductionProbeOperator {
    /// `[input [2,3], weight [2,3]]`: `matmul(input, weight^T)` in F32.
    DenseMatmul,
    /// `[input [1,256], lanes [128,256], scales [128,4], biases [128,4]]`:
    /// F16 `quantized_matmul(..., transpose=true, group_size=64, bits=4)` on
    /// the [`pack_affine_q4_lanes`] packing, one row (MLX CUDA qmv).
    AffineQ4Qmv,
    /// As [`Self::AffineQ4Qmv`] with 16 rows (MLX CUDA `qmm_sm80`).
    AffineQ4Qmm,
    /// `[input [2,4], weight [4]]`: `fast_rms_norm(input, weight, epsilon)`.
    RmsNorm,
    /// `[input [2,4], weight [4], bias [4]]`: `fast_layer_norm` with epsilon.
    LayerNorm,
    /// `[scores [2,4]]`: `softmax(scores, axis=-1)`.
    AttentionSoftmax,
    /// `[input [2,4], weight [3,4]]`: `silu(matmul(input, weight^T))`.
    SiluProjection,
    /// `[input [2,4], weight [3,4]]`: tanh GELU of `matmul(input, weight^T)`.
    GeluProjection,
    /// `[hidden [2,4], residual [2,4]]`: `add(hidden, residual)`.
    ResidualAdd,
    /// `[input [2,130]]`: inclusive forward `cumsum(input, axis=1)`.
    PrefixSum,
    /// `[mel [1,32,128,1], weight [128,3,3,1]]`: BF16 cast of both, then
    /// [`SscpConvolutionGeometry`] padding and a strided `conv2d`, BF16 output.
    Gemma3nSscpConvolution,
}

impl ProductionProbeOperator {
    /// Every operator, in report order.
    pub const ALL: [Self; 11] = [
        Self::DenseMatmul,
        Self::AffineQ4Qmv,
        Self::AffineQ4Qmm,
        Self::RmsNorm,
        Self::LayerNorm,
        Self::AttentionSoftmax,
        Self::SiluProjection,
        Self::GeluProjection,
        Self::ResidualAdd,
        Self::PrefixSum,
        Self::Gemma3nSscpConvolution,
    ];

    /// Stable command-line name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::DenseMatmul => "dense-matmul",
            Self::AffineQ4Qmv => "q4-qmv",
            Self::AffineQ4Qmm => "q4-qmm",
            Self::RmsNorm => "rms-norm",
            Self::LayerNorm => "layer-norm",
            Self::AttentionSoftmax => "softmax",
            Self::SiluProjection => "silu-projection",
            Self::GeluProjection => "gelu-projection",
            Self::ResidualAdd => "residual-add",
            Self::PrefixSum => "cumsum",
            Self::Gemma3nSscpConvolution => "gemma3n-sscp-conv2d",
        }
    }

    /// Parse a [`Self::name`].
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|operator| operator.name() == name)
    }

    /// Whether the operator runs the affine Q4 kernels, whose evidence is
    /// kept apart from the dense operators.
    #[must_use]
    pub const fn is_affine_q4(self) -> bool {
        matches!(self, Self::AffineQ4Qmv | Self::AffineQ4Qmm)
    }
}

/// Padding and stride the reference applies to the first SSCP convolution,
/// in MLX's NHWC layout (`[time, frequency]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct SscpConvolutionGeometry {
    pub padding_low: [usize; 2],
    pub padding_high: [usize; 2],
    pub stride: [usize; 2],
}

/// Operands and reference parameters of one production probe.
#[derive(Clone, Debug, PartialEq)]
pub struct ProductionProbeSpec {
    operator: ProductionProbeOperator,
    operands: Vec<NumericTensor>,
    epsilon: Option<f32>,
    group_size: Option<usize>,
    convolution: Option<SscpConvolutionGeometry>,
}

impl ProductionProbeSpec {
    #[must_use]
    pub fn operator(&self) -> ProductionProbeOperator {
        self.operator
    }

    /// The operands, in the order the operator's documentation names.
    #[must_use]
    pub fn operands(&self) -> &[NumericTensor] {
        &self.operands
    }

    /// Normalization epsilon for [`ProductionProbeOperator::RmsNorm`] and
    /// [`ProductionProbeOperator::LayerNorm`].
    #[must_use]
    pub fn epsilon(&self) -> Option<f32> {
        self.epsilon
    }

    /// Affine group size for the Q4 operators.
    #[must_use]
    pub fn group_size(&self) -> Option<usize> {
        self.group_size
    }

    /// Geometry for [`ProductionProbeOperator::Gemma3nSscpConvolution`].
    #[must_use]
    pub fn convolution(&self) -> Option<SscpConvolutionGeometry> {
        self.convolution
    }
}

/// Pack `[outputs, inputs]` integer lanes in `0..=15` into MLX's
/// least-significant-lane-first `[outputs, inputs / 8]` u32 layout.
pub fn pack_affine_q4_lanes(lanes: &NumericTensor) -> Result<Vec<u32>, String> {
    let [_, inputs] = lanes.shape() else {
        return Err(format!(
            "affine Q4 lanes must be rank-2, got {:?}",
            lanes.shape()
        ));
    };
    if !inputs.is_multiple_of(8) {
        return Err(format!(
            "affine Q4 lane width {inputs} must be divisible by 8"
        ));
    }
    lanes
        .values()
        .chunks_exact(8)
        .map(|chunk| {
            chunk
                .iter()
                .enumerate()
                .try_fold(0u32, |packed, (lane, value)| {
                    if !value.is_finite() || value.fract() != 0.0 || !(0.0..=15.0).contains(value) {
                        return Err(format!("invalid affine Q4 lane value {value}"));
                    }
                    Ok(packed | (*value as u32) << (lane * 4))
                })
        })
        .collect()
}

/// Operands and parameters for `operator`, so a caller can build the matching
/// reference before running it.
pub fn production_probe_spec(
    operator: ProductionProbeOperator,
) -> Result<ProductionProbeSpec, String> {
    Ok(production_probe(operator)?.spec)
}

/// Run `operator` with the emitted IREE operation on `device` (`local-*`,
/// `cuda`, or `hip`) against `reference`, which receives the operands of
/// [`production_probe_spec`] and returns the production result.
///
/// `reference_backend` names the reference runtime, revision, patch, build
/// flags, target, and device; it is recorded verbatim in the report.
pub fn run_production_probe<R>(
    device: &str,
    operator: ProductionProbeOperator,
    reference_backend: NumericBackendIdentity,
    reference: R,
) -> Result<NumericOracleReport, String>
where
    R: FnOnce(&[NumericTensor]) -> Result<NumericTensor, String>,
{
    let toolchain = ProbeToolchain::resolve(device)?;
    let probe = production_probe(operator)?;
    let backend = ProductionBackend::new(
        toolchain.compiler_version.clone(),
        toolchain.runtime_identity.clone(),
        device,
    );
    let operator_contract = probe.contract(&backend)?;
    execute_artifact(
        &toolchain,
        probe.artifact,
        NumericOracleClaim::ProductionMlx,
        &operator_contract,
        reference_backend,
        reference,
    )
}

struct ProductionProbe {
    spec: ProductionProbeSpec,
    artifact: ProbeArtifact,
    operators: Vec<DeclaredOperator>,
    precision: Precision,
    weights: CheckpointWeights,
}

impl ProductionProbe {
    fn contract(&self, backend: &ProductionBackend) -> Result<OperatorNumericContractSet, String> {
        production_operator_contract(
            PROBE_FAMILY,
            &self.operators,
            self.precision,
            self.weights,
            backend,
        )
    }
}

const fn weighted(id: &'static str, class: OperatorClass) -> DeclaredOperator {
    DeclaredOperator::new(
        id,
        class,
        OperandOrigin::MatrixWeight,
        InputPolicy::FollowsPrecision,
        AssociationPolicy::BackendAlgorithm,
    )
}

const fn activation(
    id: &'static str,
    class: OperatorClass,
    input: InputPolicy,
    association: AssociationPolicy,
) -> DeclaredOperator {
    DeclaredOperator::new(id, class, OperandOrigin::Activation, input, association)
}

fn production_probe(operator: ProductionProbeOperator) -> Result<ProductionProbe, String> {
    use ProductionProbeOperator as Op;
    let f32_weights = CheckpointWeights::Float(CheckpointDType::F32);
    let norm = |id, class| {
        DeclaredOperator::new(
            id,
            class,
            OperandOrigin::FloatParameter,
            InputPolicy::F32Carrier,
            AssociationPolicy::BackendAlgorithm,
        )
    };
    let (definition, declared) = match operator {
        Op::AffineQ4Qmv => return affine_q4_probe(operator, 1),
        Op::AffineQ4Qmm => return affine_q4_probe(operator, Q4_QMM_ROWS),
        Op::Gemma3nSscpConvolution => return sscp_convolution_probe(),
        Op::DenseMatmul => (
            dense_definition()?,
            weighted("dense-matmul", OperatorClass::DenseMatmul),
        ),
        Op::RmsNorm => (
            rms_norm_definition()?,
            norm("rms-norm", OperatorClass::RmsNorm),
        ),
        Op::LayerNorm => (
            layer_norm_definition()?,
            norm("layer-norm", OperatorClass::LayerNorm),
        ),
        Op::AttentionSoftmax => (
            softmax_definition()?,
            activation(
                "attention-softmax",
                OperatorClass::AttentionSoftmax,
                InputPolicy::F32Carrier,
                AssociationPolicy::BackendAlgorithm,
            ),
        ),
        Op::SiluProjection => (
            silu_projection_definition()?,
            weighted("silu-projection", OperatorClass::SiluProjection),
        ),
        Op::GeluProjection => (
            gelu_projection_definition()?,
            weighted("gelu-projection", OperatorClass::GeluProjection),
        ),
        Op::ResidualAdd => (
            residual_definition()?,
            activation(
                "residual-add",
                OperatorClass::ResidualAdd,
                InputPolicy::F32Carrier,
                AssociationPolicy::Sequential,
            ),
        ),
        Op::PrefixSum => (
            prefix_sum_definition()?,
            activation(
                "cumulative-sum",
                OperatorClass::PrefixReduction,
                InputPolicy::PinnedF32,
                AssociationPolicy::CudaOneBlockScan,
            ),
        ),
    };
    let epsilon = match operator {
        Op::RmsNorm => Some(super::RMS_EPSILON),
        Op::LayerNorm => Some(super::LAYER_NORM_EPSILON),
        _ => None,
    };
    let artifact = definition.into_artifact();
    Ok(ProductionProbe {
        spec: ProductionProbeSpec {
            operator,
            operands: artifact.operands.clone(),
            epsilon,
            group_size: None,
            convolution: None,
        },
        artifact,
        operators: vec![declared],
        precision: Precision::F32,
        weights: f32_weights,
    })
}

fn affine_q4_probe(
    operator: ProductionProbeOperator,
    rows: usize,
) -> Result<ProductionProbe, String> {
    let spec = AffineQ4MatmulProbeSpec {
        rows,
        outputs: Q4_OUTPUTS,
        inputs: Q4_INPUTS,
        group_size: Q4_GROUP_SIZE,
    };
    let operands = fixtures::affine_q4_operands(rows, Q4_OUTPUTS, Q4_INPUTS, Q4_GROUP_SIZE)?;
    let [_, lanes, scales, biases] = operands.as_slice() else {
        return Err("affine Q4 fixture must carry four operands".to_string());
    };
    let weights = vec![
        AuxiliaryWeight {
            name: "affine_q4_matmul.packed".to_string(),
            storage: AuxiliaryWeightStorage::Bytes(u32_bytes(&pack_affine_q4_lanes(lanes)?)),
            dtype: AuxiliaryWeightDType::Uint32,
            shape: vec![Q4_OUTPUTS, Q4_INPUTS / 8],
        },
        f16_weight("affine_q4_matmul.scales", scales),
        f16_weight("affine_q4_matmul.biases", biases),
    ];
    let dequantization = AffineDequantizationContract::new(
        4,
        Q4_GROUP_SIZE,
        NumericDType::U32,
        NumericDType::F16,
        PackedLaneOrder::LeastSignificantFirst,
        AffineEvaluationOrder::SeparateMultiplyThenAdd,
    )?;
    let (operation, cache_key) = match operator {
        ProductionProbeOperator::AffineQ4Qmv => (
            "micro-oracle.affine-q4-qmv.f16",
            "production-affine-q4-qmv-f16",
        ),
        _ => (
            "micro-oracle.affine-q4-qmm.f16",
            "production-affine-q4-qmm-f16",
        ),
    };
    let artifact = ProbeArtifact {
        operation,
        entry: AFFINE_Q4_MATMUL_ENTRY,
        cache_key,
        config_identity: format!(
            "rows={rows};outputs={Q4_OUTPUTS};inputs={Q4_INPUTS};bits=4;group={Q4_GROUP_SIZE};\
             metadata=f16;precision=f16;result=f16"
        ),
        mlir: emit_affine_q4_matmul_probe(spec)?,
        operands: operands.clone(),
        weights,
        dynamic_operand_indices: vec![0],
        output_shape: vec![rows, Q4_OUTPUTS],
        comparison: QWEN_SAME_INPUT_TOLERANCE,
    };
    Ok(ProductionProbe {
        spec: ProductionProbeSpec {
            operator,
            operands,
            epsilon: None,
            group_size: Some(Q4_GROUP_SIZE),
            convolution: None,
        },
        artifact,
        operators: vec![
            DeclaredOperator::new(
                "q4-weight-dequantize",
                OperatorClass::AffineQ4Dequantize,
                OperandOrigin::MatrixWeight,
                InputPolicy::F32Carrier,
                AssociationPolicy::Sequential,
            ),
            weighted("q4-projection", OperatorClass::DenseMatmul).rounded_to(NumericDType::F16),
        ],
        precision: Precision::F16,
        weights: CheckpointWeights::Affine {
            float: CheckpointDType::F16,
            dequantization,
            execution: AffineExecution::InGraph,
        },
    })
}

fn sscp_convolution_probe() -> Result<ProductionProbe, String> {
    let operands = fixtures::sscp_convolution_operands(SSCP_SPEC)?;
    let declared = gemma3n_audio_encode_operators()
        .into_iter()
        .find(|operator| operator.id == "subsample.convolution")
        .ok_or_else(|| "Gemma3n audio declares no subsample.convolution".to_string())?;
    let SscpConvolutionProbeSpec {
        time,
        frequency,
        channels,
        kernel,
        stride,
    } = SSCP_SPEC;
    let artifact = ProbeArtifact {
        operation: "micro-oracle.gemma3n-sscp-conv2d.bf16",
        entry: SSCP_CONVOLUTION_ENTRY,
        cache_key: "production-gemma3n-sscp-conv2d-bf16",
        config_identity: format!(
            "time={time};frequency={frequency};channels={channels};kernel={kernel:?};\
             stride={stride:?};pad-time=0,{};pad-frequency=1,1;precision=bf16;result=bf16",
            kernel[0] - 1
        ),
        mlir: emit_sscp_convolution_probe(SSCP_SPEC)?,
        weights: vec![resident_weight("gemma3n_sscp.conv_0.weight", &operands[1])],
        operands: operands.clone(),
        dynamic_operand_indices: vec![0],
        output_shape: SSCP_SPEC.output_shape(),
        comparison: GEMMA3N_AUDIO_STAGE_TOLERANCE,
    };
    Ok(ProductionProbe {
        spec: ProductionProbeSpec {
            operator: ProductionProbeOperator::Gemma3nSscpConvolution,
            operands,
            epsilon: None,
            group_size: None,
            convolution: Some(SscpConvolutionGeometry {
                padding_low: [0, 1],
                padding_high: [kernel[0] - 1, 1],
                stride,
            }),
        },
        artifact,
        operators: vec![declared],
        precision: Precision::Bf16,
        weights: CheckpointWeights::Float(CheckpointDType::Bf16),
    })
}

fn f16_weight(name: &str, tensor: &NumericTensor) -> AuxiliaryWeight {
    AuxiliaryWeight {
        name: name.to_string(),
        storage: AuxiliaryWeightStorage::Bytes(f16_bytes(tensor.values())),
        dtype: AuxiliaryWeightDType::Float16,
        shape: tensor.shape().to_vec(),
    }
}

#[cfg(test)]
#[path = "production_tests.rs"]
mod tests;
