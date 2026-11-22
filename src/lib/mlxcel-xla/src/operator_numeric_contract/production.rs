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

//! Builder that binds a production family's declared operators to one
//! [`OperatorNumericContractSet`] (issue #3211).
//!
//! A family declares which operator classes its StableHLO graph emits and how
//! each one reads its operands. The builder derives every dtype field from the
//! same [`Precision`] value the emitter used, the checkpoint weights the loader
//! observed, and the IREE backend identity, so changing any of them changes the
//! v3 manifest digest and forces a recompile before the module is invoked.
//!
//! This is artifact identity only. The kernel or schedule IREE selects for each
//! dispatch is not observable from the compiled module, so every entry records
//! it as explicitly unobserved; nothing here claims oracle qualification.

use crate::emitter::Precision;
use crate::numeric_dtype_contract::{NumericDType, WeightExecution};

use super::{
    AffineDequantizationContract, AffineEvaluationOrder, AssociationPolicy, BackendNumericIdentity,
    CheckpointDType, OperatorClass, OperatorNumericContract, OperatorNumericContractSet,
    PackedLaneOrder, RoundingBoundary,
};

/// Revision of the production StableHLO graph authoring conventions recorded in
/// every production operator entry. Bump it when a builder-level rule changes
/// the emitted numerics for every family (for example the contraction demotion
/// rule in `Builder::dot_general`).
const PRODUCTION_EMITTER_REVISION: &str = "mlxcel-xla-stablehlo-emitter-v1";

/// IREE exposes no per-dispatch kernel identity for a compiled VMFB.
const UNOBSERVED_ALGORITHM: &str = "unobserved-iree-dispatch-selection";

/// IREE compiler, runtime, and HAL device that build and run one artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ProductionBackend {
    compiler_version: String,
    runtime_identity: String,
    device: String,
}

impl ProductionBackend {
    pub(crate) fn new(
        compiler_version: impl Into<String>,
        runtime_identity: impl Into<String>,
        device: impl Into<String>,
    ) -> Self {
        Self {
            compiler_version: compiler_version.into(),
            runtime_identity: runtime_identity.into(),
            device: device.into(),
        }
    }

    fn identity(&self) -> Result<BackendNumericIdentity, String> {
        BackendNumericIdentity::new(
            PRODUCTION_EMITTER_REVISION,
            self.runtime_identity.as_str(),
            // `--version` output can carry the identity separator.
            self.compiler_version.replace(';', ","),
            self.device.as_str(),
            UNOBSERVED_ALGORITHM,
        )
    }
}

/// Where an affine-quantized checkpoint weight is dequantized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AffineExecution {
    /// The loader dequantizes to f32 and uploads dense weights.
    Host,
    /// The weight stays packed and `Builder::dequant_affine` unpacks it in the graph.
    InGraph,
}

/// Checkpoint dtype classes observed while a family loads its tensors.
///
/// Loaders call [`Self::observe_float`] for every floating tensor and
/// [`Self::observe_affine`] for every packed affine tensor, then
/// [`Self::resolve`] turns the observations into the [`CheckpointWeights`] the
/// contract binds. The exact per-tensor source schema stays in the config
/// identity; this records the dtype classes the operator entries name.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CheckpointSourceObserver {
    f16: usize,
    bf16: usize,
    f32: usize,
    affine: Option<AffineSource>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AffineSource {
    bits: u8,
    group_size: usize,
    scale_bias_dtype: NumericDType,
}

impl CheckpointSourceObserver {
    pub(crate) fn observe_float(&mut self, dtype: CheckpointDType) -> Result<(), String> {
        match dtype {
            CheckpointDType::F16 => self.f16 += 1,
            CheckpointDType::Bf16 => self.bf16 += 1,
            CheckpointDType::F32 => self.f32 += 1,
            CheckpointDType::AffineU4 | CheckpointDType::AffineU8 => {
                return Err(format!(
                    "{} is not a floating checkpoint dtype",
                    dtype.identity()
                ));
            }
        }
        Ok(())
    }

    /// Record one packed affine tensor. Every affine tensor of one artifact must
    /// share one bit width, group size, and scale/bias dtype.
    pub(crate) fn observe_affine(
        &mut self,
        bits: usize,
        group_size: usize,
        scale_bias_dtype: NumericDType,
    ) -> Result<(), String> {
        let bits = u8::try_from(bits).map_err(|_| format!("affine bits {bits} out of range"))?;
        let source = AffineSource {
            bits,
            group_size,
            scale_bias_dtype,
        };
        match self.affine {
            None => self.affine = Some(source),
            Some(existing) if existing == source => {}
            Some(existing) => {
                return Err(format!(
                    "checkpoint mixes affine layouts {existing:?} and {source:?} in one artifact"
                ));
            }
        }
        Ok(())
    }

    /// The floating checkpoint dtype the entries name: the most frequent class,
    /// ties resolved toward the wider type so the choice is deterministic.
    fn float_dtype(&self) -> Result<CheckpointDType, String> {
        // `max_by_key` keeps the last maximum, so list the widest dtype last.
        [
            (self.f16, CheckpointDType::F16),
            (self.bf16, CheckpointDType::Bf16),
            (self.f32, CheckpointDType::F32),
        ]
        .into_iter()
        .filter(|(count, _)| *count > 0)
        .max_by_key(|(count, _)| *count)
        .map(|(_, dtype)| dtype)
        .ok_or_else(|| "checkpoint source has no floating tensors".to_string())
    }

    pub(crate) fn resolve(&self, execution: AffineExecution) -> Result<CheckpointWeights, String> {
        let float = self.float_dtype()?;
        let Some(source) = self.affine else {
            return Ok(CheckpointWeights::Float(float));
        };
        let evaluation = AffineEvaluationOrder::SeparateMultiplyThenAdd;
        let dequantization = AffineDequantizationContract::new(
            source.bits,
            source.group_size,
            NumericDType::U32,
            source.scale_bias_dtype,
            PackedLaneOrder::LeastSignificantFirst,
            evaluation,
        )?;
        Ok(CheckpointWeights::Affine {
            float,
            dequantization,
            execution,
        })
    }
}

/// Checkpoint dtype classes one artifact's operators read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CheckpointWeights {
    /// Every tensor is floating and reaches the graph as dense f32.
    Float(CheckpointDType),
    /// Matrix weights are affine-quantized; parameters stay `float`.
    Affine {
        float: CheckpointDType,
        dequantization: AffineDequantizationContract,
        execution: AffineExecution,
    },
}

impl CheckpointWeights {
    fn float(self) -> CheckpointDType {
        match self {
            Self::Float(float) | Self::Affine { float, .. } => float,
        }
    }
}

/// What a declared operator reads from the checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OperandOrigin {
    /// A projection weight matrix, the tensor an affine checkpoint quantizes.
    MatrixWeight,
    /// Floating parameters (norm scale and shift, convolution kernels, biases).
    FloatParameter,
    /// Only activations produced inside the graph (f32 carrier).
    Activation,
}

/// How a declared operator materializes its inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InputPolicy {
    /// A contraction whose f32 inputs `Builder::dot_general` or
    /// `Builder::convolution` demote to the precision's narrow type.
    FollowsPrecision,
    /// A contraction or scan that keeps f32 inputs at every precision.
    PinnedF32,
    /// Elementwise or reduction math on the f32 activation carrier.
    F32Carrier,
}

/// One operator class a family's graph emits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DeclaredOperator {
    pub(crate) id: &'static str,
    pub(crate) class: OperatorClass,
    pub(crate) origin: OperandOrigin,
    pub(crate) input: InputPolicy,
    pub(crate) association: AssociationPolicy,
    /// Explicit narrow rounding the graph applies to the result (`round_bf16`).
    pub(crate) output_rounding: Option<NumericDType>,
}

impl DeclaredOperator {
    pub(crate) const fn new(
        id: &'static str,
        class: OperatorClass,
        origin: OperandOrigin,
        input: InputPolicy,
        association: AssociationPolicy,
    ) -> Self {
        Self {
            id,
            class,
            origin,
            input,
            association,
            output_rounding: None,
        }
    }

    #[must_use]
    pub(crate) const fn rounded_to(mut self, dtype: NumericDType) -> Self {
        self.output_rounding = Some(dtype);
        self
    }
}

const fn is_contraction(class: OperatorClass) -> bool {
    matches!(
        class,
        OperatorClass::DenseMatmul
            | OperatorClass::Q4Matmul
            | OperatorClass::SiluProjection
            | OperatorClass::GeluProjection
            | OperatorClass::Convolution
            | OperatorClass::PrefixReduction
    )
}

const fn accumulates(class: OperatorClass) -> bool {
    is_contraction(class)
        || matches!(
            class,
            OperatorClass::LayerNorm | OperatorClass::RmsNorm | OperatorClass::AttentionSoftmax
        )
}

fn narrow_dtype(precision: Precision) -> Option<NumericDType> {
    match precision {
        Precision::F32 => None,
        Precision::F16 => Some(NumericDType::F16),
        Precision::Bf16 => Some(NumericDType::Bf16),
    }
}

/// Build the operator numeric contract for one production artifact.
///
/// `family` prefixes every operation id; `precision` must be the value the
/// emitter received for the same graph.
pub(crate) fn production_operator_contract(
    family: &str,
    operators: &[DeclaredOperator],
    precision: Precision,
    weights: CheckpointWeights,
    backend: &ProductionBackend,
) -> Result<OperatorNumericContractSet, String> {
    if operators.is_empty() {
        return Err(format!("{family} declares no operators"));
    }
    let backend = backend.identity()?;
    let entries = operators
        .iter()
        .map(|operator| operator_entry(family, operator, precision, weights, &backend))
        .collect::<Result<Vec<_>, _>>()?;
    OperatorNumericContractSet::new(entries)
}

fn operator_entry(
    family: &str,
    operator: &DeclaredOperator,
    precision: Precision,
    weights: CheckpointWeights,
    backend: &BackendNumericIdentity,
) -> Result<OperatorNumericContract, String> {
    let id = format!("{family}.{}", operator.id);
    let contraction = is_contraction(operator.class);
    let narrow = match (operator.input, contraction) {
        (InputPolicy::FollowsPrecision, true) => narrow_dtype(precision),
        (InputPolicy::PinnedF32, true) | (InputPolicy::F32Carrier, false) => None,
        _ => {
            return Err(format!(
                "{id}: input policy {:?} does not fit operator class {:?}",
                operator.input, operator.class
            ));
        }
    };
    let affine = match (operator.origin, weights) {
        (
            OperandOrigin::MatrixWeight,
            CheckpointWeights::Affine {
                dequantization,
                execution,
                ..
            },
        ) => Some((dequantization, execution)),
        _ => None,
    };
    let (checkpoint_dtype, weight_execution, dequantization) = match affine {
        Some((dequantization, execution)) => (
            if dequantization.bits == 4 {
                CheckpointDType::AffineU4
            } else {
                CheckpointDType::AffineU8
            },
            match execution {
                AffineExecution::Host => WeightExecution::HostAffineDequantized,
                AffineExecution::InGraph => WeightExecution::PackedAffineInGraph,
            },
            Some(dequantization),
        ),
        None => (
            if operator.origin == OperandOrigin::Activation {
                CheckpointDType::F32
            } else {
                weights.float()
            },
            WeightExecution::Dense,
            None,
        ),
    };
    let mut rounding = Vec::new();
    if operator.origin != OperandOrigin::Activation {
        rounding.push(RoundingBoundary::CheckpointLoad);
    }
    if dequantization.is_some() {
        rounding.push(RoundingBoundary::DequantizedWeight);
    }
    if narrow.is_some() {
        rounding.push(RoundingBoundary::OperatorInput);
    }
    if accumulates(operator.class) {
        rounding.push(RoundingBoundary::AccumulatorResult);
    }
    match operator.class {
        OperatorClass::SiluProjection | OperatorClass::GeluProjection => {
            rounding.push(RoundingBoundary::ActivationResult);
        }
        OperatorClass::ResidualAdd => rounding.push(RoundingBoundary::ResidualResult),
        _ => {}
    }
    if operator.output_rounding.is_some() {
        rounding.push(RoundingBoundary::OperatorOutput);
    }
    if rounding.is_empty() {
        // Activation-only, non-accumulating math rounds once at its output.
        rounding.push(RoundingBoundary::OperatorOutput);
    }
    OperatorNumericContract::new(
        id,
        operator.class,
        checkpoint_dtype,
        narrow.unwrap_or(NumericDType::F32),
        NumericDType::F32,
        operator.output_rounding.unwrap_or(NumericDType::F32),
        rounding,
        operator.association,
        weight_execution,
        dequantization,
        backend.clone(),
    )
}
