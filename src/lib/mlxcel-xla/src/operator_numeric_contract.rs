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

//! Versioned operator-level numeric identity for StableHLO/IREE artifacts.
//!
//! The earlier dtype contract binds graph-wide materialization choices. This
//! descriptor is deliberately narrower and stronger: every entry names one
//! contract-sensitive operation and records its dtype boundaries, rounding
//! points, association policy, weight materialization, and backend algorithm
//! identity. It is artifact identity, not evidence that an oracle passed.

use crate::numeric_dtype_contract::{NumericDType, WeightExecution};

const SCHEMA: &str = "mlxcel-xla-operator-numeric-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CheckpointDType {
    F16,
    Bf16,
    F32,
    AffineU4,
    AffineU8,
}

impl CheckpointDType {
    const fn identity(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
            Self::AffineU4 => "affine-u4",
            Self::AffineU8 => "affine-u8",
        }
    }

    const fn affine_bits(self) -> Option<u8> {
        match self {
            Self::AffineU4 => Some(4),
            Self::AffineU8 => Some(8),
            Self::F16 | Self::Bf16 | Self::F32 => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OperatorClass {
    AffineQ4Dequantize,
    DenseMatmul,
    Q4Matmul,
    LayerNorm,
    RmsNorm,
    ResidualAdd,
    SiluProjection,
    GeluProjection,
    AttentionSoftmax,
    PrefixReduction,
    Convolution,
}

impl OperatorClass {
    const fn identity(self) -> &'static str {
        match self {
            Self::AffineQ4Dequantize => "affine-q4-dequantize",
            Self::DenseMatmul => "dense-matmul",
            Self::Q4Matmul => "q4-matmul",
            Self::LayerNorm => "layer-norm",
            Self::RmsNorm => "rms-norm",
            Self::ResidualAdd => "residual-add",
            Self::SiluProjection => "silu-projection",
            Self::GeluProjection => "gelu-projection",
            Self::AttentionSoftmax => "attention-softmax",
            Self::PrefixReduction => "prefix-reduction",
            Self::Convolution => "convolution",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum RoundingBoundary {
    CheckpointLoad,
    DequantizedWeight,
    OperatorInput,
    AccumulatorResult,
    BiasAdd,
    ActivationResult,
    ResidualResult,
    OperatorOutput,
}

impl RoundingBoundary {
    const fn identity(self) -> &'static str {
        match self {
            Self::CheckpointLoad => "checkpoint-load",
            Self::DequantizedWeight => "dequantized-weight",
            Self::OperatorInput => "operator-input",
            Self::AccumulatorResult => "accumulator-result",
            Self::BiasAdd => "bias-add",
            Self::ActivationResult => "activation-result",
            Self::ResidualResult => "residual-result",
            Self::OperatorOutput => "operator-output",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AssociationPolicy {
    Sequential,
    PairwiseTree,
    PrefixSequential,
    BackendAlgorithm,
}

impl AssociationPolicy {
    const fn identity(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::PairwiseTree => "pairwise-tree",
            Self::PrefixSequential => "prefix-sequential",
            Self::BackendAlgorithm => "backend-algorithm",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PackedLaneOrder {
    LeastSignificantFirst,
    MostSignificantFirst,
}

impl PackedLaneOrder {
    const fn identity(self) -> &'static str {
        match self {
            Self::LeastSignificantFirst => "least-significant-first",
            Self::MostSignificantFirst => "most-significant-first",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AffineEvaluationOrder {
    SeparateMultiplyThenAdd,
    FusedMultiplyAdd,
}

impl AffineEvaluationOrder {
    const fn identity(self) -> &'static str {
        match self {
            Self::SeparateMultiplyThenAdd => "separate-multiply-then-add",
            Self::FusedMultiplyAdd => "fused-multiply-add",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AffineDequantizationContract {
    bits: u8,
    group_size: usize,
    packed_carrier: NumericDType,
    scale_bias_dtype: NumericDType,
    lane_order: PackedLaneOrder,
    evaluation_order: AffineEvaluationOrder,
}

impl AffineDequantizationContract {
    pub(crate) fn new(
        bits: u8,
        group_size: usize,
        packed_carrier: NumericDType,
        scale_bias_dtype: NumericDType,
        lane_order: PackedLaneOrder,
        evaluation_order: AffineEvaluationOrder,
    ) -> Result<Self, String> {
        if !matches!(bits, 4 | 8) {
            return Err(format!(
                "affine dequantization bits must be 4 or 8, got {bits}"
            ));
        }
        if group_size == 0 {
            return Err("affine dequantization group size must be non-zero".to_string());
        }
        if packed_carrier != NumericDType::U32 {
            return Err("affine dequantization packed carrier must be u32".to_string());
        }
        if !matches!(scale_bias_dtype, NumericDType::F16 | NumericDType::Bf16) {
            return Err("affine dequantization scale/bias dtype must be f16 or bf16".to_string());
        }
        Ok(Self {
            bits,
            group_size,
            packed_carrier,
            scale_bias_dtype,
            lane_order,
            evaluation_order,
        })
    }

    fn canonical_identity(self) -> String {
        format!(
            "bits={},group={},carrier={},metadata={},lanes={},evaluation={}",
            self.bits,
            self.group_size,
            self.packed_carrier.identity(),
            self.scale_bias_dtype.identity(),
            self.lane_order.identity(),
            self.evaluation_order.identity(),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BackendNumericIdentity {
    framework_revision: String,
    framework_patch: String,
    compiler_build: String,
    target: String,
    kernel_or_algorithm: String,
}

impl BackendNumericIdentity {
    pub(crate) fn new(
        framework_revision: impl Into<String>,
        framework_patch: impl Into<String>,
        compiler_build: impl Into<String>,
        target: impl Into<String>,
        kernel_or_algorithm: impl Into<String>,
    ) -> Result<Self, String> {
        let identity = Self {
            framework_revision: framework_revision.into(),
            framework_patch: framework_patch.into(),
            compiler_build: compiler_build.into(),
            target: target.into(),
            kernel_or_algorithm: kernel_or_algorithm.into(),
        };
        for (name, value) in [
            ("framework revision", identity.framework_revision.as_str()),
            ("framework patch", identity.framework_patch.as_str()),
            ("compiler build", identity.compiler_build.as_str()),
            ("target", identity.target.as_str()),
            (
                "kernel or algorithm identity",
                identity.kernel_or_algorithm.as_str(),
            ),
        ] {
            validate_component(name, value)?;
        }
        Ok(identity)
    }

    fn canonical_identity(&self) -> String {
        format!(
            "framework={};patch={};compiler={};target={};algorithm={}",
            self.framework_revision,
            self.framework_patch,
            self.compiler_build,
            self.target,
            self.kernel_or_algorithm,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OperatorNumericContract {
    operation_id: String,
    class: OperatorClass,
    checkpoint_dtype: CheckpointDType,
    input_materialization: NumericDType,
    accumulator: NumericDType,
    result_materialization: NumericDType,
    rounding_boundaries: Vec<RoundingBoundary>,
    association: AssociationPolicy,
    weight_execution: WeightExecution,
    affine_dequantization: Option<AffineDequantizationContract>,
    backend: BackendNumericIdentity,
}

impl OperatorNumericContract {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        operation_id: impl Into<String>,
        class: OperatorClass,
        checkpoint_dtype: CheckpointDType,
        input_materialization: NumericDType,
        accumulator: NumericDType,
        result_materialization: NumericDType,
        rounding_boundaries: impl IntoIterator<Item = RoundingBoundary>,
        association: AssociationPolicy,
        weight_execution: WeightExecution,
        affine_dequantization: Option<AffineDequantizationContract>,
        backend: BackendNumericIdentity,
    ) -> Result<Self, String> {
        let operation_id = operation_id.into();
        validate_component("operation id", &operation_id)?;
        let mut rounding_boundaries = rounding_boundaries.into_iter().collect::<Vec<_>>();
        rounding_boundaries.sort_unstable();
        if rounding_boundaries.is_empty() {
            return Err(format!(
                "operator numeric contract {operation_id} requires at least one rounding boundary"
            ));
        }
        if rounding_boundaries
            .windows(2)
            .any(|window| window[0] == window[1])
        {
            return Err(format!(
                "operator numeric contract {operation_id} has duplicate rounding boundaries"
            ));
        }
        if let Some(expected_bits) = checkpoint_dtype.affine_bits() {
            let dequantization = affine_dequantization.ok_or_else(|| {
                format!(
                    "operator numeric contract {operation_id} requires an affine dequantization order"
                )
            })?;
            if dequantization.bits != expected_bits {
                return Err(format!(
                    "operator numeric contract {operation_id} checkpoint uses {expected_bits}-bit \
                     affine weights but dequantization uses {} bits",
                    dequantization.bits
                ));
            }
            if weight_execution == WeightExecution::Dense {
                return Err(format!(
                    "operator numeric contract {operation_id} cannot use dense weight execution \
                     for an affine checkpoint"
                ));
            }
        } else {
            if affine_dequantization.is_some() {
                return Err(format!(
                    "operator numeric contract {operation_id} has affine dequantization for a \
                     non-affine checkpoint"
                ));
            }
            if weight_execution != WeightExecution::Dense {
                return Err(format!(
                    "operator numeric contract {operation_id} requires an affine checkpoint dtype \
                     for quantized weight execution"
                ));
            }
        }
        Ok(Self {
            operation_id,
            class,
            checkpoint_dtype,
            input_materialization,
            accumulator,
            result_materialization,
            rounding_boundaries,
            association,
            weight_execution,
            affine_dequantization,
            backend,
        })
    }

    fn canonical_identity(&self) -> String {
        let rounding = self
            .rounding_boundaries
            .iter()
            .map(|boundary| boundary.identity())
            .collect::<Vec<_>>()
            .join(",");
        let dequantization = self.affine_dequantization.map_or_else(
            || "none".to_string(),
            AffineDequantizationContract::canonical_identity,
        );
        format!(
            "id={};class={};checkpoint={};input={};acc={};result={};round={};\
             association={};weights={};dequant={};{}",
            self.operation_id,
            self.class.identity(),
            self.checkpoint_dtype.identity(),
            self.input_materialization.identity(),
            self.accumulator.identity(),
            self.result_materialization.identity(),
            rounding,
            self.association.identity(),
            self.weight_execution.identity(),
            dequantization,
            self.backend.canonical_identity(),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct OperatorNumericContractSet {
    operators: Vec<OperatorNumericContract>,
}

impl OperatorNumericContractSet {
    pub(crate) fn new(
        operators: impl IntoIterator<Item = OperatorNumericContract>,
    ) -> Result<Self, String> {
        let mut operators = operators.into_iter().collect::<Vec<_>>();
        if operators.is_empty() {
            return Err("operator numeric contract set must not be empty".to_string());
        }
        operators.sort_by(|left, right| left.operation_id.cmp(&right.operation_id));
        if operators
            .windows(2)
            .any(|window| window[0].operation_id == window[1].operation_id)
        {
            return Err("operator numeric contract ids must be unique".to_string());
        }
        Ok(Self { operators })
    }

    #[must_use]
    pub(crate) fn canonical_identity(&self) -> String {
        let mut identity = SCHEMA.to_string();
        for operator in &self.operators {
            identity.push('\0');
            identity.push_str(&operator.canonical_identity());
        }
        identity
    }
}

fn validate_component(name: &str, value: &str) -> Result<(), String> {
    if value.is_empty() {
        return Err(format!("{name} must not be empty"));
    }
    if value.contains(['\0', ';', '\n', '\r']) {
        return Err(format!("{name} contains a reserved identity separator"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend() -> BackendNumericIdentity {
        BackendNumericIdentity::new(
            "mlx-r2",
            "qmm-sm80-r2",
            "iree-3.12.0rc",
            "cuda-sm80",
            "qmm_sm80_r2",
        )
        .unwrap()
    }

    fn affine_dequantization() -> AffineDequantizationContract {
        AffineDequantizationContract::new(
            4,
            64,
            NumericDType::U32,
            NumericDType::F16,
            PackedLaneOrder::LeastSignificantFirst,
            AffineEvaluationOrder::FusedMultiplyAdd,
        )
        .unwrap()
    }

    fn q4_contract(id: &str) -> OperatorNumericContract {
        OperatorNumericContract::new(
            id,
            OperatorClass::Q4Matmul,
            CheckpointDType::AffineU4,
            NumericDType::F16,
            NumericDType::F32,
            NumericDType::F32,
            [
                RoundingBoundary::DequantizedWeight,
                RoundingBoundary::OperatorInput,
                RoundingBoundary::AccumulatorResult,
            ],
            AssociationPolicy::BackendAlgorithm,
            WeightExecution::PackedAffineInGraph,
            Some(affine_dequantization()),
            backend(),
        )
        .unwrap()
    }

    #[test]
    fn canonical_identity_is_order_independent_and_stable() {
        let first =
            OperatorNumericContractSet::new([q4_contract("block.1"), q4_contract("block.0")])
                .unwrap();
        let second =
            OperatorNumericContractSet::new([q4_contract("block.0"), q4_contract("block.1")])
                .unwrap();
        assert_eq!(first.canonical_identity(), second.canonical_identity());
        assert!(first.canonical_identity().starts_with(SCHEMA));
        assert!(first.canonical_identity().contains("id=block.0"));
        assert!(first.canonical_identity().contains("algorithm=qmm_sm80_r2"));
    }

    #[test]
    fn every_materialization_and_backend_field_is_identity_sensitive() {
        let baseline_contract = q4_contract("block.0");
        let baseline = OperatorNumericContractSet::new([baseline_contract.clone()])
            .unwrap()
            .canonical_identity();
        let mut variants = Vec::new();

        let mut changed = baseline_contract.clone();
        changed.operation_id = "block.1".to_string();
        variants.push(("operation id", changed));
        let mut changed = baseline_contract.clone();
        changed.class = OperatorClass::DenseMatmul;
        variants.push(("operator class", changed));
        let mut changed = baseline_contract.clone();
        changed.checkpoint_dtype = CheckpointDType::AffineU8;
        changed.affine_dequantization.as_mut().unwrap().bits = 8;
        variants.push(("checkpoint dtype", changed));
        let mut changed = baseline_contract.clone();
        changed.input_materialization = NumericDType::Bf16;
        variants.push(("input materialization", changed));
        let mut changed = baseline_contract.clone();
        changed.accumulator = NumericDType::Bf16;
        variants.push(("accumulator", changed));
        let mut changed = baseline_contract.clone();
        changed.result_materialization = NumericDType::F16;
        variants.push(("result materialization", changed));
        let mut changed = baseline_contract.clone();
        changed.rounding_boundaries = vec![RoundingBoundary::OperatorOutput];
        variants.push(("rounding boundaries", changed));
        let mut changed = baseline_contract.clone();
        changed.association = AssociationPolicy::PairwiseTree;
        variants.push(("association policy", changed));
        let mut changed = baseline_contract.clone();
        changed.weight_execution = WeightExecution::HostAffineDequantized;
        variants.push(("weight execution", changed));
        let mut changed = baseline_contract.clone();
        changed.affine_dequantization.as_mut().unwrap().group_size = 128;
        variants.push(("dequantization group", changed));
        let mut changed = baseline_contract.clone();
        changed
            .affine_dequantization
            .as_mut()
            .unwrap()
            .scale_bias_dtype = NumericDType::Bf16;
        variants.push(("dequantization metadata", changed));
        let mut changed = baseline_contract.clone();
        changed.affine_dequantization.as_mut().unwrap().lane_order =
            PackedLaneOrder::MostSignificantFirst;
        variants.push(("packed lane order", changed));
        let mut changed = baseline_contract.clone();
        changed
            .affine_dequantization
            .as_mut()
            .unwrap()
            .evaluation_order = AffineEvaluationOrder::SeparateMultiplyThenAdd;
        variants.push(("dequantization evaluation", changed));

        for (name, value) in [
            ("framework revision", "mlx-r3"),
            ("framework patch", "qmm-sm80-r3"),
            ("compiler build", "iree-3.12.1"),
            ("target", "cuda-sm90"),
            ("kernel identity", "qmm_sm80_r3"),
        ] {
            let mut changed = baseline_contract.clone();
            match name {
                "framework revision" => changed.backend.framework_revision = value.to_string(),
                "framework patch" => changed.backend.framework_patch = value.to_string(),
                "compiler build" => changed.backend.compiler_build = value.to_string(),
                "target" => changed.backend.target = value.to_string(),
                "kernel identity" => changed.backend.kernel_or_algorithm = value.to_string(),
                _ => unreachable!(),
            }
            variants.push((name, changed));
        }

        for (name, changed) in variants {
            let changed = OperatorNumericContractSet::new([changed])
                .unwrap()
                .canonical_identity();
            assert_ne!(baseline, changed, "{name} must affect artifact identity");
        }
    }

    #[test]
    fn invalid_and_ambiguous_identities_fail_closed() {
        assert!(BackendNumericIdentity::new("", "patch", "compiler", "cpu", "kernel").is_err());
        assert!(
            BackendNumericIdentity::new("revision", "patch;other", "compiler", "cpu", "kernel")
                .is_err()
        );
        assert!(
            AffineDequantizationContract::new(
                3,
                64,
                NumericDType::U32,
                NumericDType::F16,
                PackedLaneOrder::LeastSignificantFirst,
                AffineEvaluationOrder::FusedMultiplyAdd,
            )
            .is_err()
        );
        assert!(
            AffineDequantizationContract::new(
                4,
                0,
                NumericDType::U32,
                NumericDType::F16,
                PackedLaneOrder::LeastSignificantFirst,
                AffineEvaluationOrder::FusedMultiplyAdd,
            )
            .is_err()
        );
        assert!(
            AffineDequantizationContract::new(
                4,
                64,
                NumericDType::U8,
                NumericDType::F16,
                PackedLaneOrder::LeastSignificantFirst,
                AffineEvaluationOrder::FusedMultiplyAdd,
            )
            .is_err()
        );
        assert!(
            AffineDequantizationContract::new(
                4,
                64,
                NumericDType::U32,
                NumericDType::F32,
                PackedLaneOrder::LeastSignificantFirst,
                AffineEvaluationOrder::FusedMultiplyAdd,
            )
            .is_err()
        );
        assert!(
            OperatorNumericContract::new(
                "block.0",
                OperatorClass::DenseMatmul,
                CheckpointDType::F16,
                NumericDType::F16,
                NumericDType::F32,
                NumericDType::F32,
                [],
                AssociationPolicy::Sequential,
                WeightExecution::Dense,
                None,
                backend(),
            )
            .is_err()
        );
        assert!(
            OperatorNumericContract::new(
                "block.0",
                OperatorClass::Q4Matmul,
                CheckpointDType::AffineU4,
                NumericDType::F16,
                NumericDType::F32,
                NumericDType::F32,
                [RoundingBoundary::OperatorOutput],
                AssociationPolicy::BackendAlgorithm,
                WeightExecution::PackedAffineInGraph,
                None,
                backend(),
            )
            .is_err()
        );
        assert!(
            OperatorNumericContract::new(
                "block.0",
                OperatorClass::DenseMatmul,
                CheckpointDType::F16,
                NumericDType::F16,
                NumericDType::F32,
                NumericDType::F32,
                [RoundingBoundary::OperatorOutput],
                AssociationPolicy::Sequential,
                WeightExecution::Dense,
                Some(affine_dequantization()),
                backend(),
            )
            .is_err()
        );
        assert!(
            OperatorNumericContractSet::new([q4_contract("block.0"), q4_contract("block.0")])
                .is_err()
        );
    }
}
