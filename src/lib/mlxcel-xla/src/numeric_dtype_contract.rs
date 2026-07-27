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

//! Canonical dtype/materialization identity for StableHLO/IREE artifacts.
//!
//! Shapes, model-family options, compiler identity, and emitted MLIR already
//! participate in their respective artifact identities. This descriptor binds
//! the dtype choices that otherwise look identical at those layers: where
//! quantized weights are expanded, contraction operand/accumulator/result
//! types, reduction compute type, and the activation carrier between operators.
//!
//! This is intentionally a subset of a complete operator numeric contract. It
//! does not describe rounding placement, reduction association, operation
//! ordering, or backend algorithm identity. A family cannot claim numeric
//! qualification from this descriptor alone.

const SCHEMA: &str = "mlxcel-xla-numeric-dtype-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NumericDType {
    F16,
    Bf16,
    F32,
    I32,
    U8,
    U32,
}

impl NumericDType {
    pub(crate) const fn identity(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
            Self::I32 => "i32",
            Self::U8 => "u8",
            Self::U32 => "u32",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WeightExecution {
    Dense,
    HostAffineDequantized,
    PackedAffineInGraph,
}

impl WeightExecution {
    pub(crate) const fn identity(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::HostAffineDequantized => "host-affine-dequantized",
            Self::PackedAffineInGraph => "packed-affine-in-graph",
        }
    }
}

/// Dtype and materialization choices that must match before VMFB reuse.
///
/// The type deliberately has no `Default`: adopting the versioned auxiliary
/// manifest must be an explicit decision. This contract is necessary but not
/// sufficient for a family to pass its bounded operator oracle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NumericDTypeContract {
    pub(crate) weight_execution: WeightExecution,
    pub(crate) contraction_input: NumericDType,
    pub(crate) contraction_accumulator: NumericDType,
    pub(crate) contraction_output: NumericDType,
    pub(crate) reduction_compute: NumericDType,
    pub(crate) activation_carrier: NumericDType,
}

impl NumericDTypeContract {
    #[must_use]
    pub(crate) const fn new(
        weight_execution: WeightExecution,
        contraction_input: NumericDType,
        contraction_accumulator: NumericDType,
        contraction_output: NumericDType,
        reduction_compute: NumericDType,
        activation_carrier: NumericDType,
    ) -> Self {
        Self {
            weight_execution,
            contraction_input,
            contraction_accumulator,
            contraction_output,
            reduction_compute,
            activation_carrier,
        }
    }

    /// Stable text used only as hash input; never use `Debug` output here.
    #[must_use]
    pub(crate) fn canonical_identity(self) -> String {
        format!(
            "{SCHEMA};weights={};dot-in={};dot-acc={};dot-out={};reduce={};carrier={}",
            self.weight_execution.identity(),
            self.contraction_input.identity(),
            self.contraction_accumulator.identity(),
            self.contraction_output.identity(),
            self.reduction_compute.identity(),
            self.activation_carrier.identity(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host_dequantized_f32() -> NumericDTypeContract {
        NumericDTypeContract::new(
            WeightExecution::HostAffineDequantized,
            NumericDType::F32,
            NumericDType::F32,
            NumericDType::F32,
            NumericDType::F32,
            NumericDType::F32,
        )
    }

    #[test]
    fn canonical_identity_is_stable() {
        assert_eq!(
            host_dequantized_f32().canonical_identity(),
            "mlxcel-xla-numeric-dtype-v1;weights=host-affine-dequantized;dot-in=f32;\
             dot-acc=f32;dot-out=f32;reduce=f32;carrier=f32"
        );
    }

    #[test]
    fn every_numeric_field_changes_the_identity() {
        let baseline = host_dequantized_f32();
        let variants = [
            NumericDTypeContract {
                weight_execution: WeightExecution::Dense,
                ..baseline
            },
            NumericDTypeContract {
                weight_execution: WeightExecution::PackedAffineInGraph,
                ..baseline
            },
            NumericDTypeContract {
                contraction_input: NumericDType::F16,
                ..baseline
            },
            NumericDTypeContract {
                contraction_accumulator: NumericDType::Bf16,
                ..baseline
            },
            NumericDTypeContract {
                contraction_output: NumericDType::F16,
                ..baseline
            },
            NumericDTypeContract {
                reduction_compute: NumericDType::Bf16,
                ..baseline
            },
            NumericDTypeContract {
                activation_carrier: NumericDType::F16,
                ..baseline
            },
        ];
        let baseline_identity = baseline.canonical_identity();
        for variant in variants {
            assert_ne!(variant.canonical_identity(), baseline_identity);
        }
    }
}
