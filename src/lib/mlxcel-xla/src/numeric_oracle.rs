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

//! Shared comparison and reporting for bounded MLX/IREE operator oracles.
//!
//! This module deliberately does not know how either backend executes an
//! operation. Callers provide two closures, and both receive the same immutable
//! operand set. The harness owns identity validation, timing, exact first
//! divergence reporting, and conservative qualification semantics.

use std::time::Instant;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::operator_numeric_contract::OperatorNumericContractSet;

const REPORT_SCHEMA: &str = "mlxcel-xla-micro-oracle-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NumericOracleClaim {
    ProductionMlx,
    CanonicalDecomposition,
    MathematicalCloseness,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum ComparisonMode {
    ExactBits,
    AbsoluteRelative { absolute: f64, relative: f64 },
}

impl ComparisonMode {
    fn validate(self) -> Result<Self, String> {
        if let Self::AbsoluteRelative { absolute, relative } = self {
            if !absolute.is_finite() || absolute < 0.0 {
                return Err(
                    "numeric oracle absolute tolerance must be finite and non-negative".to_string(),
                );
            }
            if !relative.is_finite() || relative < 0.0 {
                return Err(
                    "numeric oracle relative tolerance must be finite and non-negative".to_string(),
                );
            }
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "status", content = "identity", rename_all = "kebab-case")]
pub enum AlgorithmIdentity {
    Observed(String),
    Unobserved(String),
}

impl AlgorithmIdentity {
    pub fn observed(identity: impl Into<String>) -> Result<Self, String> {
        let identity = identity.into();
        validate_identity_component("observed algorithm identity", &identity)?;
        Ok(Self::Observed(identity))
    }

    pub fn unobserved(reason: impl Into<String>) -> Result<Self, String> {
        let reason = reason.into();
        validate_identity_component("unobserved algorithm reason", &reason)?;
        Ok(Self::Unobserved(reason))
    }

    fn is_observed(&self) -> bool {
        matches!(self, Self::Observed(_))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct NumericBackendIdentity {
    runtime: String,
    revision: String,
    patch: String,
    compiler: String,
    compiler_flags: Vec<String>,
    target: String,
    device: String,
    artifact_fingerprint: String,
    algorithm: AlgorithmIdentity,
}

impl NumericBackendIdentity {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        runtime: impl Into<String>,
        revision: impl Into<String>,
        patch: impl Into<String>,
        compiler: impl Into<String>,
        compiler_flags: Vec<String>,
        target: impl Into<String>,
        device: impl Into<String>,
        artifact_fingerprint: impl Into<String>,
        algorithm: AlgorithmIdentity,
    ) -> Result<Self, String> {
        let identity = Self {
            runtime: runtime.into(),
            revision: revision.into(),
            patch: patch.into(),
            compiler: compiler.into(),
            compiler_flags,
            target: target.into(),
            device: device.into(),
            artifact_fingerprint: artifact_fingerprint.into(),
            algorithm,
        };
        for (name, value) in [
            ("runtime", identity.runtime.as_str()),
            ("revision", identity.revision.as_str()),
            ("patch", identity.patch.as_str()),
            ("compiler", identity.compiler.as_str()),
            ("target", identity.target.as_str()),
            ("device", identity.device.as_str()),
            (
                "artifact fingerprint",
                identity.artifact_fingerprint.as_str(),
            ),
        ] {
            validate_identity_component(name, value)?;
        }
        if identity.compiler_flags.is_empty() {
            return Err("numeric backend compiler flags must not be empty".to_string());
        }
        for flag in &identity.compiler_flags {
            validate_identity_component("compiler flag", flag)?;
        }
        Ok(identity)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NumericTensor {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

impl NumericTensor {
    pub fn new(
        name: impl Into<String>,
        shape: Vec<usize>,
        values: Vec<f32>,
    ) -> Result<Self, String> {
        let name = name.into();
        validate_identity_component("numeric tensor name", &name)?;
        let elements = checked_elements(&shape)?;
        if values.len() != elements {
            return Err(format!(
                "numeric tensor {name} has {} values, expected {elements} for shape {shape:?}",
                values.len()
            ));
        }
        Ok(Self {
            name,
            shape,
            values,
        })
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

#[derive(Clone, Debug)]
pub struct NumericOracleCase {
    operation: String,
    claim: NumericOracleClaim,
    operator_contract_identity: String,
    reference_backend: NumericBackendIdentity,
    candidate_backend: NumericBackendIdentity,
    comparison: ComparisonMode,
}

impl NumericOracleCase {
    pub fn new(
        operation: impl Into<String>,
        claim: NumericOracleClaim,
        operator_contract_identity: impl Into<String>,
        reference_backend: NumericBackendIdentity,
        candidate_backend: NumericBackendIdentity,
        comparison: ComparisonMode,
    ) -> Result<Self, String> {
        let operation = operation.into();
        let operator_contract_identity = operator_contract_identity.into();
        validate_identity_component("numeric oracle operation", &operation)?;
        validate_contract_identity(&operator_contract_identity)?;
        Ok(Self {
            operation,
            claim,
            operator_contract_identity,
            reference_backend,
            candidate_backend,
            comparison: comparison.validate()?,
        })
    }

    #[allow(dead_code)] // Used by operation-specific probes as they migrate to this harness.
    pub(crate) fn from_operator_contract(
        operation: impl Into<String>,
        claim: NumericOracleClaim,
        operator_contract: &OperatorNumericContractSet,
        reference_backend: NumericBackendIdentity,
        candidate_backend: NumericBackendIdentity,
        comparison: ComparisonMode,
    ) -> Result<Self, String> {
        Self::new(
            operation,
            claim,
            operator_contract.canonical_identity(),
            reference_backend,
            candidate_backend,
            comparison,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TensorSummary {
    name: String,
    shape: Vec<usize>,
    dtype: &'static str,
    sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FirstDivergence {
    flat_index: usize,
    coordinate: Vec<usize>,
    reference_value: f32,
    candidate_value: f32,
    reference_bits: u32,
    candidate_bits: u32,
    absolute_error: Option<f64>,
    relative_error: Option<f64>,
}

impl FirstDivergence {
    #[must_use]
    pub fn flat_index(&self) -> usize {
        self.flat_index
    }

    #[must_use]
    pub fn coordinate(&self) -> &[usize] {
        &self.coordinate
    }

    #[must_use]
    pub fn reference_value(&self) -> f32 {
        self.reference_value
    }

    #[must_use]
    pub fn candidate_value(&self) -> f32 {
        self.candidate_value
    }

    #[must_use]
    pub fn reference_bits(&self) -> u32 {
        self.reference_bits
    }

    #[must_use]
    pub fn candidate_bits(&self) -> u32 {
        self.candidate_bits
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ComparisonSummary {
    mode: ComparisonMode,
    elements: usize,
    failures: usize,
    non_finite_failures: usize,
    max_absolute: f64,
    max_relative: f64,
    rms: f64,
}

impl ComparisonSummary {
    #[must_use]
    pub fn failures(&self) -> usize {
        self.failures
    }

    #[must_use]
    pub fn non_finite_failures(&self) -> usize {
        self.non_finite_failures
    }

    #[must_use]
    pub fn max_absolute(&self) -> f64 {
        self.max_absolute
    }

    #[must_use]
    pub fn max_relative(&self) -> f64 {
        self.max_relative
    }

    #[must_use]
    pub fn rms(&self) -> f64 {
        self.rms
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExecutionTiming {
    reference_seconds: f64,
    candidate_seconds: f64,
}

impl ExecutionTiming {
    #[must_use]
    pub fn reference_seconds(&self) -> f64 {
        self.reference_seconds
    }

    #[must_use]
    pub fn candidate_seconds(&self) -> f64 {
        self.candidate_seconds
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NumericOracleReport {
    schema: &'static str,
    operation: String,
    claim: NumericOracleClaim,
    operands: Vec<TensorSummary>,
    operator_contract_identity: String,
    operator_contract_sha256: String,
    reference_backend: NumericBackendIdentity,
    candidate_backend: NumericBackendIdentity,
    output_shape: Vec<usize>,
    comparison: ComparisonSummary,
    first_divergence: Option<FirstDivergence>,
    timing: ExecutionTiming,
    passed: bool,
    production_qualified: bool,
}

impl NumericOracleReport {
    #[must_use]
    pub fn operation(&self) -> &str {
        &self.operation
    }

    #[must_use]
    pub fn passed(&self) -> bool {
        self.passed
    }

    #[must_use]
    pub fn production_qualified(&self) -> bool {
        self.production_qualified
    }

    #[must_use]
    pub fn first_divergence(&self) -> Option<&FirstDivergence> {
        self.first_divergence.as_ref()
    }

    #[must_use]
    pub fn comparison(&self) -> &ComparisonSummary {
        &self.comparison
    }

    #[must_use]
    pub fn timing(&self) -> &ExecutionTiming {
        &self.timing
    }
}

pub fn run_bounded_numeric_oracle<R, C>(
    case: NumericOracleCase,
    operands: &[NumericTensor],
    reference: R,
    candidate: C,
) -> Result<NumericOracleReport, String>
where
    R: FnOnce(&[NumericTensor]) -> Result<NumericTensor, String>,
    C: FnOnce(&[NumericTensor]) -> Result<NumericTensor, String>,
{
    if operands.is_empty() {
        return Err("numeric oracle requires at least one operand".to_string());
    }
    let operand_summaries = operands.iter().map(tensor_summary).collect();
    let reference_started = Instant::now();
    let reference_output =
        reference(operands).map_err(|error| format!("run reference backend: {error}"))?;
    let reference_seconds = reference_started.elapsed().as_secs_f64();
    let candidate_started = Instant::now();
    let candidate_output =
        candidate(operands).map_err(|error| format!("run candidate backend: {error}"))?;
    let candidate_seconds = candidate_started.elapsed().as_secs_f64();
    if reference_output.shape != candidate_output.shape {
        return Err(format!(
            "numeric oracle output shape mismatch: reference {:?}, candidate {:?}",
            reference_output.shape, candidate_output.shape
        ));
    }
    let (comparison, first_divergence) =
        compare_outputs(&reference_output, &candidate_output, case.comparison)?;
    let passed = comparison.failures == 0;
    let production_qualified = passed
        && case.claim == NumericOracleClaim::ProductionMlx
        && case.reference_backend.algorithm.is_observed()
        && case.candidate_backend.algorithm.is_observed();
    Ok(NumericOracleReport {
        schema: REPORT_SCHEMA,
        operation: case.operation,
        claim: case.claim,
        operands: operand_summaries,
        operator_contract_sha256: sha256_hex(case.operator_contract_identity.as_bytes()),
        operator_contract_identity: case.operator_contract_identity,
        reference_backend: case.reference_backend,
        candidate_backend: case.candidate_backend,
        output_shape: reference_output.shape,
        comparison,
        first_divergence,
        timing: ExecutionTiming {
            reference_seconds,
            candidate_seconds,
        },
        passed,
        production_qualified,
    })
}

fn compare_outputs(
    reference: &NumericTensor,
    candidate: &NumericTensor,
    mode: ComparisonMode,
) -> Result<(ComparisonSummary, Option<FirstDivergence>), String> {
    if reference.values.len() != candidate.values.len() {
        return Err(format!(
            "numeric oracle output length mismatch: reference {}, candidate {}",
            reference.values.len(),
            candidate.values.len()
        ));
    }
    let mut failures = 0usize;
    let mut non_finite_failures = 0usize;
    let mut max_absolute = 0.0f64;
    let mut max_relative = 0.0f64;
    let mut squared_error = 0.0f64;
    let mut first_divergence = None;
    for (index, (&reference_value, &candidate_value)) in
        reference.values.iter().zip(&candidate.values).enumerate()
    {
        let finite = reference_value.is_finite() && candidate_value.is_finite();
        let absolute =
            finite.then(|| (f64::from(candidate_value) - f64::from(reference_value)).abs());
        let relative = absolute.map(|absolute| {
            absolute / f64::from(reference_value.abs()).max(f64::from(f32::MIN_POSITIVE))
        });
        let failed = match mode {
            ComparisonMode::ExactBits => reference_value.to_bits() != candidate_value.to_bits(),
            ComparisonMode::AbsoluteRelative {
                absolute: absolute_tolerance,
                relative: relative_tolerance,
            } => {
                !finite
                    || absolute.is_some_and(|absolute| {
                        absolute
                            > absolute_tolerance
                                + relative_tolerance * f64::from(reference_value.abs())
                    })
            }
        };
        if let Some(absolute) = absolute {
            max_absolute = max_absolute.max(absolute);
            squared_error += absolute * absolute;
        }
        if let Some(relative) = relative {
            max_relative = max_relative.max(relative);
        }
        if failed {
            failures += 1;
            if !finite {
                non_finite_failures += 1;
            }
            first_divergence.get_or_insert_with(|| FirstDivergence {
                flat_index: index,
                coordinate: row_major_coordinate(index, &reference.shape),
                reference_value,
                candidate_value,
                reference_bits: reference_value.to_bits(),
                candidate_bits: candidate_value.to_bits(),
                absolute_error: absolute,
                relative_error: relative,
            });
        }
    }
    let rms = if reference.values.is_empty() {
        0.0
    } else {
        (squared_error / reference.values.len() as f64).sqrt()
    };
    Ok((
        ComparisonSummary {
            mode,
            elements: reference.values.len(),
            failures,
            non_finite_failures,
            max_absolute,
            max_relative,
            rms,
        },
        first_divergence,
    ))
}

fn tensor_summary(tensor: &NumericTensor) -> TensorSummary {
    let mut bytes = Vec::new();
    for dimension in &tensor.shape {
        let dimension = u64::try_from(*dimension)
            .expect("validated tensor dimensions always fit the report u64 schema");
        bytes.extend_from_slice(&dimension.to_le_bytes());
    }
    for value in &tensor.values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    TensorSummary {
        name: tensor.name.clone(),
        shape: tensor.shape.clone(),
        dtype: "f32",
        sha256: sha256_hex(&bytes),
    }
}

fn row_major_coordinate(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut coordinate = vec![0; shape.len()];
    for (axis, dimension) in shape.iter().enumerate().rev() {
        coordinate[axis] = index % dimension;
        index /= dimension;
    }
    coordinate
}

fn checked_elements(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |elements, dimension| {
        if *dimension == 0 {
            return Err("numeric tensor dimensions must be non-zero".to_string());
        }
        u64::try_from(*dimension)
            .map_err(|_| "numeric tensor dimension exceeds the report u64 schema".to_string())?;
        elements
            .checked_mul(*dimension)
            .ok_or_else(|| "numeric tensor element count overflows usize".to_string())
    })
}

fn validate_identity_component(name: &str, value: &str) -> Result<(), String> {
    if value.is_empty() {
        return Err(format!("{name} must not be empty"));
    }
    if value.contains(['\0', '\n', '\r']) {
        return Err(format!("{name} contains a reserved identity separator"));
    }
    Ok(())
}

fn validate_contract_identity(value: &str) -> Result<(), String> {
    if value.is_empty() {
        return Err("numeric oracle operator contract identity must not be empty".to_string());
    }
    if value.contains(['\n', '\r']) {
        return Err(
            "numeric oracle operator contract identity contains a line separator".to_string(),
        );
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric_dtype_contract::{NumericDType, WeightExecution};
    use crate::operator_numeric_contract::{
        AffineDequantizationContract, AffineEvaluationOrder, AssociationPolicy,
        BackendNumericIdentity, CheckpointDType, OperatorClass, OperatorNumericContract,
        PackedLaneOrder, RoundingBoundary,
    };

    fn backend(runtime: &str, observed: bool) -> NumericBackendIdentity {
        let algorithm = if observed {
            AlgorithmIdentity::observed(format!("{runtime}-kernel-v1")).unwrap()
        } else {
            AlgorithmIdentity::unobserved("runtime exposes no selected kernel identity").unwrap()
        };
        NumericBackendIdentity::new(
            runtime,
            "revision-1",
            "patch-1",
            "compiler-1",
            vec!["--target=local-task".to_string()],
            "local-task",
            "cpu",
            format!("{runtime}-artifact-v1"),
            algorithm,
        )
        .unwrap()
    }

    fn operator_contract() -> OperatorNumericContractSet {
        let backend = BackendNumericIdentity::new(
            "mlx-revision",
            "patch-1",
            "iree-compiler-1",
            "local-task",
            "q4-reference-v1",
        )
        .unwrap();
        let dequantization = AffineDequantizationContract::new(
            4,
            64,
            NumericDType::U32,
            NumericDType::F16,
            PackedLaneOrder::LeastSignificantFirst,
            AffineEvaluationOrder::SeparateMultiplyThenAdd,
        )
        .unwrap();
        let operator = OperatorNumericContract::new(
            "probe.q4",
            OperatorClass::Q4Matmul,
            CheckpointDType::AffineU4,
            NumericDType::F16,
            NumericDType::F32,
            NumericDType::F32,
            [
                RoundingBoundary::DequantizedWeight,
                RoundingBoundary::AccumulatorResult,
            ],
            AssociationPolicy::BackendAlgorithm,
            WeightExecution::PackedAffineInGraph,
            Some(dequantization),
            backend,
        )
        .unwrap();
        OperatorNumericContractSet::new([operator]).unwrap()
    }

    fn case(
        claim: NumericOracleClaim,
        observed_algorithms: bool,
        mode: ComparisonMode,
    ) -> NumericOracleCase {
        NumericOracleCase::from_operator_contract(
            "probe.q4",
            claim,
            &operator_contract(),
            backend("mlx", observed_algorithms),
            backend("iree", observed_algorithms),
            mode,
        )
        .unwrap()
    }

    #[test]
    fn exact_comparison_reports_first_row_major_coordinate_and_bits() {
        let operands = [NumericTensor::new("input", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap()];
        let report = run_bounded_numeric_oracle(
            case(
                NumericOracleClaim::ProductionMlx,
                true,
                ComparisonMode::ExactBits,
            ),
            &operands,
            |_| NumericTensor::new("output", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
            |_| NumericTensor::new("output", vec![2, 2], vec![1.0, 2.5, 9.0, 4.0]),
        )
        .unwrap();
        assert!(!report.passed());
        assert!(!report.production_qualified());
        let first = report.first_divergence().unwrap();
        assert_eq!(first.flat_index, 1);
        assert_eq!(first.coordinate, vec![0, 1]);
        assert_eq!(first.reference_bits, 2.0f32.to_bits());
        assert_eq!(first.candidate_bits, 2.5f32.to_bits());
    }

    #[test]
    fn tolerance_mode_rejects_non_finite_values() {
        let operands = [NumericTensor::new("input", vec![1], vec![1.0]).unwrap()];
        let report = run_bounded_numeric_oracle(
            case(
                NumericOracleClaim::MathematicalCloseness,
                false,
                ComparisonMode::AbsoluteRelative {
                    absolute: 1e-5,
                    relative: 1e-5,
                },
            ),
            &operands,
            |_| NumericTensor::new("output", vec![2], vec![1.0, f32::INFINITY]),
            |_| NumericTensor::new("output", vec![2], vec![1.0, f32::INFINITY]),
        )
        .unwrap();
        assert!(!report.passed());
        assert_eq!(report.comparison.non_finite_failures, 1);
        assert_eq!(report.first_divergence().unwrap().coordinate, vec![1]);
    }

    #[test]
    fn qualification_is_derived_from_claim_and_observed_algorithms() {
        let operands = [NumericTensor::new("input", vec![1], vec![3.0]).unwrap()];
        for (claim, observed, expected) in [
            (NumericOracleClaim::ProductionMlx, true, true),
            (NumericOracleClaim::ProductionMlx, false, false),
            (NumericOracleClaim::CanonicalDecomposition, true, false),
            (NumericOracleClaim::MathematicalCloseness, true, false),
        ] {
            let report = run_bounded_numeric_oracle(
                case(claim, observed, ComparisonMode::ExactBits),
                &operands,
                |_| NumericTensor::new("output", vec![1], vec![7.0]),
                |_| NumericTensor::new("output", vec![1], vec![7.0]),
            )
            .unwrap();
            assert!(report.passed());
            assert_eq!(report.production_qualified(), expected);
        }
    }

    #[test]
    fn operand_digest_and_contract_identity_are_mutation_sensitive() {
        let first = [NumericTensor::new("input", vec![2], vec![1.0, 2.0]).unwrap()];
        let second = [NumericTensor::new("input", vec![2], vec![1.0, 3.0]).unwrap()];
        let run = |operands: &[NumericTensor]| {
            run_bounded_numeric_oracle(
                case(
                    NumericOracleClaim::CanonicalDecomposition,
                    true,
                    ComparisonMode::ExactBits,
                ),
                operands,
                |values| Ok(values[0].clone()),
                |values| Ok(values[0].clone()),
            )
            .unwrap()
        };
        let first_report = run(&first);
        let second_report = run(&second);
        assert_ne!(
            first_report.operands[0].sha256,
            second_report.operands[0].sha256
        );
        assert_eq!(
            first_report.operator_contract_sha256,
            second_report.operator_contract_sha256
        );
        let json = serde_json::to_value(first_report).unwrap();
        assert_eq!(json["schema"], REPORT_SCHEMA);
        assert_eq!(json["first_divergence"], serde_json::Value::Null);
        assert_eq!(json["passed"], true);
        assert_eq!(json["production_qualified"], false);
    }

    #[test]
    fn malformed_shapes_tolerances_and_identities_fail_closed() {
        assert!(NumericTensor::new("bad", vec![2, 2], vec![1.0]).is_err());
        assert!(NumericTensor::new("bad", vec![0], Vec::new()).is_err());
        assert!(
            ComparisonMode::AbsoluteRelative {
                absolute: f64::NAN,
                relative: 0.0,
            }
            .validate()
            .is_err()
        );
        assert!(AlgorithmIdentity::observed("").is_err());
        assert!(
            NumericBackendIdentity::new(
                "mlx",
                "revision",
                "patch",
                "compiler",
                vec!["--target=local-task".to_string()],
                "target",
                "",
                "artifact",
                AlgorithmIdentity::unobserved("not exposed").unwrap(),
            )
            .is_err()
        );
    }
}
