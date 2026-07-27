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
    DENSE_MATMUL_ENTRY, DenseMatmulProbeSpec, emit_dense_matmul_probe,
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

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four-byte f32 chunk")))
        .collect()
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
) -> Result<OperatorNumericContractSet, String> {
    let backend = BackendNumericIdentity::new(
        "canonical-decomposition-v1",
        runtime_identity,
        compiler_version.replace(';', ","),
        device,
        "iree-selected-contraction-unobserved",
    )?;
    let operation = OperatorNumericContract::new(
        DENSE_OPERATION,
        OperatorClass::DenseMatmul,
        CheckpointDType::F32,
        NumericDType::F32,
        NumericDType::F32,
        NumericDType::F32,
        [
            RoundingBoundary::OperatorInput,
            RoundingBoundary::AccumulatorResult,
            RoundingBoundary::OperatorOutput,
        ],
        AssociationPolicy::BackendAlgorithm,
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

fn resident_weight(weight: &NumericTensor) -> AuxiliaryWeight {
    AuxiliaryWeight {
        name: "dense.weight".to_string(),
        bytes: f32_bytes(weight.values()),
        dtype: AuxiliaryWeightDType::Float32,
        shape: weight.shape().to_vec(),
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

/// Compile and execute one deterministic dense-matmul probe through the native
/// auxiliary IREE ABI.
///
/// The reference is an explicitly reported canonical Rust decomposition, not a
/// production MLX kernel. Consequently this probe can validate the shared
/// emitter/runtime path but can never qualify a model family for production.
pub fn run_dense_matmul_probe(device: &str) -> Result<NumericOracleReport, String> {
    if !device.starts_with("local") {
        return Err(format!(
            "the initial dense matmul canonical probe supports only local IREE CPU targets, got \
             {device:?}"
        ));
    }
    let spec = DenseMatmulProbeSpec {
        rows: DENSE_ROWS,
        outputs: DENSE_OUTPUTS,
        contraction: DENSE_CONTRACTION,
    };
    let mlir = emit_dense_matmul_probe(spec)?;
    let compiler = iree_compile_bin()?;
    let flags = target_flags(device)?;
    let compiler_version = compiler_version(&compiler)?;
    let runtime_identity = runtime_archive_identity(&compiler)?;
    let operator_contract = operator_identity(&compiler_version, &runtime_identity, device)?;
    let generation_identity = format!(
        "compiler={};version={};runtime={runtime_identity};flags={flags:?};mlir_sha256={}",
        compiler.display(),
        compiler_version,
        sha256_hex(mlir.as_bytes())
    );
    let artifact_contract = AuxiliaryArtifactContract::new_with_operator_numeric_contract(
        DENSE_MATMUL_ENTRY,
        format!(
            "rows={DENSE_ROWS};outputs={DENSE_OUTPUTS};contraction={DENSE_CONTRACTION};dtype=f32"
        ),
        generation_identity,
        operator_contract.clone(),
    )?;
    let operands = dense_operands()?;
    let weights = vec![resident_weight(&operands[1])];
    let cache = std::env::temp_dir().join("mlxcel-xla-numeric-probes");
    std::fs::create_dir_all(&cache)
        .map_err(|error| format!("mkdir {}: {error}", cache.display()))?;
    let vmfb = cached_vmfb_path(&compiler, &mlir, flags, &cache, "dense-matmul-f32", 0);
    ensure_qualified_auxiliary_artifact(&vmfb, &artifact_contract, &weights, |output| {
        compile_one_to(
            &compiler,
            &mlir,
            flags,
            &cache,
            "dense-matmul-f32",
            0,
            output,
        )
    })?;
    let mut module = IreeAuxiliaryModule::load(device, &vmfb, &artifact_contract, weights)?;
    let artifact_fingerprint = format!("{:016x}", module.fingerprint());
    let reference_backend = NumericBackendIdentity::new(
        "rust-canonical-decomposition",
        env!("CARGO_PKG_VERSION"),
        "dense-matmul-v1",
        "rust-f32",
        vec!["row-major-k-sequential".to_string()],
        std::env::consts::ARCH,
        "host-cpu",
        "rust-dense-matmul-v1",
        AlgorithmIdentity::observed("row-major-k-sequential-v1")?,
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
            "IREE does not expose the selected CPU contraction kernel identity",
        )?,
    )?;
    let case = NumericOracleCase::from_operator_contract(
        DENSE_OPERATION,
        NumericOracleClaim::CanonicalDecomposition,
        &operator_contract,
        reference_backend,
        candidate_backend,
        ComparisonMode::ExactBits,
    )?;
    run_bounded_numeric_oracle(case, &operands, reference_dense_matmul, move |operands| {
        let [input, weight] = operands else {
            return Err(format!(
                "dense matmul IREE candidate requires 2 operands, got {}",
                operands.len()
            ));
        };
        if weight.shape() != [DENSE_OUTPUTS, DENSE_CONTRACTION] {
            return Err(format!(
                "dense matmul IREE resident weight shape changed to {:?}",
                weight.shape()
            ));
        }
        let input_bytes = f32_bytes(input.values());
        let input_shape = [DENSE_ROWS, DENSE_CONTRACTION];
        let output_shape = [DENSE_ROWS, DENSE_OUTPUTS];
        let mut output_bytes = vec![0u8; DENSE_ROWS * DENSE_OUTPUTS * size_of::<f32>()];
        module.invoke(
            &[AuxiliaryInput {
                bytes: &input_bytes,
                dtype: AuxiliaryTensorDType::Float32,
                shape: &input_shape,
            }],
            &mut [AuxiliaryOutput {
                bytes: &mut output_bytes,
                dtype: AuxiliaryTensorDType::Float32,
                shape: &output_shape,
            }],
        )?;
        NumericTensor::new("output", output_shape.to_vec(), decode_f32(&output_bytes))
    })
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
}
