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

//! Real-runtime acceptance probe for the generic auxiliary ABI.

use std::path::Path;
use std::process::Command;

use crate::aux::{
    AuxiliaryInput, AuxiliaryOutput, AuxiliaryTensorDType, AuxiliaryWeight, AuxiliaryWeightDType,
    IreeAuxiliaryModule,
};
use crate::aux_manifest::{AuxiliaryArtifactContract, write_auxiliary_manifest};
use crate::iree::{compile_one, iree_compile_bin, target_flags};

const SMOKE_MLIR: &str = r#"module @aux_smoke {
  func.func public @main(
      %weight: tensor<2xf32>,
      %floats: tensor<2xf32>,
      %integers: tensor<2xi32>,
      %mask: tensor<2xi1>
  ) -> (tensor<2xf32>, tensor<2xi32>, tensor<2xi1>) {
    %sum = stablehlo.add %weight, %floats : tensor<2xf32>
    return %sum, %integers, %mask : tensor<2xf32>, tensor<2xi32>, tensor<2xi1>
  }
}
"#;

#[derive(Debug, Clone, PartialEq)]
pub struct AuxiliaryAbiSmokeReport {
    pub float_output: Vec<f32>,
    pub integer_output: Vec<i32>,
    pub bool_output: Vec<bool>,
    pub too_few_outputs_rejected: bool,
    pub too_many_outputs_rejected: bool,
    pub output_type_mismatch_rejected: bool,
    pub output_shape_mismatch_rejected: bool,
    pub config_identity_mismatch_rejected: bool,
}

fn weight() -> AuxiliaryWeight {
    AuxiliaryWeight {
        name: "smoke.weight".to_string(),
        bytes: f32_bytes(&[1.5, -2.0]),
        dtype: AuxiliaryWeightDType::Float32,
        shape: vec![2],
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect()
}

fn decode_i32(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_ne_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect()
}

fn invocation_inputs<'a>(
    floats: &'a [u8],
    integers: &'a [u8],
    mask: &'a [u8],
) -> [AuxiliaryInput<'a>; 3] {
    [
        AuxiliaryInput {
            bytes: floats,
            dtype: AuxiliaryTensorDType::Float32,
            shape: &[2],
        },
        AuxiliaryInput {
            bytes: integers,
            dtype: AuxiliaryTensorDType::Int32,
            shape: &[2],
        },
        AuxiliaryInput {
            bytes: mask,
            dtype: AuxiliaryTensorDType::Bool,
            shape: &[2],
        },
    ]
}

fn generation_identity(compiler: &Path, flags: &[&str]) -> Result<String, String> {
    let version = Command::new(compiler)
        .arg("--version")
        .output()
        .map_err(|error| format!("run {} --version: {error}", compiler.display()))?;
    if !version.status.success() {
        return Err(format!(
            "{} --version failed: {}",
            compiler.display(),
            String::from_utf8_lossy(&version.stderr)
        ));
    }
    Ok(format!(
        "compiler={};version={};flags={flags:?};mlir={SMOKE_MLIR}",
        compiler.display(),
        String::from_utf8_lossy(&version.stdout).trim()
    ))
}

pub fn run_auxiliary_abi_smoke(device: &str) -> Result<AuxiliaryAbiSmokeReport, String> {
    let compiler = iree_compile_bin()?;
    let flags = target_flags(device)?;
    let cache = std::env::temp_dir().join("mlxcel-xla-aux-smoke");
    std::fs::create_dir_all(&cache)
        .map_err(|error| format!("mkdir {}: {error}", cache.display()))?;
    let vmfb = compile_one(&compiler, SMOKE_MLIR, flags, &cache, "aux-smoke", 0)?;
    let contract = AuxiliaryArtifactContract::new(
        "aux_smoke.main",
        "aux-smoke-config-v1",
        generation_identity(&compiler, flags)?,
    )?;
    write_auxiliary_manifest(&vmfb, &contract, &[weight()])?;

    let wrong_contract = AuxiliaryArtifactContract::new(
        "aux_smoke.main",
        "aux-smoke-config-v2",
        contract.generation_identity.clone(),
    )?;
    let config_identity_mismatch_rejected =
        IreeAuxiliaryModule::load(device, &vmfb, &wrong_contract, vec![weight()]).is_err();

    let mut module = IreeAuxiliaryModule::load(device, &vmfb, &contract, vec![weight()])?;
    let floats = f32_bytes(&[2.5, 8.0]);
    let integers = i32_bytes(&[7, -3]);
    let mask = [1u8, 0u8];
    let inputs = invocation_inputs(&floats, &integers, &mask);

    let mut float_bytes = vec![0u8; 8];
    let mut integer_bytes = vec![0u8; 8];
    let mut bool_bytes = vec![0u8; 2];
    module.invoke(
        &inputs,
        &mut [
            AuxiliaryOutput {
                bytes: &mut float_bytes,
                dtype: AuxiliaryTensorDType::Float32,
                shape: &[2],
            },
            AuxiliaryOutput {
                bytes: &mut integer_bytes,
                dtype: AuxiliaryTensorDType::Int32,
                shape: &[2],
            },
            AuxiliaryOutput {
                bytes: &mut bool_bytes,
                dtype: AuxiliaryTensorDType::Bool,
                shape: &[2],
            },
        ],
    )?;

    let mut two_outputs_a = vec![0u8; 8];
    let mut two_outputs_b = vec![0u8; 8];
    let too_few_outputs_rejected = module
        .invoke(
            &inputs,
            &mut [
                AuxiliaryOutput {
                    bytes: &mut two_outputs_a,
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &[2],
                },
                AuxiliaryOutput {
                    bytes: &mut two_outputs_b,
                    dtype: AuxiliaryTensorDType::Int32,
                    shape: &[2],
                },
            ],
        )
        .is_err();

    let mut extra_a = vec![0u8; 8];
    let mut extra_b = vec![0u8; 8];
    let mut extra_c = vec![0u8; 2];
    let mut extra_d = vec![0u8; 2];
    let too_many_outputs_rejected = module
        .invoke(
            &inputs,
            &mut [
                AuxiliaryOutput {
                    bytes: &mut extra_a,
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &[2],
                },
                AuxiliaryOutput {
                    bytes: &mut extra_b,
                    dtype: AuxiliaryTensorDType::Int32,
                    shape: &[2],
                },
                AuxiliaryOutput {
                    bytes: &mut extra_c,
                    dtype: AuxiliaryTensorDType::Bool,
                    shape: &[2],
                },
                AuxiliaryOutput {
                    bytes: &mut extra_d,
                    dtype: AuxiliaryTensorDType::Bool,
                    shape: &[2],
                },
            ],
        )
        .is_err();

    let mut wrong_type = vec![0u8; 8];
    let mut wrong_type_i = vec![0u8; 8];
    let mut wrong_type_b = vec![0u8; 2];
    let output_type_mismatch_rejected = module
        .invoke(
            &inputs,
            &mut [
                AuxiliaryOutput {
                    bytes: &mut wrong_type,
                    dtype: AuxiliaryTensorDType::Int32,
                    shape: &[2],
                },
                AuxiliaryOutput {
                    bytes: &mut wrong_type_i,
                    dtype: AuxiliaryTensorDType::Int32,
                    shape: &[2],
                },
                AuxiliaryOutput {
                    bytes: &mut wrong_type_b,
                    dtype: AuxiliaryTensorDType::Bool,
                    shape: &[2],
                },
            ],
        )
        .is_err();

    let mut shape_f = vec![0u8; 8];
    let mut wrong_shape = vec![0u8; 8];
    let mut shape_b = vec![0u8; 2];
    let output_shape_mismatch_rejected = module
        .invoke(
            &inputs,
            &mut [
                AuxiliaryOutput {
                    bytes: &mut shape_f,
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &[2],
                },
                AuxiliaryOutput {
                    bytes: &mut wrong_shape,
                    dtype: AuxiliaryTensorDType::Int32,
                    shape: &[1, 2],
                },
                AuxiliaryOutput {
                    bytes: &mut shape_b,
                    dtype: AuxiliaryTensorDType::Bool,
                    shape: &[2],
                },
            ],
        )
        .is_err();

    let report = AuxiliaryAbiSmokeReport {
        float_output: decode_f32(&float_bytes),
        integer_output: decode_i32(&integer_bytes),
        bool_output: bool_bytes.iter().map(|value| *value != 0).collect(),
        too_few_outputs_rejected,
        too_many_outputs_rejected,
        output_type_mismatch_rejected,
        output_shape_mismatch_rejected,
        config_identity_mismatch_rejected,
    };
    if report.float_output != [4.0, 6.0]
        || report.integer_output != [7, -3]
        || report.bool_output != [true, false]
        || !report.too_few_outputs_rejected
        || !report.too_many_outputs_rejected
        || !report.output_type_mismatch_rejected
        || !report.output_shape_mismatch_rejected
        || !report.config_identity_mismatch_rejected
    {
        return Err(format!("auxiliary ABI smoke contract failed: {report:?}"));
    }
    Ok(report)
}
