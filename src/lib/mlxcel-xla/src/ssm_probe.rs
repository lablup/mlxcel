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

//! Single-layer SSM mixer probe runner (issue #3337).
//!
//! Emits one Mamba2 mixer layer ([`crate::emitter::emit_ssm_probe_with`]),
//! compiles it for a device, and runs it through the auxiliary IREE ABI with
//! the mixer weights resident and the activations and state as inputs. The
//! `xla_ssm_probe` example feeds it the inputs `spike/openxla/ssm_oracle.py`
//! wrote and hands the outputs back to the oracle for comparison.

use std::process::Command;
use std::time::Instant;

use crate::aux_manifest::{AuxiliaryArtifactContract, write_auxiliary_manifest};
use crate::auxiliary::{
    AuxiliaryInput, AuxiliaryOutput, AuxiliaryTensorDType, AuxiliaryWeight, AuxiliaryWeightDType,
    AuxiliaryWeightStorage, IreeAuxiliaryModule,
};
use crate::emitter::{
    Precision, ProbeMode, SsmConfig, SsmPrefillForm, emit_ssm_probe_with, precision_env_override,
};
use crate::iree::{compile_one, iree_compile_bin, target_flags};

/// The probe's dimensions and mode, as `ssm_oracle.py` records them.
#[derive(Clone, Debug, PartialEq)]
pub struct SsmProbeSpec {
    pub hidden: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub state_size: usize,
    pub n_groups: usize,
    pub conv_kernel: usize,
    pub use_conv_bias: bool,
    pub use_bias: bool,
    pub time_step_limit: (f32, f32),
    pub norm_eps: f32,
    pub norm_group_size: usize,
    pub chunk_size: usize,
    /// Prefill: the padded capacity `T`. Decode: the batch rows `B`.
    pub rows: usize,
    /// `true` for one decode step, `false` for a padded prefill.
    pub decode: bool,
    /// `chunked`, `associative`, or `sequential` (prefill only).
    pub form: String,
}

impl SsmProbeSpec {
    fn config(&self) -> Result<SsmConfig, String> {
        let d_inner = self.n_heads * self.head_dim;
        let config = SsmConfig {
            n_heads: self.n_heads,
            head_dim: self.head_dim,
            state_size: self.state_size,
            n_groups: self.n_groups,
            conv_kernel: self.conv_kernel,
            conv_dim: d_inner + 2 * self.n_groups * self.state_size,
            use_conv_bias: self.use_conv_bias,
            use_bias: self.use_bias,
            time_step_limit: self.time_step_limit,
            norm_eps: self.norm_eps,
            norm_group_size: self.norm_group_size,
            chunk_size: self.chunk_size,
        };
        config.validate()?;
        Ok(config)
    }

    /// The resident weight names and shapes, in probe argument order.
    pub fn weight_layout(&self) -> Result<Vec<(&'static str, Vec<usize>)>, String> {
        let c = self.config()?;
        let (cd, h) = (c.conv_dim, c.n_heads);
        let mut layout = vec![("in_proj", vec![c.in_proj_dim(), self.hidden])];
        if c.use_bias {
            layout.push(("in_proj_b", vec![c.in_proj_dim()]));
        }
        layout.push(("conv_w", vec![cd, 1, c.conv_kernel]));
        if c.use_conv_bias {
            layout.push(("conv_b", vec![cd]));
        }
        layout.extend([
            ("dt_bias", vec![h]),
            ("a_log", vec![h]),
            ("d", vec![h]),
            ("norm_w", vec![c.d_inner()]),
            ("out_proj", vec![self.hidden, c.d_inner()]),
        ]);
        if c.use_bias {
            layout.push(("out_proj_b", vec![self.hidden]));
        }
        Ok(layout)
    }

    /// Shapes of `(x, h0, conv0)`; the outputs `(y, h_T, conv_tail)` match them.
    pub fn io_shapes(&self) -> Result<[Vec<usize>; 3], String> {
        let c = self.config()?;
        let (h, p, n) = (c.n_heads, c.head_dim, c.state_size);
        let tail = c.conv_kernel - 1;
        Ok(if self.decode {
            [
                vec![self.rows, self.hidden],
                vec![self.rows, h, p, n],
                vec![self.rows, tail, c.conv_dim],
            ]
        } else {
            [
                vec![self.rows, self.hidden],
                vec![h, p, n],
                vec![tail, c.conv_dim],
            ]
        })
    }

    /// The probe's StableHLO text. f32 contractions (the HF fp32 gate) unless
    /// `MLXCEL_XLA_PRECISION` asks for the f16 / bf16 projection path.
    pub fn emit(&self) -> Result<String, String> {
        let c = self.config()?;
        let mode = if self.decode {
            ProbeMode::Decode
        } else {
            ProbeMode::Prefill
        };
        let form = SsmPrefillForm::parse(&self.form)?;
        let precision = precision_env_override().unwrap_or(Precision::F32);
        Ok(emit_ssm_probe_with(
            &c,
            self.hidden,
            self.rows,
            mode,
            form,
            precision,
        ))
    }
}

/// The probe's activations and state, flat row-major f32.
#[derive(Clone, Debug, Default)]
pub struct SsmProbeInputs {
    /// One entry per [`SsmProbeSpec::weight_layout`] row, in order.
    pub weights: Vec<Vec<f32>>,
    pub x: Vec<f32>,
    /// Prefill only.
    pub real_len: i32,
    pub h0: Vec<f32>,
    pub conv0: Vec<f32>,
}

/// The probe results, flat row-major f32.
#[derive(Clone, Debug, Default)]
pub struct SsmProbeOutputs {
    pub y: Vec<f32>,
    pub h: Vec<f32>,
    pub conv: Vec<f32>,
    pub compile_seconds: f64,
    pub run_seconds: f64,
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn compiler_version(compiler: &std::path::Path) -> Result<String, String> {
    let out = Command::new(compiler)
        .arg("--version")
        .output()
        .map_err(|error| format!("run {} --version: {error}", compiler.display()))?;
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn check_len(label: &str, values: &[f32], shape: &[usize]) -> Result<(), String> {
    let expected: usize = shape.iter().product();
    if values.len() == expected {
        Ok(())
    } else {
        Err(format!(
            "{label} has {} values, expected {expected} for {shape:?}",
            values.len()
        ))
    }
}

/// Compile the probe for `device` and run it once on `inputs`.
pub fn run_ssm_probe(
    device: &str,
    spec: &SsmProbeSpec,
    inputs: &SsmProbeInputs,
) -> Result<SsmProbeOutputs, String> {
    let layout = spec.weight_layout()?;
    if layout.len() != inputs.weights.len() {
        return Err(format!(
            "probe takes {} weights, got {}",
            layout.len(),
            inputs.weights.len()
        ));
    }
    let [x_shape, h_shape, conv_shape] = spec.io_shapes()?;
    check_len("x", &inputs.x, &x_shape)?;
    check_len("h0", &inputs.h0, &h_shape)?;
    check_len("conv0", &inputs.conv0, &conv_shape)?;
    let mut weights = Vec::with_capacity(layout.len());
    for ((name, shape), values) in layout.iter().zip(&inputs.weights) {
        check_len(name, values, shape)?;
        weights.push(AuxiliaryWeight {
            name: format!("ssm.{name}"),
            storage: AuxiliaryWeightStorage::Float32(values.clone()),
            dtype: AuxiliaryWeightDType::Float32,
            shape: shape.clone(),
        });
    }

    let mlir = spec.emit()?;
    let compiler = iree_compile_bin()?;
    let flags = target_flags(device)?;
    let cache = crate::cache_dir::xla_cache_dir(crate::cache_dir::XlaCache::SsmProbe)?;
    let started = Instant::now();
    let vmfb = compile_one(&compiler, &mlir, &flags, &cache, "ssm-probe", spec.rows)?;
    let compile_seconds = started.elapsed().as_secs_f64();
    let contract = AuxiliaryArtifactContract::new_legacy_unqualified(
        "ssm_probe.main",
        format!("{spec:?}"),
        format!(
            "compiler={};version={};flags={flags:?}",
            compiler.display(),
            compiler_version(&compiler)?
        ),
    )?;
    write_auxiliary_manifest(&vmfb, &contract, &weights)?;
    let mut module = IreeAuxiliaryModule::load(device, &vmfb, &contract, weights)?;

    let x = f32_bytes(&inputs.x);
    let real_len = inputs.real_len.to_ne_bytes();
    let h0 = f32_bytes(&inputs.h0);
    let conv0 = f32_bytes(&inputs.conv0);
    let mut args = vec![AuxiliaryInput {
        bytes: &x,
        dtype: AuxiliaryTensorDType::Float32,
        shape: &x_shape,
    }];
    if !spec.decode {
        args.push(AuxiliaryInput {
            bytes: &real_len,
            dtype: AuxiliaryTensorDType::Int32,
            shape: &[],
        });
    }
    args.push(AuxiliaryInput {
        bytes: &h0,
        dtype: AuxiliaryTensorDType::Float32,
        shape: &h_shape,
    });
    args.push(AuxiliaryInput {
        bytes: &conv0,
        dtype: AuxiliaryTensorDType::Float32,
        shape: &conv_shape,
    });
    let mut y = vec![0u8; x.len()];
    let mut h = vec![0u8; h0.len()];
    let mut conv = vec![0u8; conv0.len()];
    let started = Instant::now();
    module.invoke(
        &args,
        &mut [
            AuxiliaryOutput {
                bytes: &mut y,
                dtype: AuxiliaryTensorDType::Float32,
                shape: &x_shape,
            },
            AuxiliaryOutput {
                bytes: &mut h,
                dtype: AuxiliaryTensorDType::Float32,
                shape: &h_shape,
            },
            AuxiliaryOutput {
                bytes: &mut conv,
                dtype: AuxiliaryTensorDType::Float32,
                shape: &conv_shape,
            },
        ],
    )?;
    Ok(SsmProbeOutputs {
        y: decode_f32(&y),
        h: decode_f32(&h),
        conv: decode_f32(&conv),
        compile_seconds,
        run_seconds: started.elapsed().as_secs_f64(),
    })
}
