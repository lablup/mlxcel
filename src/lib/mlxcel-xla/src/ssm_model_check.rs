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

//! Whole-model Mamba2 graph check (issue #3338).
//!
//! Loads a Mamba2 checkpoint through the production config reader and weight
//! loader, emits the state-threaded token prefill ([`emit_prefill_with`]),
//! compiles it, and runs it through the auxiliary IREE ABI with the weights
//! resident. `spike/openxla/ssm_model_check.py` drives it through the
//! `xla_ssm_model_check` example and compares the logits with HF
//! `Mamba2ForCausalLM` in fp32. [`compile_ssm_model_graphs`] also compiles the
//! single-sequence decode module, which the runtime of #3339 executes.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use crate::aux_manifest::{AuxiliaryArtifactContract, write_auxiliary_manifest};
use crate::auxiliary::{
    AuxiliaryInput, AuxiliaryOutput, AuxiliaryTensorDType, AuxiliaryWeight, AuxiliaryWeightDType,
    AuxiliaryWeightStorage, IreeAuxiliaryModule,
};
use crate::emitter::{
    Config, Precision, ProjectionLayout, emit_decode_buffer_abi, emit_decode_with,
    emit_prefill_with,
};
use crate::iree::{
    HostWeight, compile_one, compile_one_to, iree_compile_bin, load_host_weights, target_flags,
};

/// One threaded state tensor of the check, flat row-major f32.
#[derive(Clone, Debug, PartialEq)]
pub struct SsmModelState {
    pub name: String,
    pub dims: Vec<usize>,
    pub values: Vec<f32>,
}

/// The result of one prefill call.
#[derive(Clone, Debug, Default)]
pub struct SsmModelPrefill {
    /// Logits of the last live row (`real_len - 1`).
    pub logits: Vec<f32>,
    /// The advanced state, in `Config::state_tensors` order.
    pub states: Vec<SsmModelState>,
    pub run_seconds: f64,
}

/// Wall time of one fresh `iree-compile` per graph.
#[derive(Clone, Debug, Default)]
pub struct SsmModelCompileTimes {
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
}

/// Parse the contraction precision name (`f32`, `f16`, `bf16`).
fn parse_precision(name: &str) -> Result<Precision, String> {
    match name {
        "f32" => Ok(Precision::F32),
        "f16" => Ok(Precision::F16),
        "bf16" => Ok(Precision::Bf16),
        other => Err(format!("unknown precision {other:?}; use f32, f16 or bf16")),
    }
}

/// Parse the projection layout name. `inout` is the layout `IreeLlama` (the
/// single-sequence runtime #3339 drives) always emits and uploads; `outin` is
/// the checkpoint layout the batching engine keeps.
fn parse_layout(name: &str) -> Result<ProjectionLayout, String> {
    match name {
        "inout" => Ok(ProjectionLayout::InOut),
        "outin" => Ok(ProjectionLayout::OutIn),
        other => Err(format!("unknown layout {other:?}; use inout or outin")),
    }
}

fn load_config(model_dir: &Path, capacity: usize, layout: &str) -> Result<Config, String> {
    let mut config = Config::from_json(model_dir)?.with_context_capacity(capacity)?;
    config.projection_layout = parse_layout(layout)?;
    if !config.has_recurrent_state() {
        return Err(format!(
            "{} is not a recurrent-state model; the SSM model check runs Mamba2 only",
            model_dir.display()
        ));
    }
    Ok(config)
}

fn compiler_version(compiler: &Path) -> Result<String, String> {
    let out = Command::new(compiler)
        .arg("--version")
        .output()
        .map_err(|error| format!("run {} --version: {error}", compiler.display()))?;
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Emit the prefill and single-sequence decode graphs for `model_dir` and
/// compile each once for `device`, uncached, into `out_dir` (which also gets
/// the `.mlir` text), returning the compile wall times.
pub fn compile_ssm_model_graphs(
    device: &str,
    model_dir: &Path,
    capacity: usize,
    precision: &str,
    layout: &str,
    out_dir: &Path,
) -> Result<SsmModelCompileTimes, String> {
    let precision = parse_precision(precision)?;
    let config = load_config(model_dir, capacity, layout)?;
    let compiler = iree_compile_bin()?;
    let flags = target_flags(device)?;
    std::fs::create_dir_all(out_dir)
        .map_err(|error| format!("create {}: {error}", out_dir.display()))?;
    let mut seconds = [0.0f64; 2];
    let graphs = [
        ("prefill", emit_prefill_with(&config, true, precision)),
        ("decode", emit_decode_with(&config, true, precision)),
    ];
    for (slot, (name, mlir)) in seconds.iter_mut().zip(graphs) {
        std::fs::write(out_dir.join(format!("{name}.mlir")), &mlir)
            .map_err(|error| format!("write {name}.mlir: {error}"))?;
        let output: PathBuf = out_dir.join(format!("{name}.{device}.vmfb"));
        let started = Instant::now();
        compile_one_to(&compiler, &mlir, &flags, out_dir, name, capacity, &output)?;
        *slot = started.elapsed().as_secs_f64();
    }
    Ok(SsmModelCompileTimes {
        prefill_seconds: seconds[0],
        decode_seconds: seconds[1],
    })
}

/// Compile `mlir` for `device` (cached) and load it with the checkpoint's
/// weights resident, as a single-entry auxiliary module calling `entry`.
fn load_module(
    device: &str,
    model_dir: &Path,
    config: &Config,
    precision: Precision,
    mlir: &str,
    entry: &str,
) -> Result<(IreeAuxiliaryModule, f64), String> {
    // The runtime's rule: an f16 graph declares its projections f16-resident.
    let resident_f16 = precision == Precision::F16 && config.supports_f16_resident();
    let weights = load_host_weights(model_dir, config, resident_f16)?
        .into_iter()
        .enumerate()
        .map(|(index, (weight, shape))| {
            let (storage, dtype) = match weight {
                HostWeight::F32(values) => (
                    AuxiliaryWeightStorage::Float32(values),
                    AuxiliaryWeightDType::Float32,
                ),
                HostWeight::F16(bits) => (
                    AuxiliaryWeightStorage::Bytes(
                        bits.iter().flat_map(|b| b.to_ne_bytes()).collect(),
                    ),
                    AuxiliaryWeightDType::Float16,
                ),
            };
            AuxiliaryWeight {
                name: format!("ssm_model.w{index}"),
                storage,
                dtype,
                shape,
            }
        })
        .collect::<Vec<_>>();
    let compiler = iree_compile_bin()?;
    let flags = target_flags(device)?;
    let cache = crate::cache_dir::xla_cache_dir(crate::cache_dir::XlaCache::SsmProbe)?;
    let started = Instant::now();
    let tag = entry.replace('.', "-");
    let vmfb = compile_one(
        &compiler,
        mlir,
        &flags,
        &cache,
        &tag,
        config.context_capacity,
    )?;
    let compile_seconds = started.elapsed().as_secs_f64();
    let contract = AuxiliaryArtifactContract::new_legacy_unqualified(
        entry,
        format!("{config:?};precision={precision:?}"),
        format!(
            "compiler={};version={};flags={flags:?}",
            compiler.display(),
            compiler_version(&compiler)?
        ),
    )?;
    write_auxiliary_manifest(&vmfb, &contract, &weights)?;
    let module = IreeAuxiliaryModule::load(device, &vmfb, &contract, weights)?;
    Ok((module, compile_seconds))
}

/// Give every state global of a Mamba2 decode module a zero initializer, so a
/// single-entry auxiliary module (which cannot call `@state_import`) starts
/// from a fresh sequence. Check-only: the runtime seeds the globals through
/// `@state_import` instead.
fn zero_initialize_state_globals(mlir: &str, config: &Config) -> Result<String, String> {
    let mut text = mlir.to_string();
    for spec in config.state_tensors() {
        let declared = format!("util.global private mutable @{} : ", spec.name);
        let initialized = format!(
            "util.global private mutable @{} = dense<0.000000e+00> : ",
            spec.name
        );
        if text.matches(&declared).count() != 1 {
            return Err(format!(
                "decode module does not declare @{} once",
                spec.name
            ));
        }
        text = text.replacen(&declared, &initialized, 1);
    }
    Ok(text)
}

/// A compiled Mamba2 single-sequence logits decode step, state in module
/// globals that persist across calls (one sequence from a zero state).
pub struct SsmDecodeSession {
    config: Config,
    module: IreeAuxiliaryModule,
    steps: usize,
}

impl SsmDecodeSession {
    /// Load `model_dir` and compile its decode step.
    pub fn load(
        device: &str,
        model_dir: &Path,
        capacity: usize,
        precision: &str,
        layout: &str,
    ) -> Result<Self, String> {
        let precision = parse_precision(precision)?;
        let config = load_config(model_dir, capacity, layout)?;
        let mlir =
            zero_initialize_state_globals(&emit_decode_buffer_abi(&config, precision), &config)?;
        let (module, _) = load_module(
            device,
            model_dir,
            &config,
            precision,
            &mlir,
            "decode_step.main",
        )?;
        Ok(Self {
            config,
            module,
            steps: 0,
        })
    }

    /// Feed one token and return the logits after it.
    pub fn step(&mut self, token: i32) -> Result<Vec<f32>, String> {
        let vocab = i32::try_from(self.config.vocab).map_err(|_| "vocab exceeds i32")?;
        if !(0..vocab).contains(&token) {
            return Err(format!(
                "token {token} is outside the vocabulary of {vocab}"
            ));
        }
        let pos = i32::try_from(self.steps).map_err(|_| "decode position exceeds i32")?;
        let token_bytes = token.to_ne_bytes();
        let pos_bytes = pos.to_ne_bytes();
        // token, pos, cache_len (Mamba2 reads only the token).
        let inputs =
            [&token_bytes[..], &pos_bytes[..], &pos_bytes[..]].map(|bytes| AuxiliaryInput {
                bytes,
                dtype: AuxiliaryTensorDType::Int32,
                shape: &[],
            });
        let vocab_shape = [self.config.vocab];
        let mut logits = vec![0u8; self.config.vocab * 4];
        let mut outputs = [AuxiliaryOutput {
            bytes: &mut logits,
            dtype: AuxiliaryTensorDType::Float32,
            shape: &vocab_shape,
        }];
        self.module.invoke(&inputs, &mut outputs)?;
        self.steps += 1;
        Ok(decode_f32(&logits))
    }
}

/// A compiled Mamba2 logits prefill with its weights resident.
pub struct SsmModelSession {
    config: Config,
    module: IreeAuxiliaryModule,
    /// Wall time of the (possibly cached) compile at load.
    pub compile_seconds: f64,
}

impl SsmModelSession {
    /// Load `model_dir` at `capacity` rows and compile its logits prefill.
    pub fn load(
        device: &str,
        model_dir: &Path,
        capacity: usize,
        precision: &str,
        layout: &str,
    ) -> Result<Self, String> {
        let precision = parse_precision(precision)?;
        let config = load_config(model_dir, capacity, layout)?;
        let mlir = emit_prefill_with(&config, false, precision);
        let (module, compile_seconds) =
            load_module(device, model_dir, &config, precision, &mlir, "prefill.main")?;
        Ok(Self {
            config,
            module,
            compile_seconds,
        })
    }

    /// The state tensors a prefill takes and returns, with zeroed values.
    #[must_use]
    pub fn zero_state(&self) -> Vec<SsmModelState> {
        self.config
            .state_tensors()
            .into_iter()
            .map(|spec| SsmModelState {
                name: spec.name.to_string(),
                values: vec![0.0; spec.dims.iter().product()],
                dims: spec.dims,
            })
            .collect()
    }

    /// Run one prefill of `tokens` (at most `context_capacity`, at least one)
    /// from `state` (zeros for a fresh sequence). `real_len` is checked against
    /// the capacity here, on the host, before it reaches the graph.
    pub fn prefill(
        &mut self,
        tokens: &[i32],
        state: &[SsmModelState],
    ) -> Result<SsmModelPrefill, String> {
        let capacity = self.config.context_capacity;
        if tokens.is_empty() || tokens.len() > capacity {
            return Err(format!(
                "prefill of {} tokens is outside 1..={capacity}",
                tokens.len()
            ));
        }
        let vocab = i32::try_from(self.config.vocab).map_err(|_| "vocab exceeds i32")?;
        if let Some(bad) = tokens.iter().find(|&&t| t < 0 || t >= vocab) {
            return Err(format!("token {bad} is outside the vocabulary of {vocab}"));
        }
        let mut expected = self.zero_state();
        if state.len() != expected.len() {
            return Err(format!(
                "prefill takes {} state tensors, got {}",
                expected.len(),
                state.len()
            ));
        }
        for (want, got) in expected.iter().zip(state) {
            if want.name != got.name
                || want.dims != got.dims
                || want.values.len() != got.values.len()
            {
                return Err(format!(
                    "state {} must be {:?}, got {} {:?} with {} values",
                    want.name,
                    want.dims,
                    got.name,
                    got.dims,
                    got.values.len()
                ));
            }
        }
        let mut padded = vec![0i32; capacity];
        padded[..tokens.len()].copy_from_slice(tokens);
        let positions: Vec<i32> = (0..capacity)
            .map(|p| i32::try_from(p).map_err(|_| "capacity exceeds i32".to_string()))
            .collect::<Result<_, _>>()?;
        let real_len =
            i32::try_from(tokens.len()).map_err(|_| "real_len exceeds i32".to_string())?;
        let token_bytes = i32_bytes(&padded);
        let position_bytes = i32_bytes(&positions);
        let real_len_bytes = real_len.to_ne_bytes();
        let state_bytes: Vec<Vec<u8>> = state.iter().map(|s| f32_bytes(&s.values)).collect();
        let lp = [capacity];
        let mut inputs = vec![
            AuxiliaryInput {
                bytes: &token_bytes,
                dtype: AuxiliaryTensorDType::Int32,
                shape: &lp,
            },
            AuxiliaryInput {
                bytes: &position_bytes,
                dtype: AuxiliaryTensorDType::Int32,
                shape: &lp,
            },
            AuxiliaryInput {
                bytes: &real_len_bytes,
                dtype: AuxiliaryTensorDType::Int32,
                shape: &[],
            },
        ];
        for (bytes, tensor) in state_bytes.iter().zip(state) {
            inputs.push(AuxiliaryInput {
                bytes,
                dtype: AuxiliaryTensorDType::Float32,
                shape: &tensor.dims,
            });
        }
        let vocab_shape = [self.config.vocab];
        let mut logits = vec![0u8; self.config.vocab * 4];
        let mut state_out: Vec<Vec<u8>> = expected
            .iter()
            .map(|s| vec![0u8; s.values.len() * 4])
            .collect();
        let mut outputs = vec![AuxiliaryOutput {
            bytes: &mut logits,
            dtype: AuxiliaryTensorDType::Float32,
            shape: &vocab_shape,
        }];
        for (bytes, tensor) in state_out.iter_mut().zip(&expected) {
            outputs.push(AuxiliaryOutput {
                bytes,
                dtype: AuxiliaryTensorDType::Float32,
                shape: &tensor.dims,
            });
        }
        let started = Instant::now();
        self.module.invoke(&inputs, &mut outputs)?;
        let run_seconds = started.elapsed().as_secs_f64();
        drop(outputs);
        for (tensor, bytes) in expected.iter_mut().zip(&state_out) {
            tensor.values = decode_f32(bytes);
        }
        Ok(SsmModelPrefill {
            logits: decode_f32(&logits),
            states: expected,
            run_seconds,
        })
    }
}
