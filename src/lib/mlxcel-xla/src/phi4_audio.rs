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

//! Resident IREE runtime for Phi4MM's pinned cascaded Conformer.
//!
//! Host code supplies the #874 SpeechLib feature frames. This module owns only
//! the audio encoder/projection weights and the `audio.main` VMFB; it never
//! constructs or falls back to an MLX decoder.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::process::Command;

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};

use crate::aux::{
    AuxiliaryInput, AuxiliaryOutput, AuxiliaryTensorDType, AuxiliaryWeight, AuxiliaryWeightDType,
    IreeAuxiliaryModule,
};
use crate::aux_manifest::{AuxiliaryArtifactContract, ensure_qualified_auxiliary_artifact};
use crate::emitter::{
    Phi4AudioConfig, Precision, emit_phi4_audio_diagnostic_with, emit_phi4_audio_with,
    phi4_audio_diagnostic_specs, phi4_audio_weight_specs,
};
use crate::iree::{cached_vmfb_path, compile_one_to, iree_compile_bin, target_flags};
use crate::weights::{bf16_to_f32, f16_to_f32};

/// Immutable upstream checkpoint revision qualified by #874.
pub const PHI4MM_AUDIO_CHECKPOINT_REVISION: &str = "93f923e1a7727d1c4f446756212d9d3e8fcc5d81";

/// Static production buckets. The largest bucket maps to 500 Conformer rows.
pub const PHI4MM_AUDIO_FRAME_BUCKETS: &[usize] = &[64, 128, 256, 512, 1024, 2048, 4000];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phi4AudioProjectionMode {
    Speech,
    Vision,
}

impl Phi4AudioProjectionMode {
    fn code(self) -> i32 {
        match self {
            Self::Speech => 1,
            Self::Vision => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Phi4AudioOutput {
    /// Pinned #874 projection stage, cropped to `valid_rows`, shape
    /// `[1, valid_rows, hidden_size]`.
    pub projected: Vec<f32>,
    pub valid_rows: usize,
    pub hidden_size: usize,
    pub frame_bucket: usize,
}

#[must_use]
pub fn phi4_audio_bucket_for_frames(frame_len: usize) -> Option<usize> {
    PHI4MM_AUDIO_FRAME_BUCKETS
        .iter()
        .copied()
        .find(|bucket| frame_len > 0 && frame_len <= *bucket)
}

fn config_identity(config: &Phi4AudioConfig, frame_bucket: usize, precision: Precision) -> String {
    format!(
        "family=phi4mm-audio;revision={PHI4MM_AUDIO_CHECKPOINT_REVISION};\
         input={};dim={};heads={};blocks={};ff={};reduction={};channels={};\
         kernel={};relative={};projection={};bucket={frame_bucket};precision={precision:?};\
         projection-modes=speech,vision;abi=features-mask-length-mode-to-projection-length-v1",
        config.input_size,
        config.attention_dim,
        config.attention_heads,
        config.num_blocks,
        config.linear_units,
        config.time_reduction,
        config.conv_channels,
        config.kernel_size,
        config.relative_bias_max_distance,
        config.projection_hidden,
    )
}

fn audio_precision() -> Result<Precision, String> {
    match std::env::var("MLXCEL_PHI4MM_AUDIO_PRECISION").as_deref() {
        Ok("f32") | Err(_) => Ok(Precision::F32),
        Ok("bf16") => Ok(Precision::Bf16),
        Ok(value) => Err(format!(
            "unsupported MLXCEL_PHI4MM_AUDIO_PRECISION={value}; expected f32 or bf16"
        )),
    }
}

fn generation_identity(compiler: &Path, flags: &[&str], graph: &str) -> Result<String, String> {
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
    generation_identity_from_version(
        compiler,
        flags,
        graph,
        String::from_utf8_lossy(&version.stdout).trim(),
    )
}

fn generation_identity_from_version(
    compiler: &Path,
    flags: &[&str],
    graph: &str,
    version: &str,
) -> Result<String, String> {
    Ok(format!(
        "compiler={};compiler_sha256={};version={version};flags={flags:?};stablehlo_sha256={}",
        compiler.display(),
        sha256_file(compiler)?,
        sha256_hex(graph.as_bytes()),
    ))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex_bytes(&digest)
}

fn hex_bytes(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(hex_bytes(&digest.finalize()))
}

fn validate_finite_values(label: &str, values: &[f32]) -> Result<(), String> {
    if let Some((index, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "{label} contains non-finite value {value} at flat index {index}"
        ));
    }
    Ok(())
}

fn resolve_weight_shards(model_dir: &Path, names: &[String]) -> Result<Vec<PathBuf>, String> {
    let single = model_dir.join("model.safetensors");
    if single.is_file() {
        return Ok(vec![single; names.len()]);
    }
    let index = model_dir.join("model.safetensors.index.json");
    let text = std::fs::read_to_string(&index)
        .map_err(|error| format!("read {}: {error}", index.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| format!("parse {}: {error}", index.display()))?;
    let map = value
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| format!("{} is missing object `weight_map`", index.display()))?;
    names
        .iter()
        .map(|name| {
            map.get(name)
                .and_then(serde_json::Value::as_str)
                .map(|file| model_dir.join(file))
                .ok_or_else(|| format!("{} has no entry for `{name}`", index.display()))
        })
        .collect()
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
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four-byte f32 chunk")))
        .collect()
}

fn decode_i32_scalar(bytes: &[u8]) -> Result<i32, String> {
    let bytes: [u8; 4] = bytes
        .try_into()
        .map_err(|_| format!("Phi4MM audio scalar output has {} bytes", bytes.len()))?;
    Ok(i32::from_ne_bytes(bytes))
}

fn load_audio_weights(
    model_dir: &Path,
    config: &Phi4AudioConfig,
) -> Result<Vec<AuxiliaryWeight>, String> {
    let specs = phi4_audio_weight_specs(config);
    let names = specs
        .iter()
        .map(|spec| spec.name.clone())
        .collect::<Vec<_>>();
    let shards = resolve_weight_shards(model_dir, &names)?;
    let mut by_shard = BTreeMap::<&Path, Vec<usize>>::new();
    for (index, shard) in shards.iter().enumerate() {
        by_shard.entry(shard).or_default().push(index);
    }
    let mut loaded = (0..specs.len())
        .map(|_| None)
        .collect::<Vec<Option<AuxiliaryWeight>>>();
    for (shard, indices) in by_shard {
        let file =
            File::open(shard).map_err(|error| format!("open {}: {error}", shard.display()))?;
        // Safety: the mapping is read-only and cannot outlive `file` within
        // this block; every selected tensor is copied before the mapping drops.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|error| format!("mmap {}: {error}", shard.display()))?;
        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|error| format!("parse {}: {error}", shard.display()))?;
        for index in indices {
            let spec = &specs[index];
            let tensor = tensors.tensor(&spec.name).map_err(|error| {
                format!("read `{}` from {}: {error}", spec.name, shard.display())
            })?;
            if tensor.shape() != spec.shape {
                return Err(format!(
                    "Phi4MM audio weight `{}` has shape {:?}, expected {:?}",
                    spec.name,
                    tensor.shape(),
                    spec.shape
                ));
            }
            let values = match tensor.dtype() {
                Dtype::BF16 => bf16_to_f32(tensor.data()),
                Dtype::F16 => f16_to_f32(tensor.data()),
                Dtype::F32 => tensor
                    .data()
                    .chunks_exact(4)
                    .map(|chunk| {
                        f32::from_le_bytes(chunk.try_into().expect("four-byte safetensors f32"))
                    })
                    .collect(),
                dtype => {
                    return Err(format!(
                        "Phi4MM audio weight `{}` has unsupported dtype {dtype:?}",
                        spec.name
                    ));
                }
            };
            let expected_elements = spec.shape.iter().product::<usize>();
            if values.len() != expected_elements {
                return Err(format!(
                    "Phi4MM audio weight `{}` has {} elements, expected {expected_elements}",
                    spec.name,
                    values.len()
                ));
            }
            validate_finite_values(&format!("Phi4MM audio weight `{}`", spec.name), &values)?;
            loaded[index] = Some(AuxiliaryWeight {
                name: spec.name.clone(),
                bytes: f32_bytes(&values),
                dtype: AuxiliaryWeightDType::Float32,
                shape: spec.shape.clone(),
            });
        }
    }
    loaded
        .into_iter()
        .enumerate()
        .map(|(index, weight)| {
            weight
                .ok_or_else(|| format!("Phi4MM audio weight {} was not loaded", specs[index].name))
        })
        .collect()
}

fn load_audio_module(
    model_dir: &Path,
    device: &str,
    config: &Phi4AudioConfig,
    frame_bucket: usize,
    precision: Precision,
    graph: &str,
    artifact_name: &str,
    entry_name: &str,
) -> Result<IreeAuxiliaryModule, String> {
    let compiler = iree_compile_bin()?;
    let flags = target_flags(device)?;
    let cache = std::env::temp_dir().join("mlxcel-xla-vmfb");
    std::fs::create_dir_all(&cache)
        .map_err(|error| format!("mkdir {}: {error}", cache.display()))?;
    let vmfb = cached_vmfb_path(&compiler, graph, flags, &cache, artifact_name, frame_bucket);
    let weights = load_audio_weights(model_dir, config)?;
    let contract = AuxiliaryArtifactContract::new(
        entry_name,
        config_identity(config, frame_bucket, precision),
        generation_identity(&compiler, flags, graph)?,
    )?;
    ensure_qualified_auxiliary_artifact(&vmfb, &contract, &weights, |temporary| {
        compile_one_to(
            &compiler,
            graph,
            flags,
            &cache,
            artifact_name,
            frame_bucket,
            temporary,
        )
    })?;
    IreeAuxiliaryModule::load(device, &vmfb, &contract, weights)
}

pub struct Phi4AudioRuntime {
    module: IreeAuxiliaryModule,
    config: Phi4AudioConfig,
    frame_bucket: usize,
    encoded_bucket: usize,
}

impl Phi4AudioRuntime {
    /// Compile/load one static audio bucket and upload all 887 encoder and
    /// projection tensors as immutable resident buffers.
    pub fn load(model_dir: &Path, device: &str, frame_bucket: usize) -> Result<Self, String> {
        if !PHI4MM_AUDIO_FRAME_BUCKETS.contains(&frame_bucket) {
            return Err(format!(
                "unsupported Phi4MM audio frame bucket {frame_bucket}; expected one of {PHI4MM_AUDIO_FRAME_BUCKETS:?}"
            ));
        }
        let config_text = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|error| format!("read Phi4MM config.json: {error}"))?;
        let config = Phi4AudioConfig::from_json_str(&config_text)?;
        let encoded_bucket = config.encoded_bucket_len(frame_bucket)?;
        // Mirror #874's real mixed boundary: BF16 feature normalization,
        // followed by F32 Conformer and projection stages.
        let precision = audio_precision()?;
        let graph = emit_phi4_audio_with(&config, frame_bucket, precision)?;
        let module = load_audio_module(
            model_dir,
            device,
            &config,
            frame_bucket,
            precision,
            &graph,
            "phi4mm-audio",
            "audio.main",
        )?;
        Ok(Self {
            module,
            config,
            frame_bucket,
            encoded_bucket,
        })
    }

    #[must_use]
    pub fn frame_bucket(&self) -> usize {
        self.frame_bucket
    }

    #[must_use]
    pub fn fingerprint(&self) -> u64 {
        self.module.fingerprint()
    }

    pub fn project(
        &mut self,
        features: &[f32],
        frame_len: usize,
        mode: Phi4AudioProjectionMode,
    ) -> Result<Phi4AudioOutput, String> {
        if frame_len == 0 || frame_len > self.frame_bucket {
            return Err(format!(
                "Phi4MM audio frame length {frame_len} is outside static bucket 1..={}",
                self.frame_bucket
            ));
        }
        let expected = frame_len
            .checked_mul(self.config.input_size)
            .ok_or_else(|| "Phi4MM audio feature element count overflowed".to_string())?;
        if features.len() != expected {
            return Err(format!(
                "Phi4MM audio features have {} elements, expected {expected} for [{frame_len}, {}]",
                features.len(),
                self.config.input_size
            ));
        }
        if features.iter().any(|value| !value.is_finite()) {
            return Err("Phi4MM audio features contain non-finite values".to_string());
        }
        let mut padded = vec![0.0f32; self.frame_bucket * self.config.input_size];
        padded[..features.len()].copy_from_slice(features);
        let mut mask = vec![0i32; self.frame_bucket];
        mask[..frame_len].fill(1);
        let feature_bytes = f32_bytes(&padded);
        let mask_bytes = i32_bytes(&mask);
        let length_bytes = i32_bytes(&[i32::try_from(frame_len)
            .map_err(|_| format!("Phi4MM audio frame length {frame_len} does not fit i32"))?]);
        let mode_bytes = i32_bytes(&[mode.code()]);
        let feature_shape = [1, self.frame_bucket, self.config.input_size];
        let mask_shape = [1, self.frame_bucket];
        let scalar_shape: [usize; 0] = [];
        let output_shape = [1, self.encoded_bucket, self.config.projection_hidden];
        let mut projected_bytes =
            vec![0u8; self.encoded_bucket * self.config.projection_hidden * size_of::<f32>()];
        let mut output_length_bytes = vec![0u8; size_of::<i32>()];
        self.module.invoke(
            &[
                AuxiliaryInput {
                    bytes: &feature_bytes,
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &feature_shape,
                },
                AuxiliaryInput {
                    bytes: &mask_bytes,
                    dtype: AuxiliaryTensorDType::Int32,
                    shape: &mask_shape,
                },
                AuxiliaryInput {
                    bytes: &length_bytes,
                    dtype: AuxiliaryTensorDType::Int32,
                    shape: &scalar_shape,
                },
                AuxiliaryInput {
                    bytes: &mode_bytes,
                    dtype: AuxiliaryTensorDType::Int32,
                    shape: &scalar_shape,
                },
            ],
            &mut [
                AuxiliaryOutput {
                    bytes: &mut projected_bytes,
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &output_shape,
                },
                AuxiliaryOutput {
                    bytes: &mut output_length_bytes,
                    dtype: AuxiliaryTensorDType::Int32,
                    shape: &scalar_shape,
                },
            ],
        )?;
        let valid_rows = usize::try_from(decode_i32_scalar(&output_length_bytes)?)
            .map_err(|_| "Phi4MM audio graph returned a negative valid length".to_string())?;
        let expected_rows = self.config.encoded_valid_len(frame_len)?;
        if valid_rows != expected_rows || valid_rows > self.encoded_bucket {
            return Err(format!(
                "Phi4MM audio graph returned valid length {valid_rows}, expected {expected_rows} within bucket {}",
                self.encoded_bucket
            ));
        }
        let mut projected = decode_f32(&projected_bytes);
        projected.truncate(valid_rows * self.config.projection_hidden);
        if projected.iter().any(|value| !value.is_finite()) {
            return Err("Phi4MM audio graph returned non-finite projection values".to_string());
        }
        Ok(Phi4AudioOutput {
            projected,
            valid_rows,
            hidden_size: self.config.projection_hidden,
            frame_bucket: self.frame_bucket,
        })
    }
}

/// One exact tensor boundary returned by the real-checkpoint diagnostic entry.
///
/// This API exists only to localize parity failures; production requests use
/// [`Phi4AudioRuntime`] and never allocate these intermediate tensors.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct Phi4AudioCheckpoint {
    pub name: &'static str,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct Phi4AudioDiagnostics {
    pub checkpoints: Vec<Phi4AudioCheckpoint>,
    pub valid_rows: usize,
}

#[doc(hidden)]
pub struct Phi4AudioDiagnosticRuntime {
    module: IreeAuxiliaryModule,
    config: Phi4AudioConfig,
    frame_bucket: usize,
    encoded_bucket: usize,
}

#[doc(hidden)]
impl Phi4AudioDiagnosticRuntime {
    pub fn load(model_dir: &Path, device: &str, frame_bucket: usize) -> Result<Self, String> {
        if !PHI4MM_AUDIO_FRAME_BUCKETS.contains(&frame_bucket) {
            return Err(format!(
                "unsupported Phi4MM audio frame bucket {frame_bucket}; expected one of {PHI4MM_AUDIO_FRAME_BUCKETS:?}"
            ));
        }
        let config_text = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|error| format!("read Phi4MM config.json: {error}"))?;
        let config = Phi4AudioConfig::from_json_str(&config_text)?;
        let encoded_bucket = config.encoded_bucket_len(frame_bucket)?;
        let precision = audio_precision()?;
        let graph = emit_phi4_audio_diagnostic_with(&config, frame_bucket, precision)?;
        let module = load_audio_module(
            model_dir,
            device,
            &config,
            frame_bucket,
            precision,
            &graph,
            "phi4mm-audio-diagnostic",
            "audio.diagnostic",
        )?;
        Ok(Self {
            module,
            config,
            frame_bucket,
            encoded_bucket,
        })
    }

    pub fn capture(
        &mut self,
        features: &[f32],
        frame_len: usize,
    ) -> Result<Phi4AudioDiagnostics, String> {
        if frame_len == 0 || frame_len > self.frame_bucket {
            return Err(format!(
                "Phi4MM audio frame length {frame_len} is outside static bucket 1..={}",
                self.frame_bucket
            ));
        }
        let expected = frame_len
            .checked_mul(self.config.input_size)
            .ok_or_else(|| "Phi4MM audio feature element count overflowed".to_string())?;
        if features.len() != expected {
            return Err(format!(
                "Phi4MM audio features have {} elements, expected {expected} for [{frame_len}, {}]",
                features.len(),
                self.config.input_size
            ));
        }
        if features.iter().any(|value| !value.is_finite()) {
            return Err("Phi4MM audio features contain non-finite values".to_string());
        }
        let mut padded = vec![0.0f32; self.frame_bucket * self.config.input_size];
        padded[..features.len()].copy_from_slice(features);
        let mut mask = vec![0i32; self.frame_bucket];
        mask[..frame_len].fill(1);
        let feature_bytes = f32_bytes(&padded);
        let mask_bytes = i32_bytes(&mask);
        let length_bytes = i32_bytes(&[i32::try_from(frame_len)
            .map_err(|_| format!("Phi4MM audio frame length {frame_len} does not fit i32"))?]);
        // The diagnostic returns both projection branches; this mode input is
        // still supplied to keep its argument ABI identical to audio.main.
        let mode_bytes = i32_bytes(&[Phi4AudioProjectionMode::Speech.code()]);
        let feature_shape = [1, self.frame_bucket, self.config.input_size];
        let mask_shape = [1, self.frame_bucket];
        let scalar_shape: [usize; 0] = [];
        let specs = phi4_audio_diagnostic_specs(&self.config, self.frame_bucket)?;
        let mut buffers = specs
            .iter()
            .map(|spec| {
                spec.shape
                    .iter()
                    .try_fold(size_of::<f32>(), |bytes: usize, dimension| {
                        bytes.checked_mul(*dimension).ok_or_else(|| {
                            format!(
                                "Phi4MM audio diagnostic {} byte count overflowed",
                                spec.name
                            )
                        })
                    })
                    .map(|bytes| vec![0u8; bytes])
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut output_length_bytes = vec![0u8; size_of::<i32>()];
        let mut outputs = buffers
            .iter_mut()
            .zip(&specs)
            .map(|(bytes, spec)| AuxiliaryOutput {
                bytes,
                dtype: AuxiliaryTensorDType::Float32,
                shape: &spec.shape,
            })
            .collect::<Vec<_>>();
        outputs.push(AuxiliaryOutput {
            bytes: &mut output_length_bytes,
            dtype: AuxiliaryTensorDType::Int32,
            shape: &scalar_shape,
        });
        self.module.invoke(
            &[
                AuxiliaryInput {
                    bytes: &feature_bytes,
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &feature_shape,
                },
                AuxiliaryInput {
                    bytes: &mask_bytes,
                    dtype: AuxiliaryTensorDType::Int32,
                    shape: &mask_shape,
                },
                AuxiliaryInput {
                    bytes: &length_bytes,
                    dtype: AuxiliaryTensorDType::Int32,
                    shape: &scalar_shape,
                },
                AuxiliaryInput {
                    bytes: &mode_bytes,
                    dtype: AuxiliaryTensorDType::Int32,
                    shape: &scalar_shape,
                },
            ],
            &mut outputs,
        )?;
        drop(outputs);
        let valid_rows = usize::try_from(decode_i32_scalar(&output_length_bytes)?)
            .map_err(|_| "Phi4MM audio diagnostic returned a negative valid length".to_string())?;
        let expected_rows = self.config.encoded_valid_len(frame_len)?;
        if valid_rows != expected_rows || valid_rows > self.encoded_bucket {
            return Err(format!(
                "Phi4MM audio diagnostic returned valid length {valid_rows}, expected {expected_rows} within bucket {}",
                self.encoded_bucket
            ));
        }
        let checkpoints = specs
            .into_iter()
            .zip(buffers)
            .map(|(spec, bytes)| {
                let values = decode_f32(&bytes);
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(format!(
                        "Phi4MM audio diagnostic {} contains non-finite values",
                        spec.name
                    ));
                }
                Ok(Phi4AudioCheckpoint {
                    name: spec.name,
                    shape: spec.shape,
                    values,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Phi4AudioDiagnostics {
            checkpoints,
            valid_rows,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aux_manifest::{auxiliary_manifest_path, verify_auxiliary_manifest};

    fn published_config() -> Phi4AudioConfig {
        Phi4AudioConfig {
            input_size: 80,
            attention_dim: 1024,
            attention_heads: 16,
            num_blocks: 24,
            linear_units: 1536,
            time_reduction: 8,
            conv_channels: 1024,
            kernel_size: 3,
            relative_bias_max_distance: 500,
            projection_hidden: 3072,
        }
    }

    fn temporary_audio_path(suffix: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock must be after the Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "mlxcel-phi4mm-audio-cache-test-{}-{nonce}.{suffix}",
            std::process::id(),
        ))
    }

    #[test]
    fn bucket_selection_is_static_and_fail_closed() {
        assert_eq!(phi4_audio_bucket_for_frames(0), None);
        assert_eq!(phi4_audio_bucket_for_frames(1), Some(64));
        assert_eq!(phi4_audio_bucket_for_frames(64), Some(64));
        assert_eq!(phi4_audio_bucket_for_frames(65), Some(128));
        assert_eq!(phi4_audio_bucket_for_frames(351), Some(512));
        assert_eq!(phi4_audio_bucket_for_frames(4000), Some(4000));
        assert_eq!(phi4_audio_bucket_for_frames(4001), None);
    }

    #[test]
    fn projection_modes_match_the_pinned_abi_codes() {
        assert_eq!(Phi4AudioProjectionMode::Speech.code(), 1);
        assert_eq!(Phi4AudioProjectionMode::Vision.code(), 2);
    }

    #[test]
    fn audio_contract_reuses_qualified_cache_and_rebuilds_stale_pair_once() {
        let vmfb = temporary_audio_path("vmfb");
        let weights = vec![AuxiliaryWeight {
            name: "model.embed_tokens_extend.audio_embed.weight".to_string(),
            bytes: 1.0f32.to_ne_bytes().to_vec(),
            dtype: AuxiliaryWeightDType::Float32,
            shape: vec![1],
        }];
        let config = published_config();
        let contract = AuxiliaryArtifactContract::new(
            "audio.main",
            config_identity(&config, 512, Precision::F32),
            "compiler=test;flags=cpu;stablehlo_sha256=v1",
        )
        .unwrap();
        let mut compile_count = 0usize;
        ensure_qualified_auxiliary_artifact(&vmfb, &contract, &weights, |temporary| {
            compile_count += 1;
            std::fs::write(temporary, b"audio-vmfb-v1")
                .map_err(|error| format!("write test audio VMFB: {error}"))
        })
        .unwrap();
        assert_eq!(compile_count, 1);

        ensure_qualified_auxiliary_artifact(&vmfb, &contract, &weights, |_| {
            compile_count += 1;
            Err("qualified Phi4MM audio cache must not compile".to_string())
        })
        .unwrap();
        assert_eq!(compile_count, 1);

        let stale_replacement = AuxiliaryArtifactContract::new(
            "audio.main",
            config_identity(&config, 512, Precision::F32),
            "compiler=test;flags=cpu;stablehlo_sha256=v2",
        )
        .unwrap();
        ensure_qualified_auxiliary_artifact(&vmfb, &stale_replacement, &weights, |temporary| {
            compile_count += 1;
            std::fs::write(temporary, b"audio-vmfb-v2")
                .map_err(|error| format!("write replacement audio VMFB: {error}"))
        })
        .unwrap();
        assert_eq!(compile_count, 2);
        assert_eq!(std::fs::read(&vmfb).unwrap(), b"audio-vmfb-v2");
        verify_auxiliary_manifest(&vmfb, &stale_replacement, &weights).unwrap();

        std::fs::remove_file(auxiliary_manifest_path(&vmfb)).ok();
        std::fs::remove_file(vmfb).ok();
    }

    #[test]
    fn compiler_binary_replacement_at_same_path_and_version_rebuilds_audio_cache() {
        let compiler = temporary_audio_path("compiler");
        let vmfb = temporary_audio_path("compiler-digest.vmfb");
        let weights = vec![AuxiliaryWeight {
            name: "weight".to_string(),
            bytes: 1.0f32.to_ne_bytes().to_vec(),
            dtype: AuxiliaryWeightDType::Float32,
            shape: vec![1],
        }];
        std::fs::write(&compiler, b"compiler-build-a").unwrap();
        let first_generation =
            generation_identity_from_version(&compiler, &["--target=cpu"], "mlir", "v1").unwrap();
        assert!(
            first_generation.contains(
                "compiler_sha256=c998e735c573e64551765a30b12a0cc63d1255c2229f5943995ccdfef4939b7a"
            ),
            "generation identity should contain the compiler executable's raw SHA-256 digest"
        );
        let first_contract =
            AuxiliaryArtifactContract::new("audio.main", "config=v1", &first_generation).unwrap();
        let mut compile_count = 0usize;
        ensure_qualified_auxiliary_artifact(&vmfb, &first_contract, &weights, |temporary| {
            compile_count += 1;
            std::fs::write(temporary, b"vmfb-a")
                .map_err(|error| format!("write test VMFB: {error}"))
        })
        .unwrap();

        std::fs::write(&compiler, b"compiler-build-b").unwrap();
        let second_generation =
            generation_identity_from_version(&compiler, &["--target=cpu"], "mlir", "v1").unwrap();
        assert_ne!(first_generation, second_generation);
        let second_contract =
            AuxiliaryArtifactContract::new("audio.main", "config=v1", second_generation).unwrap();
        ensure_qualified_auxiliary_artifact(&vmfb, &second_contract, &weights, |temporary| {
            compile_count += 1;
            std::fs::write(temporary, b"vmfb-b")
                .map_err(|error| format!("write test VMFB: {error}"))
        })
        .unwrap();

        assert_eq!(compile_count, 2);
        assert_eq!(std::fs::read(&vmfb).unwrap(), b"vmfb-b");
        std::fs::remove_file(auxiliary_manifest_path(&vmfb)).ok();
        std::fs::remove_file(vmfb).ok();
        std::fs::remove_file(compiler).ok();
    }

    #[test]
    fn audio_weight_values_reject_non_finite_before_native_upload() {
        assert!(validate_finite_values("audio weight", &[0.0, -1.0, 3.0]).is_ok());
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let error = validate_finite_values("audio weight", &[0.0, value]).unwrap_err();
            assert!(error.contains("audio weight"));
            assert!(error.contains("flat index 1"));
        }
    }
}
