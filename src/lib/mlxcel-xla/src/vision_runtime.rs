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

//! Resident-weight IREE execution for the qualified LLaVA vision tower.
//!
//! The checkpoint boundary is deliberately strict: all graph arguments must be
//! present exactly once, in the emitted order, with their exact static shape
//! and one of the losslessly widenable floating-point dtypes. The artifact
//! manifest then binds that ordered schema to the config, compiler, target
//! flags, StableHLO source, and actual VMFB bytes before native loading.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};

use crate::aux::{
    AuxiliaryInput, AuxiliaryOutput, AuxiliaryTensorDType, AuxiliaryWeight, AuxiliaryWeightDType,
    IreeAuxiliaryModule,
};
use crate::aux_manifest::{AuxiliaryArtifactContract, ensure_qualified_auxiliary_artifact};
#[cfg(feature = "diagnostics")]
use crate::emitter::emit_vision_diagnostics;
use crate::emitter::{LlavaVisionConfig, VisionWeightSpec, emit_vision};
use crate::iree::{cached_vmfb_path, compile_one_to, iree_compile_bin, target_flags};
use crate::weights::{bf16_to_f32, f16_to_f32, f32_le_to_f32};

const ENTRY_NAME: &str = "vision.main";

#[derive(Debug, Clone, PartialEq)]
struct VisionProcessorContract {
    identity: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VisionExecutionMetrics {
    pub pixel_upload_bytes: usize,
    pub projected_transfer_bytes: usize,
    pub elapsed_seconds: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VisionProjection {
    pub values: Vec<f32>,
    pub shape: [usize; 2],
    pub metrics: VisionExecutionMetrics,
}

#[cfg(feature = "diagnostics")]
#[derive(Debug, Clone, PartialEq)]
pub struct VisionDiagnosticProjection {
    /// Encoder input plus every selected encoder-block output.
    pub hidden_states: Vec<Vec<f32>>,
    /// LayerNorm/QKV/attention/residual/MLP stages for encoder block zero.
    pub block0_states: Vec<Vec<f32>>,
    pub selected_vision_features: Vec<f32>,
    pub projected_image_features: Vec<f32>,
    pub hidden_shape: [usize; 2],
    pub projected_shape: [usize; 2],
    pub metrics: VisionExecutionMetrics,
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_bytes(&Sha256::digest(bytes))
}

fn hex_bytes(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn compiler_generation_identity(
    compiler: &Path,
    flags: &[&str],
    mlir: &str,
) -> Result<String, String> {
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
    compiler_generation_identity_from_version(
        compiler,
        flags,
        mlir,
        String::from_utf8_lossy(&version.stdout).trim(),
    )
}

fn compiler_generation_identity_from_version(
    compiler: &Path,
    flags: &[&str],
    mlir: &str,
    version: &str,
) -> Result<String, String> {
    Ok(format!(
        "compiler={};compiler_sha256={};version={version};flags={flags:?};mlir_sha256={}",
        compiler.display(),
        sha256_file(compiler)?,
        sha256_hex(mlir.as_bytes())
    ))
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

fn required_bool(
    object: &serde_json::Map<String, serde_json::Value>,
    name: &str,
) -> Result<bool, String> {
    object
        .get(name)
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| format!("preprocessor_config.json {name} must be a boolean"))
}

fn finite_number(
    object: &serde_json::Map<String, serde_json::Value>,
    name: &str,
) -> Result<f64, String> {
    let value = object
        .get(name)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| format!("preprocessor_config.json {name} must be a number"))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("preprocessor_config.json {name} must be finite"))
    }
}

fn finite_vector(
    object: &serde_json::Map<String, serde_json::Value>,
    name: &str,
    length: usize,
) -> Result<Vec<f64>, String> {
    let values = object
        .get(name)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("preprocessor_config.json {name} must be an array"))?;
    if values.len() != length {
        return Err(format!(
            "preprocessor_config.json {name} has {} values, expected {length}",
            values.len()
        ));
    }
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| format!("preprocessor_config.json {name}[{index}] must be finite"))
        })
        .collect()
}

fn square_size(
    object: &serde_json::Map<String, serde_json::Value>,
    name: &str,
) -> Result<usize, String> {
    let value = object
        .get(name)
        .ok_or_else(|| format!("preprocessor_config.json {name} is required"))?;
    if let Some(size) = value.as_u64() {
        return usize::try_from(size)
            .map_err(|_| format!("preprocessor_config.json {name} does not fit usize"));
    }
    let size = value
        .as_object()
        .ok_or_else(|| format!("preprocessor_config.json {name} must be a size object"))?;
    let height = size
        .get("height")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| format!("preprocessor_config.json {name}.height is required"))?;
    let width = size
        .get("width")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| format!("preprocessor_config.json {name}.width is required"))?;
    if height != width {
        return Err(format!(
            "preprocessor_config.json {name} must be square, got {height}x{width}"
        ));
    }
    usize::try_from(height)
        .map_err(|_| format!("preprocessor_config.json {name} does not fit usize"))
}

impl VisionProcessorContract {
    fn from_model_dir(model_dir: &Path, config: &LlavaVisionConfig) -> Result<Self, String> {
        let preprocessor_path = model_dir.join("preprocessor_config.json");
        let preprocessor_text = std::fs::read_to_string(&preprocessor_path)
            .map_err(|error| format!("{}: {error}", preprocessor_path.display()))?;
        let preprocessor: serde_json::Value = serde_json::from_str(&preprocessor_text)
            .map_err(|error| format!("parse {}: {error}", preprocessor_path.display()))?;
        let object = preprocessor
            .as_object()
            .ok_or_else(|| format!("{} must be a JSON object", preprocessor_path.display()))?;
        for name in ["do_resize", "do_rescale", "do_normalize"] {
            if !required_bool(object, name)? {
                return Err(format!(
                    "preprocessor_config.json {name}=false is outside the qualified vision contract"
                ));
            }
        }
        if square_size(object, "size")? != config.image_size {
            return Err(format!(
                "preprocessor size disagrees with vision image_size={}",
                config.image_size
            ));
        }
        let center_crop = object
            .get("do_center_crop")
            .map(|value| {
                value.as_bool().ok_or_else(|| {
                    "preprocessor_config.json do_center_crop must be a boolean".to_string()
                })
            })
            .transpose()?
            .unwrap_or(false);
        if center_crop && square_size(object, "crop_size")? != config.image_size {
            return Err(format!(
                "preprocessor crop_size disagrees with vision image_size={}",
                config.image_size
            ));
        }
        if object.get("resample").and_then(serde_json::Value::as_u64) != Some(3) {
            return Err("preprocessor_config.json resample must be PIL bicubic (3)".to_string());
        }
        let factor = finite_number(object, "rescale_factor")?;
        if factor <= 0.0 {
            return Err("preprocessor_config.json rescale_factor must be positive".to_string());
        }
        let mean = finite_vector(object, "image_mean", config.channels)?;
        let std = finite_vector(object, "image_std", config.channels)?;
        if std.iter().any(|value| *value <= 0.0) {
            return Err("preprocessor_config.json image_std values must be positive".to_string());
        }
        if config.channels == 3
            && object
                .get("do_convert_rgb")
                .is_some_and(|value| value.as_bool() == Some(false))
        {
            return Err(
                "preprocessor_config.json do_convert_rgb=false disagrees with 3-channel RGB input"
                    .to_string(),
            );
        }

        let processor_path = model_dir.join("processor_config.json");
        let processor_text = std::fs::read_to_string(&processor_path)
            .map_err(|error| format!("{}: {error}", processor_path.display()))?;
        let processor: serde_json::Value = serde_json::from_str(&processor_text)
            .map_err(|error| format!("parse {}: {error}", processor_path.display()))?;
        let processor_object = processor
            .as_object()
            .ok_or_else(|| format!("{} must be a JSON object", processor_path.display()))?;
        if processor_object
            .get("patch_size")
            .and_then(serde_json::Value::as_u64)
            != Some(config.patch_size as u64)
        {
            return Err(format!(
                "processor patch_size disagrees with vision patch_size={}",
                config.patch_size
            ));
        }
        let expected_strategy = if config.drop_first_token {
            "default"
        } else {
            "full"
        };
        if processor_object
            .get("vision_feature_select_strategy")
            .and_then(serde_json::Value::as_str)
            != Some(expected_strategy)
        {
            return Err(format!(
                "processor vision_feature_select_strategy disagrees with config ({expected_strategy})"
            ));
        }
        let expected_additional_tokens = usize::from(config.class_token) as u64;
        if processor_object
            .get("num_additional_image_tokens")
            .and_then(serde_json::Value::as_u64)
            != Some(expected_additional_tokens)
        {
            return Err(format!(
                "processor num_additional_image_tokens disagrees with class-token contract ({expected_additional_tokens})"
            ));
        }
        Ok(Self {
            identity: format!(
                "preprocessor={};processor={};resolved=size:{};crop:{};resample:bicubic;\
                 rescale:{:016x};mean:{mean:?};std:{std:?}",
                preprocessor,
                processor,
                config.image_size,
                center_crop,
                factor.to_bits(),
            ),
        })
    }
}

fn model_shards(model_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut shards = std::fs::read_dir(model_dir)
        .map_err(|error| format!("read {}: {error}", model_dir.display()))?
        .map(|entry| {
            entry
                .map(|entry| entry.path())
                .map_err(|error| format!("read {} entry: {error}", model_dir.display()))
        })
        .collect::<Result<Vec<_>, String>>()?;
    shards.retain(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".safetensors"))
    });
    shards.sort();
    if shards.is_empty() {
        return Err(format!(
            "no safetensors checkpoint shards found in {}",
            model_dir.display()
        ));
    }
    Ok(shards)
}

fn resolve_weight_shards(
    model_dir: &Path,
    specs: &[VisionWeightSpec],
) -> Result<Vec<PathBuf>, String> {
    let required = specs
        .iter()
        .map(|spec| spec.name.clone())
        .collect::<BTreeSet<_>>();
    let mut locations = BTreeMap::<String, PathBuf>::new();
    for shard in model_shards(model_dir)? {
        let file =
            File::open(&shard).map_err(|error| format!("open {}: {error}", shard.display()))?;
        // Safety: this read-only map lives through the safetensors header scan.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|error| format!("mmap {}: {error}", shard.display()))?;
        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|error| format!("parse {}: {error}", shard.display()))?;
        for name in tensors.names() {
            if !required.contains(name) {
                continue;
            }
            if let Some(previous) = locations.insert(name.to_string(), shard.clone()) {
                return Err(format!(
                    "vision tensor {name:?} is duplicated in {} and {}",
                    previous.display(),
                    shard.display()
                ));
            }
        }
    }
    let missing = specs
        .iter()
        .filter(|spec| !locations.contains_key(&spec.name))
        .map(|spec| spec.name.as_str())
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "checkpoint is missing {} vision tensor(s): {}",
            missing.len(),
            missing
                .iter()
                .take(8)
                .copied()
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    Ok(specs
        .iter()
        .map(|spec| {
            locations
                .get(&spec.name)
                .expect("missing tensors rejected above")
                .clone()
        })
        .collect())
}

fn native_f32_bytes(values: Vec<f32>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn transpose_patch_ohwi_to_oihw(
    values: Vec<f32>,
    output_channels: usize,
    kernel_height: usize,
    kernel_width: usize,
    input_channels: usize,
) -> Vec<f32> {
    let mut transposed = vec![0.0f32; values.len()];
    for output in 0..output_channels {
        for y in 0..kernel_height {
            for x in 0..kernel_width {
                for input in 0..input_channels {
                    let source = (((output * kernel_height + y) * kernel_width + x)
                        * input_channels)
                        + input;
                    let target = (((output * input_channels + input) * kernel_height + y)
                        * kernel_width)
                        + x;
                    transposed[target] = values[source];
                }
            }
        }
    }
    transposed
}

fn load_weights(
    model_dir: &Path,
    specs: &[VisionWeightSpec],
) -> Result<(Vec<AuxiliaryWeight>, String), String> {
    let shards = resolve_weight_shards(model_dir, specs)?;
    let mut by_shard = BTreeMap::<&Path, Vec<usize>>::new();
    for (index, shard) in shards.iter().enumerate() {
        by_shard.entry(shard.as_path()).or_default().push(index);
    }
    let mut loaded = (0..specs.len())
        .map(|_| None)
        .collect::<Vec<Option<AuxiliaryWeight>>>();
    let mut source_schema = vec![String::new(); specs.len()];
    for (shard, indices) in by_shard {
        let file =
            File::open(shard).map_err(|error| format!("open {}: {error}", shard.display()))?;
        // Safety: this read-only map lives while every selected tensor is copied.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|error| format!("mmap {}: {error}", shard.display()))?;
        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|error| format!("parse {}: {error}", shard.display()))?;
        for index in indices {
            let spec = &specs[index];
            let tensor = tensors.tensor(&spec.name).map_err(|error| {
                format!(
                    "resolved vision tensor {} in {}: {error}",
                    spec.name,
                    shard.display()
                )
            })?;
            let patch_ohwi = spec.name
                == "vision_tower.vision_model.embeddings.patch_embedding.weight"
                && spec.shape.len() == 4
                && tensor.shape() == [spec.shape[0], spec.shape[2], spec.shape[3], spec.shape[1]];
            if tensor.shape() != spec.shape && !patch_ohwi {
                return Err(format!(
                    "vision tensor {} has shape {:?}, expected {:?}",
                    spec.name,
                    tensor.shape(),
                    spec.shape
                ));
            }
            let source_dtype = tensor.dtype();
            let source_shape = tensor.shape().to_vec();
            let mut values = match source_dtype {
                Dtype::BF16 => bf16_to_f32(tensor.data()),
                Dtype::F16 => f16_to_f32(tensor.data()),
                Dtype::F32 => f32_le_to_f32(tensor.data()),
                dtype => {
                    return Err(format!(
                        "vision tensor {} has unsupported dtype {dtype:?}; expected BF16, F16, or F32",
                        spec.name
                    ));
                }
            };
            if patch_ohwi {
                values = transpose_patch_ohwi_to_oihw(
                    values,
                    spec.shape[0],
                    spec.shape[2],
                    spec.shape[3],
                    spec.shape[1],
                );
            }
            let expected = spec.shape.iter().try_fold(1usize, |count, dimension| {
                count
                    .checked_mul(*dimension)
                    .ok_or_else(|| format!("vision tensor {} element count overflowed", spec.name))
            })?;
            if values.len() != expected {
                return Err(format!(
                    "vision tensor {} decoded {} elements, expected {expected}",
                    spec.name,
                    values.len()
                ));
            }
            validate_finite_values(&format!("vision tensor {}", spec.name), &values)?;
            loaded[index] = Some(AuxiliaryWeight {
                name: spec.name.clone(),
                bytes: native_f32_bytes(values),
                dtype: AuxiliaryWeightDType::Float32,
                shape: spec.shape.clone(),
            });
            source_schema[index] = format!(
                "{}:{source_dtype:?}:{source_shape:?}:{}",
                spec.name,
                if patch_ohwi {
                    "patch-ohwi-to-oihw"
                } else {
                    "identity"
                }
            );
        }
    }
    Ok((
        loaded
            .into_iter()
            .map(|weight| weight.expect("all resolved tensors loaded"))
            .collect(),
        source_schema.join("\n"),
    ))
}

fn compile_and_load(
    model_dir: &Path,
    device: &str,
    config: &LlavaVisionConfig,
    mlir: &str,
    tag: &str,
) -> Result<IreeAuxiliaryModule, String> {
    let processor = VisionProcessorContract::from_model_dir(model_dir, config)?;
    let compiler = iree_compile_bin()?;
    if !compiler.is_file() {
        return Err(format!("iree-compile not found at {}", compiler.display()));
    }
    let flags = target_flags(device)?;
    let cache = std::env::temp_dir().join("mlxcel-xla-vision-vmfb");
    std::fs::create_dir_all(&cache)
        .map_err(|error| format!("mkdir {}: {error}", cache.display()))?;
    let (weights, checkpoint_schema) = load_weights(model_dir, &config.weight_specs())?;
    let contract = AuxiliaryArtifactContract::new(
        ENTRY_NAME,
        format!(
            "{};{};checkpoint_schema_sha256={}",
            config.fingerprint(),
            processor.identity,
            sha256_hex(checkpoint_schema.as_bytes())
        ),
        compiler_generation_identity(&compiler, flags, mlir)?,
    )?;
    let vmfb = cached_vmfb_path(&compiler, mlir, flags, &cache, tag, 0);
    ensure_qualified_auxiliary_artifact(&vmfb, &contract, &weights, |temporary| {
        compile_one_to(&compiler, mlir, flags, &cache, tag, 0, temporary)
    })?;
    IreeAuxiliaryModule::load(device, &vmfb, &contract, weights)
}

fn f32_as_bytes(values: &[f32]) -> &[u8] {
    // Safety: f32 has no invalid bit patterns and the byte slice cannot outlive
    // the immutable source slice.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
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

fn checked_f32_output(label: &str, bytes: Vec<u8>) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err(format!(
            "{label} returned {} bytes, which is not a whole number of f32 values",
            bytes.len()
        ));
    }
    let values = bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four-byte f32 chunk")))
        .collect::<Vec<_>>();
    validate_finite_values(label, &values)?;
    Ok(values)
}

fn validate_pixels(config: &LlavaVisionConfig, pixels: &[f32]) -> Result<[usize; 4], String> {
    let shape = [1, config.channels, config.image_size, config.image_size];
    let expected: usize = shape.iter().product();
    if pixels.len() != expected {
        return Err(format!(
            "vision pixels have {} elements, expected {expected} for {shape:?}",
            pixels.len()
        ));
    }
    if let Some((index, value)) = pixels
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "vision pixels contain non-finite value {value} at flat index {index}"
        ));
    }
    Ok(shape)
}

pub struct IreeVisionProjector {
    module: IreeAuxiliaryModule,
    config: LlavaVisionConfig,
}

impl IreeVisionProjector {
    pub fn load(model_dir: &Path, device: &str) -> Result<Self, String> {
        let config = LlavaVisionConfig::from_model_dir(model_dir)?;
        let mlir = emit_vision(&config);
        let module = compile_and_load(model_dir, device, &config, &mlir, "llava-vision")?;
        Ok(Self { module, config })
    }

    #[must_use]
    pub fn input_shape(&self) -> [usize; 4] {
        [
            1,
            self.config.channels,
            self.config.image_size,
            self.config.image_size,
        ]
    }

    #[must_use]
    pub fn output_shape(&self) -> [usize; 2] {
        [self.config.image_tokens(), self.config.text_hidden]
    }

    #[must_use]
    pub fn artifact_fingerprint(&self) -> u64 {
        self.module.fingerprint()
    }

    pub fn project(&mut self, pixels: &[f32]) -> Result<VisionProjection, String> {
        let input_shape = validate_pixels(&self.config, pixels)?;
        let output_shape = self.output_shape();
        let mut output =
            vec![0u8; output_shape.iter().product::<usize>() * std::mem::size_of::<f32>()];
        let started = Instant::now();
        self.module.invoke(
            &[AuxiliaryInput {
                bytes: f32_as_bytes(pixels),
                dtype: AuxiliaryTensorDType::Float32,
                shape: &input_shape,
            }],
            &mut [AuxiliaryOutput {
                bytes: &mut output,
                dtype: AuxiliaryTensorDType::Float32,
                shape: &output_shape,
            }],
        )?;
        let elapsed_seconds = started.elapsed().as_secs_f64();
        let values = checked_f32_output("IREE vision projected output", output)?;
        Ok(VisionProjection {
            values,
            shape: output_shape,
            metrics: VisionExecutionMetrics {
                pixel_upload_bytes: std::mem::size_of_val(pixels),
                projected_transfer_bytes: output_shape.iter().product::<usize>()
                    * std::mem::size_of::<f32>(),
                elapsed_seconds,
            },
        })
    }
}

#[cfg(feature = "diagnostics")]
pub struct IreeVisionDiagnosticProjector {
    module: IreeAuxiliaryModule,
    config: LlavaVisionConfig,
}

#[cfg(feature = "diagnostics")]
impl IreeVisionDiagnosticProjector {
    pub fn load(model_dir: &Path, device: &str) -> Result<Self, String> {
        let config = LlavaVisionConfig::from_model_dir(model_dir)?;
        let mlir = emit_vision_diagnostics(&config);
        let module = compile_and_load(
            model_dir,
            device,
            &config,
            &mlir,
            "llava-vision-diagnostics",
        )?;
        Ok(Self { module, config })
    }

    #[must_use]
    pub fn artifact_fingerprint(&self) -> u64 {
        self.module.fingerprint()
    }

    pub fn project(&mut self, pixels: &[f32]) -> Result<VisionDiagnosticProjection, String> {
        let input_shape = validate_pixels(&self.config, pixels)?;
        let hidden_shape = [self.config.position_count(), self.config.hidden];
        let projected_shape = [self.config.image_tokens(), self.config.text_hidden];
        let block_shapes = [
            hidden_shape,
            hidden_shape,
            hidden_shape,
            hidden_shape,
            hidden_shape,
            hidden_shape,
            hidden_shape,
            hidden_shape,
            [self.config.position_count(), self.config.intermediate],
            [self.config.position_count(), self.config.intermediate],
            hidden_shape,
            hidden_shape,
        ];
        let mut shapes = Vec::<Vec<usize>>::new();
        shapes.push(hidden_shape.to_vec());
        shapes.extend(block_shapes.iter().map(|shape| shape.to_vec()));
        shapes.extend((0..=self.config.feature_layer).map(|_| hidden_shape.to_vec()));
        shapes.push(hidden_shape.to_vec());
        shapes.push(projected_shape.to_vec());
        let mut buffers = shapes
            .iter()
            .map(|shape| vec![0u8; shape.iter().product::<usize>() * std::mem::size_of::<f32>()])
            .collect::<Vec<_>>();
        let mut outputs = buffers
            .iter_mut()
            .zip(&shapes)
            .map(|(bytes, shape)| AuxiliaryOutput {
                bytes,
                dtype: AuxiliaryTensorDType::Float32,
                shape,
            })
            .collect::<Vec<_>>();
        let started = Instant::now();
        self.module.invoke(
            &[AuxiliaryInput {
                bytes: f32_as_bytes(pixels),
                dtype: AuxiliaryTensorDType::Float32,
                shape: &input_shape,
            }],
            &mut outputs,
        )?;
        let elapsed_seconds = started.elapsed().as_secs_f64();
        drop(outputs);
        let values = buffers
            .into_iter()
            .enumerate()
            .map(|(index, bytes)| {
                checked_f32_output(&format!("IREE vision diagnostic output {index}"), bytes)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let mut values = values.into_iter();
        let hidden0 = values
            .next()
            .expect("diagnostics includes hidden state zero");
        let block0_states = values.by_ref().take(block_shapes.len()).collect::<Vec<_>>();
        let mut hidden_states = Vec::with_capacity(self.config.feature_layer + 2);
        hidden_states.push(hidden0);
        hidden_states.extend(values.by_ref().take(self.config.feature_layer + 1));
        let selected_vision_features = values
            .next()
            .expect("diagnostics includes selected features");
        let projected_image_features = values
            .next()
            .expect("diagnostics includes projected features");
        debug_assert!(values.next().is_none());
        let projected_transfer_bytes = shapes
            .iter()
            .map(|shape| shape.iter().product::<usize>() * std::mem::size_of::<f32>())
            .sum();
        Ok(VisionDiagnosticProjection {
            hidden_states,
            block0_states,
            selected_vision_features,
            projected_image_features,
            hidden_shape,
            projected_shape,
            metrics: VisionExecutionMetrics {
                pixel_upload_bytes: std::mem::size_of_val(pixels),
                projected_transfer_bytes,
                elapsed_seconds,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aux_manifest::auxiliary_manifest_path;

    fn temp_path(tag: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock must be after the Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "mlxcel-xla-vision-{tag}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn test_config() -> LlavaVisionConfig {
        LlavaVisionConfig::from_json_str(
            &serde_json::json!({
                "model_type": "llava",
                "vision_config": {
                    "model_type": "siglip_vision_model",
                    "image_size": 28,
                    "patch_size": 14,
                    "num_channels": 3,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "layer_norm_eps": 1e-6,
                    "hidden_act": "gelu_pytorch_tanh"
                },
                "text_config": {"hidden_size": 12},
                "projector_hidden_act": "gelu",
                "vision_feature_layer": -1,
                "vision_feature_select_strategy": "full"
            })
            .to_string(),
        )
        .unwrap()
    }

    #[test]
    fn pixel_contract_rejects_shape_and_non_finite_values_before_iree() {
        let config = test_config();
        let mut pixels = vec![0.0; 3 * 28 * 28];
        assert_eq!(validate_pixels(&config, &pixels).unwrap(), [1, 3, 28, 28]);
        assert!(validate_pixels(&config, &pixels[..pixels.len() - 1]).is_err());
        pixels[17] = f32::NAN;
        let error = validate_pixels(&config, &pixels).unwrap_err();
        assert!(error.contains("non-finite"));
        assert!(error.contains("17"));
    }

    #[test]
    fn weight_and_projector_outputs_reject_every_non_finite_value() {
        assert!(validate_finite_values("weight", &[0.0, -1.0, 3.0]).is_ok());
        let error = validate_finite_values("weight", &[0.0, f32::INFINITY]).unwrap_err();
        assert!(error.contains("weight"));
        assert!(error.contains("flat index 1"));

        let mut bytes = 1.0f32.to_ne_bytes().to_vec();
        bytes.extend_from_slice(&f32::NAN.to_ne_bytes());
        let error = checked_f32_output("projector", bytes).unwrap_err();
        assert!(error.contains("projector"));
        assert!(error.contains("flat index 1"));
    }

    #[test]
    fn compiler_binary_replacement_at_same_path_and_version_rebuilds_cache() {
        let compiler = temp_path("compiler");
        let vmfb = temp_path("compiler-digest").with_extension("vmfb");
        let weights = vec![AuxiliaryWeight {
            name: "weight".to_string(),
            bytes: 1.0f32.to_ne_bytes().to_vec(),
            dtype: AuxiliaryWeightDType::Float32,
            shape: vec![1],
        }];
        std::fs::write(&compiler, b"compiler-build-a").unwrap();
        let first_generation =
            compiler_generation_identity_from_version(&compiler, &["--target=cuda"], "mlir", "v1")
                .unwrap();
        let first_contract =
            AuxiliaryArtifactContract::new("vision.main", "config=v1", &first_generation).unwrap();
        let mut compile_count = 0usize;
        ensure_qualified_auxiliary_artifact(&vmfb, &first_contract, &weights, |temporary| {
            compile_count += 1;
            std::fs::write(temporary, b"vmfb-a")
                .map_err(|error| format!("write test VMFB: {error}"))
        })
        .unwrap();

        std::fs::write(&compiler, b"compiler-build-b").unwrap();
        let second_generation =
            compiler_generation_identity_from_version(&compiler, &["--target=cuda"], "mlir", "v1")
                .unwrap();
        assert_ne!(first_generation, second_generation);
        let second_contract =
            AuxiliaryArtifactContract::new("vision.main", "config=v1", second_generation).unwrap();
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
}
