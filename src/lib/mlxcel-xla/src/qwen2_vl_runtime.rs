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

//! Resident IREE execution for the Qwen2-VL vision tower and patch merger.
//!
//! Only normalized patch extraction and grid metadata remain on the host. The
//! full 32-layer vision tower and merger execute in the selected IREE target;
//! this module never constructs or calls the MLX vision encoder.

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
use crate::emitter::{
    Qwen2VlConfig, Qwen2VlHostInputs, emit_qwen2_vl, prepare_qwen2_vl_host_inputs,
};
use crate::iree::{cached_vmfb_path, compile_one_to, iree_compile_bin, target_flags};
use crate::weights::{bf16_to_f32, dequantize_affine, f16_to_f32, f32_le_to_f32};

const ENTRY_NAME: &str = "qwen2_vl_vision.main";

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen2VlVisionExecutionMetrics {
    pub patch_upload_bytes: usize,
    pub metadata_upload_bytes: usize,
    pub projected_transfer_bytes: usize,
    pub elapsed_seconds: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen2VlVisionProjection {
    pub values: Vec<f32>,
    pub shape: [usize; 2],
    pub merged_tokens_per_image: Vec<usize>,
    pub packed_cu_seqlens: Vec<usize>,
    pub metrics: Qwen2VlVisionExecutionMetrics,
}

struct ActiveModule {
    patch_bucket: usize,
    module: IreeAuxiliaryModule,
}

pub struct IreeQwen2VlProjector {
    model_dir: PathBuf,
    device: String,
    config: Qwen2VlConfig,
    processor_identity: String,
    active: Option<ActiveModule>,
}

fn hex_bytes(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        write!(output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_bytes(&Sha256::digest(bytes))
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
    Ok(format!(
        "compiler={};compiler_sha256={};version={};flags={flags:?};mlir_sha256={}",
        compiler.display(),
        sha256_file(compiler)?,
        String::from_utf8_lossy(&version.stdout).trim(),
        sha256_hex(mlir.as_bytes())
    ))
}

fn processor_identity(model_dir: &Path, config: &Qwen2VlConfig) -> Result<String, String> {
    let path = model_dir.join("preprocessor_config.json");
    let bytes = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    let object = value
        .as_object()
        .ok_or_else(|| format!("{} must contain an object", path.display()))?;
    for field in ["do_resize", "do_rescale", "do_normalize", "do_convert_rgb"] {
        if object.get(field).and_then(serde_json::Value::as_bool) != Some(true) {
            return Err(format!(
                "preprocessor_config.json {field}=true is required by Qwen2-VL IREE vision"
            ));
        }
    }
    let positive = |field: &str| -> Result<usize, String> {
        object
            .get(field)
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&value| value > 0)
            .ok_or_else(|| format!("preprocessor_config.json {field} must be a positive integer"))
    };
    for (field, expected) in [
        ("patch_size", config.patch_size),
        ("temporal_patch_size", config.temporal_patch_size),
        ("merge_size", config.spatial_merge_size),
    ] {
        let actual = positive(field)?;
        if actual != expected {
            return Err(format!(
                "preprocessor_config.json {field}={actual} disagrees with config value {expected}"
            ));
        }
    }
    if object
        .get("image_processor_type")
        .and_then(serde_json::Value::as_str)
        != Some("Qwen2VLImageProcessor")
    {
        return Err(
            "Qwen2-VL IREE vision requires image_processor_type=Qwen2VLImageProcessor".to_string(),
        );
    }
    Ok(format!(
        "qwen2-vl-processor-v1:sha256={}:patch={}:temporal={}:merge={}",
        sha256_hex(&bytes),
        config.patch_size,
        config.temporal_patch_size,
        config.spatial_merge_size
    ))
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

fn tensor_locations(model_dir: &Path) -> Result<BTreeMap<String, PathBuf>, String> {
    let mut locations = BTreeMap::new();
    for shard in model_shards(model_dir)? {
        let file =
            File::open(&shard).map_err(|error| format!("open {}: {error}", shard.display()))?;
        // Safety: the map is read-only and lives through the header scan.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|error| format!("mmap {}: {error}", shard.display()))?;
        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|error| format!("parse {}: {error}", shard.display()))?;
        for name in tensors
            .names()
            .into_iter()
            .filter(|name| name.starts_with("vision_tower."))
        {
            if let Some(previous) = locations.insert(name.to_string(), shard.clone()) {
                return Err(format!(
                    "Qwen2-VL vision tensor {name:?} is duplicated in {} and {}",
                    previous.display(),
                    shard.display()
                ));
            }
        }
    }
    Ok(locations)
}

fn native_f32_bytes(values: Vec<f32>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn validate_finite(label: &str, values: &[f32]) -> Result<(), String> {
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

fn transpose_patch_t_h_w_c_to_t_c_h_w(values: Vec<f32>, config: &Qwen2VlConfig) -> Vec<f32> {
    let out = config.hidden;
    let temporal = config.temporal_patch_size;
    let height = config.patch_size;
    let width = config.patch_size;
    let channels = config.channels;
    let mut transposed = vec![0.0; values.len()];
    for output in 0..out {
        for t in 0..temporal {
            for channel in 0..channels {
                for y in 0..height {
                    for x in 0..width {
                        let source = ((((output * temporal + t) * height + y) * width + x)
                            * channels)
                            + channel;
                        let target =
                            ((((output * temporal + t) * channels + channel) * height + y) * width)
                                + x;
                        transposed[target] = values[source];
                    }
                }
            }
        }
    }
    transposed
}

fn decode_direct_f32(
    label: &str,
    dtype: Dtype,
    data: &[u8],
    expected_shape: &[usize],
    source_shape: &[usize],
) -> Result<Vec<f32>, String> {
    if source_shape != expected_shape {
        return Err(format!(
            "{label} has shape {source_shape:?}, expected {expected_shape:?}"
        ));
    }
    let values = match dtype {
        Dtype::BF16 => bf16_to_f32(data),
        Dtype::F16 => f16_to_f32(data),
        Dtype::F32 => f32_le_to_f32(data),
        other => {
            return Err(format!(
                "{label} has unsupported dtype {other:?}; expected BF16, F16, F32, or an affine U32 projection"
            ));
        }
    };
    validate_finite(label, &values)?;
    Ok(values)
}

fn quant_sibling(name: &str, suffix: &str) -> Result<String, String> {
    let stem = name
        .strip_suffix(".weight")
        .ok_or_else(|| format!("quantized Qwen2-VL tensor {name} must end in .weight"))?;
    Ok(format!("{stem}.{suffix}"))
}

fn load_weights(
    model_dir: &Path,
    config: &Qwen2VlConfig,
) -> Result<(Vec<AuxiliaryWeight>, String), String> {
    let specs = config.weight_specs();
    let locations = tensor_locations(model_dir)?;
    let required = specs
        .iter()
        .map(|spec| spec.name.as_str())
        .collect::<BTreeSet<_>>();
    let missing = required
        .iter()
        .filter(|name| !locations.contains_key(**name))
        .copied()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "checkpoint is missing {} Qwen2-VL vision tensor(s): {}",
            missing.len(),
            missing
                .iter()
                .take(8)
                .copied()
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let mut by_shard = BTreeMap::<&Path, Vec<usize>>::new();
    for (index, spec) in specs.iter().enumerate() {
        by_shard
            .entry(
                locations
                    .get(&spec.name)
                    .expect("missing specs rejected")
                    .as_path(),
            )
            .or_default()
            .push(index);
    }
    let mut loaded = (0..specs.len())
        .map(|_| None)
        .collect::<Vec<Option<AuxiliaryWeight>>>();
    let mut source_schema = vec![String::new(); specs.len()];
    for (shard, indices) in by_shard {
        let file =
            File::open(shard).map_err(|error| format!("open {}: {error}", shard.display()))?;
        // Safety: the map is read-only and lives while selected tensors copy.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|error| format!("mmap {}: {error}", shard.display()))?;
        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|error| format!("parse {}: {error}", shard.display()))?;
        for index in indices {
            let spec = &specs[index];
            let tensor = tensors.tensor(&spec.name).map_err(|error| {
                format!(
                    "resolved Qwen2-VL tensor {} in {}: {error}",
                    spec.name,
                    shard.display()
                )
            })?;
            let source_dtype = tensor.dtype();
            let source_shape = tensor.shape().to_vec();
            let (values, transform) = if spec.name == "vision_tower.patch_embed.proj.weight" {
                let expected_source = [
                    config.hidden,
                    config.temporal_patch_size,
                    config.patch_size,
                    config.patch_size,
                    config.channels,
                ];
                if source_shape != expected_source {
                    return Err(format!(
                        "{} has source shape {source_shape:?}, expected Conv3d [out,t,h,w,c] {expected_source:?}",
                        spec.name
                    ));
                }
                let values = decode_direct_f32(
                    &spec.name,
                    source_dtype,
                    tensor.data(),
                    &expected_source,
                    &source_shape,
                )?;
                (
                    transpose_patch_t_h_w_c_to_t_c_h_w(values, config),
                    format!(
                        "conv3d-source={source_shape:?}:layout=O,T,H,W,C->O,T,C,H,W->O,{}",
                        spec.shape[1]
                    ),
                )
            } else if source_dtype == Dtype::U32 {
                let Some((bits, group_size)) = config.quantization else {
                    return Err(format!(
                        "{} is U32 affine-quantized but config.json has no quantization contract",
                        spec.name
                    ));
                };
                if spec.shape.len() != 2 {
                    return Err(format!(
                        "{} is quantized but logical shape {:?} is not rank 2",
                        spec.name, spec.shape
                    ));
                }
                let out = spec.shape[0];
                let in_packed = spec.shape[1]
                    .checked_mul(bits)
                    .ok_or_else(|| format!("{} packed width overflowed", spec.name))?
                    / 32;
                if source_shape != [out, in_packed] {
                    return Err(format!(
                        "{} has packed shape {source_shape:?}, expected [{out}, {in_packed}]",
                        spec.name
                    ));
                }
                let scales_name = quant_sibling(&spec.name, "scales")?;
                let biases_name = quant_sibling(&spec.name, "biases")?;
                for sibling in [&scales_name, &biases_name] {
                    if locations.get(sibling).map(PathBuf::as_path) != Some(shard) {
                        return Err(format!(
                            "quantized Qwen2-VL tensor {} and sibling {sibling} must share one immutable shard",
                            spec.name
                        ));
                    }
                }
                let scales = tensors
                    .tensor(&scales_name)
                    .map_err(|error| format!("load {scales_name}: {error}"))?;
                let biases = tensors
                    .tensor(&biases_name)
                    .map_err(|error| format!("load {biases_name}: {error}"))?;
                let scales_bf16 = match (scales.dtype(), biases.dtype()) {
                    (Dtype::F16, Dtype::F16) => false,
                    (Dtype::BF16, Dtype::BF16) => true,
                    (left, right) => {
                        return Err(format!(
                            "{} affine metadata dtypes {left:?}/{right:?} must be matching F16 or BF16",
                            spec.name
                        ));
                    }
                };
                let values = dequantize_affine(
                    tensor.data(),
                    scales.data(),
                    biases.data(),
                    out,
                    in_packed,
                    bits,
                    group_size,
                    scales_bf16,
                )?;
                validate_finite(&format!("Qwen2-VL tensor {}", spec.name), &values)?;
                (
                    values,
                    format!(
                        "affine-source={source_dtype:?}:{source_shape:?};scales={:?}:{:?};biases={:?}:{:?};bits={bits};group={group_size}->f32",
                        scales.dtype(),
                        scales.shape(),
                        biases.dtype(),
                        biases.shape()
                    ),
                )
            } else {
                (
                    decode_direct_f32(
                        &spec.name,
                        source_dtype,
                        tensor.data(),
                        &spec.shape,
                        &source_shape,
                    )?,
                    "identity->f32".to_string(),
                )
            };
            let expected = spec.shape.iter().try_fold(1usize, |count, dimension| {
                count
                    .checked_mul(*dimension)
                    .ok_or_else(|| format!("{} logical element count overflowed", spec.name))
            })?;
            if values.len() != expected {
                return Err(format!(
                    "{} decoded {} values, expected {expected} for logical shape {:?}",
                    spec.name,
                    values.len(),
                    spec.shape
                ));
            }
            loaded[index] = Some(AuxiliaryWeight {
                name: spec.name.clone(),
                bytes: native_f32_bytes(values),
                dtype: AuxiliaryWeightDType::Float32,
                shape: spec.shape.clone(),
            });
            source_schema[index] = format!(
                "{}:{source_dtype:?}:{source_shape:?}:{transform}",
                spec.name
            );
        }
    }
    Ok((
        loaded
            .into_iter()
            .map(|weight| weight.expect("all Qwen2-VL specs loaded"))
            .collect(),
        source_schema.join("\n"),
    ))
}

fn f32_as_bytes(values: &[f32]) -> &[u8] {
    // Safety: f32 has no invalid bit patterns and the view cannot outlive input.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn checked_f32_output(label: &str, bytes: Vec<u8>) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err(format!(
            "{label} returned {} bytes, not a whole number of f32 values",
            bytes.len()
        ));
    }
    let values = bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four-byte f32 chunk")))
        .collect::<Vec<_>>();
    validate_finite(label, &values)?;
    Ok(values)
}

fn compile_and_load(
    model_dir: &Path,
    device: &str,
    config: &Qwen2VlConfig,
    processor_identity: &str,
    patch_bucket: usize,
) -> Result<IreeAuxiliaryModule, String> {
    let mlir = emit_qwen2_vl(config, patch_bucket);
    let compiler = iree_compile_bin()?;
    if !compiler.is_file() {
        return Err(format!("iree-compile not found at {}", compiler.display()));
    }
    let flags = target_flags(device)?;
    let cache = std::env::temp_dir().join("mlxcel-xla-qwen2-vl-vmfb");
    std::fs::create_dir_all(&cache)
        .map_err(|error| format!("mkdir {}: {error}", cache.display()))?;
    let (weights, checkpoint_schema) = load_weights(model_dir, config)?;
    let contract = AuxiliaryArtifactContract::new_legacy_unqualified(
        ENTRY_NAME,
        format!(
            "{};{};checkpoint_schema_sha256={}",
            config.fingerprint(patch_bucket),
            processor_identity,
            sha256_hex(checkpoint_schema.as_bytes())
        ),
        compiler_generation_identity(&compiler, flags, &mlir)?,
    )?;
    let tag = format!("qwen2-vl-vision-{patch_bucket}");
    let vmfb = cached_vmfb_path(&compiler, &mlir, flags, &cache, &tag, 0);
    ensure_qualified_auxiliary_artifact(&vmfb, &contract, &weights, |temporary| {
        compile_one_to(&compiler, &mlir, flags, &cache, &tag, 0, temporary)
    })?;
    IreeAuxiliaryModule::load(device, &vmfb, &contract, weights)
}

impl IreeQwen2VlProjector {
    pub fn load(model_dir: &Path, device: &str) -> Result<Self, String> {
        let config = Qwen2VlConfig::from_model_dir(model_dir)?;
        let processor_identity = processor_identity(model_dir, &config)?;
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            device: device.to_string(),
            config,
            processor_identity,
            active: None,
        })
    }

    #[must_use]
    pub fn active_bucket(&self) -> Option<usize> {
        self.active.as_ref().map(|active| active.patch_bucket)
    }

    #[must_use]
    pub fn text_hidden(&self) -> usize {
        self.config.text_hidden
    }

    fn ensure_bucket(&mut self, patch_bucket: usize) -> Result<&mut IreeAuxiliaryModule, String> {
        if self.active.as_ref().map(|active| active.patch_bucket) != Some(patch_bucket) {
            let module = compile_and_load(
                &self.model_dir,
                &self.device,
                &self.config,
                &self.processor_identity,
                patch_bucket,
            )?;
            self.active = Some(ActiveModule {
                patch_bucket,
                module,
            });
        }
        Ok(&mut self
            .active
            .as_mut()
            .expect("active module installed")
            .module)
    }

    pub fn project(
        &mut self,
        temporal_patch_rows: &[f32],
        grids: &[(i32, i32, i32)],
    ) -> Result<Qwen2VlVisionProjection, String> {
        let Qwen2VlHostInputs {
            plan,
            patches,
            vision_rope_freqs,
            packed_attention_bias,
        } = prepare_qwen2_vl_host_inputs(&self.config, grids, temporal_patch_rows)?;
        let patch_width = self.config.channels
            * self.config.temporal_patch_size
            * self.config.patch_size
            * self.config.patch_size;
        let frequency_width = self.config.hidden / self.config.heads / 2;
        let patch_shape = [plan.patch_bucket, patch_width];
        let frequency_shape = [plan.patch_bucket, frequency_width];
        let bias_shape = [plan.patch_bucket, plan.patch_bucket];
        let full_output_shape = [
            plan.patch_bucket / (self.config.spatial_merge_size * self.config.spatial_merge_size),
            self.config.text_hidden,
        ];
        let mut output =
            vec![0u8; full_output_shape.iter().product::<usize>() * std::mem::size_of::<f32>()];
        let started = Instant::now();
        self.ensure_bucket(plan.patch_bucket)?.invoke(
            &[
                AuxiliaryInput {
                    bytes: f32_as_bytes(&patches),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &patch_shape,
                },
                AuxiliaryInput {
                    bytes: f32_as_bytes(&vision_rope_freqs),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &frequency_shape,
                },
                AuxiliaryInput {
                    bytes: f32_as_bytes(&packed_attention_bias),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &bias_shape,
                },
            ],
            &mut [AuxiliaryOutput {
                bytes: &mut output,
                dtype: AuxiliaryTensorDType::Float32,
                shape: &full_output_shape,
            }],
        )?;
        let elapsed_seconds = started.elapsed().as_secs_f64();
        let all_values = checked_f32_output("Qwen2-VL IREE projected output", output)?;
        let actual_tokens = plan.merged_tokens_per_image.iter().sum::<usize>();
        let actual_values = actual_tokens
            .checked_mul(self.config.text_hidden)
            .ok_or_else(|| "Qwen2-VL projected output size overflowed".to_string())?;
        let values = all_values
            .get(..actual_values)
            .ok_or_else(|| "Qwen2-VL IREE output is shorter than the actual grid".to_string())?
            .to_vec();
        Ok(Qwen2VlVisionProjection {
            values,
            shape: [actual_tokens, self.config.text_hidden],
            merged_tokens_per_image: plan.merged_tokens_per_image,
            packed_cu_seqlens: plan.packed_cu_seqlens,
            metrics: Qwen2VlVisionExecutionMetrics {
                patch_upload_bytes: std::mem::size_of_val(patches.as_slice()),
                metadata_upload_bytes: std::mem::size_of_val(vision_rope_freqs.as_slice())
                    + std::mem::size_of_val(packed_attention_bias.as_slice()),
                projected_transfer_bytes: std::mem::size_of_val(all_values.as_slice()),
                elapsed_seconds,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> Qwen2VlConfig {
        Qwen2VlConfig::from_json_str(
            r#"{
                "model_type":"qwen2_vl",
                "hidden_size":12,
                "vision_config":{
                    "depth":1,
                    "embed_dim":8,
                    "mlp_ratio":2,
                    "num_heads":2,
                    "in_chans":3,
                    "patch_size":2,
                    "spatial_merge_size":2,
                    "temporal_patch_size":2
                }
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn conv3d_layout_transform_matches_o_t_c_h_w_flattening() {
        let config = tiny_config();
        let source = (0..config.hidden
            * config.temporal_patch_size
            * config.patch_size
            * config.patch_size
            * config.channels)
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let transformed = transpose_patch_t_h_w_c_to_t_c_h_w(source.clone(), &config);
        let source_index = |o: usize, t: usize, y: usize, x: usize, c: usize| {
            ((((o * 2 + t) * 2 + y) * 2 + x) * 3) + c
        };
        let target_index = |o: usize, t: usize, c: usize, y: usize, x: usize| {
            ((((o * 2 + t) * 3 + c) * 2 + y) * 2) + x
        };
        for o in 0..8 {
            for t in 0..2 {
                for c in 0..3 {
                    for y in 0..2 {
                        for x in 0..2 {
                            assert_eq!(
                                transformed[target_index(o, t, c, y, x)],
                                source[source_index(o, t, y, x, c)]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn output_decoder_rejects_non_finite_values() {
        let mut bytes = 1.0f32.to_ne_bytes().to_vec();
        bytes.extend_from_slice(&f32::NAN.to_ne_bytes());
        assert!(
            checked_f32_output("qwen vision", bytes)
                .unwrap_err()
                .contains("flat index 1")
        );
    }
}
