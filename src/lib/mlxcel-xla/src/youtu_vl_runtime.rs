// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! Resident IREE execution for Youtu-VL's vision tower and built-in merger.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

use memmap2::Mmap;
use safetensors::SafeTensors;

use crate::aux::{
    AuxiliaryInput, AuxiliaryOutput, AuxiliaryTensorDType, AuxiliaryWeight, AuxiliaryWeightDType,
    IreeAuxiliaryModule,
};
use crate::aux_manifest::{AuxiliaryArtifactContract, ensure_qualified_auxiliary_artifact};
#[cfg(feature = "diagnostics")]
use crate::emitter::emit_youtu_vl_diagnostics;
use crate::emitter::{
    YoutuVlHostInputs, YoutuVlVisionConfig, emit_youtu_vl, prepare_youtu_vl_host_inputs,
};
use crate::iree::{cached_vmfb_path, compile_one_to, iree_compile_bin, target_flags};
use crate::qwen2_vl_runtime::{
    checked_f32_output, compiler_generation_identity, decode_direct_f32, f32_as_bytes,
    model_shards, native_f32_bytes, sha256_hex, validate_finite,
};

const ENTRY_NAME: &str = "youtu_vl_vision.main";
#[cfg(feature = "diagnostics")]
const DIAGNOSTIC_ENTRY_NAME: &str = "youtu_vl_vision_diagnostics.main";

#[derive(Debug, Clone, PartialEq)]
pub struct YoutuVlVisionExecutionMetrics {
    pub patch_upload_bytes: usize,
    pub metadata_upload_bytes: usize,
    pub projected_transfer_bytes: usize,
    pub elapsed_seconds: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct YoutuVlVisionProjection {
    pub values: Vec<f32>,
    pub shape: [usize; 2],
    pub merged_tokens_per_image: Vec<usize>,
    pub window_cu_seqlens: Vec<usize>,
    pub full_cu_seqlens: Vec<usize>,
    pub metrics: YoutuVlVisionExecutionMetrics,
}

#[cfg(feature = "diagnostics")]
#[derive(Debug, Clone, PartialEq)]
pub struct YoutuVlVisionDiagnosticProjection {
    /// Linear patch projection in IREE's window order.
    pub patch_projection: Vec<f32>,
    /// Output of window-attention layer 0 in IREE's window order.
    pub window_layer0: Vec<f32>,
    /// Output of full-attention layer 1 in IREE's window order.
    pub full_layer1: Vec<f32>,
    /// Built-in merger output before the host restores original group order.
    pub merger_window_order: Vec<f32>,
    /// Built-in merger output after restoring original per-image group order.
    pub restored_output: Vec<f32>,
    pub patch_shape: [usize; 2],
    pub merged_shape: [usize; 2],
    pub window_group_index: Vec<usize>,
    pub reverse_group_index: Vec<usize>,
}

struct ActiveModule {
    patch_bucket: usize,
    module: IreeAuxiliaryModule,
}

pub struct IreeYoutuVlProjector {
    model_dir: PathBuf,
    device: String,
    config: YoutuVlVisionConfig,
    processor_identity: String,
    active: Option<ActiveModule>,
}

fn processor_identity(model_dir: &Path, config: &YoutuVlVisionConfig) -> Result<String, String> {
    let path = model_dir.join("preprocessor_config.json");
    let bytes = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    let object = value
        .as_object()
        .ok_or_else(|| format!("{} must contain an object", path.display()))?;
    for field in ["do_resize", "do_rescale", "do_normalize"] {
        if object.get(field).and_then(serde_json::Value::as_bool) != Some(true) {
            return Err(format!(
                "Youtu-VL IREE vision requires preprocessor {field}=true"
            ));
        }
    }
    if object.get("patch_size").and_then(serde_json::Value::as_u64)
        != Some(config.patch_size as u64)
    {
        return Err("Youtu-VL processor patch_size disagrees with vision config".to_string());
    }
    Ok(format!(
        "youtu-vl-processor-v1:sha256={}:max_patches={}:resample={}",
        sha256_hex(&bytes),
        config.max_patches_per_image,
        config.resample,
    ))
}

fn tensor_locations(
    model_dir: &Path,
    required: &BTreeSet<&str>,
) -> Result<BTreeMap<String, PathBuf>, String> {
    let mut locations = BTreeMap::new();
    for shard in model_shards(model_dir)? {
        let file =
            File::open(&shard).map_err(|error| format!("open {}: {error}", shard.display()))?;
        // Safety: the read-only mapping lives through the header scan.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|error| format!("mmap {}: {error}", shard.display()))?;
        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|error| format!("parse {}: {error}", shard.display()))?;
        for name in tensors.names() {
            if !required.contains(name.as_str()) {
                continue;
            }
            if let Some(previous) = locations.insert(name.to_string(), shard.clone()) {
                return Err(format!(
                    "Youtu-VL tensor {name} is duplicated in {} and {}",
                    previous.display(),
                    shard.display()
                ));
            }
        }
    }
    Ok(locations)
}

fn load_weights(
    model_dir: &Path,
    config: &YoutuVlVisionConfig,
) -> Result<(Vec<AuxiliaryWeight>, String), String> {
    let specs = config.weight_specs();
    let required = specs
        .iter()
        .map(|spec| spec.name.as_str())
        .collect::<BTreeSet<_>>();
    let locations = tensor_locations(model_dir, &required)?;
    let missing = required
        .iter()
        .filter(|name| !locations.contains_key(**name))
        .copied()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "checkpoint is missing {} Youtu-VL vision tensor(s): {}",
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
    let mut schema = vec![String::new(); specs.len()];
    for (shard, indices) in by_shard {
        let file =
            File::open(shard).map_err(|error| format!("open {}: {error}", shard.display()))?;
        // Safety: the read-only mapping lives while selected tensors copy.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|error| format!("mmap {}: {error}", shard.display()))?;
        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|error| format!("parse {}: {error}", shard.display()))?;
        for index in indices {
            let spec = &specs[index];
            let tensor = tensors.tensor(&spec.name).map_err(|error| {
                format!(
                    "resolved Youtu-VL tensor {} in {}: {error}",
                    spec.name,
                    shard.display()
                )
            })?;
            let dtype = tensor.dtype();
            let source_shape = tensor.shape().to_vec();
            let values =
                decode_direct_f32(&spec.name, dtype, tensor.data(), &spec.shape, &source_shape)?;
            validate_finite(&spec.name, &values)?;
            loaded[index] = Some(AuxiliaryWeight {
                name: spec.name.clone(),
                bytes: native_f32_bytes(values),
                dtype: AuxiliaryWeightDType::Float32,
                shape: spec.shape.clone(),
            });
            schema[index] = format!("{}:{dtype:?}:{source_shape:?}:identity->f32", spec.name);
        }
    }
    Ok((
        loaded
            .into_iter()
            .map(|weight| weight.expect("all Youtu-VL specs loaded"))
            .collect(),
        schema.join("\n"),
    ))
}

fn compile_and_load(
    model_dir: &Path,
    device: &str,
    config: &YoutuVlVisionConfig,
    processor_identity: &str,
    patch_bucket: usize,
) -> Result<IreeAuxiliaryModule, String> {
    let mlir = emit_youtu_vl(config, patch_bucket);
    let compiler = iree_compile_bin()?;
    if !compiler.is_file() {
        return Err(format!("iree-compile not found at {}", compiler.display()));
    }
    let flags = target_flags(device)?;
    let cache = std::env::temp_dir().join("mlxcel-xla-youtu-vl-vmfb");
    std::fs::create_dir_all(&cache)
        .map_err(|error| format!("mkdir {}: {error}", cache.display()))?;
    let (weights, checkpoint_schema) = load_weights(model_dir, config)?;
    let contract = AuxiliaryArtifactContract::new(
        ENTRY_NAME,
        format!(
            "{};{};checkpoint_schema_sha256={}",
            config.fingerprint(patch_bucket),
            processor_identity,
            sha256_hex(checkpoint_schema.as_bytes())
        ),
        compiler_generation_identity(&compiler, flags, &mlir)?,
    )?;
    let tag = format!("youtu-vl-vision-{patch_bucket}");
    let vmfb = cached_vmfb_path(&compiler, &mlir, flags, &cache, &tag, 0);
    ensure_qualified_auxiliary_artifact(&vmfb, &contract, &weights, |temporary| {
        compile_one_to(&compiler, &mlir, flags, &cache, &tag, 0, temporary)
    })?;
    IreeAuxiliaryModule::load(device, &vmfb, &contract, weights)
}

#[cfg(feature = "diagnostics")]
fn compile_and_load_diagnostics(
    model_dir: &Path,
    device: &str,
    config: &YoutuVlVisionConfig,
    processor_identity: &str,
    patch_bucket: usize,
) -> Result<IreeAuxiliaryModule, String> {
    let mlir = emit_youtu_vl_diagnostics(config, patch_bucket)?;
    let compiler = iree_compile_bin()?;
    if !compiler.is_file() {
        return Err(format!("iree-compile not found at {}", compiler.display()));
    }
    let flags = target_flags(device)?;
    let cache = std::env::temp_dir().join("mlxcel-xla-youtu-vl-vmfb");
    std::fs::create_dir_all(&cache)
        .map_err(|error| format!("mkdir {}: {error}", cache.display()))?;
    let (weights, checkpoint_schema) = load_weights(model_dir, config)?;
    let contract = AuxiliaryArtifactContract::new(
        DIAGNOSTIC_ENTRY_NAME,
        format!(
            "{};diagnostic_stages=patch,window0,full1,merger;{};checkpoint_schema_sha256={}",
            config.fingerprint(patch_bucket),
            processor_identity,
            sha256_hex(checkpoint_schema.as_bytes())
        ),
        compiler_generation_identity(&compiler, flags, &mlir)?,
    )?;
    let tag = format!("youtu-vl-vision-diagnostics-{patch_bucket}");
    let vmfb = cached_vmfb_path(&compiler, &mlir, flags, &cache, &tag, 0);
    ensure_qualified_auxiliary_artifact(&vmfb, &contract, &weights, |temporary| {
        compile_one_to(&compiler, &mlir, flags, &cache, &tag, 0, temporary)
    })?;
    IreeAuxiliaryModule::load(device, &vmfb, &contract, weights)
}

impl IreeYoutuVlProjector {
    pub fn load(model_dir: &Path, device: &str) -> Result<Self, String> {
        let config = YoutuVlVisionConfig::from_model_dir(model_dir)?;
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
    pub fn text_hidden(&self) -> usize {
        self.config.text_hidden
    }

    #[must_use]
    pub fn active_bucket(&self) -> Option<usize> {
        self.active.as_ref().map(|active| active.patch_bucket)
    }

    fn ensure_bucket(&mut self, bucket: usize) -> Result<&mut IreeAuxiliaryModule, String> {
        if self.active.as_ref().map(|active| active.patch_bucket) != Some(bucket) {
            let module = compile_and_load(
                &self.model_dir,
                &self.device,
                &self.config,
                &self.processor_identity,
                bucket,
            )?;
            self.active = Some(ActiveModule {
                patch_bucket: bucket,
                module,
            });
        }
        Ok(&mut self.active.as_mut().expect("active module").module)
    }

    pub fn project(
        &mut self,
        patch_rows: &[f32],
        shapes: &[(i32, i32)],
    ) -> Result<YoutuVlVisionProjection, String> {
        let YoutuVlHostInputs {
            plan,
            patches,
            rope_freqs,
            window_attention_bias,
            full_attention_bias,
        } = prepare_youtu_vl_host_inputs(&self.config, shapes, patch_rows)?;
        let patch_shape = [
            plan.patch_bucket,
            self.config.channels * self.config.patch_size * self.config.patch_size,
        ];
        let rope_shape = [
            plan.patch_bucket,
            self.config.hidden / self.config.heads / 2,
        ];
        let bias_shape = [plan.patch_bucket, plan.patch_bucket];
        let output_shape = [plan.patch_bucket / 4, self.config.text_hidden];
        let mut output =
            vec![0u8; output_shape.iter().product::<usize>() * std::mem::size_of::<f32>()];
        let started = Instant::now();
        self.ensure_bucket(plan.patch_bucket)?.invoke(
            &[
                AuxiliaryInput {
                    bytes: f32_as_bytes(&patches),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &patch_shape,
                },
                AuxiliaryInput {
                    bytes: f32_as_bytes(&rope_freqs),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &rope_shape,
                },
                AuxiliaryInput {
                    bytes: f32_as_bytes(&window_attention_bias),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &bias_shape,
                },
                AuxiliaryInput {
                    bytes: f32_as_bytes(&full_attention_bias),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &bias_shape,
                },
            ],
            &mut [AuxiliaryOutput {
                bytes: &mut output,
                dtype: AuxiliaryTensorDType::Float32,
                shape: &output_shape,
            }],
        )?;
        let elapsed_seconds = started.elapsed().as_secs_f64();
        let all_values = checked_f32_output("Youtu-VL IREE projected output", output)?;
        let actual_tokens = plan.reverse_group_index.len();
        let mut values = Vec::with_capacity(actual_tokens * self.config.text_hidden);
        for &window_position in &plan.reverse_group_index {
            let start = window_position * self.config.text_hidden;
            values.extend_from_slice(&all_values[start..start + self.config.text_hidden]);
        }
        Ok(YoutuVlVisionProjection {
            values,
            shape: [actual_tokens, self.config.text_hidden],
            merged_tokens_per_image: plan.merged_tokens_per_image,
            window_cu_seqlens: plan.window_cu_seqlens,
            full_cu_seqlens: plan.full_cu_seqlens,
            metrics: YoutuVlVisionExecutionMetrics {
                patch_upload_bytes: std::mem::size_of_val(patches.as_slice()),
                metadata_upload_bytes: std::mem::size_of_val(rope_freqs.as_slice())
                    + std::mem::size_of_val(window_attention_bias.as_slice())
                    + std::mem::size_of_val(full_attention_bias.as_slice()),
                projected_transfer_bytes: std::mem::size_of_val(all_values.as_slice()),
                elapsed_seconds,
            },
        })
    }
}

#[cfg(feature = "diagnostics")]
pub struct IreeYoutuVlDiagnosticProjector {
    model_dir: PathBuf,
    device: String,
    config: YoutuVlVisionConfig,
    processor_identity: String,
    active: Option<ActiveModule>,
}

#[cfg(feature = "diagnostics")]
impl IreeYoutuVlDiagnosticProjector {
    pub fn load(model_dir: &Path, device: &str) -> Result<Self, String> {
        let config = YoutuVlVisionConfig::from_model_dir(model_dir)?;
        let processor_identity = processor_identity(model_dir, &config)?;
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            device: device.to_string(),
            config,
            processor_identity,
            active: None,
        })
    }

    fn ensure_bucket(&mut self, bucket: usize) -> Result<&mut IreeAuxiliaryModule, String> {
        if self.active.as_ref().map(|active| active.patch_bucket) != Some(bucket) {
            let module = compile_and_load_diagnostics(
                &self.model_dir,
                &self.device,
                &self.config,
                &self.processor_identity,
                bucket,
            )?;
            self.active = Some(ActiveModule {
                patch_bucket: bucket,
                module,
            });
        }
        Ok(&mut self.active.as_mut().expect("active module").module)
    }

    pub fn capture(
        &mut self,
        patch_rows: &[f32],
        shapes: &[(i32, i32)],
    ) -> Result<YoutuVlVisionDiagnosticProjection, String> {
        let YoutuVlHostInputs {
            plan,
            patches,
            rope_freqs,
            window_attention_bias,
            full_attention_bias,
        } = prepare_youtu_vl_host_inputs(&self.config, shapes, patch_rows)?;
        let patch_input_shape = [
            plan.patch_bucket,
            self.config.channels * self.config.patch_size * self.config.patch_size,
        ];
        let rope_shape = [
            plan.patch_bucket,
            self.config.hidden / self.config.heads / 2,
        ];
        let bias_shape = [plan.patch_bucket, plan.patch_bucket];
        let patch_shape = [plan.patch_bucket, self.config.hidden];
        let merged_bucket_shape = [plan.patch_bucket / 4, self.config.text_hidden];
        let mut buffers = [
            vec![0u8; patch_shape.iter().product::<usize>() * std::mem::size_of::<f32>()],
            vec![0u8; patch_shape.iter().product::<usize>() * std::mem::size_of::<f32>()],
            vec![0u8; patch_shape.iter().product::<usize>() * std::mem::size_of::<f32>()],
            vec![0u8; merged_bucket_shape.iter().product::<usize>() * std::mem::size_of::<f32>()],
        ];
        let output_shapes = [patch_shape, patch_shape, patch_shape, merged_bucket_shape];
        let mut outputs = buffers
            .iter_mut()
            .zip(&output_shapes)
            .map(|(bytes, shape)| AuxiliaryOutput {
                bytes,
                dtype: AuxiliaryTensorDType::Float32,
                shape,
            })
            .collect::<Vec<_>>();
        self.ensure_bucket(plan.patch_bucket)?.invoke(
            &[
                AuxiliaryInput {
                    bytes: f32_as_bytes(&patches),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &patch_input_shape,
                },
                AuxiliaryInput {
                    bytes: f32_as_bytes(&rope_freqs),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &rope_shape,
                },
                AuxiliaryInput {
                    bytes: f32_as_bytes(&window_attention_bias),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &bias_shape,
                },
                AuxiliaryInput {
                    bytes: f32_as_bytes(&full_attention_bias),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &bias_shape,
                },
            ],
            &mut outputs,
        )?;
        drop(outputs);
        let mut values = buffers
            .into_iter()
            .enumerate()
            .map(|(index, bytes)| {
                checked_f32_output(&format!("Youtu-VL IREE diagnostic output {index}"), bytes)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter();
        let patch_projection = values.next().expect("four diagnostic outputs");
        let window_layer0 = values.next().expect("four diagnostic outputs");
        let full_layer1 = values.next().expect("four diagnostic outputs");
        let merger_bucket = values.next().expect("four diagnostic outputs");
        debug_assert!(values.next().is_none());
        let actual_tokens = plan.reverse_group_index.len();
        let actual_values = actual_tokens * self.config.text_hidden;
        let merger_window_order = merger_bucket[..actual_values].to_vec();
        let mut restored_output = Vec::with_capacity(actual_values);
        for &window_position in &plan.reverse_group_index {
            let start = window_position * self.config.text_hidden;
            restored_output
                .extend_from_slice(&merger_window_order[start..start + self.config.text_hidden]);
        }
        Ok(YoutuVlVisionDiagnosticProjection {
            patch_projection,
            window_layer0,
            full_layer1,
            merger_window_order,
            restored_output,
            patch_shape,
            merged_shape: [actual_tokens, self.config.text_hidden],
            window_group_index: plan.window_group_index,
            reverse_group_index: plan.reverse_group_index,
        })
    }
}
