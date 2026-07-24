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

//! Resident packed-weight IREE runtime for Molmo v1 vision preprocessing.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::aux::{
    AuxiliaryInput, AuxiliaryOutput, AuxiliaryTensorDType, AuxiliaryWeight, AuxiliaryWeightDType,
    IreeAuxiliaryModule,
};
use crate::aux_manifest::{AuxiliaryArtifactContract, ensure_qualified_auxiliary_artifact};
use crate::emitter::{
    MolmoVisionConfig, MolmoVisionWeightDType, MolmoVisionWeightSpec, emit_molmo_vision,
};
use crate::iree::{cached_vmfb_path, compile_one_to, iree_compile_bin, target_flags};
use crate::vision_runtime::{
    checked_f32_output, compiler_generation_identity, f32_as_bytes, model_shards, native_f32_bytes,
    sha256_hex, validate_finite_values,
};
use crate::weights::{bf16_to_f32, f16_to_f32, f32_le_to_f32};

const ENTRY_NAME: &str = "molmo_vision.main";

fn checked_product(dimensions: &[usize], label: &str) -> Result<usize, String> {
    dimensions.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| format!("Molmo {label} shape overflows usize"))
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct MolmoVisionProjection {
    pub values: Vec<f32>,
    pub shape: [usize; 3],
    pub elapsed_seconds: f64,
    pub upload_bytes: usize,
    pub transfer_bytes: usize,
}

fn resolve_weight_shards(
    model_dir: &Path,
    specs: &[MolmoVisionWeightSpec],
) -> Result<Vec<PathBuf>, String> {
    let required = specs
        .iter()
        .map(|spec| spec.name.clone())
        .collect::<BTreeSet<_>>();
    let mut locations = BTreeMap::<String, PathBuf>::new();
    for shard in model_shards(model_dir)? {
        let file =
            File::open(&shard).map_err(|error| format!("open {}: {error}", shard.display()))?;
        // Safety: this read-only map lives through the header scan.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|error| format!("mmap {}: {error}", shard.display()))?;
        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|error| format!("parse {}: {error}", shard.display()))?;
        for name in tensors.names() {
            if required.contains(name)
                && let Some(previous) = locations.insert(name.to_string(), shard.clone())
            {
                return Err(format!(
                    "Molmo vision tensor {name:?} is duplicated in {} and {}",
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
            "checkpoint is missing {} Molmo vision tensor(s): {}",
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
        .map(|spec| locations[&spec.name].clone())
        .collect())
}

fn load_weights(
    model_dir: &Path,
    specs: &[MolmoVisionWeightSpec],
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
        // Safety: this map lives while every selected tensor is copied.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|error| format!("mmap {}: {error}", shard.display()))?;
        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|error| format!("parse {}: {error}", shard.display()))?;
        for index in indices {
            let spec = &specs[index];
            let tensor = tensors
                .tensor(&spec.name)
                .map_err(|error| format!("resolved Molmo vision tensor {}: {error}", spec.name))?;
            if tensor.shape() != spec.shape {
                return Err(format!(
                    "Molmo vision tensor {} has shape {:?}, expected {:?}",
                    spec.name,
                    tensor.shape(),
                    spec.shape
                ));
            }
            let (bytes, dtype) = match spec.dtype {
                MolmoVisionWeightDType::Float32 => {
                    let values = match tensor.dtype() {
                        Dtype::BF16 => bf16_to_f32(tensor.data()),
                        Dtype::F16 => f16_to_f32(tensor.data()),
                        Dtype::F32 => f32_le_to_f32(tensor.data()),
                        other => {
                            return Err(format!(
                                "Molmo tensor {} has {other:?}, expected a floating dtype",
                                spec.name
                            ));
                        }
                    };
                    validate_finite_values(&spec.name, &values)?;
                    (native_f32_bytes(values), AuxiliaryWeightDType::Float32)
                }
                MolmoVisionWeightDType::Float16 => {
                    if tensor.dtype() != Dtype::F16 {
                        return Err(format!(
                            "Molmo tensor {} has {:?}, expected F16",
                            spec.name,
                            tensor.dtype()
                        ));
                    }
                    (tensor.data().to_vec(), AuxiliaryWeightDType::Float16)
                }
                MolmoVisionWeightDType::Uint32 => {
                    if tensor.dtype() != Dtype::U32 {
                        return Err(format!(
                            "Molmo tensor {} has {:?}, expected U32",
                            spec.name,
                            tensor.dtype()
                        ));
                    }
                    (tensor.data().to_vec(), AuxiliaryWeightDType::Uint32)
                }
            };
            source_schema[index] =
                format!("{}:{:?}:{:?}", spec.name, tensor.dtype(), tensor.shape());
            loaded[index] = Some(AuxiliaryWeight {
                name: spec.name.clone(),
                bytes,
                dtype,
                shape: spec.shape.clone(),
            });
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
    config: &MolmoVisionConfig,
    mlir: &str,
) -> Result<IreeAuxiliaryModule, String> {
    let compiler = iree_compile_bin()?;
    let flags = target_flags(device)?;
    let cache = std::env::temp_dir().join("mlxcel-xla-molmo-vision-vmfb");
    std::fs::create_dir_all(&cache)
        .map_err(|error| format!("mkdir {}: {error}", cache.display()))?;
    let specs = config.weight_specs();
    let (weights, checkpoint_schema) = load_weights(model_dir, &specs)?;
    let processor_path = model_dir.join("preprocessor_config.json");
    let processor = std::fs::read(&processor_path)
        .map_err(|error| format!("{}: {error}", processor_path.display()))?;
    let contract = AuxiliaryArtifactContract::new(
        ENTRY_NAME,
        format!(
            "{};processor_sha256={};checkpoint_schema_sha256={}",
            config.fingerprint(),
            sha256_hex(&processor),
            sha256_hex(checkpoint_schema.as_bytes())
        ),
        compiler_generation_identity(&compiler, flags, mlir)?,
    )?;
    let vmfb = cached_vmfb_path(&compiler, mlir, flags, &cache, "molmo-v1-vision", 0);
    ensure_qualified_auxiliary_artifact(&vmfb, &contract, &weights, |temporary| {
        compile_one_to(
            &compiler,
            mlir,
            flags,
            &cache,
            "molmo-v1-vision",
            0,
            temporary,
        )
    })?;
    IreeAuxiliaryModule::load(device, &vmfb, &contract, weights)
}

pub struct IreeMolmoVisionProjector {
    module: IreeAuxiliaryModule,
    config: MolmoVisionConfig,
}

impl IreeMolmoVisionProjector {
    pub fn load(model_dir: &Path, device: &str) -> Result<Self, String> {
        let config = MolmoVisionConfig::from_model_dir(model_dir)?;
        let mlir = emit_molmo_vision(&config);
        let module = compile_and_load(model_dir, device, &config, &mlir)?;
        Ok(Self { module, config })
    }

    #[must_use]
    pub fn max_crops(&self) -> usize {
        self.config.max_crops
    }

    #[must_use]
    pub fn patches_per_crop(&self) -> usize {
        self.config.patches_per_crop
    }

    #[must_use]
    pub fn patch_width(&self) -> usize {
        self.config.patch_width
    }

    #[must_use]
    pub fn projected_rows_per_crop(&self) -> usize {
        self.config.projected_rows_per_crop()
    }

    #[must_use]
    pub fn text_hidden(&self) -> usize {
        self.config.text_hidden
    }

    #[must_use]
    pub fn artifact_fingerprint(&self) -> u64 {
        self.module.fingerprint()
    }

    pub fn project(
        &mut self,
        pixels: &[f32],
        masks: &[f32],
        crop_count: usize,
    ) -> Result<MolmoVisionProjection, String> {
        if crop_count == 0 || crop_count > self.config.max_crops {
            return Err(format!(
                "Molmo crop count {crop_count} is outside 1..={}",
                self.config.max_crops
            ));
        }
        let pixel_count = crop_count
            .checked_mul(self.config.patches_per_crop)
            .and_then(|count| count.checked_mul(self.config.patch_width))
            .ok_or_else(|| "Molmo pixel shape overflowed".to_string())?;
        let mask_count = crop_count
            .checked_mul(self.config.patches_per_crop)
            .ok_or_else(|| "Molmo mask shape overflowed".to_string())?;
        if pixels.len() != pixel_count || masks.len() != mask_count {
            return Err(format!(
                "Molmo processor counts pixels={}/{} masks={}/{}",
                pixels.len(),
                pixel_count,
                masks.len(),
                mask_count
            ));
        }
        validate_finite_values("Molmo pixel_values", pixels)?;
        validate_finite_values("Molmo image_masks", masks)?;
        if let Some((index, value)) = masks
            .iter()
            .enumerate()
            .find(|(_, value)| !(**value >= -1.0 && **value <= 1.0))
        {
            return Err(format!(
                "Molmo image_masks[{index}]={value} is outside [-1,1]"
            ));
        }
        let static_pixels = checked_product(
            &[
                self.config.max_crops,
                self.config.patches_per_crop,
                self.config.patch_width,
            ],
            "static pixel",
        )?;
        let static_masks = checked_product(
            &[self.config.max_crops, self.config.patches_per_crop],
            "static mask",
        )?;
        let mut padded_pixels = vec![-1.0f32; static_pixels];
        padded_pixels[..pixels.len()].copy_from_slice(pixels);
        let mut padded_masks = vec![-1.0f32; static_masks];
        padded_masks[..masks.len()].copy_from_slice(masks);
        let output_shape = [
            self.config.max_crops,
            self.config.projected_rows_per_crop(),
            self.config.text_hidden,
        ];
        let output_elements = checked_product(&output_shape, "projected output")?;
        let output_bytes = output_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "Molmo projected output byte count overflows usize".to_string())?;
        let mut output = vec![0u8; output_bytes];
        let pixel_shape = [
            self.config.max_crops,
            self.config.patches_per_crop,
            self.config.patch_width,
        ];
        let mask_shape = [self.config.max_crops, self.config.patches_per_crop];
        let started = Instant::now();
        self.module.invoke(
            &[
                AuxiliaryInput {
                    bytes: f32_as_bytes(&padded_pixels),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &pixel_shape,
                },
                AuxiliaryInput {
                    bytes: f32_as_bytes(&padded_masks),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &mask_shape,
                },
            ],
            &mut [AuxiliaryOutput {
                bytes: &mut output,
                dtype: AuxiliaryTensorDType::Float32,
                shape: &output_shape,
            }],
        )?;
        let elapsed_seconds = started.elapsed().as_secs_f64();
        let mut values = checked_f32_output("IREE Molmo projected output", output)?;
        let active_count = checked_product(
            &[
                crop_count,
                self.config.projected_rows_per_crop(),
                self.config.text_hidden,
            ],
            "active projected output",
        )?;
        values.truncate(active_count);
        let upload_bytes = static_pixels
            .checked_add(static_masks)
            .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "Molmo invocation upload byte count overflows usize".to_string())?;
        let transfer_bytes = active_count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "Molmo invocation transfer byte count overflows usize".to_string())?;
        Ok(MolmoVisionProjection {
            values,
            shape: [
                crop_count,
                self.config.projected_rows_per_crop(),
                self.config.text_hidden,
            ],
            elapsed_seconds,
            upload_bytes,
            transfer_bytes,
        })
    }
}
