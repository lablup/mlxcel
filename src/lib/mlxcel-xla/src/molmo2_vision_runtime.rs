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

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read as _;
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
use crate::emitter::emit_molmo2_vision_diagnostics;
use crate::emitter::{Molmo2VisionConfig, Molmo2VisionWeightSpec, emit_molmo2_vision};
use crate::iree::{cached_vmfb_path, compile_one_to, iree_compile_bin, target_flags};
use crate::molmo2::Molmo2SafePooling;
use crate::weights::{bf16_to_f32, f16_to_f32, f32_le_to_f32};

const ENTRY_NAME: &str = "molmo2_vision.main";
#[cfg(feature = "diagnostics")]
const MOLMO2_VIT_PROBE_LAYER: usize = 24;
#[cfg(feature = "diagnostics")]
// 591490 = 513 * checkpoint hidden width 1152 + component 514.
const MOLMO2_VIT_PROBE_FLAT_ROW: usize = 513;

#[derive(Debug, Clone, PartialEq)]
pub struct Molmo2VisionProjection {
    pub values: Vec<f32>,
    pub shape: [usize; 2],
    pub signed_pooling_indices: Vec<i32>,
    pub valid_pooling_counts: Vec<i32>,
    pub elapsed_seconds: f64,
    pub upload_bytes: usize,
    pub transfer_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Molmo2VisionInput<'a> {
    pub patches: &'a [f32],
    pub patches_shape: [usize; 3],
    pub image_token_pooling: &'a [i32],
    pub pooling_shape: [usize; 2],
    pub image_grid: [i32; 4],
    pub image_num_crops: usize,
    pub prompt_image_patch_count: usize,
}

#[cfg(feature = "diagnostics")]
#[derive(Debug, Clone, PartialEq)]
pub struct Molmo2VisionDiagnosticStage {
    pub name: String,
    pub values: Vec<f32>,
    pub shape: Vec<usize>,
}

#[cfg(feature = "diagnostics")]
#[derive(Debug, Clone, PartialEq)]
pub struct Molmo2VisionDiagnostics {
    pub stages: Vec<Molmo2VisionDiagnosticStage>,
    pub projected_values: Vec<f32>,
    pub projected_shape: [usize; 2],
    pub signed_pooling_indices: Vec<i32>,
    pub valid_pooling_counts: Vec<i32>,
    pub active_groups: Vec<usize>,
    pub elapsed_seconds: f64,
    pub upload_bytes: usize,
    pub transfer_bytes: usize,
}

struct PreparedMolmo2VisionInput {
    padded_patches: Vec<f32>,
    padded_signed_indices: Vec<i32>,
    signed_pooling_indices: Vec<i32>,
    valid_pooling_counts: Vec<i32>,
    active_groups: Vec<usize>,
    #[cfg(feature = "diagnostics")]
    crops: usize,
    #[cfg(feature = "diagnostics")]
    groups: usize,
}

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn sha256(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
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
    Ok(hex(&digest.finalize()))
}

fn generation_identity(compiler: &Path, flags: &[&str], mlir: &str) -> Result<String, String> {
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
    Ok(format!(
        "compiler={};compiler_sha256={};version={};flags={flags:?};mlir_sha256={}",
        compiler.display(),
        sha256_file(compiler)?,
        String::from_utf8_lossy(&output.stdout).trim(),
        sha256(mlir.as_bytes())
    ))
}

fn model_shards(model_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut shards = std::fs::read_dir(model_dir)
        .map_err(|error| format!("read {}: {error}", model_dir.display()))?
        .map(|entry| {
            entry
                .map(|entry| entry.path())
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
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

fn resolve_shards(
    model_dir: &Path,
    specs: &[Molmo2VisionWeightSpec],
) -> Result<Vec<PathBuf>, String> {
    let required = specs
        .iter()
        .map(|spec| spec.name.clone())
        .collect::<BTreeSet<_>>();
    let mut locations = BTreeMap::<String, PathBuf>::new();
    for shard in model_shards(model_dir)? {
        let file =
            File::open(&shard).map_err(|error| format!("open {}: {error}", shard.display()))?;
        // Safety: read-only map remains live for the header scan.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|error| format!("mmap {}: {error}", shard.display()))?;
        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|error| format!("parse {}: {error}", shard.display()))?;
        for name in tensors.names() {
            if required.contains(name)
                && let Some(previous) = locations.insert(name.to_string(), shard.clone())
            {
                return Err(format!(
                    "Molmo2 vision tensor {name} occurs in {} and {}",
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
            "checkpoint is missing {} Molmo2 vision tensor(s): {}",
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

fn native_f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn finite(label: &str, values: &[f32]) -> Result<(), String> {
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

fn load_weights(
    model_dir: &Path,
    specs: &[Molmo2VisionWeightSpec],
) -> Result<(Vec<AuxiliaryWeight>, String), String> {
    let shards = resolve_shards(model_dir, specs)?;
    let mut by_shard = BTreeMap::<&Path, Vec<usize>>::new();
    for (index, shard) in shards.iter().enumerate() {
        by_shard.entry(shard).or_default().push(index);
    }
    let mut loaded = (0..specs.len())
        .map(|_| None)
        .collect::<Vec<Option<AuxiliaryWeight>>>();
    let mut schema = vec![String::new(); specs.len()];
    for (shard, indices) in by_shard {
        let file =
            File::open(shard).map_err(|error| format!("open {}: {error}", shard.display()))?;
        // Safety: read-only map remains live while tensors are copied.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|error| format!("mmap {}: {error}", shard.display()))?;
        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|error| format!("parse {}: {error}", shard.display()))?;
        for index in indices {
            let spec = &specs[index];
            let tensor = tensors
                .tensor(&spec.name)
                .map_err(|error| format!("load Molmo2 vision tensor {}: {error}", spec.name))?;
            if tensor.shape() != spec.shape {
                return Err(format!(
                    "Molmo2 vision tensor {} has shape {:?}, expected {:?}",
                    spec.name,
                    tensor.shape(),
                    spec.shape
                ));
            }
            let values = match tensor.dtype() {
                Dtype::BF16 => bf16_to_f32(tensor.data()),
                Dtype::F16 => f16_to_f32(tensor.data()),
                Dtype::F32 => f32_le_to_f32(tensor.data()),
                dtype => {
                    return Err(format!(
                        "Molmo2 vision tensor {} has unsupported dtype {dtype:?}",
                        spec.name
                    ));
                }
            };
            finite(&format!("Molmo2 vision tensor {}", spec.name), &values)?;
            schema[index] = format!("{}:{:?}:{:?}", spec.name, tensor.dtype(), tensor.shape());
            loaded[index] = Some(AuxiliaryWeight {
                name: spec.name.clone(),
                bytes: native_f32_bytes(&values),
                dtype: AuxiliaryWeightDType::Float32,
                shape: spec.shape.clone(),
            });
        }
    }
    let loaded = loaded
        .into_iter()
        .map(|weight| {
            weight.ok_or_else(|| "resolved Molmo2 vision tensor was not loaded".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((loaded, schema.join("\n")))
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    // Safety: f32 has no invalid bit patterns and the result is borrowed.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn i32_bytes(values: &[i32]) -> &[u8] {
    // Safety: i32 has no invalid bit patterns and the result is borrowed.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn decode_output(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(4) {
        return Err("Molmo2 IREE output byte count is not f32-aligned".to_string());
    }
    let values = bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect::<Vec<_>>();
    finite("Molmo2 IREE projected output", &values)?;
    Ok(values)
}

fn prepare_vision_input(
    config: &Molmo2VisionConfig,
    input: Molmo2VisionInput<'_>,
) -> Result<PreparedMolmo2VisionInput, String> {
    let [crops, patches, patch_dim] = input.patches_shape;
    if crops != input.image_num_crops
        || crops == 0
        || crops > config.static_crops
        || patches != config.patches_per_crop
        || patch_dim != config.patch_dim
    {
        return Err(format!(
            "Molmo2 patch shape {:?}, image_num_crops={} disagrees with static [{},{},{}]",
            input.patches_shape,
            input.image_num_crops,
            config.static_crops,
            config.patches_per_crop,
            config.patch_dim
        ));
    }
    let patch_values = crops
        .checked_mul(patches)
        .and_then(|value| value.checked_mul(patch_dim))
        .ok_or_else(|| "Molmo2 patch shape overflowed".to_string())?;
    if input.patches.len() != patch_values {
        return Err(format!(
            "Molmo2 patch payload has {} values, expected {patch_values}",
            input.patches.len()
        ));
    }
    finite("Molmo2 patches", input.patches)?;
    let [groups, group_size] = input.pooling_shape;
    let grid = input
        .image_grid
        .iter()
        .try_fold((), |(), value| {
            (*value >= 0)
                .then_some(())
                .ok_or_else(|| "Molmo2 image grid contains a negative dimension".to_string())
        })
        .map(|()| input.image_grid.map(|value| value as usize))?;
    if !config.valid_runtime_geometry(crops, grid) {
        return Err(format!(
            "Molmo2 crop count {crops} and image grid {:?} disagree with processor geometry",
            input.image_grid
        ));
    }
    let [lo_h, lo_w, hi_h, hi_w] = grid;
    let grid_groups = lo_h
        .checked_mul(lo_w)
        .and_then(|low| {
            hi_h.checked_mul(hi_w)
                .and_then(|high| low.checked_add(high))
        })
        .ok_or_else(|| "Molmo2 image grid overflowed".to_string())?;
    if groups != grid_groups || groups > config.static_pool_groups || group_size != config.pool_size
    {
        return Err(format!(
            "Molmo2 pooling shape {:?} disagrees with grid {:?} and static [{},{}]",
            input.pooling_shape, input.image_grid, config.static_pool_groups, config.pool_size
        ));
    }
    let safe = Molmo2SafePooling::prepare(
        input.image_token_pooling,
        groups,
        group_size,
        crops * patches,
    )
    .map_err(|error| error.to_string())?;
    let active_groups = safe
        .active_groups_for_prompt(grid_groups, input.prompt_image_patch_count)
        .map_err(|error| error.to_string())?;
    let static_patch_values = config.static_crops * config.patches_per_crop * config.patch_dim;
    let mut padded_patches = vec![0.0f32; static_patch_values];
    padded_patches[..input.patches.len()].copy_from_slice(input.patches);
    let mut padded_signed_indices = vec![-1i32; config.static_pool_groups * config.pool_size];
    padded_signed_indices[..safe.signed_indices.len()].copy_from_slice(&safe.signed_indices);
    Ok(PreparedMolmo2VisionInput {
        padded_patches,
        padded_signed_indices,
        signed_pooling_indices: safe.signed_indices,
        valid_pooling_counts: safe.valid_counts,
        active_groups,
        #[cfg(feature = "diagnostics")]
        crops,
        #[cfg(feature = "diagnostics")]
        groups,
    })
}

fn compile_vision_module(
    model_dir: &Path,
    device: &str,
    config: &Molmo2VisionConfig,
    mlir: &str,
    tag: &str,
    diagnostic_identity: Option<&str>,
) -> Result<IreeAuxiliaryModule, String> {
    let compiler = iree_compile_bin()?;
    if !compiler.is_file() {
        return Err(format!("iree-compile not found at {}", compiler.display()));
    }
    let flags = target_flags(device)?;
    let cache = std::env::temp_dir().join("mlxcel-xla-molmo2-vision-vmfb");
    std::fs::create_dir_all(&cache)
        .map_err(|error| format!("mkdir {}: {error}", cache.display()))?;
    let (weights, checkpoint_schema) = load_weights(model_dir, &config.weight_specs())?;
    let graph_identity = diagnostic_identity
        .map(|identity| format!("{};diagnostics={identity}", config.fingerprint()))
        .unwrap_or_else(|| config.fingerprint());
    let contract = AuxiliaryArtifactContract::new(
        ENTRY_NAME,
        format!(
            "{graph_identity};checkpoint_schema_sha256={}",
            sha256(checkpoint_schema.as_bytes())
        ),
        generation_identity(&compiler, flags, mlir)?,
    )?;
    let vmfb = cached_vmfb_path(&compiler, mlir, flags, &cache, tag, 0);
    ensure_qualified_auxiliary_artifact(&vmfb, &contract, &weights, |temporary| {
        compile_one_to(&compiler, mlir, flags, &cache, tag, 0, temporary)
    })?;
    IreeAuxiliaryModule::load(device, &vmfb, &contract, weights)
}

pub struct IreeMolmo2VisionProjector {
    module: IreeAuxiliaryModule,
    config: Molmo2VisionConfig,
}

impl IreeMolmo2VisionProjector {
    pub fn load(model_dir: &Path, device: &str) -> Result<Self, String> {
        let config = Molmo2VisionConfig::from_model_dir(model_dir)?;
        let mlir = emit_molmo2_vision(&config);
        let module =
            compile_vision_module(model_dir, device, &config, &mlir, "molmo2-vision", None)?;
        Ok(Self { module, config })
    }

    pub fn image_patch_id(&self) -> i32 {
        self.config.image_patch_id
    }

    pub fn text_hidden_size(&self) -> usize {
        self.config.text_hidden
    }

    pub fn artifact_fingerprint(&self) -> u64 {
        self.module.fingerprint()
    }

    pub fn project(
        &mut self,
        input: Molmo2VisionInput<'_>,
    ) -> Result<Molmo2VisionProjection, String> {
        let prepared = prepare_vision_input(&self.config, input)?;
        let output_shape = [self.config.static_pool_groups, self.config.text_hidden];
        let mut output = vec![0u8; output_shape.iter().product::<usize>() * 4];
        let patch_shape = [
            self.config.static_crops,
            self.config.patches_per_crop,
            self.config.patch_dim,
        ];
        let pooling_shape = [self.config.static_pool_groups, self.config.pool_size];
        let started = Instant::now();
        self.module.invoke(
            &[
                AuxiliaryInput {
                    bytes: f32_bytes(&prepared.padded_patches),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &patch_shape,
                },
                AuxiliaryInput {
                    bytes: i32_bytes(&prepared.padded_signed_indices),
                    dtype: AuxiliaryTensorDType::Int32,
                    shape: &pooling_shape,
                },
            ],
            &mut [AuxiliaryOutput {
                bytes: &mut output,
                dtype: AuxiliaryTensorDType::Float32,
                shape: &output_shape,
            }],
        )?;
        let all_values = decode_output(&output)?;
        let mut values = Vec::with_capacity(prepared.active_groups.len() * self.config.text_hidden);
        for &group in &prepared.active_groups {
            let start = group * self.config.text_hidden;
            values.extend_from_slice(&all_values[start..start + self.config.text_hidden]);
        }
        Ok(Molmo2VisionProjection {
            shape: [
                values.len() / self.config.text_hidden,
                self.config.text_hidden,
            ],
            values,
            signed_pooling_indices: prepared.signed_pooling_indices,
            valid_pooling_counts: prepared.valid_pooling_counts,
            elapsed_seconds: started.elapsed().as_secs_f64(),
            upload_bytes: std::mem::size_of_val(prepared.padded_patches.as_slice())
                + std::mem::size_of_val(prepared.padded_signed_indices.as_slice()),
            transfer_bytes: output.len(),
        })
    }
}

#[cfg(feature = "diagnostics")]
struct Molmo2DiagnosticStageSpec {
    name: String,
    static_shape: Vec<usize>,
    active_shape: Vec<usize>,
}

#[cfg(feature = "diagnostics")]
fn diagnostic_stage_specs(
    config: &Molmo2VisionConfig,
    crops: usize,
    groups: usize,
) -> Vec<Molmo2DiagnosticStageSpec> {
    let hidden_static = vec![config.static_crops, config.patches_per_crop, config.hidden];
    let hidden_active = vec![crops, config.patches_per_crop, config.hidden];
    let mut specs = vec![
        Molmo2DiagnosticStageSpec {
            name: "vit.patch_embedding".to_string(),
            static_shape: hidden_static.clone(),
            active_shape: hidden_active.clone(),
        },
        Molmo2DiagnosticStageSpec {
            name: "vit.position_embedding".to_string(),
            static_shape: vec![config.position_count, config.hidden],
            active_shape: vec![config.position_count, config.hidden],
        },
        Molmo2DiagnosticStageSpec {
            name: "vit.positioned_embedding".to_string(),
            static_shape: hidden_static.clone(),
            active_shape: hidden_active.clone(),
        },
        Molmo2DiagnosticStageSpec {
            name: "vit.block.0".to_string(),
            static_shape: hidden_static.clone(),
            active_shape: hidden_active.clone(),
        },
    ];
    let mut selected_layers = config.selected_layers.clone();
    selected_layers.sort_unstable();
    let probe_split = selected_layers.partition_point(|layer| *layer < MOLMO2_VIT_PROBE_LAYER);
    specs.extend(
        selected_layers[..probe_split]
            .iter()
            .map(|layer| Molmo2DiagnosticStageSpec {
                name: format!("vit.selected.{layer}"),
                static_shape: hidden_static.clone(),
                active_shape: hidden_active.clone(),
            }),
    );
    if config.emitted_layers > MOLMO2_VIT_PROBE_LAYER
        && config.static_crops * config.patches_per_crop > MOLMO2_VIT_PROBE_FLAT_ROW
    {
        specs.extend(
            [
                "input",
                "attention_norm",
                "attention",
                "post_attention_residual",
                "ffn_norm",
                "mlp",
                "output",
            ]
            .into_iter()
            .map(|stage| Molmo2DiagnosticStageSpec {
                name: format!(
                    "vit.probe.{}.row.{}.{}",
                    MOLMO2_VIT_PROBE_LAYER, MOLMO2_VIT_PROBE_FLAT_ROW, stage
                ),
                static_shape: vec![1, config.hidden],
                active_shape: vec![1, config.hidden],
            }),
        );
    }
    specs.extend(
        selected_layers[probe_split..]
            .iter()
            .map(|layer| Molmo2DiagnosticStageSpec {
                name: format!("vit.selected.{layer}"),
                static_shape: hidden_static.clone(),
                active_shape: hidden_active.clone(),
            }),
    );
    specs.extend([
        Molmo2DiagnosticStageSpec {
            name: "vit.concatenated".to_string(),
            static_shape: vec![
                config.static_crops * config.patches_per_crop,
                config.selected_width(),
            ],
            active_shape: vec![crops * config.patches_per_crop, config.selected_width()],
        },
        Molmo2DiagnosticStageSpec {
            name: "pool.gathered_masked".to_string(),
            static_shape: vec![
                config.static_pool_groups,
                config.pool_size,
                config.selected_width(),
            ],
            active_shape: vec![groups, config.pool_size, config.selected_width()],
        },
        Molmo2DiagnosticStageSpec {
            name: "pool.valid_counts".to_string(),
            static_shape: vec![config.static_pool_groups],
            active_shape: vec![groups],
        },
        Molmo2DiagnosticStageSpec {
            name: "pool.query".to_string(),
            static_shape: vec![config.static_pool_groups, config.selected_width()],
            active_shape: vec![groups, config.selected_width()],
        },
        Molmo2DiagnosticStageSpec {
            name: "pool.output".to_string(),
            static_shape: vec![config.static_pool_groups, config.pool_hidden],
            active_shape: vec![groups, config.pool_hidden],
        },
        Molmo2DiagnosticStageSpec {
            name: "projector.w1".to_string(),
            static_shape: vec![config.static_pool_groups, config.projector_intermediate],
            active_shape: vec![groups, config.projector_intermediate],
        },
        Molmo2DiagnosticStageSpec {
            name: "projector.silu".to_string(),
            static_shape: vec![config.static_pool_groups, config.projector_intermediate],
            active_shape: vec![groups, config.projector_intermediate],
        },
        Molmo2DiagnosticStageSpec {
            name: "projector.w3".to_string(),
            static_shape: vec![config.static_pool_groups, config.projector_intermediate],
            active_shape: vec![groups, config.projector_intermediate],
        },
        Molmo2DiagnosticStageSpec {
            name: "projector.product".to_string(),
            static_shape: vec![config.static_pool_groups, config.projector_intermediate],
            active_shape: vec![groups, config.projector_intermediate],
        },
        Molmo2DiagnosticStageSpec {
            name: "projector.output_all".to_string(),
            static_shape: vec![config.static_pool_groups, config.text_hidden],
            active_shape: vec![groups, config.text_hidden],
        },
    ]);
    specs
}

#[cfg(feature = "diagnostics")]
pub struct IreeMolmo2VisionDiagnosticProjector {
    module: IreeAuxiliaryModule,
    config: Molmo2VisionConfig,
}

#[cfg(feature = "diagnostics")]
impl IreeMolmo2VisionDiagnosticProjector {
    pub fn load(model_dir: &Path, device: &str) -> Result<Self, String> {
        let config = Molmo2VisionConfig::from_model_dir(model_dir)?;
        let mlir = emit_molmo2_vision_diagnostics(&config);
        let module = compile_vision_module(
            model_dir,
            device,
            &config,
            &mlir,
            "molmo2-vision-diagnostics",
            Some("first-divergence-v3-layer24-row513"),
        )?;
        Ok(Self { module, config })
    }

    #[must_use]
    pub fn image_patch_id(&self) -> i32 {
        self.config.image_patch_id
    }

    #[must_use]
    pub fn text_hidden_size(&self) -> usize {
        self.config.text_hidden
    }

    pub fn project(
        &mut self,
        input: Molmo2VisionInput<'_>,
    ) -> Result<Molmo2VisionDiagnostics, String> {
        let prepared = prepare_vision_input(&self.config, input)?;
        let specs = diagnostic_stage_specs(&self.config, prepared.crops, prepared.groups);
        let mut buffers = specs
            .iter()
            .map(|spec| {
                vec![0u8; spec.static_shape.iter().product::<usize>() * std::mem::size_of::<f32>()]
            })
            .collect::<Vec<_>>();
        let mut outputs = buffers
            .iter_mut()
            .zip(&specs)
            .map(|(bytes, spec)| AuxiliaryOutput {
                bytes,
                dtype: AuxiliaryTensorDType::Float32,
                shape: &spec.static_shape,
            })
            .collect::<Vec<_>>();
        let patch_shape = [
            self.config.static_crops,
            self.config.patches_per_crop,
            self.config.patch_dim,
        ];
        let pooling_shape = [self.config.static_pool_groups, self.config.pool_size];
        let started = Instant::now();
        self.module.invoke(
            &[
                AuxiliaryInput {
                    bytes: f32_bytes(&prepared.padded_patches),
                    dtype: AuxiliaryTensorDType::Float32,
                    shape: &patch_shape,
                },
                AuxiliaryInput {
                    bytes: i32_bytes(&prepared.padded_signed_indices),
                    dtype: AuxiliaryTensorDType::Int32,
                    shape: &pooling_shape,
                },
            ],
            &mut outputs,
        )?;
        let elapsed_seconds = started.elapsed().as_secs_f64();
        drop(outputs);
        let transfer_bytes = buffers.iter().map(Vec::len).sum();
        let stages = buffers
            .into_iter()
            .zip(specs)
            .map(|(bytes, spec)| {
                let mut values = decode_output(&bytes)?;
                values.truncate(spec.active_shape.iter().product());
                Ok(Molmo2VisionDiagnosticStage {
                    name: spec.name,
                    values,
                    shape: spec.active_shape,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let projected_all = stages
            .last()
            .ok_or_else(|| "Molmo2 diagnostic projector output is missing".to_string())?;
        let mut projected_values =
            Vec::with_capacity(prepared.active_groups.len() * self.config.text_hidden);
        for &group in &prepared.active_groups {
            let start = group * self.config.text_hidden;
            projected_values.extend_from_slice(
                projected_all
                    .values
                    .get(start..start + self.config.text_hidden)
                    .ok_or_else(|| {
                        "Molmo2 diagnostic active projection row is truncated".to_string()
                    })?,
            );
        }
        Ok(Molmo2VisionDiagnostics {
            projected_shape: [
                projected_values.len() / self.config.text_hidden,
                self.config.text_hidden,
            ],
            projected_values,
            stages,
            signed_pooling_indices: prepared.signed_pooling_indices,
            valid_pooling_counts: prepared.valid_pooling_counts,
            active_groups: prepared.active_groups,
            elapsed_seconds,
            upload_bytes: std::mem::size_of_val(prepared.padded_patches.as_slice())
                + std::mem::size_of_val(prepared.padded_signed_indices.as_slice()),
            transfer_bytes,
        })
    }
}

#[cfg(test)]
#[path = "molmo2_vision_runtime_tests.rs"]
mod tests;
