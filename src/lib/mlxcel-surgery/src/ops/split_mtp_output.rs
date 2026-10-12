// Copyright 2025-2026 Lablup Inc.
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

//! Exclusive staging and commit support for `split-mtp` output directories.

use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, anyhow};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use safetensors::tensor::{Metadata, TensorInfo};

use super::{COMPANION_FILES, SplitMtpReport, SplitMtpResult};
use crate::SurgeryError;

const OUTPUT_FILES: &[&str] = &[
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "special_tokens_map.json",
];
const STALE_INDEX: &str = "model.safetensors.index.json";
const MAX_HEADER_SIZE: usize = 100_000_000;
static STAGING_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct OwnedTensor {
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

/// Materialize `weights` and write one exclusive `model.safetensors` with
/// the same byte layout as `safetensors::serialize_to_file`.
///
/// The crate writer uses `File::create`, which follows links and truncates
/// existing files. This streaming equivalent uses `create_new` and avoids a
/// second multi-gigabyte serialization buffer.
pub fn write_safetensors(weights: WeightMap, path: &Path) -> Result<u64, SurgeryError> {
    let mut owned: Vec<(String, OwnedTensor)> = Vec::with_capacity(weights.len());
    let mut bytes_written = 0u64;
    let mut entries: Vec<(String, UniquePtr<MlxArray>)> = weights.into_iter().collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    for (key, array) in entries {
        let dtype = safetensors_dtype(mlxcel_core::array_dtype(&array))?;
        let shape: Vec<usize> = mlxcel_core::array_shape(&array)
            .iter()
            .map(|&dimension| dimension as usize)
            .collect();
        mlxcel_core::eval(&array);
        let data = mlxcel_core::array_to_raw_bytes(&array);
        drop(array);
        bytes_written = bytes_written
            .checked_add(data.len() as u64)
            .ok_or_else(|| anyhow!("split-mtp: safetensors payload size overflow"))?;
        owned.push((key, OwnedTensor { dtype, shape, data }));
    }

    // Match safetensors 0.7's canonical order: descending dtype alignment,
    // then tensor name. Header offsets and payload order depend on it.
    owned.sort_by(|(left_name, left), (right_name, right)| {
        right.dtype.cmp(&left.dtype).then(left_name.cmp(right_name))
    });
    let mut offset = 0usize;
    let mut tensor_info = Vec::with_capacity(owned.len());
    for (name, tensor) in &owned {
        let end = offset
            .checked_add(tensor.data.len())
            .ok_or_else(|| anyhow!("split-mtp: safetensors payload offset overflow"))?;
        tensor_info.push((
            name.clone(),
            TensorInfo {
                dtype: tensor.dtype,
                shape: tensor.shape.clone(),
                data_offsets: (offset, end),
            },
        ));
        offset = end;
    }
    let mut data_info = HashMap::new();
    data_info.insert("format".to_string(), "mlx".to_string());
    let metadata = Metadata::new(Some(data_info), tensor_info)
        .map_err(|error| anyhow!("split-mtp: preparing safetensors metadata: {error}"))?;
    let mut header = serde_json::to_vec(&metadata)
        .map_err(|error| anyhow!("split-mtp: serializing safetensors metadata: {error}"))?;
    let aligned_len = header
        .len()
        .checked_add(7)
        .ok_or_else(|| anyhow!("split-mtp: safetensors header size overflow"))?
        / 8
        * 8;
    if aligned_len > MAX_HEADER_SIZE {
        return Err(anyhow!("split-mtp: safetensors header too large").into());
    }
    header.resize(aligned_len, b' ');

    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("split-mtp: creating {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&(aligned_len as u64).to_le_bytes())?;
    writer.write_all(&header)?;
    for (_, tensor) in owned {
        writer.write_all(&tensor.data)?;
    }
    writer.flush()?;
    Ok(bytes_written)
}

fn safetensors_dtype(mlx_dtype: i32) -> Result<safetensors::Dtype, SurgeryError> {
    use mlxcel_core::dtype as d;
    use safetensors::Dtype;
    Ok(match mlx_dtype {
        d::BOOL => Dtype::BOOL,
        d::UINT8 => Dtype::U8,
        d::UINT16 => Dtype::U16,
        d::UINT32 => Dtype::U32,
        d::UINT64 => Dtype::U64,
        d::INT8 => Dtype::I8,
        d::INT16 => Dtype::I16,
        d::INT32 => Dtype::I32,
        d::INT64 => Dtype::I64,
        d::FLOAT16 => Dtype::F16,
        d::FLOAT32 => Dtype::F32,
        d::FLOAT64 => Dtype::F64,
        d::BFLOAT16 => Dtype::BF16,
        other => {
            return Err(
                anyhow!("split-mtp: MLX dtype {other} has no safetensors equivalent").into(),
            );
        }
    })
}

pub(super) fn write_split_output(
    model_dir: &Path,
    output_dir: &Path,
    result: SplitMtpResult,
    source_layer: usize,
    force: bool,
) -> Result<SplitMtpReport, SurgeryError> {
    let staging = StagingDir::create(output_dir)?;
    if std::fs::canonicalize(model_dir)? == staging.output_path {
        return Err(source_alias_error(model_dir));
    }
    let tensors = result.weights.len();
    let bytes_written =
        write_safetensors(result.weights, &staging.path().join("model.safetensors"))?;
    let config_text = serde_json::to_string_pretty(&result.config)
        .map_err(|error| anyhow!("split-mtp: serializing config.json: {error}"))?;
    write_new(&staging.path().join("config.json"), config_text.as_bytes())?;

    let mut copied_files = Vec::new();
    let mut missing_files = Vec::new();
    for name in COMPANION_FILES {
        let source = model_dir.join(name);
        if source.is_file() {
            copy_new(&source, &staging.path().join(name))?;
            copied_files.push((*name).to_string());
        } else if *name == "tokenizer.json" || *name == "tokenizer_config.json" {
            missing_files.push((*name).to_string());
        }
    }

    commit_staging_dir(staging, output_dir, force)?;
    Ok(SplitMtpReport {
        output_dir: output_dir.to_path_buf(),
        source_layer,
        tensors,
        quantized_tensors: result.quantized_tensors,
        bytes_written,
        copied_files,
        missing_files,
    })
}

fn write_new(path: &Path, contents: &[u8]) -> Result<(), SurgeryError> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("split-mtp: creating {}", path.display()))?;
    file.write_all(contents)?;
    Ok(())
}

fn copy_new(source: &Path, destination: &Path) -> Result<(), SurgeryError> {
    let mut source_file = File::open(source)?;
    let mut destination_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .with_context(|| format!("split-mtp: creating {}", destination.display()))?;
    std::io::copy(&mut source_file, &mut destination_file)?;
    Ok(())
}

pub(super) struct StagingDir {
    path: PathBuf,
    output_path: PathBuf,
    committed: bool,
}

impl StagingDir {
    pub(super) fn create(output_dir: &Path) -> Result<Self, SurgeryError> {
        let resolved_output = resolve_output_path(output_dir)?;
        let parent = resolved_output
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent)?;
        let output_name = resolved_output
            .file_name()
            .ok_or_else(|| anyhow!("split-mtp: output path must name a directory"))?;

        for _ in 0..1000 {
            let sequence = STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let mut staging_name = OsString::from(".");
            staging_name.push(output_name);
            staging_name.push(format!(".split-mtp-{}-{sequence}.tmp", std::process::id()));
            let path = parent.join(staging_name);
            match create_private_directory(&path) {
                Ok(()) => {
                    return Ok(Self {
                        path,
                        output_path: resolved_output,
                        committed: false,
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(anyhow!(
            "split-mtp: could not create a unique staging directory beside {}",
            output_dir.display()
        )
        .into())
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }
}

enum UnresolvedComponent {
    Normal(OsString),
    Parent,
}

/// Resolve the longest existing prefix, then normalize missing components
/// without creating them. This preserves non-UTF-8 names and resolves `..`
/// relative to an ancestor symlink's target, matching filesystem traversal.
pub(super) fn resolve_output_path(output_dir: &Path) -> Result<PathBuf, SurgeryError> {
    let mut candidate = if output_dir.is_absolute() {
        output_dir.to_path_buf()
    } else {
        std::env::current_dir()?.join(output_dir)
    };
    let mut unresolved = Vec::new();
    loop {
        match std::fs::symlink_metadata(&candidate) {
            Ok(metadata) => {
                if unresolved.is_empty() && metadata.file_type().is_symlink() {
                    return Err(anyhow!(
                        "split-mtp: output directory {} is a symlink",
                        output_dir.display()
                    )
                    .into());
                }
                let mut resolved = std::fs::canonicalize(&candidate)?;
                for component in unresolved.into_iter().rev() {
                    match component {
                        UnresolvedComponent::Normal(name) => resolved.push(name),
                        UnresolvedComponent::Parent => {
                            resolved.pop();
                        }
                    }
                }
                return Ok(resolved);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }

        let component = candidate
            .components()
            .next_back()
            .ok_or_else(|| anyhow!("split-mtp: output path must name a directory"))?;
        match component {
            std::path::Component::Normal(name) => {
                unresolved.push(UnresolvedComponent::Normal(name.to_os_string()));
            }
            std::path::Component::ParentDir => unresolved.push(UnresolvedComponent::Parent),
            std::path::Component::CurDir => {}
            std::path::Component::RootDir | std::path::Component::Prefix(_) => {
                return Err(anyhow!("split-mtp: output path must name a directory").into());
            }
        }
        if !candidate.pop() {
            return Err(anyhow!("split-mtp: output path must name a directory").into());
        }
    }
}

fn create_private_directory(path: &Path) -> std::io::Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;

        builder.mode(0o700);
    }
    builder.create(path)
}

impl Drop for StagingDir {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

pub(super) fn commit_staging_dir(
    mut staging: StagingDir,
    output_dir: &Path,
    force: bool,
) -> Result<(), SurgeryError> {
    let current_output = resolve_output_path(output_dir)?;
    if current_output != staging.output_path {
        return Err(anyhow!(
            "split-mtp: output path {} changed while the drafter was staged",
            output_dir.display()
        )
        .into());
    }
    let output_path = staging.output_path.clone();
    let output_dir = output_path.as_path();
    match first_directory_entry(output_dir)? {
        None => {
            std::fs::rename(staging.path(), output_dir).with_context(|| {
                format!(
                    "split-mtp: committing staged drafter to {}",
                    output_dir.display()
                )
            })?;
            staging.committed = true;
            Ok(())
        }
        Some(existing) if !force => Err(anyhow!(
            "split-mtp: {} is not empty ({existing}); pass --force to overwrite drafter files",
            output_dir.display()
        )
        .into()),
        Some(_) => commit_over_existing(&mut staging, output_dir),
    }
}

pub(super) fn source_alias_error(model_dir: &Path) -> SurgeryError {
    anyhow!(
        "split-mtp: --output {} is the source checkpoint; write the drafter to a separate \
         directory, otherwise the source config.json is overwritten and its tokenizer \
         files are truncated",
        model_dir.display()
    )
    .into()
}

fn first_directory_entry(dir: &Path) -> Result<Option<String>, SurgeryError> {
    match std::fs::symlink_metadata(dir) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err(
                anyhow!("split-mtp: output directory {} is a symlink", dir.display()).into(),
            );
        }
        Ok(metadata) if !metadata.is_dir() => {
            return Err(anyhow!(
                "split-mtp: output path {} exists and is not a directory",
                dir.display()
            )
            .into());
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    }

    let mut names = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        names.push(entry?.file_name().to_string_lossy().into_owned());
    }
    names.sort();
    Ok(names.into_iter().next())
}

fn commit_over_existing(staging: &mut StagingDir, output_dir: &Path) -> Result<(), SurgeryError> {
    if let Some(file) = find_foreign_safetensors_file(output_dir)? {
        return Err(anyhow!(
            "split-mtp: {} holds weight file {file}; --force does not delete foreign weight files",
            output_dir.display()
        )
        .into());
    }
    for name in OUTPUT_FILES {
        let destination = output_dir.join(name);
        if std::fs::symlink_metadata(&destination)
            .is_ok_and(|metadata| metadata.file_type().is_symlink())
        {
            return Err(anyhow!(
                "split-mtp: refusing symlinked output file {}",
                destination.display()
            )
            .into());
        }
        if destination.is_dir() {
            return Err(anyhow!(
                "split-mtp: output path {} is a directory",
                destination.display()
            )
            .into());
        }
    }

    let stale_index = output_dir.join(STALE_INDEX);
    match std::fs::symlink_metadata(&stale_index) {
        Ok(metadata) if metadata.is_dir() => {
            return Err(anyhow!(
                "split-mtp: stale index path {} is a directory",
                stale_index.display()
            )
            .into());
        }
        Ok(_) => std::fs::remove_file(&stale_index)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }

    for name in OUTPUT_FILES {
        let source = staging.path().join(name);
        if std::fs::symlink_metadata(&source).is_ok() {
            std::fs::rename(&source, output_dir.join(name)).with_context(|| {
                format!(
                    "split-mtp: committing staged {} to {}",
                    name,
                    output_dir.display()
                )
            })?;
        }
    }
    std::fs::remove_dir(staging.path())?;
    staging.committed = true;
    Ok(())
}

fn find_foreign_safetensors_file(dir: &Path) -> Result<Option<String>, SurgeryError> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let name = entry?.file_name().to_string_lossy().into_owned();
        if name != "model.safetensors" && name.ends_with(".safetensors") {
            found.push(name);
        }
    }
    found.sort();
    Ok(found.into_iter().next())
}
