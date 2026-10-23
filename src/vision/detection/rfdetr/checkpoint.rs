// Copyright 2025-2026 Lablup Inc.
// Licensed under the Apache License, Version 2.0.

//! Bounded RF-DETR checkpoint inspection before the native loader runs.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, Metadata};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use mlxcel_core::weights::WeightMap;
use serde::Deserialize;

const MAX_INDEX_BYTES: u64 = 2 * 1024 * 1024;
const MAX_HEADER_BYTES: u64 = 4 * 1024 * 1024;
const MAX_SHARDS: usize = 128;
pub(crate) const MAX_TENSORS: usize = 1024;
pub(crate) const MAX_KEY_BYTES: usize = 512;
const MAX_RANK: usize = 8;
pub(crate) const MAX_DIMENSION: usize = 16_384;
const MAX_TENSOR_ELEMENTS: usize = 64 * 1024 * 1024;
const MAX_TOTAL_ELEMENTS: usize = 128 * 1024 * 1024;

#[derive(Debug, Deserialize)]
struct TensorHeader {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TensorSpec {
    dtype: i32,
    shape: Vec<i32>,
}

struct CheckedShard {
    path: PathBuf,
    file: File,
    identity: FileIdentity,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FileIdentity {
    len: u64,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
}

impl FileIdentity {
    fn from_metadata(metadata: &Metadata) -> Self {
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        Self {
            len: metadata.len(),
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
        }
    }
}

/// Stable shard handles and the exact metadata accepted before native loading.
pub(crate) struct CheckpointPreflight {
    shards: Vec<CheckedShard>,
    tensors: BTreeMap<String, TensorSpec>,
}

impl CheckpointPreflight {
    /// Recheck path identities and require the native loader to expose exactly
    /// the tensors inspected from the held shard handles.
    pub(crate) fn validate_loaded(&self, weights: &WeightMap) -> Result<(), String> {
        validate_weight_map(weights)?;
        if weights.len() != self.tensors.len() {
            return Err(format!(
                "RF-DETR native loader returned {} tensors; preflight inspected {}",
                weights.len(),
                self.tensors.len()
            ));
        }
        for (name, expected) in &self.tensors {
            let value = weights
                .get(name)
                .ok_or_else(|| format!("RF-DETR native loader omitted tensor {name:?}"))?;
            let actual = TensorSpec {
                dtype: mlxcel_core::array_dtype(value),
                shape: mlxcel_core::array_shape(value),
            };
            if actual != *expected {
                return Err(format!(
                    "RF-DETR tensor {name:?} changed between preflight and load: expected {expected:?}, got {actual:?}"
                ));
            }
        }
        for shard in &self.shards {
            let handle =
                FileIdentity::from_metadata(&shard.file.metadata().map_err(|error| {
                    format!("failed to restat {}: {error}", shard.path.display())
                })?);
            let path =
                FileIdentity::from_metadata(&std::fs::metadata(&shard.path).map_err(|error| {
                    format!("failed to restat {}: {error}", shard.path.display())
                })?);
            if handle != shard.identity || path != shard.identity {
                return Err(format!(
                    "RF-DETR checkpoint shard changed while loading: {}",
                    shard.path.display()
                ));
            }
        }
        Ok(())
    }
}

/// Inspect a checkpoint directory that the invoking user keeps stable for the
/// duration of model loading. Held handles plus the post-load identity check
/// detect persistent replacements; they do not authenticate a hostile writer
/// that swaps a path and restores it entirely during the native loader call.
pub(crate) fn inspect(directory: &Path) -> Result<CheckpointPreflight, String> {
    let paths = select_shards(directory)?;
    if paths.is_empty() {
        return Err(format!(
            "No safetensors files found in directory: {}",
            directory.display()
        ));
    }
    let mut tensors = BTreeMap::new();
    let mut total_elements = 0usize;
    let mut shards = Vec::with_capacity(paths.len());
    for path in paths {
        let mut file = File::open(&path)
            .map_err(|error| format!("failed to open {}: {error}", path.display()))?;
        let identity = FileIdentity::from_metadata(
            &file
                .metadata()
                .map_err(|error| format!("failed to stat {}: {error}", path.display()))?,
        );
        inspect_header(
            &path,
            &mut file,
            identity.len,
            &mut tensors,
            &mut total_elements,
        )?;
        shards.push(CheckedShard {
            path,
            file,
            identity,
        });
    }
    validate_specs(
        tensors
            .iter()
            .map(|(name, spec)| (name.as_str(), spec.dtype, spec.shape.as_slice())),
    )?;
    Ok(CheckpointPreflight { shards, tensors })
}

pub(crate) fn validate_weight_map(weights: &WeightMap) -> Result<(), String> {
    validate_specs(weights.iter().map(|(name, value)| {
        (
            name.as_str(),
            mlxcel_core::array_dtype(value),
            mlxcel_core::array_shape(value),
        )
    }))
}

fn validate_specs<I, S, H>(specs: I) -> Result<(), String>
where
    I: IntoIterator<Item = (S, i32, H)>,
    S: AsRef<str>,
    H: AsRef<[i32]>,
{
    let specs: Vec<_> = specs.into_iter().collect();
    if specs.is_empty() || specs.len() > MAX_TENSORS {
        return Err(format!(
            "RF-DETR checkpoint tensor count must be in 1..={MAX_TENSORS}; got {}",
            specs.len()
        ));
    }
    let mut total_elements = 0usize;
    for (name, dtype, shape) in specs {
        validate_spec(name.as_ref(), dtype, shape.as_ref(), &mut total_elements)?;
    }
    Ok(())
}

fn validate_spec(
    name: &str,
    dtype: i32,
    shape: &[i32],
    total_elements: &mut usize,
) -> Result<(), String> {
    if name.is_empty() || name.len() > MAX_KEY_BYTES {
        return Err(format!(
            "RF-DETR tensor key length must be in 1..={MAX_KEY_BYTES}; got {}",
            name.len()
        ));
    }
    if dtype != mlxcel_core::dtype::FLOAT32 {
        return Err(format!("RF-DETR tensor {name:?} must have F32 dtype"));
    }
    if shape.is_empty() || shape.len() > MAX_RANK {
        return Err(format!(
            "RF-DETR tensor {name:?} rank must be in 1..={MAX_RANK}; got {}",
            shape.len()
        ));
    }
    let mut elements = 1usize;
    for &dimension in shape {
        let dimension = usize::try_from(dimension)
            .map_err(|_| format!("RF-DETR tensor {name:?} has invalid shape {shape:?}"))?;
        if dimension == 0 || dimension > MAX_DIMENSION {
            return Err(format!(
                "RF-DETR tensor {name:?} dimensions must be in 1..={MAX_DIMENSION}; got {shape:?}"
            ));
        }
        elements = elements
            .checked_mul(dimension)
            .ok_or_else(|| format!("RF-DETR tensor {name:?} element count overflows"))?;
    }
    if elements > MAX_TENSOR_ELEMENTS {
        return Err(format!(
            "RF-DETR tensor {name:?} exceeds {MAX_TENSOR_ELEMENTS} elements"
        ));
    }
    *total_elements = total_elements
        .checked_add(elements)
        .ok_or("RF-DETR aggregate checkpoint element count overflows")?;
    if *total_elements > MAX_TOTAL_ELEMENTS {
        return Err(format!(
            "RF-DETR checkpoint exceeds {MAX_TOTAL_ELEMENTS} aggregate elements"
        ));
    }
    Ok(())
}

fn select_shards(directory: &Path) -> Result<Vec<PathBuf>, String> {
    let index_path = directory.join("model.safetensors.index.json");
    let candidates = if index_path.exists() {
        let bytes = read_bounded(&index_path, MAX_INDEX_BYTES, "index")?;
        let value: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|error| format!("failed to parse {}: {error}", index_path.display()))?;
        let map = value
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| format!("{}: missing weight_map object", index_path.display()))?;
        if map.is_empty() || map.len() > MAX_TENSORS {
            return Err(format!(
                "RF-DETR checkpoint index tensor count must be in 1..={MAX_TENSORS}; got {}",
                map.len()
            ));
        }
        let mut names = BTreeSet::new();
        for (tensor, shard) in map {
            if tensor.is_empty() || tensor.len() > MAX_KEY_BYTES {
                return Err(format!(
                    "RF-DETR checkpoint index tensor key is invalid: {tensor:?}"
                ));
            }
            let shard = shard.as_str().ok_or_else(|| {
                format!(
                    "{}: shard for {tensor:?} is not a string",
                    index_path.display()
                )
            })?;
            validate_shard_name(shard)?;
            if !names.contains(shard) {
                ensure_shard_capacity(names.len())?;
                names.insert(shard.to_string());
            }
        }
        let indexed: Vec<_> = names.iter().map(|name| directory.join(name)).collect();
        if indexed
            .iter()
            .all(|path| std::fs::symlink_metadata(path).is_ok())
        {
            indexed
        } else {
            let globbed = glob_safetensors(directory)?;
            if globbed.is_empty() {
                return Err(
                    "RF-DETR index references missing shards and no fallback shards exist".into(),
                );
            }
            globbed
        }
    } else {
        glob_safetensors(directory)?
    };
    filter_broken_symlinks(candidates)
}

fn validate_shard_name(name: &str) -> Result<(), String> {
    let path = Path::new(name);
    if name.as_bytes().contains(&0)
        || path.is_absolute()
        || path.components().count() != 1
        || !matches!(path.components().next(), Some(Component::Normal(_)))
    {
        return Err(format!(
            "invalid RF-DETR checkpoint shard filename {name:?}"
        ));
    }
    Ok(())
}

fn ensure_shard_capacity(current: usize) -> Result<(), String> {
    if current >= MAX_SHARDS {
        return Err(format!(
            "RF-DETR checkpoint exceeds {MAX_SHARDS} safetensors shards"
        ));
    }
    Ok(())
}

fn glob_safetensors(directory: &Path) -> Result<Vec<PathBuf>, String> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(directory)
        .map_err(|error| format!("failed to read {}: {error}", directory.display()))?
    {
        let path = entry
            .map_err(|error| format!("failed to read {}: {error}", directory.display()))?
            .path();
        if path
            .extension()
            .is_some_and(|extension| extension == "safetensors")
        {
            ensure_shard_capacity(paths.len())?;
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn filter_broken_symlinks(paths: Vec<PathBuf>) -> Result<Vec<PathBuf>, String> {
    let mut valid = Vec::new();
    for path in paths {
        let metadata = std::fs::symlink_metadata(&path)
            .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?;
        if metadata.file_type().is_symlink() && !path.exists() {
            continue;
        }
        valid.push(path);
    }
    Ok(valid)
}

fn read_bounded(path: &Path, limit: u64, kind: &str) -> Result<Vec<u8>, String> {
    let file =
        File::open(path).map_err(|error| format!("failed to open {}: {error}", path.display()))?;
    let length = file
        .metadata()
        .map_err(|error| format!("failed to stat {}: {error}", path.display()))?
        .len();
    if length > limit {
        return Err(format!(
            "RF-DETR safetensors {kind} {} exceeds {limit} bytes",
            path.display()
        ));
    }
    let mut bytes = Vec::with_capacity(usize::try_from(length).unwrap_or(0));
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > limit {
        return Err(format!(
            "RF-DETR safetensors {kind} {} exceeds {limit} bytes",
            path.display()
        ));
    }
    Ok(bytes)
}

fn inspect_header(
    path: &Path,
    file: &mut File,
    file_len: u64,
    tensors: &mut BTreeMap<String, TensorSpec>,
    total_elements: &mut usize,
) -> Result<(), String> {
    let mut length = [0; 8];
    file.read_exact(&mut length)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let header_len = u64::from_le_bytes(length);
    let data_start = header_len
        .checked_add(8)
        .filter(|&value| header_len > 0 && header_len <= MAX_HEADER_BYTES && value <= file_len)
        .ok_or_else(|| {
            format!(
                "invalid RF-DETR safetensors header length in {}",
                path.display()
            )
        })?;
    let mut bytes = vec![0; usize::try_from(header_len).map_err(|_| "header length overflow")?];
    file.read_exact(&mut bytes)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
    let data_len = file_len - data_start;
    let mut ranges = Vec::new();
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let tensor: TensorHeader = serde_json::from_value(value)
            .map_err(|error| format!("invalid RF-DETR tensor header {name:?}: {error}"))?;
        if tensor.dtype != "F32" {
            return Err(format!("RF-DETR tensor {name:?} must have F32 dtype"));
        }
        let shape: Vec<i32> = tensor
            .shape
            .iter()
            .map(|&value| i32::try_from(value).map_err(|_| format!("invalid shape for {name:?}")))
            .collect::<Result<_, _>>()?;
        if tensors.contains_key(&name) {
            return Err(format!("duplicate RF-DETR tensor {name:?} across shards"));
        }
        if tensors.len() == MAX_TENSORS {
            return Err(format!(
                "RF-DETR checkpoint tensor count exceeds {MAX_TENSORS}"
            ));
        }
        let before = *total_elements;
        validate_spec(&name, mlxcel_core::dtype::FLOAT32, &shape, total_elements)?;
        let tensor_elements = total_elements
            .checked_sub(before)
            .ok_or("RF-DETR aggregate tensor count underflow")?;
        let expected_bytes = u64::try_from(tensor_elements)
            .ok()
            .and_then(|value| value.checked_mul(4))
            .ok_or_else(|| format!("RF-DETR tensor {name:?} byte size overflows"))?;
        let [start, end] = tensor.data_offsets;
        if end.checked_sub(start) != Some(expected_bytes) || end > data_len {
            return Err(format!("invalid RF-DETR tensor offsets for {name:?}"));
        }
        tensors.insert(
            name.clone(),
            TensorSpec {
                dtype: mlxcel_core::dtype::FLOAT32,
                shape,
            },
        );
        ranges.push((start, end, name));
    }
    ranges.sort_by_key(|range| range.0);
    let mut previous_end = 0;
    for (start, end, name) in ranges {
        if start != previous_end {
            return Err(format!(
                "non-contiguous RF-DETR tensor data before {name:?}"
            ));
        }
        previous_end = end;
    }
    if previous_end != data_len {
        return Err(format!(
            "RF-DETR safetensors data length mismatch in {}",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod tests;
