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

//! Checks on the files a checkpoint names. A checkpoint can come from anyone,
//! so a name in it must not reach a file outside the model, and a byte range
//! that may lie outside the model (`source_root`) must be a tensor that a
//! safetensors header declares.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};

use serde_json::Value;

/// The limit the `safetensors` crate applies to a header.
const MAX_HEADER_BYTES: u64 = 100_000_000;

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// Whether `target` is a blob of the Hugging Face cache repository whose
/// snapshot is `root`: snapshot files are symlinks to `../../blobs/<hash>`.
fn is_snapshot_blob(root: &Path, target: &Path) -> bool {
    root.parent()
        .and_then(Path::parent)
        .is_some_and(|repo| target.parent() == Some(repo.join("blobs").as_path()))
}

/// Resolve `relative` to a regular file inside `root`, following symlinks only
/// when they stay in the model. Refuses absolute paths and `..`.
pub(crate) fn resolve_contained(root: &Path, relative: &Path) -> io::Result<PathBuf> {
    let plain = relative
        .components()
        .all(|component| matches!(component, Component::Normal(_)));
    if !plain || relative.as_os_str().is_empty() {
        return Err(invalid(format!(
            "{} must be a relative path inside the model directory",
            relative.display()
        )));
    }
    let root = if root.as_os_str().is_empty() {
        Path::new(".")
    } else {
        root
    };
    let root = root.canonicalize()?;
    let target = root.join(relative).canonicalize()?;
    if !target.starts_with(&root) && !is_snapshot_blob(&root, &target) {
        return Err(invalid(format!(
            "{} resolves outside the model directory",
            relative.display()
        )));
    }
    if !fs::metadata(&target)?.is_file() {
        return Err(invalid(format!(
            "{} is not a regular file",
            relative.display()
        )));
    }
    Ok(target)
}

/// `(payload offset, header)` of a safetensors file. The declared header size
/// is checked against the file before anything is allocated for it.
pub(crate) fn read_header(path: &Path) -> io::Result<(u64, Value)> {
    read_header_inner(path)
        .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))
}

fn read_header_inner(path: &Path) -> io::Result<(u64, Value)> {
    // Checked before opening: opening a FIFO would block.
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() {
        return Err(invalid("not a regular file"));
    }
    let mut file = File::open(path)?;
    let mut size = [0u8; 8];
    file.read_exact(&mut size)?;
    let size = u64::from_le_bytes(size);
    if size > MAX_HEADER_BYTES.min(metadata.len().saturating_sub(8)) {
        return Err(invalid(format!(
            "safetensors header size {size} is invalid"
        )));
    }
    let mut header = vec![0u8; size as usize];
    file.read_exact(&mut header)?;
    let header = serde_json::from_slice(&header).map_err(|e| invalid(e.to_string()))?;
    Ok((8 + size, header))
}

/// A tensor a safetensors header declares; `end` is an absolute file offset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DeclaredTensor {
    pub(super) end: u64,
    pub(super) dtype: String,
    pub(super) shape: Vec<u64>,
}

/// The non-empty tensors of a safetensors file by absolute start offset, read
/// once per file so that matching a manifest stays linear in its size.
pub(super) fn declared_tensors(path: &Path) -> io::Result<HashMap<u64, DeclaredTensor>> {
    let (payload, header) = read_header(path)?;
    let entries = header
        .as_object()
        .ok_or_else(|| invalid(format!("{}: header is not an object", path.display())))?;
    let mut tensors = HashMap::with_capacity(entries.len());
    for info in entries.values() {
        let (Some(start), Some(end)) = (
            info["data_offsets"][0].as_u64(),
            info["data_offsets"][1].as_u64(),
        ) else {
            continue;
        };
        let (Some(start), Some(end)) = (payload.checked_add(start), payload.checked_add(end))
        else {
            continue;
        };
        let (Some(dtype), Some(shape)) = (info["dtype"].as_str(), info["shape"].as_array()) else {
            continue;
        };
        if end > start {
            let shape = shape.iter().filter_map(Value::as_u64).collect();
            let dtype = dtype.to_string();
            tensors.insert(start, DeclaredTensor { end, dtype, shape });
        }
    }
    Ok(tensors)
}
