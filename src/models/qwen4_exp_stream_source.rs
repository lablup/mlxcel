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

//! Where each routed expert's bytes live on disk (issue #2882): the plane
//! table read from the safetensors headers, and the optional contiguous copy
//! that stores an expert's nine planes as one record.

use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::models::qwen4_exp::ple_manifest::trust::{read_header, resolve_contained};

/// Contiguous expert records, next to the checkpoint.
pub const CONTIGUOUS_FILE: &str = "expert-stream.bin";
/// Its manifest; the records are ignored unless it matches the checkpoint.
pub const CONTIGUOUS_MANIFEST: &str = "expert-stream.json";
const CONTIGUOUS_VERSION: u64 = 1;
/// Most layers a checkpoint may declare; the plane table is allocated from it
/// before anything else about the checkpoint is known.
const MAX_LAYERS: usize = 4096;

/// The nine stacked planes of a layer's routed experts, in record order.
pub(crate) const PLANE_SUFFIXES: [&str; 9] = [
    "gate_proj.weight",
    "gate_proj.scales",
    "gate_proj.biases",
    "up_proj.weight",
    "up_proj.scales",
    "up_proj.biases",
    "down_proj.weight",
    "down_proj.scales",
    "down_proj.biases",
];

/// One stacked `[num_experts, ...]` tensor on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Plane {
    pub(crate) file: usize,
    /// Absolute file offset of expert 0.
    pub(crate) offset: u64,
    pub(crate) row_bytes: u64,
    /// `[out, in]` shape of one expert.
    pub(crate) shape: Vec<i32>,
    pub(crate) dtype: i32,
}

/// Every routed-expert plane of a checkpoint, and an open file for each shard.
#[derive(Debug)]
pub struct ExpertSource {
    pub(crate) files: Vec<File>,
    /// `[layers][PLANE_SUFFIXES.len()]`.
    pub(crate) layers: Vec<Vec<Plane>>,
    pub(crate) num_experts: usize,
    contiguous: Option<Contiguous>,
}

/// Open contiguous records: `offset(layer) + expert * record_bytes(layer)`.
#[derive(Debug)]
struct Contiguous {
    file: File,
    layers: Vec<(u64, u64)>,
}

/// MLX dtype and element size in bytes.
fn dtype_of(name: &str) -> Option<(i32, u64)> {
    use mlxcel_core::dtype::{BFLOAT16, FLOAT16, FLOAT32, UINT8, UINT16, UINT32};
    Some(match name {
        "U8" => (UINT8, 1),
        "U16" => (UINT16, 2),
        "U32" => (UINT32, 4),
        "F16" => (FLOAT16, 2),
        "BF16" => (BFLOAT16, 2),
        "F32" => (FLOAT32, 4),
        _ => return None,
    })
}

/// Open a shard for unbuffered reads: an expert read must not grow the page
/// cache of the machine it is trying to save memory on.
fn open_uncached(path: &Path) -> Result<File, String> {
    let file = File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: `file` is an open descriptor; F_NOCACHE takes an int flag.
        unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
    }
    Ok(file)
}

fn shard_names(model_dir: &Path) -> Result<Vec<String>, String> {
    let index = model_dir.join("model.safetensors.index.json");
    if !index.exists() {
        return Ok(vec!["model.safetensors".to_string()]);
    }
    let text = std::fs::read_to_string(&index).map_err(|e| format!("{}: {e}", index.display()))?;
    let value: Value =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", index.display()))?;
    let map = value["weight_map"]
        .as_object()
        .ok_or_else(|| format!("{} has no weight_map", index.display()))?;
    let mut names: Vec<String> = map
        .values()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    names.sort();
    names.dedup();
    Ok(names)
}

/// Prove the sums in `expert_bytes` and `total_expert_bytes` cannot overflow.
fn check_total_bytes(layers: &[Vec<Plane>], num_experts: usize) -> Result<(), String> {
    layers
        .iter()
        .try_fold(0u64, |total, planes| {
            let record = planes
                .iter()
                .try_fold(0u64, |sum, plane| sum.checked_add(plane.row_bytes))?;
            total.checked_add(record.checked_mul(num_experts as u64)?)
        })
        .map(|_| ())
        .ok_or_else(|| "expert streaming: total expert bytes overflow".to_string())
}

/// `model.layers.{layer}.mlp.switch_mlp.{suffix}`, with or without a wrapper
/// prefix, as `(layer, suffix)`.
fn parse_expert_key(key: &str) -> Option<(usize, &str)> {
    let rest = key.split_once("layers.")?.1;
    let (layer, rest) = rest.split_once('.')?;
    let suffix = rest.strip_prefix("mlp.switch_mlp.")?;
    Some((layer.parse().ok()?, suffix))
}

impl ExpertSource {
    /// Read the plane table of `model_dir`. Fails when any layer's experts are
    /// not stored as stacked `switch_mlp` planes, the only layout streamed.
    pub fn open(model_dir: &Path, layers: usize, num_experts: usize) -> Result<Self, String> {
        if layers == 0 || layers > MAX_LAYERS {
            return Err(format!(
                "expert streaming: layer count {layers} is outside 1..={MAX_LAYERS}"
            ));
        }
        if num_experts == 0 {
            return Err("expert streaming: num_experts is 0".to_string());
        }
        let names = shard_names(model_dir)?;
        let mut planes: Vec<Vec<Option<Plane>>> = vec![vec![None; PLANE_SUFFIXES.len()]; layers];
        let mut files = Vec::with_capacity(names.len());
        for (file_index, name) in names.iter().enumerate() {
            let path = resolve_contained(model_dir, Path::new(name))
                .map_err(|e| format!("{name}: {e}"))?;
            let file = open_uncached(&path)?;
            let file_len = file
                .metadata()
                .map_err(|e| format!("{}: {e}", path.display()))?
                .len();
            let (data_start, header) = read_header(&path).map_err(|e| e.to_string())?;
            let object = header
                .as_object()
                .ok_or_else(|| format!("{}: header is not an object", path.display()))?;
            for (key, info) in object {
                let Some((layer, suffix)) = parse_expert_key(key) else {
                    continue;
                };
                let Some(position) = PLANE_SUFFIXES.iter().position(|s| *s == suffix) else {
                    continue;
                };
                if layer >= layers {
                    return Err(format!("{key}: layer index out of range"));
                }
                // A wrapper-prefixed alias must not replace the plane the
                // loader's shape checks validated.
                if planes[layer][position].is_some() {
                    return Err(format!(
                        "{key}: duplicate tensor for layer {layer} {suffix}"
                    ));
                }
                let plane = parse_plane(key, info, file_index, data_start, file_len, num_experts)?;
                planes[layer][position] = Some(plane);
            }
            files.push(file);
        }
        let layers = planes
            .into_iter()
            .enumerate()
            .map(|(layer, row)| {
                row.into_iter()
                    .enumerate()
                    .map(|(position, plane)| {
                        plane.ok_or_else(|| {
                            format!(
                                "expert streaming needs stacked switch_mlp planes, but layer {layer} has no {}",
                                PLANE_SUFFIXES[position]
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        check_total_bytes(&layers, num_experts)?;
        let mut source = Self {
            files,
            layers,
            num_experts,
            contiguous: None,
        };
        source.contiguous = source.open_contiguous(model_dir);
        Ok(source)
    }

    pub(crate) fn expert_bytes(&self, layer: usize) -> u64 {
        self.layers[layer].iter().map(|p| p.row_bytes).sum()
    }

    /// Every routed expert's bytes in the checkpoint.
    pub fn total_expert_bytes(&self) -> u64 {
        (0..self.layers.len())
            .map(|layer| self.expert_bytes(layer) * self.num_experts as u64)
            .sum()
    }

    pub fn uses_contiguous_file(&self) -> bool {
        self.contiguous.is_some()
    }

    /// Fill `dest` (one `&mut [u8]` per plane, each `row_bytes` long) with one
    /// expert's bytes. Returns the number of bytes read.
    pub(crate) fn read_expert(
        &self,
        layer: usize,
        expert: usize,
        dest: &mut [&mut [u8]],
    ) -> Result<u64, String> {
        if let Some(contiguous) = &self.contiguous {
            let (start, record) = contiguous.layers[layer];
            let mut buffer = vec![0u8; record as usize];
            read_exact_at(
                &contiguous.file,
                &mut buffer,
                start + expert as u64 * record,
            )?;
            let mut from = 0;
            for plane in dest.iter_mut() {
                let to = from + plane.len();
                plane.copy_from_slice(&buffer[from..to]);
                from = to;
            }
            return Ok(record);
        }
        let mut total = 0;
        for (plane, out) in self.layers[layer].iter().zip(dest.iter_mut()) {
            let offset = plane.offset + expert as u64 * plane.row_bytes;
            read_exact_at(&self.files[plane.file], out, offset)?;
            total += plane.row_bytes;
        }
        Ok(total)
    }

    /// Identity of the expert planes a contiguous file was written from: their
    /// layout, and each shard's size and modification time, so a shard replaced
    /// by one of the same layout (a re-quantization) invalidates the copy.
    fn fingerprint(&self) -> String {
        let mut hasher = Sha256::new();
        for file in &self.files {
            let meta = file.metadata().ok();
            hasher.update(meta.as_ref().map_or(0, |m| m.len()).to_le_bytes());
            let modified = meta
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos());
            hasher.update(modified.to_le_bytes());
        }
        hasher.update((self.num_experts as u64).to_le_bytes());
        for plane in self.layers.iter().flatten() {
            hasher.update(plane.row_bytes.to_le_bytes());
            hasher.update(plane.offset.to_le_bytes());
            hasher.update((plane.file as u64).to_le_bytes());
        }
        hex(&hasher.finalize())
    }

    fn open_contiguous(&self, model_dir: &Path) -> Option<Contiguous> {
        let ignore = |why: String| eprintln!("Ignoring {CONTIGUOUS_FILE}: {why}.");
        let manifest_path = match resolve_contained(model_dir, Path::new(CONTIGUOUS_MANIFEST)) {
            Ok(path) => path,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
            Err(e) => {
                ignore(e.to_string());
                return None;
            }
        };
        let manifest: Value = serde_json::from_slice(&std::fs::read(manifest_path).ok()?).ok()?;
        if manifest["version"].as_u64() != Some(CONTIGUOUS_VERSION)
            || manifest["num_experts"].as_u64() != Some(self.num_experts as u64)
            || manifest["fingerprint"].as_str() != Some(&self.fingerprint())
        {
            ignore("it does not match this checkpoint".to_string());
            return None;
        }
        let layers = manifest["layers"]
            .as_array()?
            .iter()
            .map(|entry| Some((entry["offset"].as_u64()?, entry["record_bytes"].as_u64()?)))
            .collect::<Option<Vec<_>>>()?;
        let file = match resolve_contained(model_dir, Path::new(CONTIGUOUS_FILE))
            .map_err(|e| e.to_string())
            .and_then(|path| open_uncached(&path))
        {
            Ok(file) => file,
            Err(why) => {
                ignore(why);
                return None;
            }
        };
        if !self.layout_matches(&layers, file.metadata().ok()?.len()) {
            ignore("its manifest does not describe this checkpoint's records".to_string());
            return None;
        }
        Some(Contiguous { file, layers })
    }

    /// Whether `layers` are this checkpoint's records, packed end to end and
    /// filling a file of `file_len` bytes.
    fn layout_matches(&self, layers: &[(u64, u64)], file_len: u64) -> bool {
        if layers.len() != self.layers.len() {
            return false;
        }
        let mut cursor = 0u64;
        for (layer, &(offset, record)) in layers.iter().enumerate() {
            if offset != cursor || record != self.expert_bytes(layer) {
                return false;
            }
            let span = record.checked_mul(self.num_experts as u64);
            match span.and_then(|span| cursor.checked_add(span)) {
                Some(next) => cursor = next,
                None => return false,
            }
        }
        cursor == file_len
    }
}

fn parse_plane(
    key: &str,
    info: &Value,
    file: usize,
    data_start: u64,
    file_len: u64,
    num_experts: usize,
) -> Result<Plane, String> {
    let (dtype, element_size) = info["dtype"]
        .as_str()
        .and_then(dtype_of)
        .ok_or_else(|| format!("{key}: unsupported dtype {}", info["dtype"]))?;
    let shape = info["shape"]
        .as_array()
        .ok_or_else(|| format!("{key}: missing shape"))?
        .iter()
        .map(|dim| {
            dim.as_i64()
                .ok_or_else(|| format!("{key}: non-integer shape entry {dim}"))
        })
        .collect::<Result<Vec<i64>, _>>()?;
    let (start, end) = match (
        info["data_offsets"][0].as_u64(),
        info["data_offsets"][1].as_u64(),
    ) {
        (Some(start), Some(end)) if end >= start => (start, end),
        _ => return Err(format!("{key}: bad data_offsets")),
    };
    if shape.len() != 3 || usize::try_from(shape[0]).ok() != Some(num_experts) {
        return Err(format!(
            "{key}: expected [{num_experts}, out, in], got {shape:?}"
        ));
    }
    let dimension = |value: i64| {
        i32::try_from(value)
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| format!("{key}: dimension {value} is out of range"))
    };
    let (out, input) = (dimension(shape[1])?, dimension(shape[2])?);
    let row_bytes = (end - start) / num_experts as u64;
    if row_bytes * num_experts as u64 != end - start {
        return Err(format!("{key}: size is not a multiple of {num_experts}"));
    }
    // MLX wraps the arena with this shape and does no length check of its own.
    let expected = (out as u64)
        .checked_mul(input as u64)
        .and_then(|cells| cells.checked_mul(element_size));
    if row_bytes == 0 || expected != Some(row_bytes) {
        return Err(format!(
            "{key}: {row_bytes} bytes per expert does not match shape {shape:?} and dtype {}",
            info["dtype"]
        ));
    }
    if data_start
        .checked_add(end)
        .is_none_or(|last| last > file_len)
    {
        return Err(format!("{key}: data_offsets run past the end of the file"));
    }
    let offset = data_start
        .checked_add(start)
        .ok_or_else(|| format!("{key}: bad data_offsets"))?;
    Ok(Plane {
        file,
        offset,
        row_bytes,
        shape: vec![out, input],
        dtype,
    })
}

#[cfg(unix)]
fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> Result<(), String> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buffer, offset)
        .map_err(|e| format!("expert read failed at byte {offset}: {e}"))
}

#[cfg(windows)]
fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> Result<(), String> {
    use std::os::windows::fs::FileExt;
    let mut filled = 0;
    while filled < buffer.len() {
        let read = file
            .seek_read(&mut buffer[filled..], offset + filled as u64)
            .map_err(|e| format!("expert read failed at byte {offset}: {e}"))?;
        if read == 0 {
            return Err(format!("expert read came up short at byte {offset}"));
        }
        filled += read;
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// What [`write_contiguous_file`] produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContiguousReport {
    pub path: PathBuf,
    pub bytes: u64,
    pub layers: usize,
}

/// Write each expert's nine planes as one record to `CONTIGUOUS_FILE` in
/// `model_dir`, then read every record back and compare it with the source by
/// hash. The manifest is written last, so an interrupted run leaves a file the
/// loader ignores.
pub fn write_contiguous_file(
    model_dir: &Path,
    layers: usize,
    num_experts: usize,
) -> Result<ContiguousReport, String> {
    let source = ExpertSource::open(model_dir, layers, num_experts)?;
    let path = model_dir.join(CONTIGUOUS_FILE);
    let manifest_path = model_dir.join(CONTIGUOUS_MANIFEST);
    // Unlinked rather than truncated: a planted symlink is not written through,
    // and a running process keeps reading the inode it has open.
    remove_if_present(&manifest_path)?;
    remove_if_present(&path)?;
    let written = write_records(&source, &path, &manifest_path);
    if written.is_err() {
        // The partial data file can be tens of GB on a disk that may be full.
        for leftover in [&manifest_path, &path] {
            let _ = std::fs::remove_file(leftover);
        }
    }
    Ok(ContiguousReport {
        bytes: written?,
        path,
        layers,
    })
}

fn remove_if_present(path: &Path) -> Result<(), String> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

/// Never follows a symlink: the path must not exist.
fn create_new(path: &Path) -> std::io::Result<File> {
    File::options().write(true).create_new(true).open(path)
}

/// Write the records and the manifest; returns the bytes of records written.
fn write_records(source: &ExpertSource, path: &Path, manifest_path: &Path) -> Result<u64, String> {
    let (layers, num_experts) = (source.layers.len(), source.num_experts);
    let context = |e: std::io::Error| format!("{}: {e}", path.display());
    let mut out = BufWriter::with_capacity(1 << 24, create_new(path).map_err(context)?);

    let mut layer_entries = Vec::with_capacity(layers);
    let mut digests = Vec::with_capacity(layers);
    let mut offset = 0u64;
    for layer in 0..layers {
        let record = source.expert_bytes(layer);
        let mut hasher = Sha256::new();
        for expert in 0..num_experts {
            let mut buffer = vec![0u8; record as usize];
            source.read_expert_plain(layer, expert, &mut buffer)?;
            hasher.update(&buffer);
            out.write_all(&buffer).map_err(context)?;
        }
        digests.push(hasher.finalize());
        layer_entries.push(json!({ "offset": offset, "record_bytes": record }));
        offset += record * num_experts as u64;
    }
    out.flush().map_err(context)?;
    drop(out);

    let mut check = File::open(path).map_err(context)?;
    for (layer, digest) in digests.iter().enumerate() {
        let record = source.expert_bytes(layer);
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; record as usize];
        check
            .seek(SeekFrom::Start(
                layer_entries[layer]["offset"].as_u64().unwrap_or(0),
            ))
            .map_err(context)?;
        for _ in 0..num_experts {
            check.read_exact(&mut buffer).map_err(context)?;
            hasher.update(&buffer);
        }
        if hasher.finalize() != *digest {
            return Err(format!(
                "{}: layer {layer} does not match the checkpoint on read-back",
                path.display()
            ));
        }
    }

    let manifest = json!({
        "version": CONTIGUOUS_VERSION,
        "fingerprint": source.fingerprint(),
        "num_experts": num_experts,
        "layers": layer_entries,
    });
    let text = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
    create_new(manifest_path)
        .and_then(|mut file| file.write_all(&text))
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    Ok(offset)
}

impl ExpertSource {
    /// One expert's record from the safetensors planes, ignoring any
    /// contiguous file.
    pub(crate) fn read_expert_plain(
        &self,
        layer: usize,
        expert: usize,
        record: &mut [u8],
    ) -> Result<(), String> {
        let mut from = 0;
        for plane in &self.layers[layer] {
            let to = from + plane.row_bytes as usize;
            read_exact_at(
                &self.files[plane.file],
                &mut record[from..to],
                plane.offset + expert as u64 * plane.row_bytes,
            )?;
            from = to;
        }
        Ok(())
    }
}
