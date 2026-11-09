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

/// Contiguous expert records, next to the checkpoint.
pub const CONTIGUOUS_FILE: &str = "expert-stream.bin";
/// Its manifest; the records are ignored unless it matches the checkpoint.
pub const CONTIGUOUS_MANIFEST: &str = "expert-stream.json";
const CONTIGUOUS_VERSION: u64 = 1;

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

fn dtype_of(name: &str) -> Option<i32> {
    use mlxcel_core::dtype::{BFLOAT16, FLOAT16, FLOAT32, UINT8, UINT16, UINT32};
    Some(match name {
        "U8" => UINT8,
        "U16" => UINT16,
        "U32" => UINT32,
        "F16" => FLOAT16,
        "BF16" => BFLOAT16,
        "F32" => FLOAT32,
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

fn read_header(path: &Path) -> Result<(u64, Value), String> {
    let context = |e: std::io::Error| format!("{}: {e}", path.display());
    let mut file = File::open(path).map_err(context)?;
    let mut length = [0u8; 8];
    file.read_exact(&mut length).map_err(context)?;
    let length = u64::from_le_bytes(length);
    let mut header = vec![0u8; length as usize];
    file.read_exact(&mut header).map_err(context)?;
    let value = serde_json::from_slice(&header).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((8 + length, value))
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
        let names = shard_names(model_dir)?;
        let mut planes: Vec<Vec<Option<Plane>>> = vec![vec![None; PLANE_SUFFIXES.len()]; layers];
        let mut files = Vec::with_capacity(names.len());
        for (file_index, name) in names.iter().enumerate() {
            let path = model_dir.join(name);
            let (data_start, header) = read_header(&path)?;
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
                let plane = parse_plane(key, info, file_index, data_start, num_experts)?;
                planes[layer][position] = Some(plane);
            }
            files.push(open_uncached(&path)?);
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

    /// Identity of the expert planes a contiguous file was written from.
    fn fingerprint(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update((self.num_experts as u64).to_le_bytes());
        for plane in self.layers.iter().flatten() {
            hasher.update(plane.row_bytes.to_le_bytes());
            hasher.update(plane.offset.to_le_bytes());
            hasher.update((plane.file as u64).to_le_bytes());
        }
        hex(&hasher.finalize())
    }

    fn open_contiguous(&self, model_dir: &Path) -> Option<Contiguous> {
        let manifest: Value =
            serde_json::from_slice(&std::fs::read(model_dir.join(CONTIGUOUS_MANIFEST)).ok()?)
                .ok()?;
        if manifest["version"].as_u64() != Some(CONTIGUOUS_VERSION)
            || manifest["fingerprint"].as_str() != Some(&self.fingerprint())
        {
            eprintln!("Ignoring {CONTIGUOUS_FILE}: it does not match this checkpoint.");
            return None;
        }
        let layers = manifest["layers"]
            .as_array()?
            .iter()
            .map(|entry| Some((entry["offset"].as_u64()?, entry["record_bytes"].as_u64()?)))
            .collect::<Option<Vec<_>>>()?;
        if layers.len() != self.layers.len() {
            return None;
        }
        let file = open_uncached(&model_dir.join(CONTIGUOUS_FILE)).ok()?;
        Some(Contiguous { file, layers })
    }
}

fn parse_plane(
    key: &str,
    info: &Value,
    file: usize,
    data_start: u64,
    num_experts: usize,
) -> Result<Plane, String> {
    let dtype = info["dtype"]
        .as_str()
        .and_then(dtype_of)
        .ok_or_else(|| format!("{key}: unsupported dtype {}", info["dtype"]))?;
    let shape: Vec<i64> = info["shape"]
        .as_array()
        .ok_or_else(|| format!("{key}: missing shape"))?
        .iter()
        .filter_map(Value::as_i64)
        .collect();
    let (start, end) = match (
        info["data_offsets"][0].as_u64(),
        info["data_offsets"][1].as_u64(),
    ) {
        (Some(start), Some(end)) if end >= start => (start, end),
        _ => return Err(format!("{key}: bad data_offsets")),
    };
    if shape.len() != 3 || shape[0] != num_experts as i64 {
        return Err(format!(
            "{key}: expected [{num_experts}, out, in], got {shape:?}"
        ));
    }
    let row_bytes = (end - start) / num_experts as u64;
    if row_bytes * num_experts as u64 != end - start {
        return Err(format!("{key}: size is not a multiple of {num_experts}"));
    }
    Ok(Plane {
        file,
        offset: data_start + start,
        row_bytes,
        shape: vec![shape[1] as i32, shape[2] as i32],
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
    match std::fs::remove_file(&manifest_path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(format!("{}: {e}", manifest_path.display())),
    }
    let context = |e: std::io::Error| format!("{}: {e}", path.display());
    let mut out = BufWriter::with_capacity(1 << 24, File::create(&path).map_err(context)?);

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

    let mut check = File::open(&path).map_err(context)?;
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
    std::fs::write(&manifest_path, text)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    Ok(ContiguousReport {
        path,
        bytes: offset,
        layers,
    })
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
