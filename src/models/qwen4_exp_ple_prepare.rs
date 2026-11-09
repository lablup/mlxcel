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

//! Conversion to the external PLE layout, equivalent to mlx-vlm's
//! `prepare_external_ple_model` plus `materialize_interleaved_ple_store`.

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use memmap2::Mmap;
use mlxcel_core::weights::declared_optiq_vision_sidecar;
use serde_json::{Map, Value, json};

use super::ple_manifest::trust::{read_header, resolve_contained};
use super::ple_manifest::{MANIFEST_VERSION, PleQuantization};

const PLE_MARKER: &str = ".ple.ple_embedding.ngram_embedding.shards.";
pub const MANIFEST_NAME: &str = "ple-store.json";
pub const DEFAULT_ROW_FILE: &str = "ple-q4.rows";
const PROVENANCE_NAME: &str = "EXTERNAL_PLE.json";
/// Files `prepare_external_ple_model` never copies into the view: upstream's
/// list, plus the names the view writes itself, so that a source file cannot
/// replace them.
const EXCLUDED_FILES: [&str; 7] = [
    "model.safetensors.index.json",
    "MANIFEST.sha256",
    "MEMORY_CONTEXT_SWEEP.jsonl",
    MANIFEST_NAME,
    DEFAULT_ROW_FILE,
    PROVENANCE_NAME,
    "ple-q4.partial",
];

/// Row-table layout written by the helper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrepareLayout {
    /// One contiguous record per row in `ple-q4.rows`.
    #[default]
    Interleaved,
    /// Point the manifest at the source shards' byte ranges; writes no row file.
    SafetensorsRanges,
}

#[derive(Debug, Clone, Default)]
pub struct PrepareOptions {
    pub layout: PrepareLayout,
    pub cache_rows: u64,
}

#[derive(Debug, Clone)]
pub struct PrepareReport {
    pub target: PathBuf,
    pub table_bytes: u64,
    pub row_count: u64,
    pub linked_files: usize,
}

/// Python's `json` writes `Infinity` and `NaN`, which serde rejects; loading
/// treats them the same way the model loader does.
pub(super) fn read_json(path: &Path) -> Result<Value> {
    let text =
        fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    let text = text.replace("Infinity", "1e38").replace("NaN", "0.0");
    serde_json::from_str(&text).with_context(|| format!("invalid JSON in {}", path.display()))
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    fs::write(path, format!("{}\n", serde_json::to_string_pretty(value)?))
        .with_context(|| format!("cannot write {}", path.display()))
}

struct HeaderCache<'a> {
    dir: &'a Path,
    weight_map: &'a Map<String, Value>,
    headers: HashMap<String, (u64, Value)>,
}

impl HeaderCache<'_> {
    fn descriptor(&mut self, key: &str) -> Result<Value> {
        let file = self
            .weight_map
            .get(key)
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("the safetensors index has no entry for {key}"))?
            .to_string();
        if !self.headers.contains_key(&file) {
            let path = resolve_contained(self.dir, Path::new(&file))
                .with_context(|| format!("the safetensors index names {file:?} for {key}"))?;
            let header = read_header(&path)?;
            self.headers.insert(file.clone(), header);
        }
        let (payload, header) = &self.headers[&file];
        let info = &header[key];
        let start = info["data_offsets"][0]
            .as_u64()
            .ok_or_else(|| anyhow!("{key}: missing data_offsets in {file}"))?;
        let offset = payload
            .checked_add(start)
            .ok_or_else(|| anyhow!("{key}: data_offsets overflow in {file}"))?;
        Ok(json!({"file": file, "offset": offset, "dtype": info["dtype"], "shape": info["shape"]}))
    }
}

/// The `shards` manifest section and table geometry, in row order.
struct RangeManifest {
    shards: Vec<Value>,
    row_width: usize,
    row_count: u64,
    quantization: PleQuantization,
}

fn build_ranges(
    source: &Path,
    config: &Value,
    weight_map: &Map<String, Value>,
) -> Result<RangeManifest> {
    let mut prefixes: Vec<(u64, String)> = Vec::new();
    let mut stems: Vec<&str> = Vec::new();
    for key in weight_map.keys().filter(|key| key.contains(PLE_MARKER)) {
        let prefix = key.rsplit_once('.').map(|(head, _)| head).unwrap_or(key);
        if prefixes.iter().any(|(_, known)| known == prefix) {
            continue;
        }
        let (stem, shard) = prefix
            .rsplit_once('.')
            .ok_or_else(|| anyhow!("malformed PLE key {key}"))?;
        let number = shard
            .parse()
            .with_context(|| format!("malformed PLE shard number in {key}"))?;
        prefixes.push((number, prefix.to_string()));
        stems.push(stem);
    }
    if prefixes.is_empty() {
        bail!("checkpoint contains no Qwen4-Exp PLE shards");
    }
    stems.sort_unstable();
    stems.dedup();
    if stems.len() > 1 {
        bail!(
            "one manifest describes one n-gram table, but the checkpoint has {} PLE layers",
            stems.len()
        );
    }
    prefixes.sort();

    let quantization_block = config
        .get("quantization_config")
        .filter(|value| !value.is_null())
        .or_else(|| config.get("quantization"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    let mut headers = HeaderCache {
        dir: source,
        weight_map,
        headers: HashMap::new(),
    };
    let mut shards = Vec::with_capacity(prefixes.len());
    let mut row_start = 0u64;
    let mut quantization = None;
    let mut row_width = 0usize;
    for (_, prefix) in &prefixes {
        let params = quantization_block
            .get(prefix.as_str())
            .filter(|value| value.is_object())
            .unwrap_or(&quantization_block);
        let current = PleQuantization::from_fields(
            params["bits"].as_u64(),
            params["group_size"].as_u64(),
            params["mode"].as_str(),
        )
        .ok_or_else(|| anyhow!("unsupported PLE quantization on {prefix}: {params}"))?;
        if quantization.is_some_and(|known| known != current) {
            bail!("all PLE shards must use the same quantization");
        }
        quantization = Some(current);
        let mut entry = Map::new();
        for &(name, dtype, _) in current.tensors() {
            let tensor = headers.descriptor(&format!("{prefix}.{name}"))?;
            if tensor["dtype"] != dtype {
                bail!(
                    "PLE tensor {prefix}.{name} has dtype {}, expected {dtype}",
                    tensor["dtype"]
                );
            }
            entry.insert(name.to_string(), tensor);
        }
        let shape: Vec<u64> = entry["weight"]["shape"]
            .as_array()
            .map_or_else(Vec::new, |dims| {
                dims.iter().filter_map(Value::as_u64).collect()
            });
        let [row_count, words] = shape[..] else {
            bail!("PLE tensor {prefix}.weight has an invalid Q4 shape layout: {shape:?}");
        };
        let width = words
            .checked_mul(8)
            .and_then(|width| usize::try_from(width).ok())
            .filter(|&width| width > 0 && words % 4 == 0)
            .ok_or_else(|| {
                anyhow!("PLE tensor {prefix}.weight has an invalid Q4 shape layout: {shape:?}")
            })?;
        if row_width != 0 && width != row_width {
            bail!(
                "PLE shards disagree on row_width: {prefix} has {width}, earlier shards have {row_width}"
            );
        }
        row_width = width;
        let groups = (row_width / current.group_size()) as u64;
        for &(name, ..) in current.tensors().iter().skip(1) {
            if entry[name]["shape"] != json!([row_count, groups]) {
                bail!("PLE tensor {prefix}.{name} has an invalid Q4 shape layout");
            }
        }
        entry.insert("row_start".to_string(), json!(row_start));
        entry.insert("row_count".to_string(), json!(row_count));
        shards.push(Value::Object(entry));
        row_start = row_start
            .checked_add(row_count)
            .ok_or_else(|| anyhow!("PLE row count overflows at {prefix}"))?;
    }
    let quantization = quantization.ok_or_else(|| anyhow!("checkpoint has no PLE shards"))?;
    // Overlapping tensors must not expand into a row file larger than the
    // checkpoint they come from.
    let mut on_disk = 0u64;
    for file in headers.headers.keys() {
        on_disk = on_disk.saturating_add(fs::metadata(source.join(file))?.len());
    }
    if row_start
        .checked_mul(quantization.row_stride(row_width) as u64)
        .is_none_or(|table| table > on_disk)
    {
        bail!("{row_start} PLE rows are more than the shards ({on_disk} bytes) hold");
    }
    Ok(RangeManifest {
        shards,
        row_width,
        row_count: row_start,
        quantization,
    })
}

/// `path` relative to `base`; both must already be canonical.
fn relative_to(path: &Path, base: &Path) -> PathBuf {
    let shared = path
        .components()
        .zip(base.components())
        .take_while(|(a, b)| a == b)
        .count();
    let mut out = PathBuf::new();
    for _ in base.components().skip(shared) {
        out.push(Component::ParentDir.as_os_str());
    }
    for component in path.components().skip(shared) {
        out.push(component.as_os_str());
    }
    if out.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        out
    }
}

/// Rewrite the scattered tensors as one record per row into `data_path`.
fn materialize_rows(source: &Path, ranges: &RangeManifest, data_path: &Path) -> Result<()> {
    let partial = data_path.with_extension("partial");
    let result = (|| -> Result<()> {
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)
            .with_context(|| format!("cannot create {}", partial.display()))?;
        let mut writer = BufWriter::with_capacity(1 << 20, file);
        let mut maps: BTreeMap<String, Mmap> = BTreeMap::new();
        for shard in &ranges.shards {
            let row_count = shard["row_count"]
                .as_u64()
                .ok_or_else(|| anyhow!("shard without row_count"))?
                as usize;
            let mut columns = Vec::new();
            for &(name, _, item) in ranges.quantization.tensors() {
                let row_bytes = ranges
                    .quantization
                    .tensor_row_bytes(name, item, ranges.row_width);
                let file_name = shard[name]["file"]
                    .as_str()
                    .ok_or_else(|| anyhow!("{name} without file"))?
                    .to_string();
                if !maps.contains_key(&file_name) {
                    let file = File::open(source.join(&file_name))?;
                    // SAFETY: read-only map of a checkpoint not modified during the conversion.
                    maps.insert(file_name.clone(), unsafe { Mmap::map(&file) }?);
                }
                let offset = shard[name]["offset"]
                    .as_u64()
                    .ok_or_else(|| anyhow!("{name} without offset"))?
                    as usize;
                let map_len = maps[&file_name].len();
                if row_count
                    .checked_mul(row_bytes)
                    .and_then(|bytes| bytes.checked_add(offset))
                    .is_none_or(|end| end > map_len)
                {
                    bail!("PLE tensor range exceeds {file_name}");
                }
                columns.push((file_name, offset, row_bytes));
            }
            for row in 0..row_count {
                for (file_name, offset, row_bytes) in &columns {
                    let start = offset + row * row_bytes;
                    let bytes = maps[file_name]
                        .get(start..start + row_bytes)
                        .ok_or_else(|| anyhow!("PLE tensor range exceeds {file_name}"))?;
                    writer.write_all(bytes)?;
                }
            }
        }
        writer
            .into_inner()
            .map_err(|e| anyhow!("cannot flush the row file: {e}"))?
            .sync_all()?;
        Ok(())
    })();
    if result.is_ok() {
        fs::rename(&partial, data_path)?;
    } else {
        let _ = fs::remove_file(&partial);
    }
    result
}

/// Build a hard-linked model view of `source` at `target` whose n-gram table
/// is read from disk.
pub fn prepare_external_ple_model(
    source: &Path,
    target: &Path,
    options: &PrepareOptions,
) -> Result<PrepareReport> {
    let source = source
        .canonicalize()
        .with_context(|| format!("cannot resolve {}", source.display()))?;
    if target.exists() && fs::read_dir(target)?.next().is_some() {
        bail!("target model directory is not empty: {}", target.display());
    }
    fs::create_dir_all(target).with_context(|| format!("cannot create {}", target.display()))?;
    let target = target.canonicalize()?;
    let index_path = source.join("model.safetensors.index.json");
    let mut index = read_json(&index_path)?;
    let weight_map = index["weight_map"]
        .as_object()
        .ok_or_else(|| anyhow!("{} has no weight_map", index_path.display()))?
        .clone();
    let mut config = read_json(&source.join("config.json"))?;

    let ranges = build_ranges(&source, &config, &weight_map)?;
    let stride = ranges.quantization.row_stride(ranges.row_width);
    let table_bytes = ranges.row_count * stride as u64;
    let source_root = relative_to(&source, &target).to_string_lossy().into_owned();
    let mut manifest = json!({
        "version": MANIFEST_VERSION,
        "source_root": source_root,
        "row_width": ranges.row_width,
        "row_count": ranges.row_count,
        "quantization": ranges.quantization.manifest_json(),
        "cache_rows": options.cache_rows,
    });
    match options.layout {
        PrepareLayout::SafetensorsRanges => {
            manifest["layout"] = json!("safetensors_ranges");
            manifest["shards"] = Value::Array(ranges.shards.clone());
        }
        PrepareLayout::Interleaved => {
            materialize_rows(&source, &ranges, &target.join(DEFAULT_ROW_FILE))?;
            manifest["layout"] = json!("interleaved_q4");
            manifest["data_file"] = json!(DEFAULT_ROW_FILE);
            manifest["row_stride"] = json!(stride);
        }
    }
    write_json(&target.join(MANIFEST_NAME), &manifest)?;

    let kept: Map<String, Value> = weight_map
        .into_iter()
        .filter(|(key, _)| !key.contains(PLE_MARKER))
        .collect();
    let mut linked: Vec<String> = kept
        .values()
        .filter_map(Value::as_str)
        .map(str::to_string)
        .collect();
    linked.sort_unstable();
    linked.dedup();
    let link = |relative: &Path| -> Result<()> {
        // Only weight files are linked: the view writes over its JSON files,
        // which would write through a link into the source checkpoint.
        if relative.extension().is_none_or(|ext| ext != "safetensors") {
            bail!(
                "refusing to link {}: not a .safetensors file",
                relative.display()
            );
        }
        let original = resolve_contained(&source, relative)
            .with_context(|| format!("refusing to link {}", relative.display()))?;
        if let Some(parent) = relative
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(target.join(parent))?;
        }
        fs::hard_link(original, target.join(relative)).with_context(|| {
            format!(
                "cannot hard-link {} (source and target must share a filesystem)",
                relative.display()
            )
        })
    };
    for file in &linked {
        link(Path::new(file))?;
    }
    // An OptiQ index may already list its vision sidecar among the shards.
    if let Some(sidecar) = declared_optiq_vision_sidecar(&source)
        && !linked.iter().any(|file| Path::new(file) == sidecar)
    {
        link(&sidecar)?;
    }

    index["weight_map"] = Value::Object(kept);
    if !index["metadata"].is_object() {
        index["metadata"] = json!({});
    }
    index["metadata"]["external_ple_bytes"] = json!(table_bytes.to_string());
    write_json(&target.join("model.safetensors.index.json"), &index)?;

    for entry in fs::read_dir(&source)? {
        let entry = entry?;
        let name = entry.file_name();
        let path = entry.path();
        let skip = !path.is_file()
            || name == "config.json"
            || path.extension().is_some_and(|ext| ext == "safetensors")
            || EXCLUDED_FILES.iter().any(|excluded| name == *excluded);
        if !skip {
            // `is_file` follows symlinks; one leading out of the model would
            // copy that file's contents into a directory meant for sharing.
            let original = resolve_contained(&source, Path::new(&name))
                .with_context(|| format!("refusing to copy {}", path.display()))?;
            fs::copy(original, target.join(&name))?;
        }
    }

    let text_config = config
        .get_mut("text_config")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| anyhow!("config.json has no text_config"))?;
    text_config.insert(
        "ple_storage".to_string(),
        json!({"manifest": MANIFEST_NAME, "cache_rows": options.cache_rows}),
    );
    if let Some(quantization) = config
        .get_mut("quantization")
        .and_then(Value::as_object_mut)
    {
        quantization.retain(|key, _| !key.contains(PLE_MARKER));
    }
    write_json(&target.join("config.json"), &config)?;
    write_json(
        &target.join(PROVENANCE_NAME),
        &json!({
            "format": "qwen4_exp_external_ple_v1",
            "source_model": source.to_string_lossy(),
            "source_index": "model.safetensors.index.json",
            "external_ple_manifest": MANIFEST_NAME,
            "linked_weight_files": linked,
        }),
    )?;
    Ok(PrepareReport {
        target,
        table_bytes,
        row_count: ranges.row_count,
        linked_files: linked.len(),
    })
}
