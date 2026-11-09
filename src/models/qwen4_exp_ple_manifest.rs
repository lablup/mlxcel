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

//! The external PLE table manifest (`ple-store.json`, version 2), as written
//! by mlx-vlm's `prepare_external_ple_model` and by
//! `mlxcel prepare-external-ple`, and the `text_config.ple_storage` block that
//! points at it.
//!
//! Every validation mirrors upstream `QuantizedMMapNGramEmbedding.__init__`
//! and every failure names the manifest field.

use std::path::{Component, Path, PathBuf};

use serde_json::Value;

pub const MANIFEST_VERSION: u64 = 2;

/// A manifest rule failure: the offending field and why it is refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("external PLE manifest field `{field}`: {reason}")]
pub struct PleManifestError {
    pub field: String,
    pub reason: String,
}

pub(crate) fn field_error(field: impl Into<String>, reason: impl Into<String>) -> PleManifestError {
    PleManifestError {
        field: field.into(),
        reason: reason.into(),
    }
}

/// The two supported quantization layouts of a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PleQuantization {
    /// `(bits, group_size, mode) = (4, 32, "affine")`: `weight` U32, `scales`
    /// BF16, `biases` BF16.
    Affine4x32,
    /// `(4, 16, "nvfp4")`: `weight` U32, `scales` U8.
    Nvfp4x16,
}

impl PleQuantization {
    pub const BITS: i32 = 4;

    pub(crate) fn from_fields(
        bits: Option<u64>,
        group: Option<u64>,
        mode: Option<&str>,
    ) -> Option<Self> {
        match (bits, group, mode) {
            (Some(4), Some(32), Some("affine")) => Some(Self::Affine4x32),
            (Some(4), Some(16), Some("nvfp4")) => Some(Self::Nvfp4x16),
            _ => None,
        }
    }

    pub fn group_size(self) -> usize {
        match self {
            Self::Affine4x32 => 32,
            Self::Nvfp4x16 => 16,
        }
    }

    pub fn mode(self) -> &'static str {
        match self {
            Self::Affine4x32 => "affine",
            Self::Nvfp4x16 => "nvfp4",
        }
    }

    /// `(name, safetensors dtype, item bytes)` in on-disk record order.
    pub fn tensors(self) -> &'static [(&'static str, &'static str, usize)] {
        match self {
            Self::Affine4x32 => &[
                ("weight", "U32", 4),
                ("scales", "BF16", 2),
                ("biases", "BF16", 2),
            ],
            Self::Nvfp4x16 => &[("weight", "U32", 4), ("scales", "U8", 1)],
        }
    }

    /// Bytes of a tensor in one row of width `row_width`.
    pub fn tensor_row_bytes(self, name: &str, item_bytes: usize, row_width: usize) -> usize {
        if name == "weight" {
            row_width / 2
        } else {
            row_width / self.group_size() * item_bytes
        }
    }

    /// Bytes of one interleaved row record.
    pub fn row_stride(self, row_width: usize) -> usize {
        self.tensors()
            .iter()
            .map(|&(name, _, item)| self.tensor_row_bytes(name, item, row_width))
            .sum()
    }

    pub(crate) fn manifest_json(self) -> Value {
        serde_json::json!({"bits": 4, "group_size": self.group_size(), "mode": self.mode()})
    }
}

/// One tensor's byte range in a shard file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PleTensorRange {
    pub path: PathBuf,
    pub offset: u64,
    pub row_bytes: usize,
}

/// A `safetensors_ranges` shard; `tensors` is in [`PleQuantization::tensors`] order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PleShardRanges {
    pub row_start: u64,
    pub row_count: u64,
    pub tensors: Vec<PleTensorRange>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PleLayout {
    Interleaved { data_path: PathBuf },
    SafetensorsRanges { shards: Vec<PleShardRanges> },
}

/// A validated `ple-store.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PleManifest {
    pub row_width: usize,
    pub row_count: u64,
    pub quantization: PleQuantization,
    pub layout: PleLayout,
    /// Upstream's row-count cache hint; 0 means none.
    pub cache_rows: u64,
}

impl PleManifest {
    pub fn row_stride(&self) -> usize {
        self.quantization.row_stride(self.row_width)
    }

    /// Bytes the table occupies on disk.
    pub fn table_bytes(&self) -> u64 {
        self.row_count.saturating_mul(self.row_stride() as u64)
    }

    /// Read and validate a manifest. `config_cache_rows` is
    /// `text_config.ple_storage.cache_rows`, which overrides the manifest's own.
    pub fn load(path: &Path, config_cache_rows: Option<u64>) -> Result<Self, PleManifestError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| field_error("manifest", format!("cannot read {}: {e}", path.display())))?;
        let value: Value = serde_json::from_str(&text).map_err(|e| {
            field_error(
                "manifest",
                format!("invalid JSON in {}: {e}", path.display()),
            )
        })?;
        Self::from_value(&value, path, config_cache_rows)
    }

    pub(crate) fn from_value(
        value: &Value,
        manifest_path: &Path,
        config_cache_rows: Option<u64>,
    ) -> Result<Self, PleManifestError> {
        let manifest = value
            .as_object()
            .ok_or_else(|| field_error("manifest", "must be a JSON object"))?;
        let missing: Vec<&str> = ["version", "source_root", "row_width", "quantization"]
            .into_iter()
            .filter(|key| !manifest.contains_key(*key))
            .collect();
        if !missing.is_empty() {
            return Err(field_error(missing.join(","), "missing from the manifest"));
        }
        let version = expect_u64(&manifest["version"], "version")?;
        if version != MANIFEST_VERSION {
            return Err(field_error(
                "version",
                format!("unsupported manifest version {version}, expected {MANIFEST_VERSION}"),
            ));
        }
        let quantization = parse_quantization(&manifest["quantization"])?;
        let source_root = manifest["source_root"]
            .as_str()
            .ok_or_else(|| field_error("source_root", "must be a string"))?;
        let base = manifest_path.parent().unwrap_or_else(|| Path::new(""));
        let source_root = base.join(source_root);
        let row_width = expect_u64(&manifest["row_width"], "row_width")? as usize;
        if row_width == 0 || !row_width.is_multiple_of(quantization.group_size()) {
            return Err(field_error(
                "row_width",
                format!(
                    "{row_width} must be positive and divisible by group_size {}",
                    quantization.group_size()
                ),
            ));
        }
        if row_width > i32::MAX as usize {
            return Err(field_error(
                "row_width",
                "exceeds the MLX i32 dimension limit",
            ));
        }
        let cache_rows = match config_cache_rows {
            Some(rows) => rows,
            None => match manifest.get("cache_rows") {
                None | Some(Value::Null) => 0,
                Some(value) => expect_u64(value, "cache_rows")?,
            },
        };
        let layout = manifest
            .get("layout")
            .and_then(Value::as_str)
            .unwrap_or("safetensors_ranges");
        let stride = quantization.row_stride(row_width);
        let (layout, row_count) = match layout {
            "interleaved_q4" => parse_interleaved(manifest, base, stride)?,
            "safetensors_ranges" => parse_ranges(manifest, &source_root, row_width, quantization)?,
            other => {
                return Err(field_error(
                    "layout",
                    format!("unsupported PLE storage layout {other:?}"),
                ));
            }
        };
        Ok(Self {
            row_width,
            row_count,
            quantization,
            layout,
            cache_rows,
        })
    }
}

fn parse_quantization(value: &Value) -> Result<PleQuantization, PleManifestError> {
    let object = value
        .as_object()
        .ok_or_else(|| field_error("quantization", "must be an object"))?;
    let bits = object.get("bits").and_then(Value::as_u64);
    let group = object.get("group_size").and_then(Value::as_u64);
    let mode = object.get("mode").and_then(Value::as_str);
    PleQuantization::from_fields(bits, group, mode)
        .ok_or_else(|| field_error("quantization", format!("unsupported PLE quantization {value}: only affine 4-bit group 32 and nvfp4 4-bit group 16 are defined")))
}

pub(crate) fn expect_u64(value: &Value, field: &str) -> Result<u64, PleManifestError> {
    value.as_u64().ok_or_else(|| {
        field_error(
            field,
            format!("must be a non-negative integer, got {value}"),
        )
    })
}

/// Only a single normal path component names a file beside the manifest or
/// under `source_root`.
fn single_file_name(name: &str, field: &str, place: &str) -> Result<(), PleManifestError> {
    let mut components = Path::new(name).components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(_)), None) => Ok(()),
        _ => Err(field_error(
            field,
            format!("must be a filename {place}, got {name:?}"),
        )),
    }
}

fn parse_interleaved(
    manifest: &serde_json::Map<String, Value>,
    base: &Path,
    stride: usize,
) -> Result<(PleLayout, u64), PleManifestError> {
    let required = |key: &str| {
        manifest
            .get(key)
            .ok_or_else(|| field_error(key, "missing from an interleaved_q4 manifest"))
    };
    let name = required("data_file")?
        .as_str()
        .ok_or_else(|| field_error("data_file", "must be a string"))?;
    single_file_name(name, "data_file", "beside the manifest")?;
    let row_count = expect_u64(required("row_count")?, "row_count")?;
    let row_stride = expect_u64(required("row_stride")?, "row_stride")?;
    if row_stride != stride as u64 {
        return Err(field_error(
            "row_stride",
            format!(
                "{row_stride} does not match the row width and quantization, expected {stride}"
            ),
        ));
    }
    let expected = row_count
        .checked_mul(row_stride)
        .ok_or_else(|| field_error("row_count", "row_count * row_stride overflows"))?;
    let data_path = base.join(name);
    let size = std::fs::metadata(&data_path)
        .map_err(|e| {
            field_error(
                "data_file",
                format!("cannot stat {}: {e}", data_path.display()),
            )
        })?
        .len();
    if size != expected {
        return Err(field_error(
            "data_file",
            format!("interleaved PLE size mismatch: expected {expected}, got {size}"),
        ));
    }
    Ok((PleLayout::Interleaved { data_path }, row_count))
}

fn parse_ranges(
    manifest: &serde_json::Map<String, Value>,
    source_root: &Path,
    row_width: usize,
    quantization: PleQuantization,
) -> Result<(PleLayout, u64), PleManifestError> {
    let entries = manifest
        .get("shards")
        .ok_or_else(|| field_error("shards", "missing from a safetensors_ranges manifest"))?
        .as_array()
        .ok_or_else(|| field_error("shards", "must be an array"))?;
    let groups = (row_width / quantization.group_size()) as u64;
    let mut shards = Vec::with_capacity(entries.len());
    let mut expected_start = 0u64;
    for (index, entry) in entries.iter().enumerate() {
        let field = |name: &str| format!("shards[{index}].{name}");
        let get = |name: &str| {
            entry
                .get(name)
                .ok_or_else(|| field_error(field(name), "missing"))
        };
        let row_start = expect_u64(get("row_start")?, &field("row_start"))?;
        let row_count = expect_u64(get("row_count")?, &field("row_count"))?;
        if row_start != expected_start {
            return Err(field_error(
                field("row_start"),
                format!(
                    "shards must be contiguous and ordered: expected {expected_start}, got {row_start}"
                ),
            ));
        }
        if row_count == 0 {
            return Err(field_error(field("row_count"), "shards must be positive"));
        }
        let mut tensors = Vec::new();
        for &(name, dtype, item_bytes) in quantization.tensors() {
            let tensor = get(name)?;
            let member = |key: &str| {
                tensor
                    .get(key)
                    .ok_or_else(|| field_error(format!("shards[{index}].{name}.{key}"), "missing"))
            };
            let tensor_field = |key: &str| format!("shards[{index}].{name}.{key}");
            let found = member("dtype")?.as_str().unwrap_or_default();
            if found != dtype {
                return Err(field_error(
                    tensor_field("dtype"),
                    format!("expected {dtype}, got {found:?}"),
                ));
            }
            let file = member("file")?.as_str().unwrap_or_default();
            single_file_name(file, &tensor_field("file"), "under source_root")?;
            let offset = expect_u64(member("offset")?, &tensor_field("offset"))?;
            let shape: Vec<u64> = member("shape")?
                .as_array()
                .ok_or_else(|| field_error(tensor_field("shape"), "must be an array"))?
                .iter()
                .map(|dim| expect_u64(dim, &tensor_field("shape")))
                .collect::<Result<_, _>>()?;
            if shape.first() != Some(&row_count) {
                return Err(field_error(
                    tensor_field("shape"),
                    format!("{shape:?} does not start with the shard's row count {row_count}"),
                ));
            }
            let expected_width = if name == "weight" {
                (row_width / 8) as u64
            } else {
                groups
            };
            if shape.len() != 2 || shape[1] != expected_width {
                let what = if name == "weight" {
                    "packed weight width does not match row_width"
                } else {
                    "auxiliary tensor shape does not match the Q4 group layout"
                };
                return Err(field_error(
                    tensor_field("shape"),
                    format!("{what}: {shape:?}, expected [{row_count}, {expected_width}]"),
                ));
            }
            let bytes = row_count
                .checked_mul(shape[1])
                .and_then(|items| items.checked_mul(item_bytes as u64))
                .ok_or_else(|| field_error(tensor_field("shape"), "byte count overflows"))?;
            let path = source_root.join(file);
            let size = std::fs::metadata(&path)
                .map_err(|e| {
                    field_error(
                        tensor_field("file"),
                        format!("cannot stat {}: {e}", path.display()),
                    )
                })?
                .len();
            if offset.checked_add(bytes).is_none_or(|end| size < end) {
                return Err(field_error(
                    tensor_field("offset"),
                    format!("byte range exceeds {}", path.display()),
                ));
            }
            tensors.push(PleTensorRange {
                path,
                offset,
                row_bytes: quantization.tensor_row_bytes(name, item_bytes, row_width),
            });
        }
        expected_start = row_start
            .checked_add(row_count)
            .ok_or_else(|| field_error(field("row_count"), "row range overflows"))?;
        shards.push(PleShardRanges {
            row_start,
            row_count,
            tensors,
        });
    }
    Ok((PleLayout::SafetensorsRanges { shards }, expected_start))
}

#[cfg(test)]
#[path = "qwen4_exp_ple_manifest_tests.rs"]
mod tests;
