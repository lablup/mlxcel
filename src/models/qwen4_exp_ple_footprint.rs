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

//! Memory accounting for a checkpoint whose n-gram table is external.

use std::collections::HashMap;
use std::path::Path;

use mlxcel_core::weights::declared_optiq_vision_sidecar;
use serde_json::Value;

use super::ple_io::PleIoMode;
use super::ple_manifest::PleManifest;
use super::ple_prepare::{read_header, read_json};
use super::ple_storage::{DEFAULT_PLE_CACHE_BYTES, resolve_cache_budget};

/// What an external-table checkpoint keeps resident.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternalPleFootprint {
    /// Bytes of the tensors the weight index still names, sidecar included.
    pub weight_bytes: u64,
    /// The row cache's budget, or the whole table under `ram`.
    pub row_memory_bytes: u64,
    pub io_mode: PleIoMode,
}

impl ExternalPleFootprint {
    pub fn resident_bytes(&self) -> u64 {
        self.weight_bytes.saturating_add(self.row_memory_bytes)
    }
}

/// The footprint of `model_dir` when its config declares `ple_storage`, or
/// `None` otherwise (and when the manifest cannot be read, leaving the load to
/// report that).
pub fn external_ple_footprint(model_dir: &Path) -> Option<ExternalPleFootprint> {
    let config = read_json(&model_dir.join("config.json")).ok()?;
    let storage = config.get("text_config")?.get("ple_storage")?;
    let manifest_name = storage.get("manifest")?.as_str()?;
    let manifest_path = model_dir.join(manifest_name);
    let manifest = PleManifest::load(
        &manifest_path,
        storage.get("cache_rows").and_then(Value::as_u64),
    )
    .ok()?;
    let weight_bytes = index_weight_bytes(model_dir)?;
    let io_mode = PleIoMode::from_env().unwrap_or_default();
    let row_memory_bytes = match io_mode {
        PleIoMode::Ram => manifest.table_bytes(),
        PleIoMode::Mmap | PleIoMode::Direct => {
            resolve_cache_budget(manifest.cache_rows, manifest.row_stride())
                .unwrap_or(DEFAULT_PLE_CACHE_BYTES)
        }
    };
    Some(ExternalPleFootprint {
        weight_bytes,
        row_memory_bytes,
        io_mode,
    })
}

/// Sum of the tensors the index maps, read from the shard headers: the view's
/// shards still hold the table's bytes, so whole-file sums would count it.
fn index_weight_bytes(model_dir: &Path) -> Option<u64> {
    let index = read_json(&model_dir.join("model.safetensors.index.json")).ok()?;
    let weight_map = index.get("weight_map")?.as_object()?;
    let mut per_file: HashMap<&str, Vec<&str>> = HashMap::new();
    for (key, file) in weight_map {
        per_file.entry(file.as_str()?).or_default().push(key);
    }
    let mut total = 0u64;
    let tensor_size = |info: &Value| -> Option<u64> {
        let offsets = info.get("data_offsets")?.as_array()?;
        Some(
            offsets
                .get(1)?
                .as_u64()?
                .saturating_sub(offsets.first()?.as_u64()?),
        )
    };
    for (file, keys) in per_file {
        let (_, header) = read_header(&model_dir.join(file)).ok()?;
        for key in keys {
            total = total.saturating_add(tensor_size(header.get(key)?)?);
        }
    }
    if let Some(sidecar) = declared_optiq_vision_sidecar(model_dir) {
        let (_, header) = read_header(&model_dir.join(sidecar)).ok()?;
        total = total.saturating_add(
            header
                .as_object()?
                .values()
                .filter_map(tensor_size)
                .sum::<u64>(),
        );
    }
    Some(total)
}
