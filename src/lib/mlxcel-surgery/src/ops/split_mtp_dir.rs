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

//! Checkpoint directory driver for `split-mtp`.

use std::path::Path;

use anyhow::anyhow;
use serde_json::Value;

use super::{SplitMtpOptions, SplitMtpReport, nextn_layer_index, output, split_mtp};
use crate::SurgeryError;

/// Whether the safetensors index names any `model.layers.{layer}.*` tensor.
///
/// One JSON parse, no shard opened. `Ok(true)` for a directory without an
/// index: the filtered load below then answers the question from the shard
/// headers and [`split_mtp`] reports the same refusal on an empty result.
fn index_names_nextn_layer(model_dir: &Path, layer: usize) -> Result<bool, SurgeryError> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        return Ok(true);
    }
    let prefix = format!("model.layers.{layer}.");
    let raw = std::fs::read_to_string(&index_path)?;
    let value: Value = serde_json::from_str(&raw)
        .map_err(|error| anyhow!("split-mtp: parsing {}: {error}", index_path.display()))?;
    let map = value
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("{}: missing weight_map object", index_path.display()))?;
    Ok(map.keys().any(|key| key.starts_with(&prefix)))
}

/// Refuse when `output_dir` resolves to the same directory as `model_dir`.
///
/// A sharded raw checkpoint carries no `model.safetensors`, so the CLI's
/// overwrite guard does not fire on it. The refusal runs before any mutation,
/// since the CLI cannot distinguish a source shard from a foreign output.
pub fn refuse_output_is_source(model_dir: &Path, output_dir: &Path) -> Result<(), SurgeryError> {
    if std::fs::canonicalize(model_dir)? == output::resolve_output_path(output_dir)? {
        return Err(output::source_alias_error(model_dir));
    }
    Ok(())
}

/// Split the nextn block out of `model_dir` into `output_dir`.
///
/// Reads `config.json`, refuses a checkpoint whose index names no nextn
/// tensor, loads only the shards that hold the block, performs the transform,
/// and sends the complete result through the exclusive staging writer.
pub fn split_mtp_dir(
    model_dir: &Path,
    output_dir: &Path,
    opts: &SplitMtpOptions,
) -> Result<SplitMtpReport, SurgeryError> {
    refuse_output_is_source(model_dir, output_dir)?;
    let config_path = model_dir.join("config.json");
    let config: Value = serde_json::from_str(&std::fs::read_to_string(&config_path)?)
        .map_err(|error| anyhow!("split-mtp: parsing {}: {error}", config_path.display()))?;
    let source_layer = nextn_layer_index(&config)?;

    if !index_names_nextn_layer(model_dir, source_layer)? {
        return Err(anyhow!(
            "split-mtp: {} names no `model.layers.{source_layer}.*` tensor; this checkpoint was \
             converted without its MTP layer; use the raw zai-org checkpoint",
            model_dir.display()
        )
        .into());
    }

    let layer_prefix = format!("model.layers.{source_layer}.");
    let weights = mlxcel_core::weights::load_weights_from_dir_index_filtered(model_dir, |name| {
        name.starts_with(&layer_prefix)
    })
    .map_err(|error| anyhow!("split-mtp: {error}"))?;
    let result = split_mtp(weights, &config, opts)?;

    output::write_split_output(model_dir, output_dir, result, source_layer, opts.force)
}
