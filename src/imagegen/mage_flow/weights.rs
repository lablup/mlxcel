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

//! Component weight loading for Mage-Flow, a port of `weights.py` in
//! mlx-vlm `73d366ec` (`mlx_vlm/models/mage_flow`).
//!
//! The published trees (`transformer/`, `vae/`) load as-is: the transformer
//! keys are renamed to upstream's module names and the VAE keys are mapped
//! from the training checkpoint layout (`student.dconv_encoder.*`,
//! `pipeline.*`). An mlx-vlm converted tree loads too: its quantized
//! transformer projections stay packed and run quantized matmuls, with the
//! mode, bits and group size read the way upstream's `_metadata_value` reads
//! them (the component `config.json` merged with the safetensors metadata).

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::weights::WeightMap;
use serde_json::{Map, Value};

use super::config::TransformerConfig;
use super::layers::LinearQuant;
use super::transformer::MageFlowTransformer;
use super::vae::MageVae;
use crate::imagegen::quant::{QuantSpec, quantized_prefixes, validate_quantized_weights};
use crate::imagegen::vae::{ConvLayout, probe_conv_layout};

/// Marker an mlx-vlm conversion writes into the component metadata.
pub const MLX_VLM_FORMAT: &str = "mage_flow";

/// `__metadata__` of one safetensors file (string values), or empty.
fn safetensors_metadata(path: &Path) -> Map<String, Value> {
    use std::io::Read;
    let read = || -> Option<Map<String, Value>> {
        let mut file = std::fs::File::open(path).ok()?;
        let mut len = [0u8; 8];
        file.read_exact(&mut len).ok()?;
        let len = u64::from_le_bytes(len);
        // A corrupt header length must not allocate unbounded memory.
        if len > 256 * 1024 * 1024 {
            return None;
        }
        let mut header = vec![0u8; usize::try_from(len).ok()?];
        file.read_exact(&mut header).ok()?;
        let header: Value = serde_json::from_slice(&header).ok()?;
        header.get("__metadata__")?.as_object().cloned()
    };
    read().unwrap_or_default()
}

/// Upstream `_load_safetensors` metadata: the index `metadata`, then every
/// shard's `__metadata__`, later keys winning.
pub fn component_metadata(dir: &Path) -> Result<Map<String, Value>> {
    let mut metadata = Map::new();
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let text = std::fs::read_to_string(&index)
            .with_context(|| format!("cannot read {}", index.display()))?;
        let value: Value = serde_json::from_str(&text)
            .with_context(|| format!("Invalid safetensor index: {}", index.display()))?;
        if let Some(m) = value.get("metadata").and_then(Value::as_object) {
            metadata.extend(m.clone());
        }
    }
    let mut shards: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("cannot list {}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "safetensors")
                && !p
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("._"))
        })
        .collect();
    shards.sort();
    for shard in shards {
        metadata.extend(safetensors_metadata(&shard));
    }
    Ok(metadata)
}

/// Upstream `_metadata_value`: the first of `keys` at the top level, then in
/// a `quantization` or `quantization_config` object.
fn metadata_value<'a>(metadata: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a Value> {
    let mut candidates = vec![metadata];
    for container in ["quantization", "quantization_config"] {
        if let Some(m) = metadata.get(container).and_then(Value::as_object) {
            candidates.push(m);
        }
    }
    candidates
        .into_iter()
        .find_map(|c| keys.iter().find_map(|k| c.get(*k)))
}

/// Upstream `_metadata_int`: a number or a numeric string.
fn metadata_int(metadata: &Map<String, Value>, keys: &[&str]) -> Result<Option<i32>> {
    let Some(value) = metadata_value(metadata, keys) else {
        return Ok(None);
    };
    let parsed = match value {
        Value::Number(n) => n.as_i64().and_then(|n| i32::try_from(n).ok()),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    };
    parsed
        .map(Some)
        .with_context(|| format!("Invalid quantization metadata value: {value}"))
}

/// Upstream `infer_quantization_config`: `None` for a dense checkpoint (no
/// `.scales`), else the validated spec; every packed tensor is checked before
/// MLX sees it.
pub fn infer_quantization(
    weights: &WeightMap,
    metadata: &Map<String, Value>,
) -> Result<Option<LinearQuant>> {
    if quantized_prefixes(weights).is_empty() {
        return Ok(None);
    }
    let mode = metadata_value(metadata, &["mode", "quantization_mode", "q_mode"])
        .context("Quantized checkpoint is missing its quantization mode")?;
    let mode = match mode {
        Value::String(s) => s.to_lowercase(),
        other => other.to_string().to_lowercase(),
    };
    let bits = metadata_int(
        metadata,
        &["bits", "num_bits", "quantization_level", "q_bits"],
    )?;
    let group_size = metadata_int(
        metadata,
        &["group_size", "quantization_group_size", "q_group_size"],
    )?;
    let (Some(bits), Some(group_size)) = (bits, group_size) else {
        bail!("Quantized checkpoint is missing its bits or group size");
    };
    let spec = QuantSpec::validated("Mage-Flow", &mode, bits, group_size)?;
    validate_quantized_weights(weights, &spec)?;
    LinearQuant::from_spec(&spec).map(Some)
}

/// Upstream `sanitize_transformer_weights`: Diffusers-style names to the
/// module names of [`MageFlowTransformer`].
#[must_use]
pub fn sanitize_transformer_key(key: &str) -> String {
    const RENAMES: [(&str, &str); 7] = [
        (".img_mod.1.", ".img_mod.linear."),
        (".txt_mod.1.", ".txt_mod.linear."),
        (".img_mlp.net.0.proj.", ".img_mlp.linear_in."),
        (".img_mlp.net.2.", ".img_mlp.linear_out."),
        (".txt_mlp.net.0.proj.", ".txt_mlp.linear_in."),
        (".txt_mlp.net.2.", ".txt_mlp.linear_out."),
        (".attn.to_out.0.", ".attn.to_out."),
    ];
    RENAMES
        .iter()
        .fold(key.to_string(), |k, (from, to)| k.replace(from, to))
}

/// Read `<root>/transformer` (config and weights) into a transformer.
pub fn load_transformer(root: &Path) -> Result<MageFlowTransformer> {
    let dir = root.join("transformer");
    let config_path = dir.join("config.json");
    let config: Value = serde_json::from_str(
        &std::fs::read_to_string(&config_path)
            .with_context(|| format!("cannot read {}", config_path.display()))?,
    )
    .with_context(|| format!("cannot parse {}", config_path.display()))?;
    let transformer_config = TransformerConfig::from_json(&config)?;
    let raw = mlxcel_core::weights::load_weights_from_dir(&dir)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("cannot load Mage-Flow weights from {}", dir.display()))?;
    let weights: WeightMap = raw
        .into_iter()
        .map(|(k, v)| (sanitize_transformer_key(&k), v))
        .collect();
    // `{**config, **metadata}`: the safetensors metadata wins.
    let mut metadata = config.as_object().cloned().unwrap_or_default();
    metadata.extend(component_metadata(&dir)?);
    let quant = infer_quantization(&weights, &metadata)?;
    MageFlowTransformer::from_weights(transformer_config, &weights, quant)
}

/// Upstream `_map_vae_key`: a training-checkpoint VAE key to a [`MageVae`]
/// key, or `None` for a tensor the VAE never reads.
#[must_use]
pub fn map_vae_key(key: &str) -> Option<String> {
    let key = if let Some(rest) = key.strip_prefix("student.dconv_encoder.") {
        format!("dconv_encoder.{rest}")
    } else if key.starts_with("dconv_encoder.") || key.starts_with("decoder_model.") {
        key.to_string()
    } else if key.starts_with("pipeline.y_embedder.encoder.") {
        return None;
    } else {
        format!("decoder_model.{}", key.strip_prefix("pipeline.")?)
    };
    let mut key = [
        (".adaLN_modulation.1.", ".adaLN_modulation.linear."),
        (".ca.1.", ".ca_conv."),
        (".t_embedder.mlp.0.", ".t_embedder.linear_1."),
        (".t_embedder.mlp.2.", ".t_embedder.linear_2."),
        (".x_embedder.embedder.0.", ".x_embedder.linear."),
    ]
    .iter()
    .fold(key, |k, (from, to)| k.replace(from, to));
    // `(\.dec_net\.res_blocks\.\d+)\.mlp\.(0|2)\.` to `\1.linear_(1|2).`.
    for (from, to) in [(".mlp.0.", ".linear_1."), (".mlp.2.", ".linear_2.")] {
        if let Some(pos) = key.find(".dec_net.res_blocks.") {
            let digits_start = pos + ".dec_net.res_blocks.".len();
            let digits = key[digits_start..]
                .bytes()
                .take_while(u8::is_ascii_digit)
                .count();
            let at = digits_start + digits;
            if digits > 0 && key[at..].starts_with(from) {
                key = format!("{}{to}{}", &key[..at], &key[at + from.len()..]);
            }
        }
    }
    Some(key)
}

/// Upstream `sanitize_vae_weights`: map the keys, drop the unread tensors and
/// `num_batches_tracked`, and put 4-D conv weights in MLX layout.
pub(crate) fn sanitize_vae_weights(raw: WeightMap, layout: ConvLayout) -> WeightMap {
    raw.into_iter()
        .filter(|(k, _)| !k.ends_with(".num_batches_tracked"))
        .filter_map(|(k, v)| {
            let key = map_vae_key(&k)?;
            let value = if layout == ConvLayout::Torch && mlxcel_core::array_ndim(&v) == 4 {
                mlxcel_core::transpose_axes(&v, &[0, 2, 3, 1])
            } else {
                v
            };
            Some((key, value))
        })
        .collect()
}

/// The conv layout of a VAE: MLX when the metadata says so (upstream's only
/// rule), else probed from `conv_in`'s shape, so a converted tree without
/// metadata is not permuted a second time.
fn vae_layout(weights: &WeightMap, metadata: &Map<String, Value>) -> Result<ConvLayout> {
    let says = |k: &str, v: &str| metadata.get(k).and_then(Value::as_str) == Some(v);
    if says("tensor_layout", "mlx_nhwc") || says("mlx_vlm_format", MLX_VLM_FORMAT) {
        return Ok(ConvLayout::Mlx);
    }
    probe_conv_layout(
        weights,
        &[
            ("pipeline.y_embedder.decoder.conv_in.weight", 384, 128, 3),
            (
                "decoder_model.y_embedder.decoder.conv_in.weight",
                384,
                128,
                3,
            ),
        ],
    )
}

/// Read `<root>/vae` into a [`MageVae`]; the encoder only when
/// `include_encoder` (edit variants).
pub fn load_vae(root: &Path, include_encoder: bool) -> Result<MageVae> {
    let dir = root.join("vae");
    let raw = mlxcel_core::weights::load_weights_from_dir_filtered(&dir, |key| {
        map_vae_key(key).is_some_and(|k| include_encoder || !k.starts_with("dconv_encoder."))
    })
    .map_err(anyhow::Error::msg)
    .with_context(|| format!("cannot load Mage-VAE weights from {}", dir.display()))?;
    if !quantized_prefixes(&raw).is_empty() {
        bail!(
            "quantized Mage-VAE weights are not supported (mlx-vlm quantizes only the transformer)"
        );
    }
    let metadata = component_metadata(&dir)?;
    let layout = vae_layout(&raw, &metadata)?;
    MageVae::from_weights(&sanitize_vae_weights(raw, layout), include_encoder)
}

#[cfg(test)]
#[path = "weights_tests.rs"]
mod tests;
