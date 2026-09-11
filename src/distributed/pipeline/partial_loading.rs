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

//! Partial model loading for pipeline parallelism.
//!
//! When a pipeline stage only needs a subset of transformer layers (plus
//! optionally the embedding table and lm_head), this module provides the
//! tools to:
//!
//! 1. Build a [`LayerFilter`] from a [`StageAssignment`].
//! 2. Classify weight keys as needed or skippable via [`classify_weight_key`].
//! 3. Identify which SafeTensors shard files are required using a weight map
//!    index ([`identify_required_shards`]).
//! 4. Filter an already-loaded [`WeightMap`] to drop unneeded tensors
//!    ([`filter_weight_map`]).
//! 5. Estimate memory for a partial load and validate against device capacity.
//!
//! Stage-local LoRA adapter loading lives in
//! [`super::partial_loading_adapter`] and is re-exported from this module
//! so callers can import a single namespace.
//!
//! Used by: pipeline startup, distributed model loading

use std::collections::HashSet;
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{Result, ensure};

use super::partition::{ModelProfile, StageAssignment};

// ---------------------------------------------------------------------------
// LayerFilter
// ---------------------------------------------------------------------------

/// Describes which parts of a model a pipeline stage needs.
///
/// Created from a [`StageAssignment`] and used to filter weight keys during
/// partial loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerFilter {
    /// Half-open range of transformer layer indices to load (e.g. `16..32`).
    pub layer_range: Range<usize>,
    /// Whether to load the embedding table (first pipeline stage).
    pub has_embedding: bool,
    /// Whether to load the lm_head / output projection (last pipeline stage).
    pub has_lm_head: bool,
}

impl LayerFilter {
    /// Create a filter from a stage assignment.
    pub fn from_stage(stage: &StageAssignment) -> Self {
        Self {
            layer_range: stage.layer_range.clone(),
            has_embedding: stage.has_embedding,
            has_lm_head: stage.has_lm_head,
        }
    }

    /// Number of layers this filter covers.
    pub fn num_layers(&self) -> usize {
        self.layer_range.end.saturating_sub(self.layer_range.start)
    }

    /// Returns `true` if this filter covers all layers of a model with
    /// `total_layers` layers and includes both embedding and lm_head (i.e.
    /// partial loading is not actually needed).
    pub fn is_full_model(&self, total_layers: usize) -> bool {
        self.layer_range == (0..total_layers) && self.has_embedding && self.has_lm_head
    }
}

// ---------------------------------------------------------------------------
// Weight key classification
// ---------------------------------------------------------------------------

/// Known weight key prefix families used by supported architectures.
///
/// Most HuggingFace models use `model.layers.N.` for transformer blocks,
/// `model.embed_tokens` for the embedding, and `lm_head` for the output head.
/// Some models (VLMs, Gemma3n, etc.) nest under `language_model.model.` or
/// use other prefixes. This list covers the common patterns; models with
/// non-standard prefixes should be added here as they are discovered.
const LAYER_PREFIXES: &[&str] = &[
    "model.layers.",
    "language_model.model.layers.",
    "transformer.h.",
    "transformer.layers.",
    "backbone.layers.",
];

const EMBEDDING_PREFIXES: &[&str] = &[
    "model.embed_tokens",
    "language_model.model.embed_tokens",
    "transformer.wte",
    "transformer.embd",
    "backbone.embedding",
    "model.word_embeddings",
];

const LM_HEAD_PREFIXES: &[&str] = &["lm_head", "language_model.lm_head", "output."];

/// Non-layer model-level weights that should be loaded on every stage
/// (e.g. final layer norm). These are small and essential for correct
/// inference when the stage hosts the lm_head or needs to normalize
/// before handing off activations.
const NORM_PREFIXES: &[&str] = &[
    "model.norm",
    "model.final_layernorm",
    "language_model.model.norm",
    "transformer.ln_f",
    "backbone.norm",
];

/// Result of classifying a weight key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightClass {
    /// A transformer block weight belonging to layer N.
    Layer(usize),
    /// An embedding table weight.
    Embedding,
    /// An lm_head / output projection weight.
    LmHead,
    /// A model-level normalization weight (final norm, etc.).
    Norm,
    /// Any other weight (vision encoder, connector, etc.).
    Other,
}

/// Classify a weight key into its functional category.
///
/// Returns `WeightClass::Layer(n)` for transformer block weights, or one
/// of the other variants for embedding/lm_head/norm/other weights.
pub fn classify_weight_key(key: &str) -> WeightClass {
    // Check layer prefixes first (most common).
    for prefix in LAYER_PREFIXES {
        if let Some(rest) = key.strip_prefix(prefix) {
            // rest starts with the layer index, e.g. "0.self_attn.q_proj.weight"
            if let Some(idx) = parse_leading_usize(rest) {
                return WeightClass::Layer(idx);
            }
        }
    }

    // Embedding.
    for prefix in EMBEDDING_PREFIXES {
        if key.starts_with(prefix) {
            return WeightClass::Embedding;
        }
    }

    // lm_head.
    for prefix in LM_HEAD_PREFIXES {
        if key.starts_with(prefix) {
            return WeightClass::LmHead;
        }
    }

    // Norm.
    for prefix in NORM_PREFIXES {
        if key.starts_with(prefix) {
            return WeightClass::Norm;
        }
    }

    WeightClass::Other
}

/// Parse a leading unsigned integer from a string (stops at the first
/// non-digit character). Returns `None` if the string does not start
/// with a digit.
fn parse_leading_usize(s: &str) -> Option<usize> {
    let digits: &str = &s[..s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len())];
    if digits.is_empty() {
        return None;
    }
    digits.parse().ok()
}

/// Determine whether a weight key should be loaded given the filter.
///
/// - Layer weights: loaded only if the layer index is within `filter.layer_range`.
/// - Embedding weights: loaded only if `filter.has_embedding`.
/// - LmHead weights: loaded only if `filter.has_lm_head`.
/// - Norm weights: loaded if the stage has lm_head (needs final norm for output).
/// - Other weights: loaded on the first stage by default (vision encoder,
///   connectors, etc. typically pair with embedding).
pub fn should_load_key(key: &str, filter: &LayerFilter) -> bool {
    match classify_weight_key(key) {
        WeightClass::Layer(idx) => filter.layer_range.contains(&idx),
        WeightClass::Embedding => filter.has_embedding,
        WeightClass::LmHead => filter.has_lm_head,
        WeightClass::Norm => filter.has_lm_head,
        WeightClass::Other => filter.has_embedding,
    }
}

/// Filter a set of weight keys, returning only those needed by this stage.
pub fn filter_weight_keys<'a>(
    keys: impl Iterator<Item = &'a str>,
    filter: &LayerFilter,
) -> Vec<String> {
    keys.filter(|k| should_load_key(k, filter))
        .map(|k| k.to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// SafeTensors index: shard file identification
// ---------------------------------------------------------------------------

/// A parsed SafeTensors index (`model.safetensors.index.json`).
///
/// Maps each weight key to the shard filename that contains it.
#[derive(Debug, Clone)]
pub struct SafeTensorsIndex {
    /// Map from weight key to shard filename (e.g. "model-00001-of-00004.safetensors").
    pub weight_to_shard: Vec<(String, String)>,
}

impl SafeTensorsIndex {
    /// Parse the index from a JSON string (the content of
    /// `model.safetensors.index.json`).
    ///
    /// Expects a top-level `"weight_map"` object mapping tensor names to
    /// shard filenames.
    pub fn from_json(json_str: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| anyhow::anyhow!("failed to parse safetensors index: {e}"))?;

        let weight_map = value
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow::anyhow!("safetensors index missing 'weight_map' object"))?;

        let entries: Vec<(String, String)> = weight_map
            .iter()
            .map(|(k, v)| {
                let shard = v
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("weight_map value for '{k}' is not a string"))?;
                // Reject shard filenames with path traversal components.
                // Valid shard names are plain filenames like "model-00001-of-00004.safetensors".
                ensure!(
                    !shard.contains('/') && !shard.contains('\\') && !shard.contains(".."),
                    "shard filename for key '{k}' contains path traversal: '{shard}'"
                );
                Ok((k.clone(), shard.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            weight_to_shard: entries,
        })
    }

    /// Try to load the index from a model directory.
    ///
    /// Looks for `model.safetensors.index.json` in the given directory.
    /// Returns `None` if the file does not exist (single-shard model).
    pub fn load_from_dir(model_dir: &Path) -> Result<Option<Self>> {
        let index_path = model_dir.join("model.safetensors.index.json");
        if !index_path.exists() {
            return Ok(None);
        }
        let content = std::fs::read_to_string(&index_path)
            .map_err(|e| anyhow::anyhow!("failed to read safetensors index: {e}"))?;
        Self::from_json(&content).map(Some)
    }

    /// Return the set of shard filenames needed for the given filter.
    ///
    /// Only shards containing at least one needed weight key are returned.
    pub fn required_shards(&self, filter: &LayerFilter) -> HashSet<String> {
        let mut shards = HashSet::new();
        for (key, shard) in &self.weight_to_shard {
            if should_load_key(key, filter) {
                shards.insert(shard.clone());
            }
        }
        shards
    }
}

/// Identify required shard file paths for a partial load.
///
/// If an index is available, returns only the paths of shards containing
/// needed weights. If no index exists (single shard), returns all
/// `.safetensors` files in the directory.
pub fn identify_required_shards(
    model_dir: &Path,
    index: Option<&SafeTensorsIndex>,
    filter: &LayerFilter,
) -> Result<Vec<PathBuf>> {
    match index {
        Some(idx) => {
            let needed = idx.required_shards(filter);
            let mut paths: Vec<PathBuf> = Vec::with_capacity(needed.len());
            for name in needed {
                // Defensive: ensure shard name is a plain filename with no
                // directory components (path traversal was already rejected
                // during index parsing, but belt-and-suspenders).
                let shard_path = std::path::Path::new(&name);
                ensure!(
                    shard_path.file_name() == Some(shard_path.as_os_str()),
                    "shard name '{name}' is not a plain filename"
                );
                paths.push(model_dir.join(&name));
            }
            paths.sort();
            Ok(paths)
        }
        None => {
            // No index => single-shard model; load all safetensors files.
            let mut paths = Vec::new();
            let entries = std::fs::read_dir(model_dir)
                .map_err(|e| anyhow::anyhow!("failed to read model dir: {e}"))?;
            for entry in entries {
                let entry = entry.map_err(|e| anyhow::anyhow!("failed to read entry: {e}"))?;
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "safetensors") {
                    paths.push(path);
                }
            }
            paths.sort();
            Ok(paths)
        }
    }
}

// ---------------------------------------------------------------------------
// In-place WeightMap filtering
// ---------------------------------------------------------------------------

/// Remove all weight entries from `weights` that are not needed by `filter`.
///
/// This is useful when the full weight map has already been loaded and we
/// want to free memory for unneeded layers. Returns the number of keys
/// removed.
pub fn filter_weight_map(
    weights: &mut mlxcel_core::weights::WeightMap,
    filter: &LayerFilter,
) -> usize {
    let keys_to_remove: Vec<String> = weights
        .keys()
        .filter(|k| !should_load_key(k, filter))
        .cloned()
        .collect();
    let count = keys_to_remove.len();
    for key in keys_to_remove {
        weights.remove(&key);
    }
    count
}

// ---------------------------------------------------------------------------
// Memory estimation and validation
// ---------------------------------------------------------------------------

/// Estimate the memory required for a partial load using the model profile.
///
/// The estimate is based on the per-layer cost from [`ModelProfile`] plus
/// embedding and lm_head costs if the stage hosts them.
pub fn estimate_partial_memory(filter: &LayerFilter, profile: &ModelProfile) -> u64 {
    let layer_bytes = (filter.num_layers() as u64).saturating_mul(profile.layer_param_bytes);
    let embed_bytes = if filter.has_embedding {
        profile.embedding_param_bytes
    } else {
        0
    };
    let head_bytes = if filter.has_lm_head {
        profile.lm_head_param_bytes
    } else {
        0
    };
    layer_bytes
        .saturating_add(embed_bytes)
        .saturating_add(head_bytes)
}

/// Validate that a partial load fits within available device memory.
///
/// `available_bytes` should reflect the device's free memory after accounting
/// for OS overhead, KV cache budget, etc.
pub fn validate_partial_memory(
    filter: &LayerFilter,
    profile: &ModelProfile,
    available_bytes: u64,
) -> Result<u64> {
    let estimated = estimate_partial_memory(filter, profile);
    ensure!(
        estimated <= available_bytes,
        "partial load requires {estimated} bytes but only {available_bytes} bytes available \
         (layers {}..{}, embedding={}, lm_head={})",
        filter.layer_range.start,
        filter.layer_range.end,
        filter.has_embedding,
        filter.has_lm_head,
    );
    Ok(estimated)
}

#[cfg(test)]
#[path = "partial_loading_tests.rs"]
mod tests;
