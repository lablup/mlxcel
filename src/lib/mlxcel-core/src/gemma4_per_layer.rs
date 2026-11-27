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

//! Gemma 4 `per_layer_config`: the full-attention layers' head size and
//! KV-head count as transformers 5.15+ writes them.
//!
//! Older Gemma 4 configs carry the full-attention geometry in two flat keys,
//! `global_head_dim` and `num_global_key_value_heads`. transformers 5.15+
//! drops those and writes `per_layer_config: {"<layer index>": {"head_dim":
//! H, "num_key_value_heads": K}}` with one entry per full-attention layer. A
//! reader that knows only the flat keys builds those layers with the sliding
//! head size and KV-head count, so weight loading fails on shape or the
//! attention is wrong.
//!
//! Port of `global_attention_from_per_layer_config` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/config.py>
//! (Blaizzy/mlx-vlm#2719).
//!
//! The function lives here rather than in the model crate so the Gemma 4
//! target config there and the MTP drafter config in this crate resolve the
//! block with one rule, the same reason [`crate::rope_parameters`] is here.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

/// The layer type whose shape `per_layer_config` describes.
const FULL_ATTENTION: &str = "full_attention";

/// The only keys an entry may carry.
const HEAD_DIM: &str = "head_dim";
const NUM_KEY_VALUE_HEADS: &str = "num_key_value_heads";

/// Longest stretch of a checkpoint-controlled key echoed into an error.
const MAX_ERROR_KEY_CHARS: usize = 64;

/// The `(head_dim, num_key_value_heads)` every full-attention layer resolves
/// to under `per_layer_config`.
///
/// - `per_layer` absent: `Ok(None)`. The caller keeps `global_head_dim` and
///   `num_global_key_value_heads` as the config wrote them.
/// - Keys are layer indices read as integers, so `"05"` and `"5"` both mean
///   layer 5. An entry for a layer that is out of range or is not
///   `full_attention` is an error, and so is a key other than `head_dim` and
///   `num_key_value_heads`.
/// - Each full-attention layer takes its entry's `head_dim`, else the base
///   `head_dim`, and its entry's `num_key_value_heads`, else the base
///   `num_key_value_heads`. A layer with no entry inherits the base values,
///   not the legacy global ones, so `{}` resolves to
///   `(head_dim, num_key_value_heads)`.
/// - All full-attention layers must resolve to one pair, because the models
///   that read this take one head size and one KV-head count for all of
///   them; differing shapes are an error.
///
/// `Ok(None)` is also returned when `layer_types` names no full-attention
/// layer, since there is then no shape to report (upstream returns
/// `(None, None)` and overwrites nothing).
///
/// Where this is stricter than upstream, which passes these through to a
/// later failure: an index that is not an integer, an entry that is not an
/// object, a value that is not a positive integer, and one layer named by
/// two spellings (`"5"` and `"05"`) are each a named error. Upstream resolves
/// the last case by JSON key order, which a parsed map here does not keep in
/// every build.
///
/// Used by: Gemma 4 (`gemma4::TextConfig::apply_per_layer_config`, and
/// through it the text, VLM, Unified, DiffusionGemma and tensor-parallel
/// loaders), the memory preflight (`execution::kv_arch`), and the Gemma 4
/// assistant drafter (`drafter::gemma4_assistant::config`).
pub fn gemma4_global_attention_from_per_layer_config(
    per_layer: Option<&Map<String, Value>>,
    layer_types: &[&str],
    head_dim: usize,
    num_key_value_heads: usize,
) -> Result<Option<(usize, usize)>, String> {
    let Some(per_layer) = per_layer else {
        return Ok(None);
    };

    let mut layer_overrides: BTreeMap<usize, &Map<String, Value>> = BTreeMap::new();
    for (index, overrides) in per_layer {
        let Ok(layer) = index.trim().parse::<i64>() else {
            return Err(format!(
                "per_layer_config has a layer index {:?} that is not an integer",
                printable(index)
            ));
        };
        let full_attention_layer = usize::try_from(layer)
            .ok()
            .filter(|layer| layer_types.get(*layer).copied() == Some(FULL_ATTENTION));
        let Some(layer) = full_attention_layer else {
            return Err(format!(
                "per_layer_config overrides layer {layer}, which is not a full-attention layer; \
                 this is not supported"
            ));
        };
        let Some(overrides) = overrides.as_object() else {
            return Err(format!(
                "per_layer_config entry for layer {layer} is not an object"
            ));
        };
        let mut unknown: Vec<&str> = overrides
            .keys()
            .map(String::as_str)
            .filter(|key| *key != HEAD_DIM && *key != NUM_KEY_VALUE_HEADS)
            .collect();
        if !unknown.is_empty() {
            unknown.sort_unstable();
            let unknown: Vec<String> = unknown
                .into_iter()
                .map(|key| format!("'{}'", printable(key)))
                .collect();
            return Err(format!(
                "per_layer_config has unsupported keys [{}]",
                unknown.join(", ")
            ));
        }
        if layer_overrides.insert(layer, overrides).is_some() {
            return Err(format!(
                "per_layer_config names layer {layer} more than once"
            ));
        }
    }

    let mut resolved: Option<(usize, usize)> = None;
    for (layer, layer_type) in layer_types.iter().enumerate() {
        if *layer_type != FULL_ATTENTION {
            continue;
        }
        let overrides = layer_overrides.get(&layer).copied();
        let shape = (
            override_or(overrides, layer, HEAD_DIM, head_dim)?,
            override_or(overrides, layer, NUM_KEY_VALUE_HEADS, num_key_value_heads)?,
        );
        match resolved {
            None => resolved = Some(shape),
            Some(first) if first != shape => {
                return Err(
                    "per_layer_config gives full-attention layers different shapes; \
                     this is not supported"
                        .to_string(),
                );
            }
            Some(_) => {}
        }
    }
    Ok(resolved)
}

/// One field of one layer's entry, or `base` when the entry or the field is
/// absent.
fn override_or(
    overrides: Option<&Map<String, Value>>,
    layer: usize,
    key: &str,
    base: usize,
) -> Result<usize, String> {
    let Some(value) = overrides.and_then(|overrides| overrides.get(key)) else {
        return Ok(base);
    };
    value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("per_layer_config layer {layer} `{key}` must be a positive integer"))
}

/// Checkpoint-controlled text made safe for one line of an error message.
fn printable(text: &str) -> String {
    let cleaned: String = text
        .chars()
        .take(MAX_ERROR_KEY_CHARS)
        .map(|c| if c.is_control() { '\u{fffd}' } else { c })
        .collect();
    if text.chars().nth(MAX_ERROR_KEY_CHARS).is_some() {
        format!("{cleaned}...")
    } else {
        cleaned
    }
}

#[cfg(test)]
#[path = "gemma4_per_layer_tests.rs"]
mod tests;
