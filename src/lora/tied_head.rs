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

//! The one verdict on an output-head LoRA pair against a tied-embeddings
//! checkpoint (issue #1577).
//!
//! The single-process loaders validate an adapter against the raw safetensors
//! map, where a tied checkpoint has no `lm_head.weight`, so an `lm_head` pair
//! failed with "no base weight". The pipeline stage loader validates against
//! `models::load_text_weights`, whose `sanitize_tied_embeddings` has already
//! synthesized `lm_head.weight` from the embeddings, so the same pair fused,
//! and then the Llama stage built its output projection from
//! `model.embed_tokens` and never read the fused copy. Same adapter, same
//! checkpoint, a refusal on one path and a silent no-op on the other.
//!
//! The answer chosen is to refuse on every path, and to take the verdict from
//! the checkpoint itself (its `config.json` and its stored tensor names)
//! rather than from either path's weight map, so both reach it identically and
//! before any weight is read. Applying the pair instead would mean writing a
//! delta into a copy of the embedding matrix, which silently unties the input
//! and output projections the checkpoint was trained with. `mlx_lm.lora` only
//! wraps modules inside `model.layers`, so no adapter it trains carries such a
//! pair, which is what makes refusing cheap.
//!
//! Used by: `loading::load_model_with_adapter`,
//! `loading::load_model_with_adapter_specs`, the pipeline Llama stage loader

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::Result;

/// Output-projection weight stems an adapter pair can address, for a text
/// checkpoint and for a VLM that nests its language model.
const OUTPUT_HEADS: [&str; 2] = ["lm_head", "language_model.lm_head"];

/// Largest safetensors header this probe reads. An adapter header is a few
/// kilobytes; anything past this is a malformed file the real loader reports.
const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

/// Refuse an adapter that carries a LoRA pair for the output head of a
/// checkpoint whose output head is tied to its input embeddings.
///
/// A checkpoint counts as tied for a head when its `config.json` sets
/// `tie_word_embeddings: true` (top level or `text_config`), or when it stores
/// no `<head>.weight` tensor of its own, which is the case the loaders fill by
/// copying the embeddings. Only the adapter's safetensors header and the
/// checkpoint's headers are read, never tensor data.
///
/// An adapter whose file cannot be found or parsed is passed through: the
/// adapter loader that runs next reports that with its own wording, and this
/// probe should not become a second, differently worded source of it.
///
/// # Errors
///
/// Returns an error naming every targeted head when the checkpoint is tied,
/// or when the checkpoint's `config.json` or headers cannot be read.
pub fn reject_tied_output_head_adapter(model_dir: &Path, adapter_path: &Path) -> Result<()> {
    let Some(heads) = targeted_output_heads(adapter_path) else {
        return Ok(());
    };
    if heads.is_empty() {
        return Ok(());
    }

    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|err| anyhow::anyhow!("failed to read {}: {err}", config_path.display()))?;
    let config: serde_json::Value =
        serde_json::from_str(&crate::models::sanitize_config_json(&config_str))
            .map_err(|err| anyhow::anyhow!("failed to parse {}: {err}", config_path.display()))?;

    let reason = if declares_tied_embeddings(&config) {
        "its config.json sets tie_word_embeddings".to_string()
    } else {
        let mut missing = Vec::new();
        for head in &heads {
            let stored = format!("{head}.weight");
            let present =
                mlxcel_core::weights::dir_has_tensor_name(model_dir, |name| name == stored)
                    .map_err(anyhow::Error::msg)?;
            if !present {
                missing.push(stored);
            }
        }
        if missing.is_empty() {
            return Ok(());
        }
        format!(
            "it stores no {} of its own and reuses the embedding matrix",
            missing.join(" / ")
        )
    };

    let heads: Vec<&str> = heads.into_iter().collect();
    anyhow::bail!(
        "Adapter at {} carries a LoRA pair for {}, but this checkpoint ties its output projection \
         to its input embeddings ({reason}). An output-head LoRA on a tied checkpoint is refused \
         on every load path: the model reads its logits through the embedding matrix, so the \
         pair would either never reach the forward pass or silently untie the two projections. \
         Remove the {} pair from the adapter (mlx-lm's LoRA training never targets it).",
        adapter_path.display(),
        heads.join(", "),
        heads.join(" / "),
    );
}

/// Whether the config declares the output projection tied, at the top level or
/// in a VLM's `text_config`. Only an explicit `true` counts: an absent key is
/// decided by whether the checkpoint stores the head, because family defaults
/// for the flag differ.
fn declares_tied_embeddings(config: &serde_json::Value) -> bool {
    let explicit = |value: &serde_json::Value| {
        value
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
    };
    explicit(config)
        .or_else(|| config.get("text_config").and_then(explicit))
        .unwrap_or(false)
}

/// The output heads the adapter's tensors address, or `None` when its weights
/// file cannot be located or its header cannot be read.
fn targeted_output_heads(adapter_path: &Path) -> Option<BTreeSet<&'static str>> {
    let weights_path =
        crate::distributed::pipeline::resolve_adapter_weights_path(adapter_path).ok()?;
    let names = safetensors_tensor_names(&weights_path)?;
    Some(
        names
            .iter()
            .filter_map(|name| output_head_of(name))
            .collect(),
    )
}

/// The output head an adapter tensor name addresses, if any.
///
/// Accepts the mlx-lm `.lora_a` / `.lora_b` spelling and a PEFT
/// `.base_layer` wrapper, the same stems the base-weight resolver accepts.
fn output_head_of(tensor_name: &str) -> Option<&'static str> {
    let stem = tensor_name
        .strip_suffix(".lora_a")
        .or_else(|| tensor_name.strip_suffix(".lora_b"))?;
    let stem = stem.strip_suffix(".base_layer").unwrap_or(stem);
    OUTPUT_HEADS.into_iter().find(|head| *head == stem)
}

/// Tensor names from a safetensors header, without reading tensor data.
fn safetensors_tensor_names(path: &Path) -> Option<Vec<String>> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).ok()?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf).ok()?;
    let header_len = u64::from_le_bytes(len_buf);
    if header_len == 0 || header_len > MAX_HEADER_BYTES {
        return None;
    }
    let mut header = vec![0u8; usize::try_from(header_len).ok()?];
    file.read_exact(&mut header).ok()?;
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&header).ok()?;
    Some(
        header
            .into_iter()
            .map(|(name, _)| name)
            .filter(|name| name != "__metadata__")
            .collect(),
    )
}

#[cfg(test)]
#[path = "tied_head_tests.rs"]
mod tests;
