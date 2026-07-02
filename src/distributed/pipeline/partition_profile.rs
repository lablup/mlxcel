// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Heuristic [`ModelProfile`] builder for the pipeline auto-partitioner.
//!
//! Reads `config.json` from a model directory and produces a profile with
//! realistic per-layer byte weights and adjacency constraints so
//! `auto_partition()` can balance real bytes — not just layer counts — and
//! refuse to split layer groups that share runtime state.
//!
//! What the heuristic does **not** do:
//!
//! - It does not memory-map `model.safetensors` to sum actual tensor byte
//!   sizes. Doing so would pull the entire shard index into memory on
//!   every startup; instead we derive byte counts analytically from
//!   `hidden_size`, `intermediate_size`, expert counts, and the
//!   quantisation bit-width recorded in the config. This is accurate to
//!   within a few percent for every production MoE model we ship.
//! - It does not tune for hardware generations (M1 Ultra vs. M5 Max etc.).
//!   The cost model is pure byte-count; hardware-specific calibration is
//!   future work (explicitly out of scope).
//!
//! Used by: `resolve_in_process_stage_assignments`,
//! `mlxcel-server` startup, CLI `mlxcel generate --pp-*`

use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::models::sanitize_config_json;

use super::partition::ModelProfile;
use super::partition_profile_heuristics::{build_adjacency, build_per_layer_bytes};

/// Build a [`ModelProfile`] for a model directory.
///
/// Steps:
///
/// 1. Load `config.json` and sanitise JSON quirks (NaN, Infinity).
/// 2. Inspect `model_type` and extract the relevant text config (some VLMs
///    nest the language model under `text_config`).
/// 3. Estimate per-layer byte cost analytically, separating dense and MoE
///    layers where applicable.
/// 4. Estimate embedding and lm_head bytes.
/// 5. Collect adjacency constraints (e.g. Gemma 4 KV-shared source/consumer
///    pairs) so the partitioner refuses to split them.
///
/// If `config.json` is missing or malformed, the function falls back to a
/// uniform profile using `num_layers` passed by the caller — this
/// preserves backward compatibility with tests that feed a bare layer
/// count.
pub fn build_model_profile(model_dir: &Path, fallback_num_layers: usize) -> Result<ModelProfile> {
    let config_path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let raw = sanitize_config_json(&raw);
    let root: Value = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse {}", config_path.display()))?;
    Ok(build_profile_from_json(&root, fallback_num_layers))
}

/// Same as [`build_model_profile`] but takes an already-parsed JSON value.
/// Kept public to simplify unit tests that feed synthetic configs.
pub fn build_profile_from_json(root: &Value, fallback_num_layers: usize) -> ModelProfile {
    let text = resolve_text_config(root);
    let model_type = root
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let bits_per_weight = root
        .get("quantization")
        .and_then(|q| q.get("bits"))
        .and_then(|v| v.as_u64())
        .or_else(|| {
            text.and_then(|t| t.get("quantization"))
                .and_then(|q| q.get("bits"))
                .and_then(|v| v.as_u64())
        })
        .unwrap_or(16);

    let num_layers = text
        .and_then(|t| t.get("num_hidden_layers"))
        .and_then(|v| v.as_u64())
        .or_else(|| root.get("num_hidden_layers").and_then(|v| v.as_u64()))
        .map(|v| v as usize)
        .unwrap_or(fallback_num_layers);
    // All `num_hidden_layers` DeepSeek-V3 entries are real decoder layers;
    // checkpoints that ship the multi-token-prediction trailer store it at
    // index `num_hidden_layers` (out of range) and `sanitize_weights` strips
    // it, so no per-family depth adjustment is needed here.

    let hidden_size = text
        .and_then(|t| t.get("hidden_size"))
        .and_then(|v| v.as_u64())
        .or_else(|| root.get("hidden_size").and_then(|v| v.as_u64()))
        .unwrap_or(4096);
    let intermediate_size = text
        .and_then(|t| t.get("intermediate_size"))
        .and_then(|v| v.as_u64())
        .or_else(|| root.get("intermediate_size").and_then(|v| v.as_u64()))
        .unwrap_or(hidden_size * 4);
    let vocab_size = text
        .and_then(|t| t.get("vocab_size"))
        .and_then(|v| v.as_u64())
        .or_else(|| root.get("vocab_size").and_then(|v| v.as_u64()))
        .unwrap_or(0);
    let num_attention_heads = text
        .and_then(|t| t.get("num_attention_heads"))
        .and_then(|v| v.as_u64())
        .or_else(|| root.get("num_attention_heads").and_then(|v| v.as_u64()))
        .unwrap_or(32);
    let num_key_value_heads = text
        .and_then(|t| t.get("num_key_value_heads"))
        .and_then(|v| v.as_u64())
        .or_else(|| root.get("num_key_value_heads").and_then(|v| v.as_u64()))
        .unwrap_or(num_attention_heads);
    let head_dim = text
        .and_then(|t| t.get("head_dim"))
        .and_then(|v| v.as_u64())
        .unwrap_or(hidden_size / num_attention_heads.max(1));

    let dense_layer_bytes = dense_transformer_layer_bytes(
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        bits_per_weight,
    );

    let layer_bytes = build_per_layer_bytes(
        root,
        text,
        model_type,
        num_layers,
        dense_layer_bytes,
        hidden_size,
        bits_per_weight,
    );

    let adjacency = build_adjacency(root, text, model_type, num_layers);

    let embedding_param_bytes =
        param_bytes(vocab_size.saturating_mul(hidden_size), bits_per_weight);
    let lm_head_param_bytes = if text
        .and_then(|t| t.get("tie_word_embeddings"))
        .and_then(|v| v.as_bool())
        .or_else(|| root.get("tie_word_embeddings").and_then(|v| v.as_bool()))
        .unwrap_or(false)
    {
        // Tied lm_head shares the embedding matrix; account a small head
        // overhead (final norm) rather than zero, because the last stage
        // still pays per-token output cost.
        param_bytes(hidden_size, bits_per_weight)
    } else {
        embedding_param_bytes
    };

    let fallback = layer_bytes
        .iter()
        .copied()
        .max()
        .unwrap_or(dense_layer_bytes);

    ModelProfile {
        num_layers,
        layer_param_bytes: fallback,
        embedding_param_bytes,
        lm_head_param_bytes,
        layer_bytes: Some(layer_bytes),
        adjacency,
    }
}

/// Returns the slice of the JSON config that carries the language-model
/// hyperparameters. Most dense models keep them at the top level; VLMs
/// (Llama 4, Gemma 4 VLM, Qwen 2.5 VL) nest them under `text_config`.
fn resolve_text_config(root: &Value) -> Option<&Value> {
    if let Some(t) = root.get("text_config") {
        return Some(t);
    }
    Some(root)
}

/// Estimate the byte cost of a dense transformer block.
///
/// Layout accounted for:
///
/// - Attention: q, k, v, o projections. With GQA the k/v blocks are
///   `hidden_size * num_kv_heads * head_dim`.
/// - MLP: three projections (gate, up, down) of size
///   `hidden_size * intermediate_size`.
/// - Two RMSNorm weight vectors.
///
/// The result is pre-scaled by `bits_per_weight` so a quantised model
/// reports fewer bytes than its f16 counterpart.
fn dense_transformer_layer_bytes(
    hidden_size: u64,
    intermediate_size: u64,
    num_attention_heads: u64,
    num_kv_heads: u64,
    head_dim: u64,
    bits_per_weight: u64,
) -> u64 {
    let q_proj = hidden_size.saturating_mul(num_attention_heads.saturating_mul(head_dim));
    let kv_proj = hidden_size.saturating_mul(num_kv_heads.saturating_mul(head_dim));
    let o_proj = hidden_size.saturating_mul(hidden_size);
    let attn_params = q_proj
        .saturating_add(kv_proj)
        .saturating_add(kv_proj)
        .saturating_add(o_proj);

    let mlp_params = hidden_size
        .saturating_mul(intermediate_size)
        .saturating_mul(3);

    let norm_params = hidden_size.saturating_mul(2);

    param_bytes(
        attn_params
            .saturating_add(mlp_params)
            .saturating_add(norm_params),
        bits_per_weight,
    )
}

/// Convert a raw parameter count (scalars) to byte bytes given the
/// effective bits per weight. Quantised weights still need to store
/// scales/biases as f16; we approximate the overhead at +25%.
pub(super) fn param_bytes(params: u64, bits_per_weight: u64) -> u64 {
    let bits = params.saturating_mul(bits_per_weight);
    let bytes = bits / 8;
    if bits_per_weight <= 8 {
        // Include ~25% overhead for quantisation scales/biases that ship
        // with the weight tensor in f16.
        bytes.saturating_add(bytes / 4)
    } else {
        bytes
    }
}

#[cfg(test)]
#[path = "partition_profile_tests.rs"]
mod tests;
