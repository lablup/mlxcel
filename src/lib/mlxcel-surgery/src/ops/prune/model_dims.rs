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

//! Structural dimensions read from `config.json` for the prune op.
//!
//! Centralizes the model-shape facts the pruners depend on: number of
//! attention heads, number of KV heads (for GQA reasoning), head
//! dimension, MLP intermediate size, and number of transformer
//! blocks. Also handles VLM nesting (where the language-model
//! dimensions live under `text_config`).
//!
//! Used by: `super::PruneOp::apply`, the per-granularity pruners in
//! `super::granularity`.

use super::PruneSelector;
use crate::SurgeryError;

/// Structural dimensions the op needs to slice tensors correctly.
///
/// All fields are filled from the parsed `config.json` passed through
/// [`crate::SurgeryOp::apply`]. VLM configs may nest the language-
/// model dimensions under `text_config`; this helper looks there if
/// the top-level keys are absent so the same op spec works on both.
///
/// `head_dim` may be specified explicitly (Llama 4, DeepSeek V3, some
/// Mixtral checkpoints) or derived as `hidden_size / num_heads` when
/// absent (the common case).
#[derive(Debug, Clone, Copy)]
pub(super) struct ModelDims {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
}

impl ModelDims {
    pub(super) fn from_config(cfg: &serde_json::Value) -> Result<Self, SurgeryError> {
        // Prefer top-level fields; fall back to `text_config` for VLMs.
        let root = if cfg.get("num_attention_heads").is_some() {
            cfg
        } else if let Some(tc) = cfg.get("text_config") {
            tc
        } else {
            cfg
        };

        let num_heads = read_usize(root, "num_attention_heads")?;
        // num_key_value_heads defaults to num_heads for MHA models.
        let num_kv_heads = match read_usize_opt(root, "num_key_value_heads")? {
            Some(n) => n,
            None => num_heads,
        };
        let hidden_size = read_usize_opt(root, "hidden_size")?;
        let head_dim = match read_usize_opt(root, "head_dim")? {
            Some(d) => d,
            None => {
                let h = hidden_size.ok_or_else(|| {
                    SurgeryError::Other(anyhow::anyhow!(
                        "prune: config.json must provide head_dim or hidden_size + num_attention_heads"
                    ))
                })?;
                if num_heads == 0 {
                    return Err(SurgeryError::Other(anyhow::anyhow!(
                        "prune: num_attention_heads must be > 0"
                    )));
                }
                if !h.is_multiple_of(num_heads) {
                    return Err(SurgeryError::Other(anyhow::anyhow!(
                        "prune: hidden_size ({h}) not divisible by num_attention_heads ({num_heads})"
                    )));
                }
                h / num_heads
            }
        };
        let intermediate_size = read_usize_opt(root, "intermediate_size")?.unwrap_or(0);
        let num_hidden_layers = read_usize_opt(root, "num_hidden_layers")?.unwrap_or(usize::MAX);

        if num_heads == 0 {
            return Err(SurgeryError::Other(anyhow::anyhow!(
                "prune: num_attention_heads must be > 0"
            )));
        }
        if num_kv_heads == 0 {
            return Err(SurgeryError::Other(anyhow::anyhow!(
                "prune: num_key_value_heads must be > 0"
            )));
        }
        if head_dim == 0 {
            return Err(SurgeryError::Other(anyhow::anyhow!(
                "prune: head_dim must be > 0"
            )));
        }

        Ok(Self {
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            num_hidden_layers,
        })
    }

    /// `true` when the model uses Grouped-Query Attention
    /// (`num_kv_heads < num_heads`). Diagnostic only — the op already
    /// chooses the safe GQA policy unconditionally.
    pub(super) fn is_gqa(&self) -> bool {
        self.num_kv_heads < self.num_heads
    }
}

fn read_usize(cfg: &serde_json::Value, key: &str) -> Result<usize, SurgeryError> {
    read_usize_opt(cfg, key)?.ok_or_else(|| {
        SurgeryError::Other(anyhow::anyhow!(
            "prune: config.json missing required field `{key}`"
        ))
    })
}

fn read_usize_opt(cfg: &serde_json::Value, key: &str) -> Result<Option<usize>, SurgeryError> {
    let Some(v) = cfg.get(key) else {
        return Ok(None);
    };
    let n = v.as_u64().ok_or_else(|| {
        SurgeryError::Other(anyhow::anyhow!(
            "prune: config.json `{key}` must be an unsigned integer; got {v}"
        ))
    })?;
    Ok(Some(n as usize))
}

/// Validate that every id in a [`PruneSelector`] is within the model's
/// dimensions. Called by [`super::PruneOp::apply`] after parsing the
/// config so the per-granularity pruners can assume all ids are valid.
pub(super) fn validate_ids(
    selector: &PruneSelector,
    model: &ModelDims,
) -> Result<(), SurgeryError> {
    match selector {
        PruneSelector::Layer { layer_ids } => {
            for &id in layer_ids {
                if id >= model.num_hidden_layers {
                    return Err(SurgeryError::Other(anyhow::anyhow!(
                        "prune: layer_id {id} out of range (num_hidden_layers={})",
                        model.num_hidden_layers
                    )));
                }
            }
        }
        PruneSelector::AttentionHead { head_ids } => {
            for &id in head_ids {
                if id >= model.num_heads {
                    return Err(SurgeryError::Other(anyhow::anyhow!(
                        "prune: head_id {id} out of range (num_attention_heads={})",
                        model.num_heads
                    )));
                }
            }
        }
        PruneSelector::MlpChannel { channel_ids } => {
            if model.intermediate_size == 0 {
                return Err(SurgeryError::Other(anyhow::anyhow!(
                    "prune: mlp_channel granularity requires intermediate_size in config.json"
                )));
            }
            for &id in channel_ids {
                if id >= model.intermediate_size {
                    return Err(SurgeryError::Other(anyhow::anyhow!(
                        "prune: channel_id {id} out of range (intermediate_size={})",
                        model.intermediate_size
                    )));
                }
            }
        }
    }
    Ok(())
}
