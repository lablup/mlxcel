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

//! Configuration of the Qwen3.8-Flash-Next native MTP drafter
//! (`model_type: "qwen4_exp_mtp"`, issue #2880).
//!
//! Mirrors upstream `Qwen4ExpMTPConfig`
//! (`mlx_vlm/speculative/drafters/qwen4_exp_mtp/config.py`): a small top
//! level (`block_size`, `tie_word_embeddings`) plus a nested `text_config`
//! that is a copy of the target's text config, as `mlxcel split-mtp`'s
//! upstream counterpart (`split.py`) writes it.
//!
//! ## The head's layer layout
//!
//! Upstream builds the head from the target text config with
//! `num_hidden_layers = 1`, `layer_types = ["qwen_sparse_attention"]`,
//! `full_attention_interval = 1` and `ple_layer_ids = []`
//! (`qwen4_exp_mtp.py`), and reads nothing from the target config's `mtp`
//! sub-object. The released `Qwen3.8-Flash-Next` config carries that
//! sub-object with `"layer_types": ["full_attention"]`, which is the
//! checkpoint spelling of the same QSA layer (`Qwen4ExpConfig::resolve`
//! renames it), `"num_hidden_layers": 1` and `"hybrid": true`; the two
//! descriptions agree, and this port follows the code: the overrides below,
//! not the sub-object.

use serde_json::{Map, Value, json};

use mlxcel_core::drafter::{DrafterError, QWEN4_EXP_MTP_MODEL_TYPE};

use super::qwen4_exp::config::QWEN_SPARSE_ATTENTION;
use super::qwen4_exp::{Qwen4ExpConfig, ResolvedQwen4ExpConfig};

/// The one head depth this port implements. A deeper head needs one cache
/// and one capture per layer and is refused by name until it exists.
const SUPPORTED_MTP_DEPTH: usize = 1;

#[derive(Debug, Clone)]
pub struct Qwen4ExpMtpConfig {
    /// Total verify-round budget including the bonus token. Upstream
    /// defaults it to `mtp_num_hidden_layers + 1` (so a one-layer head
    /// drafts one token per round at its configured depth); the runtime
    /// expands toward the requested `--draft-block-size` adaptively, since the
    /// drafter does not prefer the requested size (upstream
    /// `prefer_requested_block_size = False`).
    pub block_size: usize,
    pub tie_word_embeddings: bool,
    /// The target text config resolved with the head's layer layout: one QSA
    /// layer at index 0, no PLE.
    pub layer: ResolvedQwen4ExpConfig,
}

impl Qwen4ExpMtpConfig {
    /// Parse a drafter `config.json`.
    pub fn from_json(full: &Value) -> Result<Self, String> {
        let object = full
            .as_object()
            .ok_or_else(|| "Qwen4ExpMtpConfig: config.json must be a JSON object".to_string())?;
        let model_type = object
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or(QWEN4_EXP_MTP_MODEL_TYPE);
        if model_type != QWEN4_EXP_MTP_MODEL_TYPE {
            return Err(format!(
                "Qwen4ExpMtpConfig: model_type must be {QWEN4_EXP_MTP_MODEL_TYPE:?}, got {model_type:?}"
            ));
        }
        let mut text: Map<String, Value> = object
            .get("text_config")
            .and_then(Value::as_object)
            .cloned()
            .ok_or_else(|| {
                "Qwen4ExpMtpConfig.text_config must be set: the head is sized from the target \
                 text config"
                    .to_string()
            })?;
        let depth = text
            .get("mtp_num_hidden_layers")
            .map(|value| {
                value.as_u64().map(|d| d as usize).ok_or_else(|| {
                    format!(
                        "Qwen4ExpMtpConfig.text_config.mtp_num_hidden_layers must be a positive \
                         integer, got {value}"
                    )
                })
            })
            .transpose()?
            .unwrap_or(SUPPORTED_MTP_DEPTH);
        if depth != SUPPORTED_MTP_DEPTH {
            return Err(format!(
                "Qwen4ExpMtpConfig.text_config.mtp_num_hidden_layers = {depth} is not supported: \
                 this drafter implements the one-layer head the released checkpoint ships \
                 (a multi-layer head needs one cache and one rollback capture per layer)"
            ));
        }
        let block_size = match object.get("block_size") {
            None | Some(Value::Null) => depth + 1,
            Some(value) => value
                .as_u64()
                .map(|b| b as usize)
                .filter(|&b| b >= 2)
                .ok_or_else(|| {
                    format!("Qwen4ExpMtpConfig.block_size must be an integer >= 2, got {value}")
                })?,
        };
        // A top-level `quantization` block applies to the head's modules when
        // the text config carries none, as the target loader inherits it.
        if !text.contains_key("quantization")
            && let Some(quantization) = object.get("quantization")
        {
            text.insert("quantization".to_string(), quantization.clone());
        }
        text.insert("num_hidden_layers".to_string(), json!(1));
        text.insert("layer_types".to_string(), json!([QWEN_SPARSE_ATTENTION]));
        text.insert("full_attention_interval".to_string(), json!(1));
        text.insert("ple_layer_ids".to_string(), json!([]));
        text.remove("ple_storage");
        let tie_word_embeddings = text
            .get("tie_word_embeddings")
            .and_then(Value::as_bool)
            .or_else(|| object.get("tie_word_embeddings").and_then(Value::as_bool))
            .unwrap_or(false);
        let config: Qwen4ExpConfig = serde_json::from_value(Value::Object(text))
            .map_err(|error| format!("Qwen4ExpMtpConfig.text_config: {error}"))?;
        let layer = config
            .resolve()
            .map_err(|error| format!("Qwen4ExpMtpConfig.text_config: {error}"))?;
        Ok(Self {
            block_size,
            tie_word_embeddings,
            layer,
        })
    }

    /// Read and parse `<dir>/config.json`.
    pub fn from_dir(path: &std::path::Path) -> Result<Self, DrafterError> {
        let cfg_path = path.join("config.json");
        let bytes = std::fs::read(&cfg_path).map_err(|e| DrafterError::ConfigIo {
            path: cfg_path.display().to_string(),
            source: e,
        })?;
        let value: Value =
            serde_json::from_slice(&bytes).map_err(|e| DrafterError::ConfigParse {
                path: cfg_path.display().to_string(),
                source: e,
            })?;
        Self::from_json(&value).map_err(DrafterError::Config)
    }

    pub fn hidden_size(&self) -> usize {
        self.layer.raw.hidden_size
    }

    pub fn hc_count(&self) -> usize {
        self.layer.raw.hc_count
    }

    /// Width of the residual streams the head consumes and produces.
    pub fn stream_width(&self) -> usize {
        self.hc_count() * self.hidden_size()
    }

    pub fn vocab_size(&self) -> usize {
        self.layer.raw.vocab_size
    }
}
