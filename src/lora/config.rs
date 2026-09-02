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

//! LoRA adapter configuration parsing
//!
//! Parses adapter_config.json files from HuggingFace-compatible LoRA adapters.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// LoRA-specific parameters
#[derive(Debug, Clone, Deserialize)]
pub struct LoRAParameters {
    /// Low-rank dimension (r)
    #[serde(default = "default_rank")]
    pub rank: usize,

    /// Dropout rate during training (ignored for inference)
    #[serde(default)]
    pub dropout: f32,

    /// Scaling factor for LoRA updates
    #[serde(default = "default_scale")]
    pub scale: f32,

    /// Alpha parameter (alternative to scale: scale = alpha / rank)
    #[serde(rename = "lora_alpha")]
    pub alpha: Option<f32>,
}

fn default_rank() -> usize {
    8
}

fn default_scale() -> f32 {
    20.0
}

impl LoRAParameters {
    /// Get the effective scale factor
    /// If alpha is provided, scale = alpha / rank
    /// Otherwise, use the explicit scale value
    pub fn effective_scale(&self) -> f32 {
        if let Some(alpha) = self.alpha {
            alpha / self.rank as f32
        } else {
            self.scale
        }
    }
}

impl Default for LoRAParameters {
    fn default() -> Self {
        Self {
            rank: default_rank(),
            dropout: 0.0,
            scale: default_scale(),
            alpha: None,
        }
    }
}

/// Fine-tuning type
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum FineTuneType {
    /// Low-Rank Adaptation
    #[default]
    LoRA,
    /// Weight-Decomposed Low-Rank Adaptation (magnitude-preserving)
    DoRA,
    /// Full fine-tuning (all weights trainable)
    Full,
}

/// Adapter configuration from adapter_config.json
#[derive(Debug, Clone, Deserialize)]
pub struct AdapterConfig {
    /// Base model identifier
    #[serde(default)]
    pub model: Option<String>,

    /// Type of fine-tuning applied
    #[serde(default)]
    pub fine_tune_type: FineTuneType,

    /// Number of layers that have adapters (-1 for all)
    #[serde(default = "default_num_layers")]
    pub num_layers: i32,

    /// LoRA-specific parameters
    #[serde(default)]
    pub lora_parameters: LoRAParameters,

    /// Target modules for LoRA (e.g., ["q_proj", "v_proj"])
    #[serde(default)]
    pub target_modules: Option<Vec<String>>,

    /// Training batch size (ignored for inference)
    #[serde(default)]
    pub batch_size: Option<usize>,

    /// Training iterations (ignored for inference)
    #[serde(default)]
    pub iters: Option<usize>,
}

fn default_num_layers() -> i32 {
    -1 // All layers
}

impl AdapterConfig {
    /// Load adapter configuration from a directory
    pub fn load(adapter_path: &Path) -> Result<Self> {
        let config_path = adapter_path.join("adapter_config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read adapter config: {:?}", config_path))?;

        let config: AdapterConfig = serde_json::from_str(&config_str)
            .with_context(|| "Failed to parse adapter_config.json")?;

        Ok(config)
    }

    /// Whether this adapter is plain LoRA, the only kind this build can apply.
    ///
    /// DoRA is deliberately excluded. This used to be `is_lora()`, which
    /// answered `true` for DoRA as well, so a DoRA adapter was accepted and
    /// then applied as if its magnitude vectors did not exist, producing
    /// weights that match neither the base model nor the fine-tune (issue
    /// #1328). Rejecting it is [`crate::lora::AdapterConfig`]'s job only in the
    /// sense of reporting the type; the message lives with the loaders.
    pub fn is_fusable_lora(&self) -> bool {
        self.fine_tune_type == FineTuneType::LoRA
    }

    /// Get the effective LoRA scale
    pub fn effective_scale(&self) -> f32 {
        self.lora_parameters.effective_scale()
    }

    /// Get the LoRA rank
    pub fn rank(&self) -> usize {
        self.lora_parameters.rank
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_lora_params() {
        let params = LoRAParameters::default();
        assert_eq!(params.rank, 8);
        assert_eq!(params.scale, 20.0);
        assert_eq!(params.dropout, 0.0);
    }

    #[test]
    fn test_effective_scale_with_alpha() {
        let params = LoRAParameters {
            rank: 16,
            alpha: Some(32.0),
            scale: 20.0, // Should be ignored
            dropout: 0.0,
        };
        assert_eq!(params.effective_scale(), 2.0); // 32 / 16 = 2
    }

    #[test]
    fn test_parse_minimal_config() {
        let json = r#"{"fine_tune_type": "lora"}"#;
        let config: AdapterConfig = serde_json::from_str(json).unwrap();
        assert!(config.is_fusable_lora());
        assert_eq!(config.num_layers, -1);
    }

    #[test]
    fn dora_and_full_are_not_fusable_lora() {
        for json in [
            r#"{"fine_tune_type": "dora"}"#,
            r#"{"fine_tune_type": "full"}"#,
        ] {
            let config: AdapterConfig = serde_json::from_str(json).unwrap();
            assert!(
                !config.is_fusable_lora(),
                "must not be treated as plain LoRA: {json}"
            );
        }
    }
}
