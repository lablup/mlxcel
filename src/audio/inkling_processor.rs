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

//! Inkling host processor configuration and prompt-token metadata.

use std::path::Path;

use serde::Deserialize;

use super::inkling_dmel::{
    DEFAULT_DMEL_MAX, DEFAULT_DMEL_MIN, DEFAULT_MEL_VOCAB_SIZE, InklingFeatureExtractorConfig,
};

fn default_num_dmel_bins() -> usize {
    DEFAULT_MEL_VOCAB_SIZE
}
fn default_dmel_min() -> f32 {
    DEFAULT_DMEL_MIN
}
fn default_dmel_max() -> f32 {
    DEFAULT_DMEL_MAX
}
fn default_audio_token() -> String {
    "<|unused_200053|>".into()
}
fn default_audio_bos_token() -> String {
    "<|content_audio_input|>".into()
}
fn default_audio_end_token() -> String {
    "<|audio_end|>".into()
}

/// Inkling processor fields used by host preprocessing and prompt expansion.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct InklingProcessorConfig {
    #[serde(default)]
    pub feature_extractor: InklingFeatureExtractorConfig,
    #[serde(default = "default_num_dmel_bins")]
    pub num_dmel_bins: usize,
    #[serde(default = "default_dmel_min")]
    pub dmel_min_value: f32,
    #[serde(default = "default_dmel_max")]
    pub dmel_max_value: f32,
    #[serde(default = "default_audio_token")]
    pub audio_token: String,
    #[serde(default = "default_audio_bos_token")]
    pub audio_bos_token: String,
    #[serde(default = "default_audio_end_token")]
    pub audio_end_token: String,
}

impl Default for InklingProcessorConfig {
    fn default() -> Self {
        Self {
            feature_extractor: InklingFeatureExtractorConfig::default(),
            num_dmel_bins: default_num_dmel_bins(),
            dmel_min_value: default_dmel_min(),
            dmel_max_value: default_dmel_max(),
            audio_token: default_audio_token(),
            audio_bos_token: default_audio_bos_token(),
            audio_end_token: default_audio_end_token(),
        }
    }
}

impl InklingProcessorConfig {
    pub fn from_model_path(model_path: &Path) -> Result<Self, String> {
        let path = model_path.join("processor_config.json");
        let raw = match std::fs::read_to_string(&path) {
            Ok(raw) => raw,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self::default());
            }
            Err(error) => {
                return Err(format!(
                    "Failed to read Inkling processor config {}: {error}",
                    path.display()
                ));
            }
        };
        let config: Self = serde_json::from_str(&raw)
            .map_err(|error| format!("Failed to parse Inkling processor_config.json: {error}"))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.feature_extractor.validate()?;
        if self.num_dmel_bins != DEFAULT_MEL_VOCAB_SIZE {
            return Err(format!(
                "Inkling num_dmel_bins must be {DEFAULT_MEL_VOCAB_SIZE}, got {}",
                self.num_dmel_bins
            ));
        }
        if self.dmel_min_value != DEFAULT_DMEL_MIN || self.dmel_max_value != DEFAULT_DMEL_MAX {
            return Err(format!(
                "Inkling dMel range must be [{DEFAULT_DMEL_MIN}, {DEFAULT_DMEL_MAX}]"
            ));
        }
        if self.audio_token.is_empty()
            || self.audio_bos_token.is_empty()
            || self.audio_end_token.is_empty()
        {
            return Err("Inkling audio processor tokens must not be empty".into());
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "inkling_processor_tests.rs"]
mod tests;
