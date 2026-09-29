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

//! Loaded Nemotron VoiceChat checkpoint: the four networks plus tokenizer.
//!
//! Port of `Model` in upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_voicechat/model.py.
//! The duplex timeline loops live in [`super::session`]; this module owns
//! loading and the `LanguageModel` delegation that keeps the checkpoint
//! usable by generic tooling (warmup, capability probes).
//!
//! Used by: `LoadedModel::NemotronVoiceChat`, the `mlxcel generate --audio`
//! VoiceChat branch

use std::path::Path;

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::VoiceChatConfig;
use super::llm::VoiceChatLanguageModel;
use crate::models::nemotron_h::NemotronHConfig;
use crate::tokenizer::{MlxcelTokenizer, load_tokenizer};

/// A loaded VoiceChat checkpoint.
pub struct NemotronVoiceChatModel {
    pub(crate) config: VoiceChatConfig,
    pub(crate) lm: VoiceChatLanguageModel,
    pub(crate) tokenizer: MlxcelTokenizer,
}

impl NemotronVoiceChatModel {
    /// Load every network of the checkpoint at `model_path`.
    pub fn load(model_path: &Path) -> anyhow::Result<Self> {
        let config_text = std::fs::read_to_string(model_path.join("config.json"))?;
        let config = VoiceChatConfig::from_json(&config_text).map_err(anyhow::Error::msg)?;
        let text_config: NemotronHConfig = serde_json::from_value(config.text_config.clone())
            .map_err(|e| anyhow::anyhow!("nemotron_voicechat text_config: {e}"))?;
        let mut weights =
            mlxcel_core::weights::load_weights_from_dir(model_path).map_err(anyhow::Error::msg)?;
        let lm = VoiceChatLanguageModel::from_weights(
            text_config,
            config.default_quantization(),
            &mut weights,
            config.function_channel_weight,
        )
        .map_err(anyhow::Error::msg)?;
        let tokenizer = load_tokenizer(model_path)?;
        Ok(Self {
            config,
            lm,
            tokenizer,
        })
    }

    /// Timeline-level configuration.
    pub fn config(&self) -> &VoiceChatConfig {
        &self.config
    }

    /// The LLM tokenizer (`tokenizer.json`).
    pub fn tokenizer(&self) -> &MlxcelTokenizer {
        &self.tokenizer
    }
}

impl LanguageModel for NemotronVoiceChatModel {
    /// Text-only Nemotron-H forward over the shared embedding table and the
    /// text head. It exists for trait completeness (warmup, tooling); the
    /// duplex timeline needs the fused audio channel and runs through
    /// [`super::session`] instead, which the CLI routes to before the
    /// autoregressive loop.
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward(self.lm.backbone(), input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        LanguageModel::make_caches(self.lm.backbone())
    }

    fn num_layers(&self) -> usize {
        LanguageModel::num_layers(self.lm.backbone())
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![self.config.eos_token_id]
    }

    fn supports_batching(&self) -> bool {
        false
    }
}
