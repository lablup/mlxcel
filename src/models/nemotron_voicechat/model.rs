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
use serde::de::DeserializeOwned;
use serde_json::Value;

use super::config::VoiceChatConfig;
use super::llm::VoiceChatLanguageModel;
use super::tts::{RvqEarTtsModel, SpeechDecoderAssets, TtsConfig};
use crate::audio::fastconformer::{ConformerArgs, VoiceChatPerception};
use crate::audio::nemotron_codec::{CodecConfig, NemotronCodec};
use crate::audio::nemotron_mel::MelArgs;
use crate::audio::rnnt::{JointArgs, PredictArgs, RnntDecoder, load_vocabulary_json};
use crate::models::nemotron_h::NemotronHConfig;
use crate::tokenizer::{MlxcelTokenizer, load_tokenizer};

/// Default RNNT symbol cap per encoder frame (`audio_config.max_symbols`).
const DEFAULT_MAX_SYMBOLS: usize = 10;

/// A loaded VoiceChat checkpoint.
pub struct NemotronVoiceChatModel {
    pub(crate) config: VoiceChatConfig,
    pub(crate) mel_args: MelArgs,
    pub(crate) perception: VoiceChatPerception,
    pub(crate) rnnt: RnntDecoder,
    pub(crate) rnnt_vocabulary: Vec<String>,
    pub(crate) max_symbols: usize,
    pub(crate) lm: VoiceChatLanguageModel,
    pub(crate) tts: RvqEarTtsModel,
    pub(crate) tts_config: TtsConfig,
    pub(crate) assets: SpeechDecoderAssets,
    pub(crate) codec: NemotronCodec,
    pub(crate) tokenizer: MlxcelTokenizer,
}

/// Deserialize `value[key]` into `T`, or `T::default()` when absent.
fn sub_config<T: DeserializeOwned + Default>(value: &Value, key: &str) -> anyhow::Result<T> {
    match value.get(key) {
        Some(v) if !v.is_null() => serde_json::from_value(v.clone())
            .map_err(|e| anyhow::anyhow!("nemotron_voicechat {key}: {e}")),
        _ => Ok(T::default()),
    }
}

fn whole_config<T: DeserializeOwned + Default>(value: &Value, name: &str) -> anyhow::Result<T> {
    if value.is_null() {
        return Ok(T::default());
    }
    serde_json::from_value(value.clone())
        .map_err(|e| anyhow::anyhow!("nemotron_voicechat {name}: {e}"))
}

impl NemotronVoiceChatModel {
    /// Load every network of the checkpoint at `model_path`.
    pub fn load(model_path: &Path) -> anyhow::Result<Self> {
        let config_text = std::fs::read_to_string(model_path.join("config.json"))?;
        let config = VoiceChatConfig::from_json(&config_text).map_err(anyhow::Error::msg)?;
        let text_config: NemotronHConfig = serde_json::from_value(config.text_config.clone())
            .map_err(|e| anyhow::anyhow!("nemotron_voicechat text_config: {e}"))?;
        let audio = &config.audio_config;
        let mel_args: MelArgs = sub_config(audio, "preprocessor")?;
        let conformer_args: ConformerArgs = sub_config(audio, "encoder")?;
        let predict_args: PredictArgs = sub_config(audio, "decoder")?;
        let joint_args: JointArgs = sub_config(audio, "joint")?;
        let max_symbols = audio
            .get("max_symbols")
            .and_then(Value::as_u64)
            .map_or(DEFAULT_MAX_SYMBOLS, |v| v as usize);
        let tts_config: TtsConfig = whole_config(&config.tts_config, "tts_config")?;
        let codec_config: CodecConfig = whole_config(&config.codec_config, "codec_config")?;

        let mut weights =
            mlxcel_core::weights::load_weights_from_dir(model_path).map_err(anyhow::Error::msg)?;
        let msg = anyhow::Error::msg;
        let perception =
            VoiceChatPerception::from_weights(&weights, "stt_model.perception", &conformer_args)
                .map_err(msg)?;
        let rnnt = RnntDecoder::from_weights(
            &weights,
            "stt_model.rnnt_decoder",
            "stt_model.rnnt_joint",
            &predict_args,
            &joint_args,
        )
        .map_err(msg)?;
        let mut tts = RvqEarTtsModel::from_weights(&weights, "tts_model.tts_model", &tts_config)
            .map_err(msg)?;
        let assets =
            SpeechDecoderAssets::from_weights(&weights, "tts_model", &tts_config).map_err(msg)?;
        let codec = NemotronCodec::from_weights(&weights, "tts_model.audio_codec", &codec_config)
            .map_err(msg)?;
        weights.retain(|k, _| {
            !k.starts_with("tts_model.")
                && !k.starts_with("stt_model.perception.")
                && !k.starts_with("stt_model.rnnt_")
        });
        let lm = VoiceChatLanguageModel::from_weights(
            text_config,
            config.default_quantization(),
            &mut weights,
            config.function_channel_weight,
        )
        .map_err(msg)?;
        drop(weights);

        let tokenizer = load_tokenizer(model_path)?;
        let vocab = tokenizer
            .hf_tokenizer()
            .ok_or_else(|| {
                anyhow::anyhow!("nemotron_voicechat requires a HuggingFace tokenizer.json")
            })?
            .get_vocab(true);
        tts.set_vocabulary(&vocab).map_err(msg)?;

        let rnnt_vocabulary = if config.rnnt_vocabulary.is_empty() {
            let path = model_path.join("rnnt_tokenizer").join("vocab.json");
            if path.is_file() {
                load_vocabulary_json(&path).map_err(msg)?
            } else {
                Vec::new()
            }
        } else {
            config.rnnt_vocabulary.clone()
        };

        Ok(Self {
            config,
            mel_args,
            perception,
            rnnt,
            rnnt_vocabulary,
            max_symbols,
            lm,
            tts,
            tts_config,
            assets,
            codec,
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
