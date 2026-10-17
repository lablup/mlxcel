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

//! Shared Gemma3n host-front-end contract for OpenXLA CLI and serving.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use mlxcel_xla::{
    GEMMA3N_AUDIO_MAX_CLIPS, GEMMA3N_AUDIO_MEL_BINS, GEMMA3N_AUDIO_SOFT_TOKENS, Gemma3nAudioInput,
    Gemma3nXlaAudioConfig, select_gemma3n_audio_frame_bucket, validate_gemma3n_audio_checkpoint,
};

use super::{Gemma3nAudioFeatureBatch, Gemma3nAudioFeatureExtractor};
use crate::audio::{AudioFamilyPolicy, AudioWaveformBatch};
use crate::tokenizer::MlxcelTokenizer;

/// Canonical request input produced before invoking the split IREE audio graph.
pub struct Gemma3nXlaPreparedAudioInput {
    pub input: Gemma3nAudioInput,
    pub token_ids: Vec<i32>,
    pub clips: usize,
}

/// Model-derived token and preprocessing policy shared by CLI and server paths.
pub struct Gemma3nXlaAudioPreparer {
    context_capacity: usize,
    policy: AudioFamilyPolicy,
    wrapper_tokens: Vec<i32>,
    audio_token_id: i32,
    boa_token_id: i32,
    eoa_token_id: i32,
    end_of_turn_token_id: Option<i32>,
}

fn bucket_audio_features(features: Gemma3nAudioFeatureBatch) -> Result<Gemma3nAudioInput, String> {
    let frame_bucket =
        select_gemma3n_audio_frame_bucket(features.frames).map_err(|error| error.to_string())?;
    let mut mel = vec![0.0f32; features.batch_size * frame_bucket * GEMMA3N_AUDIO_MEL_BINS];
    let mut valid_mask = vec![0u8; features.batch_size * frame_bucket];
    let mut frame_lengths = Vec::with_capacity(features.batch_size);
    for clip in 0..features.batch_size {
        let mut valid_frames = 0usize;
        for frame in 0..features.frames {
            let source_row = clip * features.frames + frame;
            let target_row = clip * frame_bucket + frame;
            valid_mask[target_row] = u8::from(features.valid_mask[source_row]);
            valid_frames += usize::from(features.valid_mask[source_row]);
            let source = source_row * GEMMA3N_AUDIO_MEL_BINS;
            let target = target_row * GEMMA3N_AUDIO_MEL_BINS;
            mel[target..target + GEMMA3N_AUDIO_MEL_BINS]
                .copy_from_slice(&features.features[source..source + GEMMA3N_AUDIO_MEL_BINS]);
        }
        frame_lengths.push(valid_frames);
    }
    Gemma3nAudioInput::new(mel, valid_mask, frame_lengths, frame_bucket)
        .map_err(|error| error.to_string())
}

impl Gemma3nXlaAudioPreparer {
    /// Build an exact Gemma3n audio contract when the checkpoint declares one.
    pub fn from_model(
        model_path: &Path,
        context_capacity: usize,
        tokenizer: &MlxcelTokenizer,
    ) -> Result<Option<Self>, String> {
        let Some(audio_config) = Gemma3nXlaAudioConfig::from_model_dir(model_path)? else {
            return Ok(None);
        };
        let config_path = model_path.join("config.json");
        let config_text = std::fs::read_to_string(&config_path)
            .map_err(|error| format!("read {}: {error}", config_path.display()))?;
        let config: serde_json::Value = serde_json::from_str(&config_text)
            .map_err(|error| format!("parse {}: {error}", config_path.display()))?;
        let text_hidden = config
            .get("text_config")
            .and_then(|text| text.get("hidden_size"))
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                format!(
                    "{} has no valid text_config.hidden_size",
                    config_path.display()
                )
            })?;
        validate_gemma3n_audio_checkpoint(model_path, &audio_config, text_hidden)
            .map_err(|error| error.to_string())?;

        let processor = ["preprocessor_config.json", "processor_config.json"]
            .into_iter()
            .find_map(|name| {
                let path = model_path.join(name);
                std::fs::read_to_string(path)
                    .ok()
                    .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
            });
        let mut policy = AudioFamilyPolicy::from_gemma3n_configs(
            &config,
            processor.as_ref(),
            GEMMA3N_AUDIO_SOFT_TOKENS,
        )
        .map_err(|error| error.to_string())?;
        policy.max_clips = policy.max_clips.min(GEMMA3N_AUDIO_MAX_CLIPS);

        let wrapper_tokens = tokenizer
            .encode("\n\n", false)
            .map_err(|error| format!("tokenize Gemma3n audio wrapper: {error}"))?
            .into_iter()
            .map(|token| token as i32)
            .collect::<Vec<_>>();
        if wrapper_tokens.is_empty() {
            return Err("Gemma3n audio wrapper tokenized to an empty sequence".to_string());
        }
        let token = |name: &str, fallback: i32| -> Result<i32, String> {
            config
                .get(name)
                .and_then(serde_json::Value::as_i64)
                .map_or(Ok(fallback), |value| {
                    i32::try_from(value)
                        .map_err(|_| format!("Gemma3n {name}={value} does not fit i32"))
                })
        };
        let audio_token_id = token("audio_token_id", 262_273)?;
        let boa_token_id = token("boa_token_id", 256_000)?;
        let eoa_token_id = token("eoa_token_id", 262_272)?;
        if audio_token_id != audio_config.vocab_offset + 1
            || eoa_token_id != audio_config.vocab_offset
        {
            return Err(format!(
                "Gemma3n audio token ids {audio_token_id}/{eoa_token_id} do not match \
                 audio vocab offset {}",
                audio_config.vocab_offset
            ));
        }
        let configured_soft_tokens = config
            .get("audio_soft_tokens_per_image")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(GEMMA3N_AUDIO_SOFT_TOKENS as u64);
        if configured_soft_tokens != GEMMA3N_AUDIO_SOFT_TOKENS as u64 {
            return Err(format!(
                "Gemma3n audio soft-token count {configured_soft_tokens} is unsupported; \
                 expected {GEMMA3N_AUDIO_SOFT_TOKENS}"
            ));
        }
        let end_of_turn_token_id = ["<end_of_turn>", "<turn|>"].iter().find_map(|marker| {
            tokenizer
                .encode(marker, false)
                .ok()
                .filter(|tokens| tokens.len() == 1)
                .map(|tokens| tokens[0] as i32)
        });

        Ok(Some(Self {
            context_capacity,
            policy,
            wrapper_tokens,
            audio_token_id,
            boa_token_id,
            eoa_token_id,
            end_of_turn_token_id,
        }))
    }

    #[must_use]
    pub fn policy(&self) -> AudioFamilyPolicy {
        self.policy
    }

    #[must_use]
    pub fn audio_token_id(&self) -> i32 {
        self.audio_token_id
    }

    /// Convert bounded normalized waveforms into the exact static IREE input.
    pub fn prepare(
        &self,
        extractor: &Gemma3nAudioFeatureExtractor,
        waveforms: AudioWaveformBatch,
        mut token_ids: Vec<i32>,
        cancelled: &AtomicBool,
    ) -> Result<Gemma3nXlaPreparedAudioInput, String> {
        if waveforms.family != self.policy.family {
            return Err(format!(
                "Gemma3n audio producer received {} waveforms",
                waveforms.family
            ));
        }
        let clips = waveforms
            .clips
            .into_iter()
            .map(|clip| clip.samples)
            .collect::<Vec<_>>();
        let features = extractor.extract_batch_cancellable(&clips, cancelled)?;
        if cancelled.load(Ordering::Acquire) {
            return Err("Gemma3n audio feature extraction was cancelled".to_string());
        }
        let input = bucket_audio_features(features)?;
        crate::vlm_runtime::expand_gemma3n_audio_tokens(
            &mut token_ids,
            self.audio_token_id,
            self.boa_token_id,
            self.eoa_token_id,
            clips.len(),
            GEMMA3N_AUDIO_SOFT_TOKENS,
            &self.wrapper_tokens,
            self.end_of_turn_token_id,
        )
        .map_err(|error| error.to_string())?;
        if token_ids.len() > self.context_capacity {
            return Err(format!(
                "Gemma3n audio expanded prompt has {} tokens; context capacity is {}",
                token_ids.len(),
                self.context_capacity
            ));
        }
        Ok(Gemma3nXlaPreparedAudioInput {
            input,
            token_ids,
            clips: clips.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_preserves_processor_left_padding_features_and_zeros_only_static_tail() {
        let mut features = vec![-6.25; 2 * GEMMA3N_AUDIO_MEL_BINS];
        features[GEMMA3N_AUDIO_MEL_BINS..].fill(1.5);
        let input = bucket_audio_features(Gemma3nAudioFeatureBatch {
            features,
            valid_mask: vec![false, true],
            batch_size: 1,
            frames: 2,
        })
        .unwrap();

        assert_eq!(input.frame_bucket(), 8);
        assert_eq!(
            &input.mel()[..GEMMA3N_AUDIO_MEL_BINS],
            vec![-6.25; GEMMA3N_AUDIO_MEL_BINS]
        );
        assert!(
            input.mel()[2 * GEMMA3N_AUDIO_MEL_BINS..]
                .iter()
                .all(|&value| value == 0.0)
        );
        assert_eq!(&input.valid_mask()[..2], &[0, 1]);
    }
}
