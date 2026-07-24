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

//! Gemma3n host-mel to IREE audio producer for bounded server admission.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use mlxcel_xla::Gemma3nAudioIreeRuntime;

use super::observability::BatchObservability;
use super::xla_audio_preprocess::{
    AudioFeatureProducer, AudioPreparedRequest, AudioPreprocessLimits, AudioPreprocessStage,
};
use crate::audio::gemma3n::{Gemma3nAudioFeatureExtractor, Gemma3nXlaAudioPreparer};
use crate::audio::{AudioFamilyPolicy, AudioWaveformBatch};
use crate::tokenizer::MlxcelTokenizer;

pub(super) struct Gemma3nAudioStageConfig {
    model_path: PathBuf,
    device: String,
    context_capacity: usize,
    language_fingerprint: u64,
    preparer: Gemma3nXlaAudioPreparer,
}

impl Gemma3nAudioStageConfig {
    pub(super) fn from_model(
        model_path: &Path,
        device: &str,
        context_capacity: usize,
        language_fingerprint: u64,
        tokenizer: &MlxcelTokenizer,
    ) -> Result<Option<Self>, String> {
        let Some(preparer) =
            Gemma3nXlaAudioPreparer::from_model(model_path, context_capacity, tokenizer)?
        else {
            return Ok(None);
        };
        if language_fingerprint == 0 {
            return Err("Gemma3n audio requires a verified language bundle".to_string());
        }
        Ok(Some(Self {
            model_path: model_path.to_path_buf(),
            device: device.to_string(),
            context_capacity,
            language_fingerprint,
            preparer,
        }))
    }

    pub(super) fn policy(&self) -> AudioFamilyPolicy {
        self.preparer.policy()
    }

    pub(super) fn spawn(
        self,
        limits: AudioPreprocessLimits,
        observability: Arc<BatchObservability>,
    ) -> Result<AudioPreprocessStage, String> {
        AudioPreprocessStage::spawn_with_loader(limits, observability, move || {
            Ok(Gemma3nAudioProducer::new(self))
        })
    }
}

struct Gemma3nAudioProducer {
    config: Gemma3nAudioStageConfig,
    extractor: Gemma3nAudioFeatureExtractor,
    runtime: Option<((usize, usize), Gemma3nAudioIreeRuntime)>,
}

impl Gemma3nAudioProducer {
    fn new(config: Gemma3nAudioStageConfig) -> Self {
        Self {
            config,
            extractor: Gemma3nAudioFeatureExtractor::new(),
            runtime: None,
        }
    }

    fn runtime(
        &mut self,
        frame_bucket: usize,
        clips: usize,
    ) -> Result<&mut Gemma3nAudioIreeRuntime, String> {
        let key = (frame_bucket, clips);
        if self
            .runtime
            .as_ref()
            .is_none_or(|(loaded, _)| *loaded != key)
        {
            let runtime = Gemma3nAudioIreeRuntime::load(
                &self.config.model_path,
                &self.config.device,
                self.config.context_capacity,
                frame_bucket,
                clips,
                self.config.language_fingerprint,
            )?;
            // Keep a single shape-specialized runtime resident. This bounds
            // conditional audio weight memory even when request buckets vary.
            self.runtime = Some((key, runtime));
        }
        Ok(&mut self.runtime.as_mut().expect("runtime inserted").1)
    }
}

impl AudioFeatureProducer for Gemma3nAudioProducer {
    fn prepare(
        &mut self,
        waveforms: AudioWaveformBatch,
        token_ids: Vec<i32>,
        images: Vec<image::DynamicImage>,
        cancelled: &AtomicBool,
    ) -> Result<AudioPreparedRequest, String> {
        if !images.is_empty() {
            return Err("Gemma3n audio producer does not accept image inputs".to_string());
        }
        let prepared =
            self.config
                .preparer
                .prepare(&self.extractor, waveforms, token_ids, cancelled)?;
        if cancelled.load(Ordering::Acquire) {
            return Err("Gemma3n audio request was cancelled before IREE invocation".to_string());
        }
        let audio_token_id = self.config.preparer.audio_token_id();
        let prepared = self
            .runtime(prepared.input.frame_bucket(), prepared.clips)?
            .invoke_prepared(&prepared.input, prepared.token_ids, audio_token_id)?;
        Ok(AudioPreparedRequest::Gemma3n(prepared))
    }
}
