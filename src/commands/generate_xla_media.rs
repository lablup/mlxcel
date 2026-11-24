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

//! OpenXLA CLI media preparation, run before the static shape is compiled.
//!
//! The CLI compiles exactly one context shape per run, so it prepares the
//! request's images first and sizes that shape from the prompt length the
//! preprocessor produced (issue #3361), the same length the serve worker routes
//! on. See [`mlxcel::multimodal::xla_image_prefill`] for the rule and for the
//! family bounds that stay fail-closed.

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, ensure};
use image::DynamicImage;
use mlxcel::models::ModelType;
use mlxcel::{HostMultimodalPreprocessor, PreparedImagePrefill};
#[cfg(feature = "xla-iree")]
use mlxcel_core::session::PreparedPrefill;

/// A request's media, prepared as far as it can be before a session exists.
pub(super) enum XlaCliMedia {
    /// No media: the prompt is the token ids as given.
    Text,
    /// Gemma3n audio. Its runtime is loaded through the session, so it is
    /// prepared after the session is built and sized from the text prompt, as
    /// before.
    Gemma3nAudio,
    /// An image request prepared by the family preprocessor. The preprocessor
    /// is handed to the session once the shape is known.
    Image {
        preprocessor: Option<Box<dyn HostMultimodalPreprocessor>>,
        request: PreparedImagePrefill,
        /// Time spent loading the vision weights and compiling the vision
        /// graph, which sat in untimed session creation before the request was
        /// prepared first. It is not part of the reported preparation time.
        load_time: Duration,
    },
    /// Phi4MM image and/or audio, prepared by its own producer.
    #[cfg(feature = "xla-iree")]
    Phi4mm(PreparedPrefill),
}

impl XlaCliMedia {
    /// The prompt length the compiled shape and admission are sized with.
    pub(super) fn effective_prompt_len(&self, text_prompt_len: usize) -> usize {
        match self {
            Self::Text | Self::Gemma3nAudio => text_prompt_len,
            Self::Image { request, .. } => request.sequence_len(),
            #[cfg(feature = "xla-iree")]
            Self::Phi4mm(prepared) => prepared.sequence_len,
        }
    }

    /// One-time preprocessor load time, to leave out of the preparation time.
    pub(super) fn untimed_load(&self) -> Duration {
        match self {
            Self::Image { load_time, .. } => *load_time,
            _ => Duration::ZERO,
        }
    }

    /// Move the image preprocessor out so the session can own it.
    pub(super) fn take_preprocessor(&mut self) -> Option<Box<dyn HostMultimodalPreprocessor>> {
        match self {
            Self::Image { preprocessor, .. } => preprocessor.take(),
            _ => None,
        }
    }
}

/// Prepare the request's media before any OpenXLA shape is compiled.
///
/// `images` are already decoded and bounded by the configured input limits.
pub(super) fn prepare_xla_cli_media(
    model_path: &Path,
    prompt: &str,
    prompt_tokens: &[i32],
    tokenizer: &mlxcel::tokenizer::MlxcelTokenizer,
    images: &[DynamicImage],
    audio_path: Option<&Path>,
) -> Result<XlaCliMedia> {
    let model_type = mlxcel::models::get_model_type(model_path)?;
    if model_type == ModelType::Phi4MMVLM && (audio_path.is_some() || !images.is_empty()) {
        return prepare_phi4mm_media(model_path, prompt, tokenizer, images, audio_path);
    }
    if audio_path.is_some() {
        ensure!(
            images.is_empty(),
            "the loaded OpenXLA Gemma3n runtime has not qualified mixed image+audio requests"
        );
        return Ok(XlaCliMedia::Gemma3nAudio);
    }
    if images.is_empty() {
        return Ok(XlaCliMedia::Text);
    }
    let load_start = Instant::now();
    let preprocessor = mlxcel::load_xla_image_preprocessor(model_path)
        .map_err(|error| anyhow!("OpenXLA image preprocessor load failed: {error}"))?
        .ok_or_else(|| {
            anyhow!("the loaded OpenXLA model/runtime bundle does not support image input")
        })?;
    let load_time = load_start.elapsed();
    let request = mlxcel::prepare_image_prefill(preprocessor.as_ref(), prompt_tokens, images)
        .map_err(|error| anyhow!("OpenXLA image preprocessing failed: {error}"))?;
    Ok(XlaCliMedia::Image {
        preprocessor: Some(preprocessor),
        request,
        load_time,
    })
}

/// The sequence bound the Phi4MM producer enforces while preparing.
///
/// A pinned capacity keeps its early, specific refusal. Unpinned, the producer
/// is bounded by `min(checkpoint window, top of the context ladder)` (it takes
/// the minimum with the window), and the shape compiled afterwards is sized
/// from what it produced.
#[cfg(feature = "xla-iree")]
fn phi4mm_preparation_capacity() -> Result<usize> {
    if std::env::var_os(mlxcel_xla::CONTEXT_CAPACITY_ENV).is_some() {
        mlxcel_xla::context_capacity_from_env().map_err(anyhow::Error::msg)
    } else {
        // Bound by the top of the context ladder so a prompt no shape could
        // hold is refused before the producer computes its embeddings.
        Ok(mlxcel_xla::CONTEXT_CAPACITY_LADDER
            .last()
            .copied()
            .unwrap_or(mlxcel_xla::DEFAULT_CONTEXT_CAPACITY))
    }
}

#[cfg(not(feature = "xla-iree"))]
fn prepare_phi4mm_media(
    _model_path: &Path,
    _prompt: &str,
    _tokenizer: &mlxcel::tokenizer::MlxcelTokenizer,
    _images: &[DynamicImage],
    _audio_path: Option<&Path>,
) -> Result<XlaCliMedia> {
    anyhow::bail!("OpenXLA Phi4MM media input requires a build with the xla-iree feature")
}

#[cfg(feature = "xla-iree")]
fn prepare_phi4mm_media(
    model_path: &Path,
    prompt: &str,
    tokenizer: &mlxcel::tokenizer::MlxcelTokenizer,
    images: &[DynamicImage],
    audio_path: Option<&Path>,
) -> Result<XlaCliMedia> {
    use mlxcel::multimodal::phi4mm_xla_audio::{Phi4MmXlaAudioProducer, load_phi4mm_audio_policy};

    let mut tokenization_error = None;
    let normalized = mlxcel::phi4mm_prompt::prepare_phi4mm_prompt_tokens(
        prompt,
        images.len(),
        usize::from(audio_path.is_some()),
        |text, add_special| match tokenizer.encode(text, add_special) {
            Ok(tokens) => tokens.into_iter().map(|token| token as i32).collect(),
            Err(error) => {
                tokenization_error = Some(error.to_string());
                Vec::new()
            }
        },
    )
    .map_err(|error| anyhow!("Phi4MM prompt normalization failed: {error}"))?;
    if let Some(error) = tokenization_error {
        return Err(anyhow!("Phi4MM tokenization failed: {error}"));
    }
    let cancelled = std::sync::atomic::AtomicBool::new(false);
    let device = std::env::var("MLXCEL_XLA_DEVICE")
        .unwrap_or_else(|_| mlxcel_xla::default_device().to_string());
    let capacity = phi4mm_preparation_capacity()?;
    let mut producer = if images.is_empty() {
        Phi4MmXlaAudioProducer::load(model_path, &device, capacity)
    } else {
        Phi4MmXlaAudioProducer::load_multimodal(model_path, &device, capacity)
    }
    .map_err(anyhow::Error::msg)?;
    let prepared = if let Some(audio_path) = audio_path {
        let policy = load_phi4mm_audio_policy(model_path).map_err(anyhow::Error::msg)?;
        let waveforms = mlxcel::audio::preprocess_wav_file(audio_path, policy, &cancelled)
            .map_err(|error| anyhow!("OpenXLA audio waveform preprocessing failed: {error}"))?;
        producer.prepare_media(waveforms, normalized.tokens, images, &cancelled)
    } else {
        producer.prepare_images(normalized.tokens, images, &cancelled)
    }
    .map_err(|error| anyhow!("OpenXLA Phi4MM media preprocessing failed: {error}"))?;
    Ok(XlaCliMedia::Phi4mm(prepared))
}

/// Run a Gemma3n audio request, whose runtime is loaded through the session.
pub(super) fn generate_xla_gemma3n_audio(
    session: &mut mlxcel::backend::session::XlaBackendSession,
    model_path: &Path,
    prompt_tokens: &[i32],
    tokenizer: &mlxcel::tokenizer::MlxcelTokenizer,
    audio_path: &Path,
    max_tokens: usize,
    eos: &[i32],
) -> Result<Vec<i32>> {
    #[cfg(feature = "xla-iree")]
    {
        let preparer = mlxcel::audio::gemma3n::Gemma3nXlaAudioPreparer::from_model(
            model_path,
            session.context_capacity(),
            tokenizer,
        )
        .map_err(|error| anyhow!("OpenXLA audio configuration failed: {error}"))?
        .ok_or_else(|| {
            anyhow!("the loaded OpenXLA model/runtime bundle does not support audio input")
        })?;
        let cancelled = std::sync::atomic::AtomicBool::new(false);
        let waveforms =
            mlxcel::audio::preprocess_wav_file(audio_path, preparer.policy(), &cancelled)
                .map_err(|error| anyhow!("OpenXLA audio preprocessing failed: {error}"))?;
        let prepared_input = preparer
            .prepare(
                &mlxcel::audio::gemma3n::Gemma3nAudioFeatureExtractor::new(),
                waveforms,
                prompt_tokens.to_vec(),
                &cancelled,
            )
            .map_err(|error| anyhow!("OpenXLA audio feature preparation failed: {error}"))?;
        let mut runtime = session
            .load_gemma3n_audio(prepared_input.input.frame_bucket(), prepared_input.clips)
            .map_err(|error| anyhow!("OpenXLA audio runtime load failed: {error}"))?;
        let prepared = runtime
            .invoke_prepared(
                &prepared_input.input,
                prepared_input.token_ids,
                preparer.audio_token_id(),
            )
            .map_err(|error| anyhow!("OpenXLA audio execution failed: {error}"))?;
        session
            .generate_gemma3n_prepared_greedy(prepared.request(), max_tokens, eos)
            .map_err(|error| anyhow!("OpenXLA audio generation failed: {error}"))
    }
    #[cfg(not(feature = "xla-iree"))]
    {
        let _ = (
            session,
            model_path,
            prompt_tokens,
            audio_path,
            tokenizer,
            max_tokens,
            eos,
        );
        anyhow::bail!("OpenXLA audio execution requires the xla-iree feature");
    }
}
