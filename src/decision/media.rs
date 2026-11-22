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

//! Media decoding shared by the route and the CLI.
//!
//! Audio reaches a model as 16 kHz mono samples, decoded with the same
//! container sniffing the chat audio path uses (wav, mp3, flac). The CLI
//! reads local paths directly and sends URLs and data URIs through the
//! server's guarded fetchers, so a remote image or clip obeys the same size
//! and address limits it would on `POST /v1/decisions`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};

use super::DecisionMedia;

/// Sample rate every decision model reads audio at.
pub const DECISION_AUDIO_SAMPLE_RATE: u32 = 16_000;

/// Decode one encoded clip to 16 kHz mono samples.
///
/// The error is the decoder's own message, which the route returns as a 400.
pub fn decode_audio_16k(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let (samples, rate) = crate::audio::feature_extractor::load_wav_from_bytes(bytes)?;
    if samples.is_empty() {
        return Err("audio decoded to no samples".to_string());
    }
    Ok(if rate == DECISION_AUDIO_SAMPLE_RATE {
        samples
    } else {
        crate::audio::whisper_mel::resample_to_16k(&samples, rate)
    })
}

/// Whether a CLI media argument is fetched rather than read from disk.
fn is_remote(source: &str) -> bool {
    source.starts_with("http://") || source.starts_with("https://") || source.starts_with("data:")
}

/// A CLI path argument, with an optional `file://` prefix dropped.
fn local_path(source: &str) -> &Path {
    Path::new(source.strip_prefix("file://").unwrap_or(source))
}

fn block_on<F: std::future::Future>(future: F) -> Result<F::Output> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to start the media fetch runtime")?;
    Ok(runtime.block_on(future))
}

fn load_image(source: &str) -> Result<image::DynamicImage> {
    let bytes = if is_remote(source) {
        let limits = crate::server::media::current_image_input_limits();
        block_on(crate::server::media::try_read_image_url_with_limits(
            source, limits,
        ))??
        .ok_or_else(|| anyhow!("{source}: unsupported image URL scheme"))?
    } else {
        let path = local_path(source);
        std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?
    };
    crate::multimodal::image_orientation::load_upright_from_memory(&bytes)
        .with_context(|| format!("failed to decode image {source}"))
}

fn load_audio(source: &str) -> Result<Vec<f32>> {
    let bytes = if is_remote(source) {
        block_on(crate::server::media::read_decision_audio(source))?
            .map_err(|err| anyhow!("{source}: {err}"))?
    } else {
        let path = local_path(source);
        std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?
    };
    decode_audio_16k(&bytes).map_err(|err| anyhow!("failed to decode audio {source}: {err}"))
}

/// Load the media of one `mlxcel decide` call.
pub fn load_cli_media(
    images: &[String],
    videos: &[PathBuf],
    audio: Option<&str>,
) -> Result<DecisionMedia> {
    let images = images
        .iter()
        .map(|source| load_image(source))
        .collect::<Result<Vec<_>>>()?;
    let videos = videos
        .iter()
        .map(|path| {
            if path.is_file() {
                Ok(crate::multimodal::video::VideoSource::from_path(
                    path.clone(),
                ))
            } else {
                Err(anyhow!("video file does not exist: {}", path.display()))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let audio = audio.map(load_audio).transpose()?;
    Ok(DecisionMedia {
        images,
        videos,
        audio,
    })
}
